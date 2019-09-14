import pathlib

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.ticker

import util.plot.auxiliary
import util.logging


# *** plot types *** #

def data(file, data, land_value=np.nan, no_data_value=np.inf, land_brightness=0,
         use_log_scale=False, v_min=None, v_max=None,
         contours=False, contours_text_brightness=0.5, contour_power_limit=3,
         colorbar=True, colormap=None,
         **kwargs):
    # prepare data
    data = np.asanyarray(data)

    def get_masks(data, land_value, no_data_value):
        def get_value_mask(array, value):
            if value is None:
                mask = np.zeros_like(array).astype('bool')
            elif value is np.nan:
                mask = np.isnan(array)
            else:
                mask = array == value
            return mask

        land_mask = get_value_mask(data, land_value)
        no_data_mask = get_value_mask(data, no_data_value)

        return (land_mask, no_data_mask)

    util.logging.debug(f'Plotting data with v_min {v_min}, v_max {v_max}, use_log_scale {use_log_scale}, colorbar {colorbar}, contours {contours}, land_value {land_value}, no_data_value {no_data_value}.')

    # reshape data
    original_shape = data.shape
    original_dim = len(original_shape)
    if original_dim == 2:
        data = data.reshape((1,) + original_shape + (1,))
    elif original_dim == 3:
        data = data.reshape((1,) + original_shape)

    # shape of data
    t_len = data.shape[0]
    t_len_str = str(t_len)
    z_len = data.shape[3]
    z_len_str = str(z_len)

    # get masks
    (land_mask, no_data_mask) = get_masks(data, land_value=land_value, no_data_value=no_data_value)

    # convert data to float if int
    if data.dtype == np.float128 or np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64, copy=True)
    else:
        data = data.copy()

    # asserts
    assert (np.isnan(land_value) and np.all(np.isnan(data[land_mask]))) or np.all(data[land_mask] == land_value)
    assert (np.isnan(no_data_value) and np.all(np.isnan(data[no_data_mask]))) or np.all(data[no_data_mask] == no_data_value)
    del land_value, no_data_value

    # make data as masked array
    data = np.ma.masked_where(np.logical_or(land_mask, no_data_mask), data)

    # prepare filename
    file = pathlib.PurePath(file)
    file_extension = file.suffix
    file_root = str(file.with_suffix(''))
    file_add_depth_info = '{depth}' in file_root
    if z_len > 1 and not file_add_depth_info:
        file_root += '_depth_{depth}'
        file_add_depth_info = True
    file_add_time_info = '{time}' in file_root
    if t_len > 1 and not file_add_time_info:
        file_root += '_time_{time}'
        file_add_time_info = True
    file = file_root + file_extension

    # set colormap
    if colormap is None:
        colormap = plt.cm.plasma
    colormap.set_bad(color='w', alpha=0.0)

    # plot each depth
    for z in range(z_len):
        # append depth to filename
        if file_add_depth_info:
            depth_str = z_len_str + '_' + str(z + 1).zfill(len(z_len_str))
            if file_add_time_info:
                current_file_with_z = file.format(depth=depth_str, time='{time}')
            else:
                current_file_with_z = file.format(depth=depth_str)
        else:
            current_file_with_z = file

        for t in range(t_len):
            # append time to filename
            if file_add_time_info:
                time_str = t_len_str + '_' + str(t + 1).zfill(len(t_len_str))
                current_file = current_file_with_z.format(time=time_str)
            else:
                current_file = current_file_with_z

            # check if file should be saved
            def plot_function(fig):
                util.logging.debug(f'Plotting file {current_file}.')

                data_i = data[t, :, :, z]

                # make no_data with 1 where no data, 0.5 where water at surface, 0 where land and nan where data (1 is white, 0 is black)
                no_data_array = np.ma.masked_all(data_i.shape, np.float32)
                no_data_array[no_data_mask[t, :, :, z]] = 1
                no_data_array[land_mask[t, :, :, z]] = (1 - land_brightness) / 2
                no_data_array[land_mask[t, :, :, 0]] = land_brightness

                # make axes
                axes = plt.gca()

                # plot no data mask
                colormap_no_data = plt.cm.gray
                colormap_no_data.set_bad(color='w', alpha=0.0)
                axes_image = plt.imshow(no_data_array.transpose(), origin='lower', aspect='equal', cmap=colormap_no_data, vmin=0, vmax=1)

                # choose v_min and v_max
                if v_min is None:
                    v_min_i = util.plot.auxiliary.v_min(data_i)
                else:
                    v_min_i = v_min
                if v_max is None:
                    v_max_i = util.plot.auxiliary.v_max(data_i)
                else:
                    v_max_i = v_max
                if use_log_scale:
                    if v_min_i < 1:
                        v_min_i = 1

                util.logging.debug(f'Using v_min {v_min_i} and v_max {v_max_i}.')
                assert np.isfinite(v_min_i)
                assert np.isfinite(v_max_i)
                assert v_min_i <= v_max_i

                # choose norm
                if use_log_scale:
                    norm = matplotlib.colors.LogNorm(vmin=v_min_i, vmax=v_max_i)
                else:
                    if issubclass(data.dtype.type, np.integer):
                        norm = matplotlib.colors.BoundaryNorm(np.arange(v_min_i, v_max_i + 1), colormap.N)
                    else:
                        norm = None

                # plot data
                axes_image = plt.imshow(data_i.transpose(), origin='lower', aspect='equal', cmap=colormap, vmin=v_min_i, vmax=v_max_i, norm=norm)

                # disable axis labels
                axes.axis('off')

                # plot add_colorbar
                util.plot.auxiliary.add_colorbar(axes_image, colorbar=colorbar)

                # plot contours
                if contours:
                    if use_log_scale:
                        # choose tick base
                        if colorbar:
                            v_max_tick = v_max_i
                        else:
                            data_max_i = np.nanmax(data_i)
                            v_max_tick = min([v_max_i, data_max_i])
                            if data_max_i == 0:
                                v_max_tick = 1
                        v_min_tick = v_min_i
                        tick_base_exp = int(np.ceil(np.log10(v_max_tick))) - 1
                        tick_base = 10 ** tick_base_exp

                        # decrease tick if too few data above tick
                        if (data_i[np.logical_not(np.isnan(data_i))] > tick_base).sum() < (np.logical_not(np.isnan(data_i))).sum() / 100:
                            tick_base = tick_base / 10

                        # decrease tick if too few ticks
                        if v_max_tick / tick_base - v_min_tick / tick_base < 3:
                            tick_base = tick_base / 10

                        # choose locator
                        tick_locator = plt.LogLocator(base=tick_base, subs=(tick_base / 10,))

                        # choose label format
                        if np.abs(tick_base_exp) >= contour_power_limit:
                            label_fmt = '%.0e'
                        elif tick_base_exp < 0:
                            label_fmt = '%1.{:d}f'.format(np.abs(tick_base_exp))
                        else:
                            label_fmt = '%d'
                    else:
                        tick_locator = None
                        label_fmt = '%1.3f'

                    data_min_i = np.nanmin(data_i)
                    data_max_i = np.nanmax(data_i)
                    if data_min_i < data_max_i:
                        contour_plot = plt.contour(data_i.transpose(), origin='lower', locator=tick_locator, colors='k', linestyles=['dashed', 'solid'], linewidths=0.5, norm=norm)
                        axes.clabel(contour_plot, contour_plot.levels[1::2], fontsize=8, fmt=label_fmt, colors=str(contours_text_brightness))

            util.plot.auxiliary.generic(current_file, plot_function, **kwargs)

    util.logging.debug('Plot completed.')


def line(file, x, y,
         x_order=0, line_label=None, line_width=1, line_style='-', line_color='r', xticks=None, use_log_scale=False, x_label=None, y_label=None,
         **kwargs):
    try:
        kwargs['x_min']
    except KeyError:
        calculate_x_min = True
    else:
        calculate_x_min = False
    try:
        kwargs['x_max']
    except KeyError:
        calculate_x_max = True
    else:
        calculate_x_max = False

    def plot_function(fig):
        nonlocal x, y
        util.logging.debug(f'Plotting line to file {file}.')

        # check if multi line
        try:
            y[0][0]
        except LookupError:
            is_multi_line = False
        else:
            is_multi_line = True

        # check input and prepare
        if len(x) != len(y):
            if not is_multi_line:
                ValueError('For single line plot, x and y must have same length but length x is {} and length y is {}.'.format(len(x), len(y)))
            else:
                x = np.asarray(x)
                y = np.asarray(y)
                assert y.ndim == 2
                if x.ndim != 1:
                    raise ValueError('For multiline plot, if x and y have two dims, the x and y must have same length but length x is {} and length y is {}.'.format(len(x), len(y)))
                else:
                    if x.shape[0] != y.shape[1]:
                        raise ValueError('For multiline plot, if x has one dim and y has two dims, the first dim of x must have the same size as the second dim of y, but shape x is {} and shape y is {}.'.format(x.shape, y.shape))
                    else:
                        x = np.tile(x, y.shape[1]).reshape(y.shape[1], -1)

        # prepare line(s)
        if len(y) > 0:
            # multiple lines
            if is_multi_line:
                xs = x
                ys = y

                # copy line setups for each line if only one line setup passed
                number_of_lines = len(ys)
                line_setups = []
                for line_setup in (line_label, line_width, line_style, line_color):
                    if np.asarray(line_setup).ndim == 0:
                        line_setup = (line_setup,) * len(y)
                    line_setups.append(line_setup)

                line_labels = line_setups[0]
                line_widths = line_setups[1]
                line_styles = line_setups[2]
                line_colors = line_setups[3]
            # one line
            else:
                number_of_lines = 1
                xs = x.reshape((1,) + x.shape)
                ys = y.reshape((1,) + y.shape)
                line_labels = [line_label]
                line_widths = [line_width]
                line_styles = [line_style]
                line_colors = [line_color]
        else:
            number_of_lines = 0

        # init x_min and x_max
        if calculate_x_min:
            x_min = np.inf
        if calculate_x_max:
            x_max = -np.inf

        # plot each line
        for i in range(number_of_lines):
            # get line and setup
            x_i = np.asanyarray(xs[i])
            y_i = np.asanyarray(ys[i])
            line_label_i = line_labels[i]

            if len(x) > 0:
                if line_label_i is None:
                    line_label_i = '_nolegend_'
                line_width_i = line_widths[i]
                line_style_i = line_styles[i]
                line_color_i = line_colors[i]

                # sort values
                if x_order != 0:
                    sorted_indices = np.argsort(x)
                    if x_order < 0:
                        sorted_indices = sorted_indices[::-1]
                    x = x[sorted_indices]
                    y = y[sorted_indices]

                # update x_min and x_max
                if calculate_x_min:
                    x_min = min([x_min, x_i[0]])
                if calculate_x_max:
                    x_max = max([x_max, x_i[-1]])

                # plot line
                plt.plot(x_i, y_i, line_style_i, color=line_color_i, linewidth=line_width_i, markersize=line_width_i * 3, label=line_label_i)

        # set axis labels
        if x_label is not None:
            plt.xlabel(x_label, fontweight='bold')
        if y_label is not None:
            plt.ylabel(y_label, fontweight='bold')

        # set tickts
        if xticks is not None:
            plt.xticks(xticks)

        # log scale
        if use_log_scale:
            plt.yscale('log')

        # set axis limits
        if calculate_x_min:
            util.plot.auxiliary.set_axis_limits(x_min=x_min)
        if calculate_x_max:
            util.plot.auxiliary.set_axis_limits(x_max=x_max)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def fill_between(file, x, y1, y2=0, color='b', xticks=None, **kwargs):
    # set x_min and x_max
    x = np.asanyarray(x)
    try:
        kwargs['x_min']
    except KeyError:
        kwargs['x_min'] = x.min()
    try:
        kwargs['x_max']
    except KeyError:
        kwargs['x_max'] = x.max()

    # plot
    def plot_function(fig):
        plt.fill_between(x, y1, y2, color=color)
        if xticks is not None:
            plt.xticks(xticks)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def scatter(file, x, y, z=None, point_size=20, plot_3d=False,
            **kwargs):
    def plot_function(fig):
        util.logging.debug(f'Plotting scatter plot to file {file}.')

        # check and prepare input
        assert x.ndim == 1
        assert y.ndim == 1
        assert z is None or z.ndim == 1

        # plot
        if z is None:
            plt.scatter(x, y, s=point_size)
        else:
            if plot_3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, s=point_size)
            else:
                axes_image = plt.scatter(x, y, c=z, s=point_size)
                util.plot.auxiliary.add_colorbar(axes_image)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def histogram(file, data,
              bins=None, step_size=None, x_min=None, x_max=None, weights=None,
              use_log_scale=False, histtype='bar',
              density=False, add_kde=False, kde_linewidth=3,
              **kwargs):

    if add_kde and not density:
        raise ValueError('Adding the kernel density estimation (add_kde is true) only makes sense if the density is plotted (density is true)')

    def plot_function(fig):
        nonlocal bins, step_size, x_min, x_max, density, add_kde
        util.logging.debug(f'Plotting histogram to file {file}.')

        # make bins
        if bins is None:
            if step_size is None or step_size in ('variable', 'FreedmanDiaconis1992'):
                bins = 'fd'  # choose Freedman and Diaconis (1981) rule
            else:
                assert step_size > 0
                if x_min is None:
                    x_min = np.floor(np.min(data) / step_size) * step_size
                if x_max is None:
                    x_max = np.ceil(np.max(data) / step_size) * step_size
                bins = np.arange(x_min, x_max + step_size, step_size)

        # plot histogram
        (ys, bins, patches) = plt.hist(data, bins=bins, weights=weights, log=use_log_scale, histtype=histtype, density=density)
        x_min = bins[0]
        x_max = bins[-1]

        # plot kernel density estimation
        if add_kde:
            import scipy.stats
            density_function = scipy.stats.gaussian_kde(data)
            step_size = bins[1] - bins[0]
            x_min -= step_size
            x_max += step_size
            xs = np.linspace(x_min, x_max, 200)
            ys = density_function(xs)
            plt.plot(xs, ys, linewidth=kde_linewidth)

        # set plot limits
        plt.xlim(x_min, x_max)

        # disable y axis on density plot
        if density:
            plt.gca().axes.get_yaxis().set_visible(False)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def bar(file, data, x_values=None, data_labels=None, x_labels=None, width=None, use_log_scale=False, color=None, **kwargs):
    assert data.ndim <= 2
    assert data_labels is None or len(data_labels) == len(data)

    def plot_function(fig, data, x_values, width):
        if data.ndim < 2:
            data = data.reshape(1, -1)
        n, m = data.shape
        if x_values is None:
            x_values = np.arange(m)
        if width is None:
            width = 1 / (n + 1)
        assert width <= 1 / n
        ax = plt.gca()

        for i, data_i in enumerate(data):
            # choose color
            if color is not None:
                color_i = color[i]
            else:
                color_i = None
            # choose x values and parameters
            x = x_values + ((i - n / 2) * width)
            tick_label = None
            if n % 2 == 0:
                align = 'edge'
                if i == n / 2:
                    tick_label = x_labels
            else:
                x += 0.5 * width
                align = 'center'
                if i == (n - 1) / 2:
                    tick_label = x_labels
            # choose data label
            if data_labels is not None:
                data_label = data_labels[i]
            else:
                data_label = None
            # make bar plot
            ax.bar(x, data_i, width=width, align=align, label=data_label, tick_label=tick_label, log=use_log_scale, color=color_i)

    use_legend = data_labels is not None
    util.plot.auxiliary.generic(file, lambda fig: plot_function(fig, data, x_values, width), use_legend=use_legend, **kwargs)


def _get_positions_and_dataset_value_from_data(data, value_function, use_abs=False):
    if data.ndim != 2:
        raise ValueError(f'The parameter data has to be a two dimensional array but its shape is {data.shape}.')
    if data.shape[1] not in (2, 3):
        raise ValueError(f'The second dimension of the parameter data has to be two or three but its shape is {data.shape}.')

    x = data[:, :-1]
    y = data[:, -1]
    positions = np.unique(x, axis=0)
    if use_abs:
        abs_function = np.abs
    else:
        def abs_function(x):
            return x
    dataset = np.array(tuple(value_function(abs_function(y[np.all(x == p, axis=1)])) for p in positions))

    return positions, dataset


def scatter_dataset_values(file, data, value_function, use_abs=False, point_size=20, **kwargs):
    def plot_function(fig):
        util.logging.debug(f'Plotting scatter dataset values to file {file}.')

        positions, dataset = _get_positions_and_dataset_value_from_data(data, value_function, use_abs=use_abs)
        plt.scatter(positions.T[0], dataset, s=point_size)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def imshow_dataset_values(file, data, value_function, use_abs=False, colorbar=True, v_min=None, v_max=None, **kwargs):
    def plot_function(fig):
        util.logging.debug(f'Plotting dataset values to file {file}.')

        # check input
        if data.ndim != 2:
            raise ValueError(f'The parameter data has to be a two dimensional array but its shape is {data.shape}.')
        if data.shape[1] != 3:
            raise ValueError(f'The second dimension of the parameter data has to be three but its shape is {data.shape}.')

        # generate positions and dataset means
        positions, dataset = _get_positions_and_dataset_value_from_data(data, value_function, use_abs=use_abs)

        # make image array with nan where no data
        positions_x, positions_x_indices = np.unique(positions[:, 0], return_inverse=True)
        positions_y, positions_y_indices = np.unique(positions[:, 1], return_inverse=True)

        n = len(positions_x)
        m = len(positions_y)
        im_array_dtype = np.promote_types(dataset.dtype, np.min_scalar_type(np.nan))
        im_array = np.ones([n, m], dtype=im_array_dtype) * np.nan
        for i, j, a in zip(positions_x_indices, positions_y_indices, dataset):
            im_array[i, j] = a

        # plot image data
        axes_image = plt.imshow(im_array, origin='lower', aspect='equal', interpolation='nearest', vmin=v_min, vmax=v_max)

        # get axes
        axes = plt.gca()

        def ticks_to_labels(ticks):
            def tick_to_label(f):
                if f != 0:
                    f = np.round(f, decimals=-(int(np.floor(np.log10(np.abs(f)))) - 4))
                return f'{f:g}'
            return [tick_to_label(value) for value in ticks]

        # update x tick labels
        ticks = axes.get_xticks()
        mask = np.logical_and(ticks >= 0, ticks < n)
        ticks[mask] = positions_x[ticks[mask].astype(np.min_scalar_type(n))]
        axes.set_xticklabels(ticks_to_labels(ticks))

        # update y tick labels
        ticks = axes.get_yticks()
        mask = np.logical_and(ticks >= 0, ticks < m)
        ticks[mask] = positions_y[ticks[mask].astype(np.min_scalar_type(m))]
        axes.set_yticklabels(ticks_to_labels(ticks))

        # make colorbar
        util.plot.auxiliary.add_colorbar(axes_image, colorbar=colorbar)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def dense_matrix_pattern(file, A, markersize=1, axis_labels=False, colorbar=True, **kwargs):
    def plot_function(fig):
        util.logging.debug(f'Plotting values for matrix {A!r} to file {file}.')

        # plot matrix values
        v_abs_max = np.abs(A).max()
        axes_image = plt.imshow(A, cmap=plt.cm.bwr, interpolation='nearest', vmin=-v_abs_max, vmax=v_abs_max)

        # disable ticks
        axes = plt.gca()
        axes.set_xticks([])
        axes.set_yticks([])

        # make colorbar
        cb = util.plot.auxiliary.add_colorbar(axes_image, colorbar=colorbar)
        cb.ax.tick_params(labelsize=font_size)

        # set power limits
        if axis_labels:
            util.plot.auxiliary.set_tick_power_limit_scientific(power_limit=3)
        # disable axis labels
        else:
            plt.axis('off')

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def _draw_sparse_matrix_pattern(A, markersize=1, color='k', **kwargs):
    return plt.spy(A, markersize=markersize, marker=',', color=color, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0, precision='present', **kwargs)


def sparse_matrix_pattern(file, A, markersize=1, axis_labels=False, **kwargs):
    def plot_function(fig):
        util.logging.debug(f'Plotting sparsity pattern differences for matrix {A!r} with markersize {markersize} to file {file}.')

        # plot sparsity_pattern
        _draw_sparse_matrix_pattern(A, markersize=markersize)

        # set power limits
        if axis_labels:
            util.plot.auxiliary.set_tick_power_limit_scientific(power_limit=3)
        # disable axis labels
        else:
            plt.axis('off')

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def sparse_matrices_patterns_with_differences(file, A, B, markersize=1,
                                              colors=((1, 0, 0), (0, 0, 1), (0.5, 0, 0.5), (1, 0, 1)),
                                              labels=('A only nonzero', 'B only nonzero', 'A and B nonzero and unequal', 'A and B nonzero and equal'),
                                              **kwargs):
    def plot_function(fig, use_legend=True):
        nonlocal A, B
        util.logging.debug(f'Plotting sparsity pattern differences for matrix {A!r} and {B!r} with markersize {markersize} to file {file}.')

        # calculate sparsity patterns
        pattern_not_equal = A != B
        pattern_A_nonzero = A != 0
        pattern_B_nonzero = B != 0
        del A
        del B
        pattern_A_xor_B_nonzero = pattern_A_nonzero - pattern_B_nonzero
        pattern_only_A_nonzero = pattern_A_nonzero.multiply(pattern_A_xor_B_nonzero)
        assert pattern_only_A_nonzero.nnz <= pattern_A_xor_B_nonzero.nnz
        pattern_only_B_nonzero = pattern_B_nonzero.multiply(pattern_A_xor_B_nonzero)
        assert pattern_only_B_nonzero.nnz <= pattern_A_xor_B_nonzero.nnz
        del pattern_A_xor_B_nonzero
        pattern_not_equal_both_nonzero = (pattern_not_equal).multiply(pattern_A_nonzero).multiply(pattern_B_nonzero)
        assert pattern_not_equal_both_nonzero.nnz <= pattern_not_equal.nnz
        assert pattern_not_equal_both_nonzero.nnz <= pattern_A_nonzero.nnz
        assert pattern_not_equal_both_nonzero.nnz <= pattern_B_nonzero.nnz
        del pattern_not_equal, pattern_B_nonzero
        pattern_equal_both_nonzero = pattern_A_nonzero - pattern_not_equal_both_nonzero
        del pattern_A_nonzero

        # plot sparsity_pattern
        patterns = (pattern_only_A_nonzero, pattern_only_B_nonzero, pattern_not_equal_both_nonzero, pattern_equal_both_nonzero)
        legend_handles = []
        for pattern, color, label in zip(patterns, colors, labels):
            if pattern.nnz > 0:
                _draw_sparse_matrix_pattern(pattern, markersize=markersize, color=color, label=label)
                legend_handles.append(matplotlib.patches.Patch(color=color, label=label))

        plt.axis('off')

        if len(legend_handles) > 0:
            util.plot.auxiliary.add_legend(handles=legend_handles, use_legend=use_legend, transparent=True)

    try:
        use_legend = kwargs['use_legend']
    except KeyError:
        use_legend = True
    else:
        del kwargs['use_legend']
    util.plot.auxiliary.generic(file, lambda fig: plot_function(fig, use_legend=use_legend), **kwargs)


def intervals(file, intervals, use_percent_ticks=False, **kwargs):
    def plot_function(fig):
        util.logging.debug(f'Plotting intervals to file {file}.')

        # prepare intervalls
        intervals_array = np.asanyarray(intervals)
        assert intervals_array.ndim == 2
        assert len(intervals_array) == 2

        # calculate data
        intervals_half_sizes = (intervals_array[1] - intervals_array[0]) / 2
        means = intervals_half_sizes + intervals_array[0]
        n = len(intervals_half_sizes)

        # plot
        linewidth = 3
        plt.errorbar(np.arange(1, n + 1), means, xerr=0, yerr=intervals_half_sizes, linestyle='', linewidth=linewidth, elinewidth=linewidth,
                     capsize=linewidth * 3, capthick=linewidth, marker='.', markersize=linewidth * 5)

        # set y limits
        plt.xlim(0.5, n + 0.5)

        # y tick formatter
        if use_percent_ticks:
            fmt = '%.0f%%'
            yticks = matplotlib.ticker.FormatStrFormatter(fmt)
            fig.gca().yaxis.set_major_formatter(yticks)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def violin(file, positions, dataset, **kwargs):
    assert len(positions) == len(dataset)

    def plot_function(fig):
        util.logging.debug(f'Plotting violin to file {file}.')

        # make violin plot
        if len(positions) > 1:
            widths = (positions[1:] - positions[:-1]).min() / 2
        else:
            widths = 0.5
        axes = plt.gca()
        plot_parts = axes.violinplot(dataset, positions=positions, widths=widths, showextrema=True, showmedians=True, showmeans=False)

        # add quantile lines
        color = plot_parts['cmedians'].get_color()
        segment = plot_parts['cmedians'].get_segments()[0]
        major_percentile_line_length = segment[1, 0] - segment[0, 0]
        minor_percentile_line_length = major_percentile_line_length / 2
        for (position, data) in zip(positions, dataset):
            for percentile in np.percentile(data, [25, 75]):
                axes.hlines(percentile, position - minor_percentile_line_length / 2, position + minor_percentile_line_length / 2, color=color)

    util.plot.auxiliary.generic(file, plot_function, **kwargs)


def percentiles(file, positions, dataset, percentiles_offsets=(2.5, 25), **kwargs):
    assert len(positions) == len(dataset)

    def plot_function(fig):
        util.logging.debug(f'Plotting percentiles to file {file}.')

        # sort and check percentiles_offsets
        percentiles_offsets_sorted = sorted(percentiles_offsets)
        assert len(percentiles_offsets_sorted) == 0 or (percentiles_offsets_sorted[0] >= 0 and percentiles_offsets_sorted[-1] < 50)

        # calculate percentiles
        n = len(percentiles_offsets_sorted)
        x = len(dataset)
        y = n * 2 + 1
        percentiles = np.empty((x, y))
        for i in range(x):
            for j in range(n):
                percentiles[i, j] = np.percentile(dataset[i], percentiles_offsets_sorted[j])
                percentiles[i, y - 1 - j] = np.percentile(dataset[i], 100 - percentiles_offsets_sorted[j])
            percentiles[i, n] = np.percentile(dataset[i], 50)

        # color
        cmap = plt.get_cmap('hot')

        def get_color(percentil_value):
            assert percentil_value >= 0 and percentil_value <= 50
            color_strength = (50 - percentil_value) / 50 * 0.9
            return cmap(color_strength)

        # plot percentiles
        axes = plt.gca()
        for i in range(n):
            axes.fill_between(positions, percentiles[:, i], percentiles[:, y - 1 - i], color=get_color(percentiles_offsets_sorted[i]))
        # plot median
        axes.plot(positions, percentiles[:, n], color='black')

    util.plot.auxiliary.generic(file, plot_function, **kwargs)
