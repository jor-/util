import pathlib

import numpy as np

import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.text
import matplotlib.ticker
import mpl_toolkits.axes_grid1.axes_divider
import mpl_toolkits.axes_grid1.colorbar

import util.io.fs
import util.logging

# *** default values *** #

_DEFAULT_VALUES = {'transparent': True,
                   'make_read_only': True,
                   'overwrite': False,
                   'dpi': 800,
                   'font_size': 20}


# *** util functions *** #

def _add_colorbar(axes, axes_image, orientation='right', size='3%', pad='1.5%', colorbar=True, **kwargs):
    # plot colorbar
    if colorbar:
        # make place for colorbar with same height as plot
        axes_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(axes)
        cax = axes_divider.append_axes(orientation, size=size, pad=pad)
        # plot colorbar
        cb = mpl_toolkits.axes_grid1.colorbar.colorbar(axes_image, cax=cax, **kwargs)
        return cb
    else:
        return None


# *** plot types *** #

def data(file, data, land_value=np.nan, no_data_value=np.inf, land_brightness=0,
         use_log_scale=False, v_min=None, v_max=None, caption=None, power_limit=3,
         contours=False, contours_text_brightness=0.5, colorbar=True, colormap=None,
         overwrite=True, **kwargs):

    # init
    _set_default_kwargs(kwargs)
    _set_global_font_size_with_kwargs(**kwargs)

    # prepare data
    data = np.asanyarray(data)

    def get_masks(data, land_value=np.nan, no_data_value=0):
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

    util.logging.debug('Plotting data.')

    # reshape data
    original_shape = data.shape
    original_dim = len(original_shape)
    if original_dim == 2:
        data = data.reshape((1,) + original_shape + (1,))
    elif original_dim == 3:
        data = data.reshape((1,) + original_shape)

    # get masks
    (land_mask, no_data_mask) = get_masks(data, land_value=land_value, no_data_value=no_data_value)
    del land_value
    del no_data_value

    # convert data to float if int
    if data.dtype == np.float128 or np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64, copy=True)
    else:
        data = data.copy()

    # set land and no data with specific values
    data[land_mask] = np.nan
    data[no_data_mask] = np.nan

    # get v_min and v_max
    if v_min is None or v_max is None or contours:
        data_mask = np.logical_not(np.logical_or(no_data_mask, land_mask))
        data_min = np.min(data[data_mask])
        data_max = np.max(data[data_mask])

    if v_min is None:
        v_min = data_min
    if v_max is None:
        v_max = data_max

    if use_log_scale:
        if v_min <= 1:
            v_min = 1

    util.logging.debug('Using {} as v_min and {} as v_max.'.format(v_min, v_max))

    # splite filename
    file = pathlib.PurePath(file)
    file_extension = file.suffix
    file_root = str(file.with_suffix(''))

    # prepare no_data_array
    no_data_array = np.empty_like(data[0, :, :, 0])

    t_len = data.shape[0]
    t_len_str = str(t_len)
    z_len = data.shape[3]
    z_len_str = str(z_len)

    # set colormap
    if colormap is None:
        colormap = plt.cm.jet
    colormap.set_bad(color='w', alpha=0.0)

    # plot each layer
    for z in range(z_len):
        current_file_with_z = file_root

        # append depth to filename
        if z_len > 1:
            current_file_with_z += '_layer_' + z_len_str + '_' + str(z + 1).zfill(len(z_len_str))

        for t in range(t_len):
            current_file = current_file_with_z

            # append time to filename
            if t_len > 1:
                current_file += '_time_' + t_len_str + '_' + str(t + 1).zfill(len(t_len_str))

            current_file += file_extension

            # check if file should be saved
            if _check_file(current_file, overwrite=overwrite):

                # make no_data with 1 where no data, 0.5 where water at surface, 0 where land and nan where data (1 is white, 0 is black)
                no_data_array = no_data_array * np.nan
                no_data_array[no_data_mask[t, :, :, z]] = 1
                no_data_array[land_mask[t, :, :, z]] = (1 - land_brightness) / 2
                no_data_array[land_mask[t, :, :, 0]] = land_brightness

                # chose norm
                if use_log_scale:
                    norm = matplotlib.colors.LogNorm(vmin=v_min, vmax=v_max)
                else:
                    if issubclass(data.dtype.type, np.integer):
                        norm = matplotlib.colors.BoundaryNorm(np.arange(v_min, v_max + 1), colormap.N)
                    else:
                        norm = None

                # make figure
                fig, axes = plt.subplots(1, 1)
                if colorbar:
                    plt.subplots_adjust(left=0, bottom=0, right=0.9, top=1, wspace=0, hspace=0)
                else:
                    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

                # plot no data mask
                colormap_no_data = plt.cm.gray
                colormap_no_data.set_bad(color='w', alpha=0.0)
                axes_image = axes.imshow(no_data_array.transpose(), origin='lower', aspect='equal', cmap=colormap_no_data, vmin=0, vmax=1)

                # plot data
                current_data = data[t, :, :, z].transpose()
                axes_image = axes.imshow(current_data, origin='lower', aspect='equal', cmap=colormap, vmin=v_min, vmax=v_max, norm=norm)

                # disable axis labels
                axes.axis('off')

                # choose tick locator
                if use_log_scale and (colorbar or contours):
                    # choose tick base
                    if colorbar:
                        v_max_tick = v_max
                    else:
                        current_data_max = np.nanmax(current_data)
                        v_max_tick = min([v_max, current_data_max])
                        if current_data_max == 0:
                            contours = False
                            v_max_tick = 1
                    v_min_tick = v_min
                    tick_base_exp = int(np.ceil(np.log10(v_max_tick))) - 1
                    tick_base = 10 ** tick_base_exp

                    # decrease tick if too few data above tick
                    if (current_data[np.logical_not(np.isnan(current_data))] > tick_base).sum() < (np.logical_not(np.isnan(current_data))).sum() / 100:
                        tick_base = tick_base / 10

                    # decrease tick if too few ticks
                    if v_max_tick / tick_base - v_min_tick / tick_base < 3:
                        tick_base = tick_base / 10

                    # choose locator
                    tick_locator = plt.LogLocator(base=tick_base, subs=(tick_base / 10,))
                else:
                    tick_locator = None

                # plot colorbar
                _add_colorbar(axes, axes_image, colorbar=colorbar, ticks=tick_locator)

                # plot contours
                if contours:
                    contour_plot = plt.contour(current_data, locator=tick_locator, colors='k', linestyles=['dashed', 'solid'], linewidths=0.5, norm=norm)
                    if np.abs(tick_base_exp) >= power_limit:
                        label_fmt = '%.0e'
                    elif tick_base_exp < 0:
                        label_fmt = '%1.{:d}f'.format(np.abs(tick_base_exp))
                    else:
                        label_fmt = '%d'
                    axes.clabel(contour_plot, contour_plot.levels[1::2], fontsize=8, fmt=label_fmt, colors=str(contours_text_brightness))

                # set caption
                if caption is not None:
                    plt.xlabel(caption)

                # save and close
                _save_and_close_fig_with_kwargs(fig, current_file, **kwargs)

    util.logging.debug('Plot completed.')


def line(file, x, y,
         x_order=0, line_label=None, line_width=1, line_style='-', line_color='r', y_min=None, y_max=None, xticks=None, spine_line_width=1, use_log_scale=False,
         tick_font_size=_DEFAULT_VALUES['font_size'], legend_font_size=_DEFAULT_VALUES['font_size'], x_label=None, y_label=None, axis_label_font_size=_DEFAULT_VALUES['font_size'],
         overwrite=True, **kwargs):
    util.logging.debug('Plotting line.')

    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

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

        # make figure
        fig = plt.figure()
        x_min = np.inf
        x_max = -np.inf

        # plot each line
        for i in range(number_of_lines):
            # get line and setup
            x = np.asanyarray(xs[i])
            y = np.asanyarray(ys[i])
            line_label = line_labels[i]

            if len(x) > 0:
                if line_label is None:
                    line_label = '_nolegend_'
                line_width = line_widths[i]
                line_style = line_styles[i]
                line_color = line_colors[i]

                # sort values
                if x_order != 0:
                    sorted_indices = np.argsort(x)
                    if x_order < 0:
                        sorted_indices = sorted_indices[::-1]
                    x = x[sorted_indices]
                    y = y[sorted_indices]

                # update x_min x_max
                x_min = min([x_min, x[0]])
                x_max = max([x_max, x[-1]])

                # plot line
                plt.plot(x, y, line_style, color=line_color, linewidth=line_width, markersize=line_width * 3, label=line_label)

        # set axis labels
        if x_label is not None:
            plt.xlabel(x_label, fontsize=axis_label_font_size, fontweight='bold')
        if y_label is not None:
            plt.ylabel(y_label, fontsize=axis_label_font_size, fontweight='bold')

        # set tickts
        if xticks is not None:
            plt.xticks(xticks)
        set_tick_font(fig.gca(), size=tick_font_size)

        # set spine line_width
        set_spine_line_size(fig, spine_line_width)

        # log scale
        if use_log_scale:
            plt.yscale('log')

        # set axis limits
        if y_min is not None:
            plt.ylim(bottom=y_min)
        if y_max is not None:
            plt.ylim(top=y_max)
        plt.xlim(x_min, x_max)

        # legend
        legend = plt.legend(loc=0)
        if legend is not None:
            transparent = kwargs['transparent']
            legend.get_frame().set_alpha(float(not transparent))
            set_legend_font(fig, size=legend_font_size)

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def scatter(file, x, y, z=None, point_size=20, plot_3d=False,
            overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # check and prepare input
        assert x.ndim == 1
        assert y.ndim == 1
        assert z is None or z.ndim == 1

        # make figure
        fig, axes = plt.subplots(1, 1)

        # plot
        if z is None:
            plt.scatter(x, y, s=point_size)
        else:
            if plot_3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, s=point_size)
            else:
                axes_image = plt.scatter(x, y, c=z, s=point_size)
                _add_colorbar(axes, axes_image)

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def _get_positions_and_dataset_means_from_data(data, use_abs=False, overwrite=True):
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
    dataset = np.array(tuple(np.mean(abs_function(y[np.all(x == p, axis=1)])) for p in positions))

    return positions, dataset


def scatter_dataset_means(file, data, use_abs=False, overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):
        positions, dataset = _get_positions_and_dataset_means_from_data(data, use_abs=use_abs)
        scatter(file, *positions.T, dataset, **kwargs)
    return file


def imshow_dataset_means(file, data, use_abs=False, colorbar=True, overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # check input
        if data.ndim != 2:
            raise ValueError(f'The parameter data has to be a two dimensional array but its shape is {data.shape}.')
        if data.shape[1] != 3:
            raise ValueError(f'The second dimension of the parameter data has to be three but its shape is {data.shape}.')

        # generate positions and dataset means
        positions, dataset = _get_positions_and_dataset_means_from_data(data, use_abs=use_abs)

        # make image array with nan where no data
        positions_x, positions_x_indices = np.unique(positions[:, 0], return_inverse=True)
        positions_y, positions_y_indices = np.unique(positions[:, 1], return_inverse=True)

        n = len(positions_x)
        m = len(positions_y)
        im_array_dtype = np.promote_types(dataset.dtype, np.min_scalar_type(np.nan))
        im_array = np.ones([n, m], dtype=im_array_dtype) * np.nan
        for i, j, a in zip(positions_x_indices, positions_y_indices, dataset):
            im_array[i, j] = a

        # make figure
        fig, axes = plt.subplots(1, 1)

        # plot image data
        axes_image = plt.imshow(im_array, origin='lower', aspect='equal', interpolation='nearest')

        # get axes
        def ticks_to_labels(ticks):
            def tick_to_label(f):
                f = np.round(f, decimals=-(int(np.floor(np.log10(f))) - 4))
                return '{:g}'.format()
            return [ticks_to_labels(value) for value in ticks]

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
        _add_colorbar(axes, axes_image, colorbar=colorbar)

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)
    return file


def histogram(file, data,
              bins=None, step_size=None, x_min=None, x_max=None, weights=None,
              use_log_scale=False, type='bar', tick_font_size=_DEFAULT_VALUES['font_size'], tick_power=None, tick_number=None,
              overwrite=True, **kwargs):
    util.logging.debug('Plotting histogram.')

    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # make fig
        fig = plt.figure()

        # make bins
        if bins is None:
            if step_size is None:
                step_size = (np.percentile(data, 95) - np.percentile(data, 5)) / 10
                step_size = np.round(step_size, decimals=int(-np.floor(np.log10(step_size))))
            if x_min is None:
                x_min = np.floor(np.min(data) / step_size) * step_size
            if x_max is None:
                x_max = np.ceil(np.max(data) / step_size) * step_size
            bins = np.arange(x_min, x_max + step_size, step_size)

        # plot
        (n, bins, patches) = plt.hist(data, bins=bins, weights=weights, log=use_log_scale, histtype=type)
        plt.xlim(bins[0], bins[-1])

        # set axis label size
        if tick_power is not None:
            set_tick_power_fix(fig.gca(), axis='y', power=tick_power)
        set_tick_font(fig.gca(), size=tick_font_size)

        # set tick number
        if tick_number is not None:
            set_number_of_ticks(fig.gca(), tick_number, axis='x')

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def dense_matrix_pattern(file, A, markersize=1, axis_labels=False, colorbar=True, overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)
        font_size = kwargs['font_size']

        # make figure
        fig, axes = plt.subplots(1, 1)

        # plot matrix values
        util.logging.debug('Plotting values for matrix {!r} to file {}.'.format(A, file))
        v_abs_max = np.abs(A).max()
        axes_image = plt.imshow(A, cmap=plt.cm.bwr, interpolation='nearest', vmin=-v_abs_max, vmax=v_abs_max)
        # disable ticks
        axes.set_xticks([])
        axes.set_yticks([])
        # make colorbar
        cb = _add_colorbar(axes, axes_image, colorbar=colorbar)
        cb.ax.tick_params(labelsize=font_size)

        # set power limits
        if axis_labels:
            formatter = plt.ScalarFormatter()
            formatter.set_powerlimits((-3, 3))
            fig.gca().xaxis.set_major_formatter(formatter)
            fig.gca().yaxis.set_major_formatter(formatter)
        # disable axis labels
        else:
            plt.axis('off')

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def _draw_sparse_matrix_pattern(A, markersize=1, **kwargs):
    plt.spy(A, markersize=markersize, marker=',', markerfacecolor='k', markeredgecolor='k', markeredgewidth=0, precision='present', **kwargs)


def sparse_matrix_pattern(file, A, markersize=1, axis_labels=False, overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # make figure
        fig = plt.figure()

        # plot sparsity_pattern
        util.logging.debug('Plotting sparsity pattern for matrix {!r} with markersize {} to file {}.'.format(A, markersize, file))
        _draw_sparse_matrix_pattern(A, markersize=markersize)

        # set power limits
        if axis_labels:
            formatter = plt.ScalarFormatter()
            formatter.set_powerlimits((-3, 3))
            fig.gca().xaxis.set_major_formatter(formatter)
            fig.gca().yaxis.set_major_formatter(formatter)
        # disable axis labels
        else:
            plt.axis('off')

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def sparse_matrices_patterns_with_differences(file, A, B, markersize=1,
                                              colors=((1, 0, 0), (0, 0, 1), (0.5, 0, 0.5), (1, 0, 1)),
                                              labels=('A only nonzero', 'B only nonzero', 'A and B nonzero and unequal', 'A and B nonzero and equal'),
                                              overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # calculate sparsity patterns
        not_equal_pattern = A != B
        nonzero_pattern_A = A != 0
        nonzero_pattern_B = B != 0
        del A
        del B
        nonzero_pattern_A_or_B = nonzero_pattern_A - nonzero_pattern_B
        nonzero_pattern_A_only = nonzero_pattern_A.multiply(nonzero_pattern_A_or_B)
        nonzero_pattern_B_only = nonzero_pattern_B.multiply(nonzero_pattern_A_or_B)
        del nonzero_pattern_A_or_B
        not_equal_nonzeros_pattern = (not_equal_pattern).multiply(nonzero_pattern_A).multiply(nonzero_pattern_B)
        del nonzero_pattern_B
        equal_pattern = nonzero_pattern_A - (nonzero_pattern_A).multiply(not_equal_pattern)
        del not_equal_pattern, nonzero_pattern_A

        # make figure
        fig = plt.figure()

        # plot sparsity_pattern
        patterns = (nonzero_pattern_A_only, nonzero_pattern_B_only, not_equal_nonzeros_pattern, equal_pattern)
        for pattern, color, label in zip(patterns, colors, labels):
            if pattern.nnz > 0:
                _draw_sparse_matrix_pattern(pattern, markersize=markersize, color=color, label=label)

        plt.legend()
        plt.axis('off')

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def intervals(file, intervals, use_percent_ticks=False, caption=None, overwrite=True, **kwargs):
    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # prepare intervalls
        intervals = np.asanyarray(intervals)
        assert intervals.ndim == 2
        assert len(intervals) == 2

        # calculate data
        intervals_half_sizes = (intervals[1] - intervals[0]) / 2
        means = intervals_half_sizes + intervals[0]
        n = len(intervals_half_sizes)

        # make figure
        fig = plt.figure()

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

        # set caption
        if caption is not None:
            plt.xlabel(caption, fontsize=kwargs['font_size'], fontweight='bold')

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def violin(file, positions, dataset, overwrite=True, **kwargs):
    assert len(positions) == len(dataset)

    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # make figure
        fig = plt.figure()
        axes = plt.gca()

        # make violin plot
        if len(positions) > 1:
            widths = (positions[1:] - positions[:-1]).min() / 2
        else:
            widths = 0.5
        plot_parts = axes.violinplot(dataset, positions=positions, widths=widths, showextrema=True, showmedians=True, showmeans=False)

        # add quantile lines
        color = plot_parts['cmedians'].get_color()
        segment = plot_parts['cmedians'].get_segments()[0]
        major_percentile_line_length = segment[1, 0] - segment[0, 0]
        minor_percentile_line_length = major_percentile_line_length / 2
        for (position, data) in zip(positions, dataset):
            for percentile in np.percentile(data, [25, 75]):
                axes.hlines(percentile, position - minor_percentile_line_length / 2, position + minor_percentile_line_length / 2, color=color)

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


def percentiles(file, positions, dataset, percentiles_offsets=(2.5, 25), overwrite=True, **kwargs):
    assert len(positions) == len(dataset)

    # check if file should be saved
    if _check_file(file, overwrite=overwrite):

        # init
        _set_default_kwargs(kwargs)
        _set_global_font_size_with_kwargs(**kwargs)

        # make figure
        fig = plt.figure()
        axes = plt.gca()

        # sort and check percentiles_offsets
        percentiles_offsets = sorted(percentiles_offsets)
        assert len(percentiles_offsets) == 0 or (percentiles_offsets[0] >= 0 and percentiles_offsets[-1] < 50)

        # calculate percentiles
        n = len(percentiles_offsets)
        x = len(dataset)
        y = n * 2 + 1
        percentiles = np.empty((x, y))
        for i in range(x):
            for j in range(n):
                percentiles[i, j] = np.percentile(dataset[i], percentiles_offsets[j])
                percentiles[i, y - 1 - j] = np.percentile(dataset[i], 100 - percentiles_offsets[j])
            percentiles[i, n] = np.percentile(dataset[i], 50)

        # color
        cmap = plt.get_cmap('hot')

        def get_color(percentil_value):
            assert percentil_value >= 0 and percentil_value <= 50
            color_strength = (50 - percentil_value) / 50 * 0.9
            return cmap(color_strength)

        # plot percentiles
        for i in range(n):
            axes.fill_between(positions, percentiles[:, i], percentiles[:, y - 1 - i], color=get_color(percentiles_offsets[i]))
        # plot median
        axes.plot(positions, percentiles[:, n], color='black')

        # save and close
        _save_and_close_fig_with_kwargs(fig, file, **kwargs)


# *** kwargs functions *** #

def _set_default_kwargs(kwargs):
    for key in kwargs.keys():
        if key not in _DEFAULT_VALUES:
            raise ValueError(f'Keyword parameter {key} is unknown.')
    for (key, value) in _DEFAULT_VALUES.items():
        kwargs.setdefault(key, value)


def _save_and_close_fig_with_kwargs(*args, **kwargs):
    KEYS = ('transparent', 'make_read_only', 'overwrite', 'dpi')
    kwargs = {key: kwargs[key] for key in KEYS}
    return save_and_close_fig(*args, **kwargs)


def _set_global_font_size_with_kwargs(**kwargs):
    KEYS = ('font_size',)
    kwargs = {key: kwargs[key] for key in KEYS}
    return set_global_font_size(**kwargs)


# *** auxiliary functions *** #

def _check_file(file, overwrite=True):
    file_exists = pathlib.Path(file).exists()
    if overwrite and file_exists:
        util.io.fs.make_writable(file)
    save = overwrite or not file_exists
    return save


def save_and_close_fig(fig, file, transparent=_DEFAULT_VALUES['transparent'], make_read_only=_DEFAULT_VALUES['make_read_only'], overwrite=_DEFAULT_VALUES['overwrite'], dpi=_DEFAULT_VALUES['dpi']):
    # prepare file
    file = pathlib.Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and file.exists():
        file.unlink()
    # plot
    plt.savefig(file, transparent=transparent, dpi=dpi)
    plt.close(fig)
    # make read only
    if make_read_only:
        util.io.fs.make_read_only(file)
    util.logging.debug('Plot saved to {}.'.format(file))


def set_spine_line_size(fig, line_width):
    axes = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(line_width)


def set_font_all(fig, family='sans-serif', weight='bold', size=12):
    font_properties = matplotlib.font_manager.FontProperties(family=family, weight=weight, size=size)

    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontproperties(font_properties)


def set_tick_font(axes, family='sans-serif', weight='bold', size=12):
    font_properties = matplotlib.font_manager.FontProperties(family=family, weight=weight, size=size)
    plt.setp(axes.get_xticklabels(), fontproperties=font_properties)
    plt.setp(axes.get_xaxis().get_offset_text(), fontproperties=font_properties)
    plt.setp(axes.get_yticklabels(), fontproperties=font_properties)
    plt.setp(axes.get_yaxis().get_offset_text(), fontproperties=font_properties)


def set_tick_power_limits(axis='y', power_limits=(-2, 2)):
    plt.ticklabel_format(style='sci', axis=axis, scilimits=power_limits)


def set_tick_power_fix(axes, axis='y', power=3):
    f = FixedOrderFormatter(power)
    if axis == 'y' or axis == 'both':
        axes.yaxis.set_major_formatter(f)
    if axis == 'x' or axis == 'both':
        axes.xaxis.set_major_formatter(f)


def set_number_of_ticks(axes, number, axis='x'):
    axes.locator_params(tight=True, axis=axis, nbins=number)


class FixedOrderFormatter(matplotlib.ticker.ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""
    def __init__(self, order_of_magnitude=0, use_offset=True, use_math_text=None):
        self._order_of_magnitude = order_of_magnitude
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=use_offset, useMathText=use_math_text)
        self.orderOfMagnitude = order_of_magnitude

    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_magnitude


def set_legend_font(fig, family='sans-serif', weight='bold', size=12):
    font_properties = matplotlib.font_manager.FontProperties(family=family, weight=weight, size=size)
    axes = fig.gca()
    legend = axes.get_legend()
    plt.setp(legend.get_texts(), fontproperties=font_properties)


def set_global_font_size(font_size=_DEFAULT_VALUES['font_size']):
    util.logging.debug('Setting font size for plots to {}.'.format(font_size))
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': font_size}
    matplotlib.rc('font', **font)


def get_colors(n, colormap_name='gist_rainbow'):
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i / (n - 1)) for i in range(n)]
    return colors
