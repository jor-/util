import pathlib

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import mpl_toolkits.axes_grid1.axes_divider

import util.io.fs
import util.logging


def generic(file, plot_function, font_size=20, transparent=True, caption=None, use_legend=False,
            tick_power_limit_scientific=None, tick_power_limit_scientific_x=None, tick_power_limit_scientific_y=None,
            tick_power_limit_fix=None, tick_power_limit_fix_x=None, tick_power_limit_fix_y=None,
            tick_number=None, tick_number_x=None, tick_number_y=None,
            tick_interger_only_x=False, tick_interger_only_y=False,
            tick_transform_x=None, tick_transform_y=None,
            x_min=None, x_max=None, y_min=None, y_max=None,
            overwrite=True, make_read_only=True, dpi=800, backend=None,
            invert_x_axis=False, invert_y_axis=False,
            colorbar=False, colorbar_tick_number=None, colorbar_tick_step_size=None,
            colorbar_tick_transform=None, colorbar_tick_power_limit_scientific=None):

    # check if file should be saved
    if check_file(file, overwrite=overwrite):
        # use backend
        if backend is not None:
            plt.switch_backend(backend)

        # font size
        set_global_font_size(font_size=font_size)

        # make fig
        fig = plt.figure()

        # make plot
        plot_function(fig)

        # get axes
        axes = fig.gca()

        # power limits
        set_tick_power_limit_scientific(tick_power_limit_scientific, axis='both')
        set_tick_power_limit_scientific(tick_power_limit_scientific_x, axis='x')
        set_tick_power_limit_scientific(tick_power_limit_scientific_y, axis='y')
        set_tick_power_limit_fix(tick_power_limit_fix, axis='both')
        set_tick_power_limit_fix(tick_power_limit_fix_x, axis='x')
        set_tick_power_limit_fix(tick_power_limit_fix_y, axis='y')

        # set number of ticks
        ticks_set_number(tick_number, axis='both')
        ticks_set_number(tick_number_x, axis='x')
        ticks_set_number(tick_number_y, axis='y')

        # set integer only for ticks
        ticks_set_integer_only(tick_interger_only_x, axis='x')
        ticks_set_integer_only(tick_interger_only_y, axis='y')

        # set caption
        if caption is not None:
            plt.xlabel(caption, fontsize=font_size, fontweight='bold')

        # legend
        add_legend(use_legend=use_legend, transparent=transparent)

        # set axis limits
        set_axis_limits(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # invert axis
        invert_axis(x_axis=invert_x_axis, y_axis=invert_y_axis, axes=axes)

        # transform tick labels
        ticks_transform_labels(tick_transform_x=tick_transform_x, tick_transform_y=tick_transform_y, axes=axes)

        # add colorbar
        if colorbar_tick_number is None:
            colorbar_tick_number = tick_number
        if colorbar_tick_power_limit_scientific is None:
            colorbar_tick_power_limit_scientific = tick_power_limit_scientific
        add_colorbar(axes=axes, colorbar=colorbar,
                     tick_number=colorbar_tick_number, tick_step_size=colorbar_tick_step_size,
                     tick_transform=colorbar_tick_transform, tick_power_limit_scientific=colorbar_tick_power_limit_scientific)

        # save and close
        save_and_close_fig(fig, file, transparent=transparent, make_read_only=make_read_only, overwrite=overwrite, dpi=dpi)


def check_file(file, overwrite=True):
    file_exists = pathlib.Path(file).exists()
    if overwrite and file_exists:
        util.io.fs.make_writable(file)
    save = overwrite or not file_exists
    return save


def save_and_close_fig(fig, file, transparent=True, make_read_only=True, overwrite=False, dpi=800):
    # prepare file
    file = pathlib.Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and file.exists():
        file.unlink()
    # plot
    fig.savefig(file, transparent=transparent, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    # make read only
    if make_read_only:
        util.io.fs.make_read_only(file)
    util.logging.debug('Plot saved to {}.'.format(file))


def set_global_font_size(font_size=None, weight='bold'):
    if font_size is not None:
        util.logging.debug(f'Setting global font size to {font_size}.')
        font_size = int(font_size)
        matplotlib.rcParams.update({'font.size': font_size})
    if weight is not None:
        util.logging.debug(f'Setting global font weight {weight}.')
        matplotlib.rcParams.update({'font.weight': weight})
        if weight == 'bold':
            matplotlib.rcParams['text.latex.preamble'] = ['\\boldmath']


def _get_axes_list(axes=None):
    if axes is None:
        axes = [plt.gca()]
    try:
        len(axes)
    except TypeError:
        axes = [axes]
    return axes


def _parse_scientific_power_limit(power_limit):
    if isinstance(power_limit, str):
        power_limit = int(power_limit)
    try:
        length = len(power_limit)
    except TypeError:
        power_limit = (-power_limit, power_limit)
    else:
        if length != 2:
            raise ValueError(f'power_limit has to be a tuple with two values but it is {power_limit}.')
    return power_limit


def set_tick_power_limit_scientific(power_limit, axis='both', axes=None):
    if power_limit is not None:
        # set axes
        axes = _get_axes_list(axes=axes)
        # set axis
        if axis is None:
            axis = 'both'
        # parse power limits
        power_limit = _parse_scientific_power_limit(power_limit)
        # apply power limit
        util.logging.debug(f'Setting tick power limit scientific for axis {axis} to {power_limit}.')
        for axes_i in axes:
            axes_i.ticklabel_format(style='scientific', axis=axis, scilimits=power_limit)


def set_tick_power_limit_fix(power_limit, axis='both', axes=None):

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

    if power_limit is not None:
        # set axes
        axes = _get_axes_list(axes=axes)
        # set axis
        if axis is None:
            axis = 'both'
        # apply power limit
        util.logging.debug(f'Setting tick power limit fix for axis {axis} to {power_limit}.')
        f = FixedOrderFormatter(power_limit)
        for axes_i in axes:
            if axis == 'y' or axis == 'both':
                axes_i.yaxis.set_major_formatter(f)
            if axis == 'x' or axis == 'both':
                axes_i.xaxis.set_major_formatter(f)


def ticks_set_number(number, axis='both', axes=None):
    if number is not None:
        number = int(number)
        # set axes
        axes = _get_axes_list(axes=axes)
        # set axis
        if axis is None:
            axis = 'both'
        # apply tick number
        for axes_i in axes:
            axes_i.locator_params(axis=axis, tight=True, nbins=number + 1, min_n_ticks=number)


def ticks_set_integer_only(integer_only, axis='both', axes=None):
    integer_only = bool(integer_only)
    if integer_only:
        # set axes
        axes = _get_axes_list(axes=axes)
        # set axis
        if axis is None:
            axis = 'both'
        # apply tick number
        for axes_i in axes:
            axes_i.locator_params(axis=axis, integer=integer_only)


def set_axis_limits(x_min=None, x_max=None, y_min=None, y_max=None):
    if x_min is not None:
        plt.xlim(left=x_min)
    if x_max is not None:
        plt.xlim(right=x_max)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)


def get_colors(n, colormap_name='gist_rainbow'):
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i / (n - 1)) for i in range(n)]
    return colors


def ticks_set_number_for_locator(locator, tick_number=None):
    if tick_number is not None:
        tick_number = int(tick_number)
        # LinearLocator, LogLocator, SymmetricalLogLocator
        try:
            locator.set_params(numticks=tick_number)
        except TypeError:
            pass
        # MaxNLocator
        try:
            locator.set_params(nbins=tick_number + 1)
            locator.set_params(min_n_ticks=tick_number)
        except TypeError:
            pass


def add_colorbar(axes_image=None, axes=None, colorbar=True,
                 orientation='right', size='3%', pad='1.5%',
                 tick_number=None, tick_step_size=None, tick_transform=None, tick_power_limit_scientific=None):
    # plot colorbar
    if colorbar:
        if axes is None:
            axes = plt.gca()
        if axes_image is None:
            axes_images = axes.get_images()
            axes_image = axes_images[-1]
        # make place for colorbar with same height as plot
        axes_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(axes)
        cax = axes_divider.append_axes(orientation, size=size, pad=pad)
        # plot colorbar
        cb = plt.colorbar(axes_image, cax=cax)
        # set ticks
        if tick_number is not None and tick_step_size is not None:
            raise ValueError('Please use only tick_number or tick_step_size, not both.')
        if tick_number is not None:
            ticks_set_number_for_locator(cb.locator, tick_number=tick_number)
            cb.update_ticks()
        if tick_step_size is not None:
            tick_step_size = float(tick_step_size)
            locator = matplotlib.ticker.MultipleLocator(base=tick_step_size)
            cb.locator = locator
            cb.update_ticks()
        if tick_transform is not None:
            ticks = cb.ax.get_yticks()
            tick_lables = [tick_transform(tick) for tick in ticks]
            cb.ax.set_yticklabels(tick_lables)
        if tick_power_limit_scientific is not None:
            tick_power_limit_scientific = _parse_scientific_power_limit(tick_power_limit_scientific)
            cb.formatter.set_powerlimits(tick_power_limit_scientific)
            cb.update_ticks()
        return cb
    else:
        return None


def add_legend(labels=None, handles=None, use_legend=True, transparent=True):
    if use_legend:
        legend = plt.legend(labels=labels, handles=handles, loc='best')
        if legend is not None:
            legend.get_frame().set_alpha(float(not transparent))


def _v_min_max(data, percentile, significant_digits=2):
    data = data[np.isfinite(data)]
    v = np.percentile(data, percentile)
    if significant_digits is not None and v != 0:
        exp = np.log10(np.abs(v))
        exp = - np.sign(exp) * np.round(np.abs(exp)) + significant_digits
        v = np.round(v * 10**exp) * 10**-exp
    assert np.isfinite(v)
    return v


def v_min(data, percentile=1, significant_digits=2):
    return _v_min_max(data, percentile, significant_digits=significant_digits)


def v_max(data, percentile=99, significant_digits=2):
    return _v_min_max(data, percentile, significant_digits=significant_digits)


def invert_axis(x_axis=False, y_axis=False, axes=None):
    if x_axis or y_axis:
        if axes is None:
            axes = plt.gca()
        if x_axis:
            axes.invert_xaxis()
        if y_axis:
            axes.invert_yaxis()


def ticks_transform_labels(tick_transform_x=None, tick_transform_y=None, axes=None):
    if tick_transform_x or tick_transform_y:
        if axes is None:
            axes = plt.gca()

        if tick_transform_x:
            x_ticks = axes.get_xticks()
            x_tick_lables = [tick_transform_x(tick) for tick in x_ticks]
            axes.set_xticklabels(x_tick_lables)

        if tick_transform_y:
            y_ticks = axes.get_yticks()
            y_tick_lables = [tick_transform_y(tick) for tick in y_ticks]
            axes.set_yticklabels(y_tick_lables)


def tick_transform_function_exponent_notation(data, v_max=None):
    # calculate factor
    if v_max is not None:
        v_max_for_tick_factor = v_max
    else:
        v_max_for_tick_factor = util.plot.auxiliary.v_max(data)
    tick_exponent = int(np.floor(np.log10(v_max_for_tick_factor)))
    if tick_exponent >= 0:
        tick_exponent_str = f'e\\!\\!{tick_exponent}$'
    else:
        tick_exponent_str = f'e\\!\\!-\\!\\!{-tick_exponent}$'
    tick_factor = 10**tick_exponent

    # create transform function
    def tick_transform(tick):
        tick /= tick_factor
        tick = int(np.round(tick, decimals=0))
        tick_label = f'${tick}{tick_exponent_str}'
        return tick_label

    # return
    return tick_transform
