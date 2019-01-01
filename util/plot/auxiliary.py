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
            overwrite=True, make_read_only=True, dpi=800):

    # check if file should be saved
    if check_file(file, overwrite=overwrite):

        # font size
        set_global_font_size(font_size=font_size)

        # make fig
        fig = plt.figure()

        # make plot
        plot_function(fig)

        # power limits
        set_tick_power_limit_scientific(tick_power_limit_scientific, axis='both')
        set_tick_power_limit_scientific(tick_power_limit_scientific_x, axis='x')
        set_tick_power_limit_scientific(tick_power_limit_scientific_y, axis='y')
        set_tick_power_limit_fix(tick_power_limit_fix, axis='both')
        set_tick_power_limit_fix(tick_power_limit_fix_x, axis='x')
        set_tick_power_limit_fix(tick_power_limit_fix_y, axis='y')

        # set number of ticks
        set_number_of_ticks(tick_number, axis='both')
        set_number_of_ticks(tick_number_x, axis='x')
        set_number_of_ticks(tick_number_y, axis='y')

        # set caption
        if caption is not None:
            plt.xlabel(caption, fontsize=font_size, fontweight='bold')

        # legend
        if use_legend:
            legend = plt.legend(loc='best')
            if legend is not None:
                legend.get_frame().set_alpha(float(not transparent))

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
    plt.savefig(file, transparent=transparent, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    # make read only
    if make_read_only:
        util.io.fs.make_read_only(file)
    util.logging.debug('Plot saved to {}.'.format(file))


def set_global_font_size(font_size):
    if font_size is not None:
        util.logging.debug(f'Setting global font size to {font_size}.')
        font = {'family': 'sans-serif',
                'weight': 'bold',
                'size': font_size}
        matplotlib.rc('font', **font)


def _get_axes_list(axes=None):
    if axes is None:
        axes = [plt.gca()]
    try:
        len(axes)
    except TypeError:
        axes = [axes]
    return axes


def set_tick_power_limit_scientific(power_limit, axis='both', axes=None):
    if power_limit is not None:
        # set axes
        axes = _get_axes_list(axes=axes)
        # set axis
        if axis is None:
            axis = 'both'
        # set power limits
        try:
            length = len(power_limit)
        except TypeError:
            power_limit = (-power_limit, power_limit)
        else:
            if length != 2:
                raise ValueError(f'power_limit has to be a tuple with two values but it is {power_limit}.')
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


def set_number_of_ticks(number, axis='both', axes=None):
    if number is not None:
        # set axes
        axes = _get_axes_list(axes=axes)
        # set axis
        if axis is None:
            axis = 'both'
        # apply tick number
        for axes_i in axes:
            axes_i.locator_params(tight=True, axis=axis, nbins=number)


def get_colors(n, colormap_name='gist_rainbow'):
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i / (n - 1)) for i in range(n)]
    return colors


def add_colorbar(axes_image, orientation='right', size='3%', pad='1.5%', colorbar=True, axes=None):
    # plot colorbar
    if colorbar:
        if axes is None:
            axes = plt.gca()
        # make place for colorbar with same height as plot
        axes_divider = mpl_toolkits.axes_grid1.axes_divider.make_axes_locatable(axes)
        cax = axes_divider.append_axes(orientation, size=size, pad=pad)
        # plot colorbar
        cb = plt.colorbar(axes_image, cax=cax)
        return cb
    else:
        return None


def _v_min_max(data, percentile, significant_digits=2):
    data = data[~np.isnan(data)]
    assert np.all(np.isfinite(data))
    v_max = np.percentile(data, percentile)
    if significant_digits is not None:
        exp = np.log10(v_max)
        exp = - np.sign(exp) * np.round(np.abs(exp)) + significant_digits
        v_max = np.round(v_max * 10**exp) * 10**-exp
    return v_max


def v_min(data, percentile=1, significant_digits=2):
    return _v_min_max(data, percentile, significant_digits=significant_digits)


def v_max(data, percentile=99, significant_digits=2):
    return _v_min_max(data, percentile, significant_digits=significant_digits)