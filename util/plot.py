import os.path
import subprocess

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.font_manager
import matplotlib.text
import matplotlib.ticker
import mpl_toolkits.axes_grid1

import util.io.fs
import util.logging
logger = util.logging.logger

DEFAULT_FONT_SIZE = 20


## plot types

def data(data, file, land_value=np.nan, no_data_value=np.inf, land_brightness=0, use_log_scale=False, v_min=None, v_max=None, caption=None, tick_font_size=DEFAULT_FONT_SIZE, power_limit=3, dpi=100, contours=False, contours_text_brightness=0.5, colorbar=True):
    
    if data.dtype == np.float128:
        data = data.astype(np.float64)
    

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


    logger.debug('Plotting data.')

    ## set font size
    set_global_font_size(tick_font_size)


    ## reshape data
    original_shape = data.shape
    original_dim = len(original_shape)
    if original_dim == 2:
        data = data.reshape((1,) + original_shape + (1,))
    elif original_dim == 3:
        data = data.reshape((1,) + original_shape)


    ## get masks and v_min and v_max
    (land_mask, no_data_mask) = get_masks(data, land_value=land_value, no_data_value=no_data_value)
    if v_min is None or v_max is None or contours:
        data_mask = np.logical_not(np.logical_or(no_data_mask, land_mask))
        data_min = np.min(data[data_mask])
        data_max = np.max(data[data_mask])

    ## v_min and v_max
    if v_min is None:
        v_min = data_min
    if v_max is None:
        v_max = data_max
    logger.debug('Using {} as v_min and {} as v_max.'.format(v_min, v_max))


    ## remove negative values for log plot
    if use_log_scale:
        data = np.array(data, copy=True)
        data[land_mask] = np.nan
        land_value = np.nan
        data[no_data_mask] = np.inf
        no_data_value = np.inf
        if v_min <= 1:
            v_min = 1


    ## splite filename
    file_root, file_extension = os.path.splitext(file)


    ## prepare no_data_array
    no_data_array = np.empty_like(data[0,:,:,0])

    t_len = data.shape[0]
    t_len_str = str(t_len)
    z_len = data.shape[3]
    z_len_str = str(z_len)

    for z in range(z_len):
        current_file_with_z = file_root

        ## append depth to filename
        if z_len > 1:
            current_file_with_z += '_layer_' + z_len_str + '_' + str(z+1).zfill(len(z_len_str))

        for t in range(t_len):
            current_file = current_file_with_z

            ## append time to filename
            if t_len > 1:
                current_file += '_time_' + t_len_str + '_' + str(t+1).zfill(len(t_len_str))

            current_file += file_extension

            ## make no_data with 1 where no data, 0.5 where water at surface, 0 where land and nan where data (1 is white, 0 is black)
            no_data_array = no_data_array * np.nan
            no_data_array[no_data_mask[t,:,:,z]] = 1
            no_data_array[land_mask[t,:,:,z]] = (1 + 9 * land_brightness) / 10
            no_data_array[land_mask[t,:,:,0]] = land_brightness
            

            ## make figure
            fig = plt.figure()

            ## chose colormap
            colormap = plt.cm.jet
            colormap.set_bad(color='w', alpha=0.0)

            ## chose norm
            if use_log_scale:
                norm = matplotlib.colors.LogNorm(vmin=v_min, vmax=v_max)
            else:
                if issubclass(data.dtype.type, np.integer):
                    norm = matplotlib.colors.BoundaryNorm(np.arange(v_min, v_max+1), colormap.N)
                else:
                    norm = None
                

            ## plot no data mask
            colormap_no_data = plt.cm.gray
            colormap_no_data.set_bad(color='w', alpha=0.0)
            axes_image = plt.imshow(no_data_array.transpose(), origin='lower', aspect='equal', cmap=colormap_no_data, vmin=0, vmax=1)

            ## plot data
            current_data = data[t,:,:,z].transpose()
            axes_image = plt.imshow(current_data, origin='lower', aspect='equal', cmap=colormap, vmin=v_min, vmax=v_max, norm=norm)

            ## disable axis labels
            plt.axis('off')
            
            ## chose tick locator
            if colorbar or contours:
                ## chose tick base
                if colorbar:
                    v_max_tick = v_max
                else:
                    current_data_max = np.nanmax(current_data)
                    v_max_tick = min([v_max, current_data_max])
                    if current_data_max == 0:
                        contours = False
                        v_max_tick = 1
                tick_base_exp = int(np.ceil(np.log10(v_max_tick))) - 1
                tick_base = 10 ** tick_base_exp
                
                if (current_data > tick_base).sum() < (~np.isnan(current_data)).sum() / 100:    # decrease tick if too few data above tick
                    tick_base = tick_base / 10
                
                ## chose locator
                if use_log_scale:
                    tick_locator = plt.LogLocator(base=tick_base, subs=tick_base/10)
                else:
                    tick_locator = plt.MultipleLocator(base=tick_base)

            ## plot colorbar
            if colorbar:
                divider = mpl_toolkits.axes_grid1.make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="3%", pad=0.1) 
                               
                cb = plt.colorbar(axes_image, cax=cax)
                cb.locator = tick_locator
                cb.update_ticks()
                
                if not use_log_scale:
                    cb.formatter.set_powerlimits((-power_limit, power_limit))
                    cb.update_ticks()
            
            ## plot contours
            if contours:
                contour_plot = plt.contour(current_data, locator=tick_locator, colors='k', linestyles=['dashed', 'solid'], linewidths=0.5, norm=norm)
                if np.abs(tick_base_exp) >= power_limit:
                    label_fmt = '%.0e'
                elif tick_base_exp < 0:
                    label_fmt = '%1.{:d}f'.format(np.abs(tick_base_exp))
                else:
                    label_fmt = '%d'
                plt.clabel(contour_plot, contour_plot.levels[1::2], fontsize=8, fmt=label_fmt, colors=str(contours_text_brightness))
            
    
            ## set caption
            if caption is not None:
                plt.xlabel(caption, fontsize=font_size, fontweight='bold')

            ## save and close
            save_and_close_fig(fig, current_file, dpi=dpi)

    logger.debug('Plot completed.')





def line(x, y, file, x_order=0, line_label=None, line_width=1, line_style='-', line_color='r', y_min=None, y_max=None, xticks=None, spine_line_width=1, use_log_scale=False, transparent=True, tick_font_size=DEFAULT_FONT_SIZE, legend_font_size=DEFAULT_FONT_SIZE, x_label=None, y_label=None, axis_label_font_size=DEFAULT_FONT_SIZE, dpi=800):
    logger.debug('Plotting line.')
    
    ## check if multi line
    try:
        y[0][0]
    except LookupError:
        is_multi_line = False
    else:
        is_multi_line = True
    
    ## check input and prepare
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
        

    ## prepare line(s)
    if len(y) > 0:
        ## multiple lines
        if is_multi_line:
            xs = x
            ys = y
    
            ## copy line setups for each line if only one line setup passed
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
        ## one line
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

    ## make figure
    fig = plt.figure()
    x_min = np.inf
    x_max = -np.inf

    ## plot each line
    for i in range(number_of_lines):
        ## get line and setup
        x = np.asanyarray(xs[i])
        y = np.asanyarray(ys[i])
        line_label = line_labels[i]

        if len(x) > 0:
            if line_label is None:
                line_label = '_nolegend_'
            line_width = line_widths[i]
            line_style = line_styles[i]
            line_color = line_colors[i]

            ## sort values
            if x_order != 0:
                sorted_indices = np.argsort(x)
                if x_order < 0:
                    sorted_indices = sorted_indices[::-1]
                x = x[sorted_indices]
                y = y[sorted_indices]

            ## update x_min x_max
            x_min = min([x_min, x[0]])
            x_max = max([x_max, x[-1]])

            ## plot line
            plt.plot(x, y, line_style, color=line_color, linewidth=line_width, markersize=line_width*3, label=line_label)


    ## set axis labels
    if x_label is not None:
        plt.xlabel(x_label, fontsize=axis_label_font_size, fontweight='bold')
    if y_label is not None:
        plt.ylabel(y_label, fontsize=axis_label_font_size, fontweight='bold')

    ## set tickts
    if xticks is not None:
        plt.xticks(xticks)
    set_tick_font(fig.gca(), size=tick_font_size)

    ## set spine line_width
    set_spine_line_size(fig, spine_line_width)

    ## log scale
    if use_log_scale:
        plt.yscale('log')

    ## set axis limits
    if y_min is not None:
        plt.ylim(ymin=y_min)
    if y_max is not None:
        plt.ylim(ymax=y_max)
    plt.xlim(x_min, x_max)

    ## legend
    legend = plt.legend(loc=0)
    if legend is not None:
        legend.get_frame().set_alpha(float(not transparent))
        set_legend_font(fig, size=legend_font_size)

    ## save and close
    save_and_close_fig(fig, file, dpi=dpi)



def scatter(x, y, file, point_size=20, dpi=800):
    ## check and prepare input
    if x.ndim == 2 and x.shape[1] > 2:
        raise ValueError('Scatter plots for x dim {} is not supported.'.format(x.shape[1]))
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:,0]

    ## make figure
    fig = plt.figure()

    ## plot
    if x.ndim == 1:
        plt.scatter(x, y, s=point_size)
    if x.ndim == 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:,0], x[:,1], y, s=point_size)

    ## save and close
    save_and_close_fig(fig, file, dpi=dpi)



def histogram(data, file, bins=None, step_size=None, x_min=None, x_max=None, weights=None, use_log_scale=False, type='bar', tick_font_size=DEFAULT_FONT_SIZE, tick_power=None, dpi=800):
    logger.debug('Plotting histogram.')

    ## make fig
    fig = plt.figure()

    ##
    if bins is None:
        if step_size is None:
            raise ValueError('Either "bins" or "step_size" has to be defined.')
        if x_min is None:
            x_min = np.floor(np.min(data) / step_size) * step_size
        if x_max is None:
            x_max = np.ceil(np.max(data) / step_size) * step_size
        bins = np.arange(x_min, x_max+step_size, step_size)

    ## plot
    (n, bins, patches) = plt.hist(data, bins=bins, weights=weights, log=use_log_scale, histtype=type)
    plt.xlim(bins[0], bins[-1])

    ## set axis label size
    if tick_power is not None:
        set_tick_power_fix(fig.gca(), axis='y', power=tick_power)
    set_tick_font(fig.gca(), size=tick_font_size)

    ## save and close
    save_and_close_fig(fig, file, dpi=dpi)



def spy(A, file, markersize=1, axis_labels=True, caption=None, font_size=DEFAULT_FONT_SIZE, dpi=800):
    import scipy.sparse

    ## set font size
    set_global_font_size(font_size)

    ## make figure
    fig = plt.figure()
    
    ## plot sparsity_pattern
    if scipy.sparse.issparse(A):
        logger.debug('Plotting sparsity pattern for matrix {!r} with markersize {} and dpi {} to file {}.'.format(A, markersize, dpi, file))
        plt.spy(A, markersize=markersize, marker=',', markeredgecolor='k', markerfacecolor='k')
    
    ## plot matrix values
    else:
        logger.debug('Plotting values for matrix {!r} with dpi {} to file {}.'.format(A, dpi, file))
        v_abs_max = np.abs(A).max()
        axes_image = plt.imshow(A, cmap=plt.cm.bwr, interpolation='nearest', vmin=-v_abs_max, vmax=v_abs_max)   
        cb = fig.colorbar(axes_image)
        cb.ax.tick_params(labelsize=font_size) 
        fig.gca().set_xticks([]) 
        fig.gca().set_yticks([]) 

    ## set power limits
    if axis_labels:
        formatter = plt.ScalarFormatter()
        formatter.set_powerlimits((-3,3))
        fig.gca().xaxis.set_major_formatter(formatter)
        fig.gca().yaxis.set_major_formatter(formatter)
    
    ## disable axis labels
    else:
        plt.axis('off')
    
    ## set caption
    if caption is not None:
        plt.xlabel(caption, fontsize=font_size, fontweight='bold')
    
    ## save and close
    save_and_close_fig(fig, file, dpi=dpi)




def intervals(intervals, file, use_percent_ticks=False, caption=None, font_size=DEFAULT_FONT_SIZE, dpi=800):
    intervals = np.asanyarray(intervals)
    assert intervals.ndim == 2
    assert len(intervals) == 2
    
    ## calculate data
    intervals_half_sizes = (intervals[1] - intervals[0]) / 2
    means = intervals_half_sizes + intervals[0]
    n = len(intervals_half_sizes)

    ## set font size
    set_global_font_size(font_size)

    ## make figure
    fig = plt.figure()
    
    ## plot
    linewidth = 3
    plt.errorbar(np.arange(1, n+1), means, xerr=0, yerr=intervals_half_sizes, linestyle='', linewidth=linewidth, elinewidth=linewidth, capsize=linewidth*3, capthick=linewidth, marker='.', markersize=linewidth*5)
    
    ## set y limits
    plt.xlim(0.5, n + 0.5)
    
    ## y tick formatter
    if use_percent_ticks:
        fmt = '%.0f%%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        fig.gca().yaxis.set_major_formatter(yticks)

    ## set caption
    if caption is not None:
        plt.xlabel(caption, fontsize=font_size, fontweight='bold')
    
    ## save and close
    save_and_close_fig(fig, file, dpi=dpi)




## auxiliary functions

def trim(file):
    logger.debug('Trimming plot %s.' % file)
    subprocess.check_output(('convert', '-trim', file, file))


def save_and_close_fig(fig, file, transparent=True, dpi=800, make_read_only=True):
    plt.tight_layout()
    plt.savefig(file, bbox_inches='tight', transparent=transparent, dpi=dpi)
    plt.close(fig)
    trim(file)
    if make_read_only:
        util.io.fs.make_read_only(file)
    logger.debug('Plot saved to {}.'.format(file))



def set_spine_line_size(fig, line_width):
    axes = fig.gca()
    for axis in ['top','bottom','left','right']:
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


def set_tick_power_limits(axis='y', power_limits=(-2,2)):
    plt.ticklabel_format(style='sci', axis=axis, scilimits=power_limits)

def set_tick_power_fix(axes, axis='y', power=3):
    f = FixedOrderFormatter(power)
    if axis == 'y' or axis == 'both':
        axes.yaxis.set_major_formatter(f)
    if axis == 'x' or axis == 'both':
        axes.xaxis.set_major_formatter(f)

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


def set_global_font_size(size=20):
    logger.debug('Setting font size for plots to {}.'.format(size))

    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : size
    }
    matplotlib.rc('font', **font)


def get_colors(n, colormap_name='gist_rainbow'):
    colormap = plt.get_cmap(colormap_name)
    colors = [colormap(i/(n-1)) for i in range(n)]
    return colors

