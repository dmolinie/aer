""" Shorthand core plotting functions

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2025
Last revised: April 2026

License: GPLv3
"""

__all__ = ['plot_core', 'bar_core']

import numpy as np

from aer.tools import check_keys, get_ndim, make_iter


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

def _get_x_y(x_y):
    """ Parse a tuple into 1 or 2 items to allow to omit an argument
    in function signatures using the star operator """
    if len(x_y) == 1:
        xvalues = None
        yvalues = x_y[0]
    else:
        xvalues = x_y[0]
        yvalues = x_y[1]
    return xvalues, yvalues

##############################################################################



##############################################################################
##                         Core Plotting Functions                          ##
##############################################################################

#---------------------------   1-Plot x-values   ----------------------------#
def _core_plot(axis, xvals, yvals, fct, **kwargs):
    """ Use the `fct` function to plot `yvals` against `xvals` on a figure
    `axis` and using the `kwargs` parameters. If `yvals` is 2D, plot every
    subarray against `xvals` if it is 1D, or its subarrays else; the `fct`
    function should handle the case when `xvals` is None specifically. """
    dims = (get_ndim(yvals), get_ndim(xvals))
    # 1 array to plot with the same x-axis values
    if xvals is None or dims[0] == 1 and dims[1] <= 1 or len(xvals) == len(yvals):
        fct(axis, xvals, yvals, **kwargs)
    # 1 array to plot with several x-axis values
    elif dims[0] == 1 and dims[1] >= 2:
        fct(axis, xvals[0], yvals, **kwargs) # Rare case (set for completeness)
    # N arrays to plot with the same x-axis values
    elif dims[0] >= 2 and dims[1] <= 1:
        for y in yvals:
            fct(axis, xvals, y, **kwargs)
    # N arrays to plot with different x-axis values
    elif dims[0] >= 2 and dims[1] >= 1:
        for x, y in zip(xvals, yvals):
            fct(axis, x, y, **kwargs)
#----------------------------------------------------------------------------#

#---------------------------   N-Plot x-values   ----------------------------#
def _core_plot_row(axs, xvals, yvals, fct, **kwargs):
    """ Iterate `axs`, `xvals` and `yvals` and pass every pair of these
    3 items (+ `fct`) to `_core_plot`. If `xvals` is None or 1D, pass it
    directly without iterating over. """
    # All subplots share the same x-scale
    if xvals is None or get_ndim(xvals) == 1 or len(xvals) == 1:
        for axis, ypos in zip(axs, yvals):
            _core_plot(axis, xvals, ypos, fct, **kwargs)
    # 1 (set of) x-scale(s) per axis
    else:
        for axis, xpos, ypos in zip(axs, xvals, yvals):
            _core_plot(axis, xpos, ypos, fct, **kwargs)
#----------------------------------------------------------------------------#

#--------------------------   NxM-Plot x-values   ---------------------------#
def _core_nplots(axs, xvals, yvals, fct, **kwargs):
    """ Iterate `axs`, `xvals` and `yvals` and pass every pair of these
    three items (in addition to `fct`) to `_core_plot` if `axs` contains
    only one Axis, or to `_core_plot_row` else (row by row). If `xvals`
    is None or 1D, pass it directly without iterating over. """
    # 1 axis only
    if np.ndim(axs) == 0:
        _core_plot(axs, xvals, yvals, fct, **kwargs)
    # 1D set of axes (row or column)
    elif np.ndim(axs) == 1:
        _core_plot_row(axs, xvals, yvals, fct, **kwargs)
    # 2D set of axes (NxM-plot grid)
    elif np.ndim(axs) == 2:
        # All rows share the same x-scale
        if xvals is None or get_ndim(xvals) == 1 or len(xvals) == 1:
            for axis_row, ypos in zip(axs, yvals):
                _core_plot_row(axis_row, xvals, ypos, fct, **kwargs)
        # 1 (set of) x-scale(s) per row of axes
        else:
            for axis_row, xpos, ypos in zip(axs, xvals, yvals):
                _core_plot_row(axis_row, xpos, ypos, fct, **kwargs)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                            Plotting Functions                            ##
##############################################################################

#---------------------------   1-Plot Shorthand   ---------------------------#
def _plot1d_real(axis, xvals, yvals, fmt, plot_params):
    """ Plot 1D real-valued data on the axis's graph (1 plot) """
    # If no x-values are provided, generate a vector of indexes
    if xvals is None:
        xvals = np.arange(np.shape(yvals)[-1])
    # Plot `yvals` against `xvals`
    axis.plot(xvals, yvals, fmt, **plot_params)

def _plot1d_cplx(axis, xvals, yvals, fmt, plot_params):
    """ Plot 1D complex-valued data on the axis's graph (2 plots) """
    # If no x-values are provided, generate a vector of indexes
    if xvals is None:
        xvals = np.arange(np.shape(yvals)[-1])
    # Plot `yvals` against `xvals`
    axis.plot(xvals, yvals.real, fmt, **plot_params)
    axis.plot(xvals, yvals.imag, fmt, **plot_params)

def _plot1d(axis, xvals, yvals, fmt='', **plot_params):
    """ Plot 1D real- or complex-valued data on a single Axis """
    dims = (get_ndim(yvals), get_ndim(xvals))
    # 1 array to plot with the same x-axis values
    if dims[0] == 1 and dims[1] <= 1:
        plot_fct = _plot1d_cplx if yvals.dtype == complex else _plot1d_real
        plot_fct(axis, xvals, yvals, fmt, plot_params)
    # 1 array to plot with several x-axis values
    elif dims[0] == 1 and dims[1] >= 2:
        plot_fct = _plot1d_cplx if yvals.dtype == complex else _plot1d_real
        plot_fct(axis, xvals[0], yvals, fmt, plot_params)
    # N arrays to plot with the same x-axis values
    elif dims[0] >= 2 and dims[1] <= 1:
        for ypos in yvals:
            plot_fct = _plot1d_cplx if ypos.dtype == complex else _plot1d_real
            plot_fct(axis, xvals, ypos, fmt, plot_params)
    # N arrays to plot with different x-axis values
    elif dims[0] >= 2 and dims[1] >= 1:
        for xpos, ypos in zip(xvals, yvals):
            plot_fct = _plot1d_cplx if ypos.dtype == complex else _plot1d_real
            plot_fct(axis, xpos, ypos, fmt, plot_params)
#----------------------------------------------------------------------------#

#---------------------------   N-Plot Shorthand   ---------------------------#
def _plotnd(axs, xvals, yvals, fmt='', **plot_params):
    """ Plot real- or complex-valued data on a set of Axis """
    _core_nplots(axs, xvals, yvals, _plot1d, fmt=fmt, **plot_params)
#----------------------------------------------------------------------------#

#------------------   Core Statements for Plots/Subplots   ------------------#
def plot_core(axs, /, *x_y, **plot_params):
    """ Plot (a set of) data on (a set of) figure' axes

    Take a (set of) Matplotlib's Axis, the y-values and their x-values,
    and plot the data on the axes. If `axs` is a single Axis, make the
    plots on a 1-plot figure; else, make them on an NxM-plot figure.

    The `axs` axes can be a single Axis or an NxM set of Axis. The axes
    are iterated in row-major order; to plot the data in column-major
    order, transpose `axs` before passing it to this present function.

    Considering that `axs` is an NxM grid of Matplotlib's Axis, iterate
    `axs` and `y` (and `x`, if any) at the same time, and plot the data
    at the ij-th index on the ij-th axis. Considering the ij-th item of
    `axs` is an Axis, `x` and `y` can both be 1D arrays, ND arrays or
    sets of 1D arrays; all the plots are made on the same Axis.

    Assuming `y` is an array of arrays, let refer to its ij-th array as
    `y_ij`; if it is a 1D array, plot it on the ij-th axis; if it is a
    2D array, plot any of its subarrays on the axis.

    In addition, `x` can have several shapes:
      - single 1D array: use it for any array of `y`;
      - 2D array: use its i-th array as x-values for any array of the
            i-th (set of) array(s) of `y`;
      - 3D array: use the ij-th array of `x` as x-values for the ij-th
            array of `y`.
    If None, generate a list of indexes for every set of `y`.

    N.B.: the additional `plot_params` arguments are common to any axes.

    N.B.: in case of NxM-plot figure, `axs`, `y` and `x`, if any, are
        iterated in row-major order (i.e. row by row).

    Parameters
    ----------
    axis : (1D or 2D array of) Matplotlib's Axis
        The figure axes to plot the data on. Can be a single Axis, or
        an NxM set of Axis. The axes are iterated in row-major order;
        to pass the data in column-major order, transpose `axs` prior.

    Positional Parameters
    ---------------------
    *x_y : inline arguments
        The x-values `x` and y-values `y`. If one argument, consider it
        is `y`; if two arguments, consider it is `x` and `y`.
        - x : (1D or 2D array of) 1D array_like, optional
            The x-values; if omitted or None, use a list of indexes as
            abscissa. If single array, use it for every array of data;
            if array of arrays, plot every pair (x_i, y_i).
        - y : (1D or 2D array of) ND array_like
            The y-values. If array, plot it against `x`; if array of ar-
            rays, plot any of them against either `x` if it is an array,
            or every array of it if it is an array of arrays.

    Other Parameters
    ----------------
    **plot_params : inline keyword arguments, optional
        The parameters for the Axis' `plot` methods; main options:
          - `fmt` (str): plot format (markers, lines, etc.)
          - `linewidth` (float): the width of the plots
          - `rasterized` (bool): plot rasterization (as raster graphic)
            :Default: default Axis' `plot` method parameters.

    Returns
    -------
    None : directly plot the data on the figure's axis.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Generate dummy data
    >>> tstp = np.linspace(0., 10., 100)
    >>> sin, cos = np.sin(tstp), np.cos(tstp)

    #--- 1-plot figure
    # Plot the sine with dynamic inputs
    >>> fig = plt.figure(figsize=(19.20, 10.80))
    >>> plot_core(fig.gca(), sin, fmt='--', linewidth=3)    # Auto xvalues
    >>> plot_core(fig.gca(), tstp, sin, linewidth=3)        # Explicit xvalues

    # Plot the sine & cosine with the same time scale on the same plot
    >>> fig = plt.figure(figsize=(19.20, 10.80))
    >>> plot_core(fig.gca(),
    ...     tstp, (cos, sin),
    ...     fmt='--', linewidth=3)

    # Plot the sine & cosine with the specific time scales on the same plot
    >>> fig = plt.figure(figsize=(19.20, 10.80))
    >>> plot_core(fig.gca(),
    ...     (tstp, tstp), (cos, sin),
    ...     fmt='--', linewidth=3)

    #--- N-plot figure
    # Plot the sine & cosine with the same time scale on different plots
    >>> fig, axs = plt.subplots(2, 1, figsize=(19.20, 10.80))
    >>> plot_core(axs,
    ...     tstp, (sin, cos),
    ...     fmt='--', linewidth=3)

    # Plot the sine & cosine with specific time scales on different plots
    >>> fig, axs = plt.subplots(2, 1, figsize=(19.20, 10.80))
    >>> plot_core(axs,
    ...     (tstp, tstp),
    ...     (cos, sin),
    ...     fmt='--', linewidth=3)

    # Plot the data on an NxM-plot graph in column-major order
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> plot_core(axs.T,        # Transpose the axes for column-major order
    ...     (tstp, tstp),
    ...     ((sin, cos, (sin, cos)), (sin, cos,( sin, cos))),
    ...     fmt='--', linewidth=3)

    >>> plt.show()
    """
    # Plot the data on the axes
    if len(make_iter(axs)) == 1:
        _plot1d(axs, *_get_x_y(x_y), **plot_params)
    else:
        _plotnd(axs, *_get_x_y(x_y), **plot_params)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                            Bar Plot Functions                            ##
##############################################################################

#----------------------   1-Bar Plot Core Statements   ----------------------#
def _core_single_bars(axis, xpos, vals, mean=True, bar_params=None):
    r""" Plot a set of `vals` as single bars at `xpos` x-values with the
    `bar_params` parameters; if `mean`, set the last bar color to black,
    the last x-label to '\mu' and force its display """

    # Build a list of locations for the x-axis, if so
    if xpos is None:
        xpos = np.arange(len(vals))

    # Plot the bars
    bars = axis.bar(xpos, vals, **bar_params)

    if mean:
        # Color the last bar (representing the mean) in black
        bars[-1].set_color('k')

        # If `sharex` is active, skip the addition of 'mu' to the xticks
        if len(axis.get_xticklabels()) != 0:
            # Remove the x-ticks that are outside the `xpos` scale and append
            # the last value (the mean) at its very end that may be missing
            xticks = [x for x in axis.get_xticks() if xpos[0] <= x < xpos[-1]]
            axis.set_xticks(xticks + [xpos[-1]])
            # Replace the last x-tick label by '\mu'
            tick_labels = axis.get_xticklabels()        # Retrieve the labels
            if len(tick_labels) > 1:
                tick_labels[-1] = r"$\overline{\mu}$"   # Set the mean at last
                axis.set_xticklabels(tick_labels)       # Set the new labels
#----------------------------------------------------------------------------#

#----------------------   N-Bar Plot Core Statements   ----------------------#
def _core_double_bars(axis, xpos, vals, mean=True, bar_params=None):
    r""" Plot a set of `vals` as two side-by-side bars at `xpos` x-values
    with the `bar_params` parameters; if `mean`, set the last bars colors
    to 'gray' & 'dimgray', the last x-tick to "\mu" and force its display """

    # Do a local copy of the parameters for the possible `pop`
    bar_params = bar_params.copy()

    # Build a list of locations for the x-axis, if so
    if xpos is None:
        xpos = np.arange(len(vals))

    # Retrieve the `width` and `alpha` values from `bar_params` if they exist,
    # or set default values else
    width = bar_params['width'] if 'width' in bar_params else 0.4
    alpha = bar_params.pop('alpha') if 'alpha' in bar_params else 1.0

    # If the x-ticks labels are provided, retrieve them to write them only once
    if 'tick_label' in bar_params:
        labels = bar_params.pop('tick_label')

    # Plot the bars
    bars_lf = axis.bar(xpos - 0.5*width, vals[:, 0], alpha=alpha, **bar_params)
    bars_rg = axis.bar(xpos + 0.5*width, vals[:, 1], alpha=0.5*alpha, **bar_params)

    # If any, place the x-ticks labels at the center of the two bars by plotting
    # a new transparent bar chart. Note that it must be done AFTER the main plots
    # in case of the `_decorations`'s `set_bartop_text` function is used.
    if 'tick_label' in bar_params:
        axis.bar(xpos, np.full_like(xpos, np.min(vals[:, 0])),
            tick_label=labels, alpha=0., fill=False)

    if mean:
        # Color the last bars (representing the mean) in nuances of gray
        bars_lf[-1].set_color('dimgray')
        bars_rg[-1].set_color('gray')

        # If `sharex` is active, skip the addition of 'mu' to the xticks
        if len(axis.get_xticklabels()) != 0:
            # Remove the x-ticks that are outside the `xpos` scale and append
            # the last value (the mean) at its very end that may be missing
            xticks = [x for x in axis.get_xticks() if xpos[0] <= x < xpos[-1]]
            axis.set_xticks(xticks + [xpos[-1]])        # Append last `xpos` value
            # Replace the last x-tick label by "\mu"
            tick_labels = axis.get_xticklabels()        # Retrieve the labels
            tick_labels[-1] = r"$\overline{\mu}$"       # Set the mean at last
            axis.set_xticklabels(tick_labels)           # Set the new labels
#----------------------------------------------------------------------------#

#---------------------   Core Statements for Bar Plot   ---------------------#
def bar_core(axs, /, *x_y, mean=False, double_bars=False, **bar_params):
    r""" Plot data bars on a figure

    Take the axes of a figure, the y-values `y` to plot & possibly
    their x-values `x` and plot the data bars on the axes at `x`.
    If `x` is None, use a set of indexes instead (starting from 0).

    If `double_bars` is False, the plot are single bars; else, they are
    two half-sized side-by-side bars. In the former case, the data are
    expected to be 1D arrays; in the latter case, they are expected to
    be 2D arrays with shape Nx2 (2 columns, one per bar); `x` shape
    remains the same no matter the state of `double_bars`.

    If `mean` is True, color the last bar in black if single bars or in
    nuances of gray if double bars, plus change the last x-tick label to
    "\overline{mu}" and force its display. Notice that the display of the
    xticks and x-labels is handled by Matplotlib, except the mean, if so.

    The function works on 1-plot, N-plot and NxM-plot figures:
      - `axs` is a single Axis: `y` must be a 1D array,
            e.g. (1, 2, 3)
      - `axs` is a 1D array of Axis: `y` must be a 2D array,
            e.g. ((1, 2, 3), (4, 5, 6))
      - `axs` is an 2D array of Axis: `y` must be an array of
            2D arrays, e.g. (((1, 2), (3, 4)), ((5, 6), (7, 8)))
    The same rule applies to `x`, if any; in the case of `double_bars`
    is set to True, the last depth values of `y` must be arrays of two
    values, e.g. ((1, 2), (3, 4), (5, 6)) for the first case (1 Axis);
    this does not apply to `x`.

    N.B.: in case of NxM-plot figure, `axs`, `y` and `x`, if any, are
        iterated in row-major order (i.e. row by row).

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to plot the data bars on. Can be a single Axis,
        a 1D array of Axis or an 2D array of Axis.

    Positional Parameters
    ---------------------
    *x_y : inline arguments
        The x-values `x` and y-values `y`. If one argument, consider it
        is `y`; if two arguments, consider it is `x` and `y`.
        - x : (1D or 2D array of) 1D array_like, optional
            The bar abscissa. Same shape as `y` when `double_bars`
            is False. If None, use a set of indexes instead.
        - y : (1D or 2D array of) 1D or 2D array_like
            The bar heights. Can be a 1D array if `axs` is an Axis, a 2D
            array if `axs` is a 1D array, or a 3D array if `axs` is a 2D
            array. The dimension must be increased by 1 if `double_bars`
            is True, and last depth must contain exactly 2 values.

    Keyword Parameters
    ------------------
    [OPT] mean : bool
        If True, the last value of each set of `y` is assumed to be
        the mean of this set; as such, color the last bar in black or
        in nuances of gray (depending on `double_bars`), append "\mu"
        to the x-labels and force its display. Do nothing if False.
            :Default: False
    [OPT] double_bars : bool
        If the bars should be single bars or sets of two half-sized bars
        set side by side. False acts the same as the Axis' `bar` method;
        if True, the last depth of `y` should be arrays of 2 values,
        one for the left bar and the other one for the right bar.
            :Defaut: False

    Other Parameters
    ----------------
    **bar_params : inline keyword arguments, optional
        The parameters for the Axis' `bar` method; main options:
          - `width` (float): the bar width; let n be the `x` step
                (1 as default), `width` should be < n with single
                bars or < n/2 with double bars.
          - `tick_label` (array): the x-tick labels
          - `color` ((array of) string(s)): the bar colors; if single
                color and `double_bars` is True, the right bar alpha
                is set to half that of the left bar.
            :Default: bar_params={'width': 0.4 if double_bars else 0.8}
                + Axis' `bar` method default parameters.

    Returns
    -------
    None : directly plot the bars; decorations can be added afterwards.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt


    #--- Single bar plots examples
    # Generate dummy data & add the mean
    >>> vals = np.arange(1, 30)                         # 1D data
    >>> vals = np.hstack((vals, vals.mean()))           # Append the mean

    # Generate simple x-ticks (same as those automatically generated)
    >>> xpos = np.arange(1, len(vals)+1)                # +1 for the mean

    # Generate simple x-labels (optional)
    >>> tick_labels = np.array(np.arange(1, len(xpos)+1), dtype=str)

    # 1-plot figure
    >>> fig = plt.figure()
    >>> bar_core(fig.gca(),
    ...     xpos,           # optional
    ...     vals,
    ...     mean=True,
    ...     # Axis' `bar` method's parameters
    ...     tick_label=tick_labels,
    ...     color='blue')

    # 2-plot figure
    >>> fig, axs = plt.subplots(2, 1)
    >>> bar_core(axs,
    ...     xpos,           # or (xpos, xpos)
    ...     (vals, vals),
    ...     mean=True)

    # 3x2-plot figure in row-major order
    >>> fig, axs = plt.subplots(3, 2)
    >>> bar_core(axs,       # row-major
    ...     ((vals, vals), (vals, vals), (vals, vals)), # 3 tuples of 2 arrays
    ...     mean=True)

    # 3x2-plot figure in column-major order
    >>> fig, axs = plt.subplots(3, 2)
    >>> bar_core(axs.T,                                 # column-major
    ...     ((vals, vals, vals), (vals, vals, vals)),   # 2 tuples of 3 arrays
    ...     mean=True)

    >>> plt.show()


    #--- Double bar plots examples
    # Generate dummy data
    >>> vals = np.arange(1, 31).reshape(-1, 2)          # 2D data
    >>> vals = np.vstack((vals, vals.mean(0)))          # Append the mean

    # Generate simple x-ticks (same as those automatically generated)
    >>> xpos = np.arange(1, len(vals)+1)                # +1 for the mean

    # Generate simple x-labels (optional)
    >>> tick_labels = np.array(np.arange(1, len(xpos)+1), dtype=str)

    # 1-plot figure
    >>> fig, axs = plt.subplots(1, 1)
    >>> bar_core(axs,
    ...     xpos,   # optional
    ...     vals,
    ...     mean=True,
    ...     double_bars=True,
    ...     # Axis' `bar` method's parameters
    ...     tick_label=tick_labels,
    ...     color='blue',
    ...     width=0.3)

    # 2-plot figure
    >>> fig, axs = plt.subplots(2, 1)
    >>> bar_core(axs,
    ...     xpos,   # or (xpos, xpos)
    ...     (vals, vals),
    ...     mean=True,
    ...     double_bars=True)

    # 2x2-plot figure with auto xpos
    >>> fig, axs = plt.subplots(2, 2)
    >>> bar_core(axs,
    ...     ((vals, vals), (vals, vals)),
    ...     mean=True,
    ...     double_bars=True)

    # 2x2-plot figure
    >>> fig, axs = plt.subplots(2, 2)
    >>> bar_core(axs,
    ...     xpos,   # or (xpos, xpos) or ((xpos, xpos), (xpos, xpos))
    ...     ((vals, vals), (vals, vals)),
    ...     mean=True,
    ...     double_bars=True)

    >>> plt.show()
    """

    # Default parameters
    # Ensure that `width` exists (0.8 is Axis' `bar` default)
    params = {'width': 0.4 if double_bars else 0.8}
    params = check_keys(bar_params, params)

    # Plot the bars on the plots
    fct = _core_double_bars if double_bars else _core_single_bars
    _core_nplots(axs, *_get_x_y(x_y), fct,
        mean=mean, bar_params=params)
#----------------------------------------------------------------------------#

##############################################################################
