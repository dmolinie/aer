""" Shorthand plotting functions

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2026
Last revised: April 2026

License: GPLv3
"""

__all__ = ['plot', 'bar_plot']

import numpy as np
import matplotlib.pyplot as plt

from aer.tools import check_keys, check_path
from ._core_plot import plot_core, bar_core
from ._decorations import (
    set_decorations, set_bartop_text, remove_spaces, set_margins)


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

def _retrieve_figure(axs):
    """ Search for the first Axis in a set and return the Figure attached """
    axis = axs
    # Search for any Axis in `axs`
    while np.ndim(axis) != 0:
        axis = axis[0]
    # Retrieve the figure attached to this Axis
    return axis.get_figure()

##############################################################################



##############################################################################
##                      General-Purpose Plot Functions                      ##
##############################################################################

def plot(axs, /, *x_y,
    labels=None, titles=None, legends=None, rtexts=None, **imgparams):
    """ Plot `y` against `x` on an NxM-graph figure

    Take the axes of an NxM-plot figure and plot the `y` (ordinates) on
    the different graphs against `x` (abscissa). Adjust the figure para-
    meters using the `imgparams` keyword arguments. The `axs` may be set
    to None; in such a case, instantiate a figure using the `fig_params`
    keyword argument, if any, or with default parameters else, plot the
    data on it and return it. See the `plot_core` and `set_decorations`
    functions for complementary information.

    The figures axes can either be a single Matplotlib's Axis, or a 1D
    or 2D set of Axis. For any Axis, there are four possibilities:
      - `y` is a 1D array:
          - `x` is a 1D array: plot `y` against `x`;
          - `x` is a set of 1D arrays: plot `y` against the first array
                of `x`;
      - `y` is a set of 1D arrays:
          - `x` is a 1D array: plot every array of `y` against the same
                `x` array;
          - `x` is set of 1D arrays: iterate `x` and `y` and plot every
                pair (x_i, y_i).
    If `x` is omitted or None, generate a 1D set of indexes as `x`.

    If `axs` is an array of Axis, iterate `axs`, `y` & `x` once; if it
    is an array of arrays of Axis, iterate them twice; for every axis,
    follow the above-mentioned rules. However, the same `x` can be used
    for several axes & y-values:
      - 1D array: use it for every array of `y`;
      - set of 1D arrays: use every array for a set of `axs` & `y`
            if they are 2D or more;
      - set of arrays of 1D arrays: iterate them along `axs` & `y`.

    If general, the dimension of `y` should be the same or 1 more than
    that of `axs`; the dimension of `x` is more flexible to avoid redun-
    dancies. See the `plot_core` function for more details, and the
    `Examples` section below for examples of use.

    Figure decorations can be passed using the `labels` (x and ylabels),
    `titles` (plot titles), `legends` (plot legends) and `rtexts` (right
    y-axis side texts), whose specific sets of parameters are optional
    keyword arguments (cf. `Other Parameters` section below). Refer to
    the `set_decorations` function for details. The shape of these arg-
    uments typically follows that of `axs`.

    Additionally, the plots' margins and spaces between the axes (when
    there are several axes on the figure) can be dynamically adjusted
    using the `margin_params` and `space_params` keyword arguments;
    cf. section `Other Parameters` below. Refer to the `set_margins`
    and `remove_spaces` functions for details.

    If the `fname` keyword is provided, automatically save the image with
    this name and using the parameters dictionary listed in `save_params`
    inline keyword argument; leave it empty to skip autosaving. Return
    the image, that can be displayed afterwards using `fig.show()`.

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axis : (1D or 2D array of) Matplotlib's Axis
        The figure axes to plot the data on. Can be a single Axis, or
        an NxM set of Axis. The axes are iterated in row-major order;
        to pass the data in column-major order, transpose `axs` prior.
        If None, check that the `fig_params` keyword argument is not
        empty; if not, instantiate a figure with those parameters.

    Positional Parameters
    ---------------------
    *x_y : inline arguments
        The x-values `x`, y-values `y` and line format `fmt`. Mimic the
        behavior of the Matplotlib's `plot` function: if one argument,
        consider it is `y`; if two arguments, consider it is `x` & `y`;
        if any, use the argument directly following `y` as `fmt`.
        - x : (1D or 2D array of) 1D array_like, optional
            The x-values; if omitted or None, use a list of indexes as
            abscissa. If single array, use it for every array of data;
            if array of arrays, plot every pair (x_i, y_i).
        - y : (1D or 2D array of) ND array_like
            The y-values. If array, plot it against `x`; if array of ar-
            rays, plot any of them against either `x` if it is an array,
            or every array of it if it is an array of arrays.

    Keyword Parameters
    ------------------
    [OPT] labels : 2-array of (1D or 2D arrays of) string(s)
        The labels of the axes. If None, write no label. Else, assume it
        is organized as (xlabels, ylabels). If single strings, set the
        labels of a 1-plot figure; if set of tuples, set the labels to
        an N-plot figure; if set of set of tuples, set the labels to an
        NxM-plot figure. The labels may be repeated along the axes of
        a row (cf. the `label_params` keyword argument below).
            :Default: None
    [OPT] titles : (1D or 2D arrays of) string(s)
        The subplot titles. If direct string, use it as title for the
        first subplot (`axs[0]`/`axs[0, 0]`); else, use the ij-th str
        as title for the ij-th graph. If 1D set, ignore counter `j`.
            :Default: None
    [OPT] legends : (1D or 2D arrays of) string(s)
        The list of legends; the strings at index ij are used as legend
        for the ij-th figure graph. If 1D set, ignore counter `j`.
            :Default: None
    [OPT] rtexts : (1D or 2D arrays of) string(s)
        The side texts to write along the right y-axis. The ij-th string
        is used as right y-axis side text for the ij-th graph. If 1D set,
        ignore counter `j`.
            :Default: None

    Other Parameters
    ----------------
    [OPT] fname : str
        The image-to-save pathname (path+name+ext).
            :Default: None (do not save the figure)
    [OPT] fig_params : dict
        The figure parameters; used only if `axs` is None; in such as
        case, pass it to the Matplotlib's `subplots` function:
          - `nrows`, `ncols` (int): the number of rows/cols
          - `figsize` (2-tuple of floats): the figure dimensions
            :Default: default `subplots` parameters
    [OPT] save_params : dict
        The parameters to save the figure, passed to the Matplotlib's
        `savefig`; used only if `fname` is not empty nor None:
          - `bbox_inches` (str): the margins of the saved graphic image
          - `dpi` (int): image DPI (used if saved as a raster graphic)
            :Default: {'bbox_inches': 'tight'}
    [OPT] plot_params : dict
        The plot parameters passed to the `plot_core` function:
          - `fmt` (str): plot format (markers, lines, etc.)
          - `linewidth` (float): the width of the plots
          - `rasterized` (bool): plot rasterization (as raster graphic)
            :Default: default `plot_core` parameters
    [OPT] label_params : dict
        The labels parameters passed to the `set_labels` function:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
          - `repeat_labels` (2 bools): if the xlabel and ylabel should
                be repeated on any axes, as (rep_xlabel, rep_ylabel)
            :Default: {'size': 18, 'repeat_labels': (False, False)}
    [OPT] title_params : dict
        The titles parameters passed to the `set_titles` function:
          - `size` (float): title font size
          - `pad` (float): title bounding box space pad
            :Default: {'size': 21}
    [OPT] legend_params : dict
        The legend parameters passed to the `set_legends` function:
          - `loc` (str): the position of the legend box
          - `fontsize` (float): legend font size
            :Default: {'size': 14}
    [OPT] rtext_params : dict
        The right y-axis side text parameters passed to the
        `set_right_yaxis_texts` function:
          - `size` (float): right y-axis sidetext font size
          - `labelpad` (float): right y-axis side text bounding box space pad
          - `stacked` (bool): vertically stack characters or rotate label
            :Default: default `set_right_yaxis_texts` parameters
    [OPT] margin_params : dict
        The margins parameters passed to the Matplotlib's `Axis`' `margins`
        method through the `set_margins` function:
          - `x`, `y` (float): the x/y-axis margin value
            :Default: default `set_margins` parameters
    [OPT] space_params : dict
        The space between plots passed to the `remove_spaces` function:
          - `no_xspace` (bool): remove horizontal space between plots
          - `no_yspace` (bool): remove vertical space between plots
            :Default: {'no_xspace': False, 'no_yspace': False}

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Generate dummy data
    >>> tstp1 = np.linspace(0., 1., 100)    # From 0s to 1s -- 100 points
    >>> tstp2 = np.linspace(0., 1., 200)    # From 0s to 1s -- 200 points
    >>> sin_10 = np.sin(2*np.pi*10.*tstp1)  # 10 Hz sine (100 pts)
    >>> sin_50 = np.sin(2*np.pi*50.*tstp1)  # 50 Hz sine (100 pts)
    >>> cos_10 = np.cos(2*np.pi*10.*tstp1)  # 10 Hz cosine (100 pts)
    >>> cos_50 = np.cos(2*np.pi*50.*tstp2)  # 50 Hz cosine (200 pts)

    # Simple plot with optional x-values and line style
    >>> fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    >>> fig = plot(fig.gca(), sin_10)                   # Auto xvals
    >>> fig = plot(fig.gca(), 100.*tstp1, sin_10+1.)    # Explicit xvals
    >>> fig.show()

    # Plot a single data array on a 1-plot figure
    >>> fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    >>> fig = plot(fig.gca(),
    ...     tstp1, sin_10,
    ...     labels=("Time", "Amplitude"),
    ...     titles="Signal amplitude against time",
    ...     rtexts="Time domain",
    ...     fname=None,             # Don't save the Figure in a file
    ...     plot_params={'fmt': '--', 'linewidth': 2},
    ...     label_params={'size': 20},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     rtext_params={'size': 18, 'labelpad': 12., 'stacked': False},
    ...     margin_params={'x': 0.01, 'y': 0.01})
    >>> fig.show()

    # Plot several data on the same 1-plot figure
    >>> fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    >>> fig = plot(fig.gca(),
    ...     (tstp1, tstp2), (sin_10, cos_50),
    ...     labels=("Time", "Amplitude"),
    ...     titles="Signal amplitudes against time",
    ...     legends=("10 Hz Sine", "50 Hz Cosine"),
    ...     fname="Sine_Cosine.pdf",    # Save the figure in a file
    ...     save_params={'bbox_inches': 'tight'},
    ...     plot_params={'fmt': '--', 'linewidth': 2},
    ...     label_params={'size': 20},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     legend_params={'fontsize': 14, 'loc': 'upper center'},
    ...     margin_params={'x': 0.01, 'y': 0.01})
    >>> fig.show()

    # Plot data arrays on an N-plot figure (one set of arrays per plot)
    # Here, generate the Figure in the function with the 'fig_params' dict. arg.
    >>> fig = plot(None,                # Generate the Figure in the function
    ...     tstp1,
    ...     (sin_10, cos_10, (sin_10, cos_10)),
    ...     labels=("Time", ("Amp. Sine", "Amp. Cosine", "Amp. Sine & Cosine")),
    ...     titles="Signal amplitudes against time",
    ...     rtexts=("Sine", "Cosine", "Sine & Cosine"),
    ...     legends=("", "", ("Sine", "Cosine")),
    ...     fname="Sine_Cosine.pdf",    # Save the figure; set to None to ignore
    ...     fig_params={'nrows': 3, 'ncols': 1, 'figsize': (19.20, 10.80)},
    ...     save_params={'bbox_inches': 'tight'},
    ...     plot_params={'linewidth': 2},
    ...     label_params={'size': 18, 'repeat_labels': (True, False)},
    ...     title_params={'size': 24},
    ...     legend_params={'loc': 'upper right'},
    ...     margin_params={'x': 0.01, 'y': 0.1},
    ...     space_params={'no_xspace': False, 'no_yspace': False})
    >>> fig.show()

    # Plot data arrays on an N-plot figure (one set of arrays per plot)
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> fig = plot(axs.T,                               # column-major
    ...     (tstp1,                                     # Left plots x-values
    ...      (tstp1, tstp2, (tstp1, tstp2))),           # Right plots x-values
    ...     ((sin_10, sin_50, (sin_10, sin_50)),        # Left plots y-values
    ...      (cos_10, cos_50, (cos_10, cos_50))),       # Right plots y-values
    ...     labels=(
    ...         ("Common x-label", ("Y_11", "Y_21", "Y_31")),     # Left labels
    ...         (("X_11", "X_21", "X_31"), "Common y-label")),    # Right labels
    ...     titles=(
    ...         ("10 Hz Sine", "50 Hz Sine", "10 & 50 Hz Sines"), # Left titles
    ...         "10 & 50 Hz Cosines"),                            # Right titles
    ...     legends=(
    ...         ("10 Hz", "50 Hz", ""),                 # Left plots legends
    ...         ("", "", ("10 Hz", "50 Hz"))),          # Right plots legends
    ...     rtexts=(
    ...         "",                                     # Right y-axis side texts
    ...         ("10 Hz", "50 Hz", "10 & 50 Hz")),      # Left y-axis side texts
    ...     fname="Sines_Cosines.pdf",                  # Save figure (None to skip)
    ...     save_params={'bbox_inches': 'tight'},
    ...     plot_params= {'fmt': '--', 'linewidth': 2},
    ...     label_params={'size': 17,'repeat_labels': (False, True)},
    ...     title_params={'size': 20},
    ...     legend_params={'fontsize': 15, 'loc': 'upper right'},
    ...     rtext_params={'size': 14, 'stacked': True},
    ...     margin_params={'x': 0.01, 'y': 0.1},
    ...     space_params={'no_xspace': False, 'no_yspace': False})
    >>> fig.show()
    """

    # Default parameters
    params = {'fname': None, 'fig_params': {},
              'save_params': {'bbox_inches': 'tight'},
              'plot_params': {},
              'label_params': {'size': 18, 'repeat_labels': (False, False)},
              'title_params': {'size': 21}, 'rtext_params': {'size': 14},
              'legend_params': {'fontsize': 15}, 'margin_params': {},
              'space_params': {'no_xspace': False, 'no_yspace': False}}

    # Append the default parameters to those possibly provided as arguments
    params = check_keys(imgparams, params)

    # Check that a set of Axis was provided; if not, if `fig_params` was
    # provided, generate a subplots using these parameters
    if axs is None:
        fig, axs = plt.subplots(**params['fig_params'])
    else:
        # Retrieve the Figure from the axes
        fig = _retrieve_figure(axs)

    # Plot the values
    plot_core(axs, *x_y, **params['plot_params'])

    # Write the decorations (labels, titles & legends)
    set_decorations(axs, labels, titles, legends, rtexts,
        label_params=params['label_params'],
        title_params=params['title_params'],
        rtext_params=params['rtext_params'],
        legend_params=params['legend_params'])

    # Reduce the space margins on the right & left of the plots
    if params['margin_params'] != {}:
        set_margins(axs, **params['margin_params'])

    # Remove the white spaces between plots (works only on subplots)
    if np.ndim(axs) > 0:
        remove_spaces(fig, **params['space_params'])

    # Save the figure in the specified file
    # If the filename contains directories, create them if they do not exist
    if params['fname'] not in (None, ''):
        fig.savefig(check_path(params['fname']), **params['save_params'])

    return fig

##############################################################################



##############################################################################
##                         Bar Chart Plot Functions                         ##
##############################################################################

# pylint: disable-next=too-many-arguments
def bar_plot(axs, /, *x_y,
    labels=None, titles=None, legends=None, rtexts=None, bartext=None,
    **imgparams):
    """ Plot values as bar charts on an NxM-graph figure

    Take the axes of an NxM-plot figure and plot the bars with height
    `y` on the graphs at the `x` locations. Adjust the figure with the
    `imgparams` arguments. The `axs` may None; in such a case, instan-
    tiate a figure using the `fig_params` keyword argument if any, and
    plot the data on it and return the figure. See the `bar_core` and
    `set_decorations` functions for complementary information.

    The bars can either be a set of single bars or two half-sized side-
    by-side bars to allow displaying two values per index (e.g. scores
    obtained on the same dataset). This option can be activated by spe-
    cifying the `double_bars` key in the `bar_params` keyword argument
    (cf. the `Other Parameters` section below). Another key for this
    dictionary is `mean`, that allows specifying that the last value
    (or row if `double_bars` is True) of `y` is the array's mean;
    notice that this mean must be computed and appended separately.

    The figures axes can either be a single Matplotlib's Axis, or a 1D
    or 2D set of Axis. For any Axis, the bars heights `y` must be a 1D
    array if `double_bars` is False, or a 2D array whose 2nd dimension
    is 2 (i.e. Nx2 values) if True. If provided, the `x` locations must
    be a 1D array; if it is omitted or None, use a set of indexes.

    If `axs` is an array of Axis, iterate `axs`, `y` & `x` once; if it
    is an array of arrays of Axis, iterate them twice; for every axis,
    follow the above-mentioned rules. However, the same `x` can be used
    for several axes and bar heights:
      - 1D array: use it for every array of `y`;
      - set of 1D arrays: use every array for a set of `axs` & `y`;
      - set of arrays of 1D arrays: iterate them along `axs` & `y`.

    Figure decorations can be passed using the `labels` (x and ylabels),
    `titles` (plot titles), `legends` (plot legends) and `rtexts` (right
    y-axis side texts); some (simple) texts can also be written at the
    bars' top as `bartext`, that can for instance be used to display
    the bars' height. The parameters for any of these arguments can be
    specified using optional keyword arguments (cf. `Other Parameters`
    section below); a set of parameters is common to any argument of a
    kind. Refer to the `set_decorations` & `set_bartop_text` functions.
    The shape of these arguments typically follows that of `axs`.

    Additionally, the plots' margins and spaces between the axes (when
    there are several axes on the figure) can be dynamically adjusted
    using the `margin_params` and `space_params` keyword arguments;
    cf. section `Other Parameters` below. Refer to the `set_margins`
    and `remove_spaces` functions for details.

    If the `fname` keyword is provided, automatically save the image with
    this name and using the parameters dictionary listed in `save_params`
    inline keyword argument; leave it empty to skip autosaving. Return
    the image, that can be displayed afterwards using `fig.show()`.

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axis : (1D or 2D array of) Matplotlib's Axis
        The figure axes to plot the data on. Can be a single Axis, or
        an NxM set of Axis. The axes are iterated in row-major order;
        to pass the data in column-major order, transpose `axs` prior.
        If None, check that the `fig_params` keyword argument is not
        empty; if not, instantiate a figure with those parameters.

    Positional Parameters
    ---------------------
    *x_y : inline arguments
        The x-values `x` and y-values `y`. If one argument, consider it
        is `y`; if two arguments, consider it is `x` and `y`.
        - x : (1D or 2D array of) 1D array_like, optional
            The bars locations along the x-axis; if omitted or None, use
            a list of indexes instead. If single array, use it for every
            array of `y`; if array of arrays, plot every pair (x_i, y_i).
            Same shape as `y` when `bar_params['double_bars']` is False.
        - y : (1D or 2D array of) 1D or 2D array_like
            The bar heights. Can be a 1D array if `axs` is an Axis, a 2D
            array if it is a 1D array, or a 3D array if it is a 2D array.
            Dimension must be increased by 1 if `bar_params['double_bars']`
            is True, and the last depth must contain exactly 2 values.

    Keyword Parameters
    ------------------
    [OPT] labels : 2-array of (1D or 2D arrays of) string(s)
        The labels of the axes. If None, write no label. Else, assume it
        is organized as (xlabels, ylabels). If single strings, set the
        labels of a 1-plot figure; if set of tuples, set the labels to
        an N-plot figure; if set of set of tuples, set the labels to an
        NxM-plot figure. The labels may be repeated along the axes of
        a row (cf. the `label_params` keyword argument below).
            :Default: None
    [OPT] titles : (1D or 2D arrays of) string(s)
        The subplot titles. If direct string, use it as title for the
        first subplot (`axs[0]`/`axs[0, 0]`); else, use the ij-th str
        as title for the ij-th graph. If 1D set, ignore counter `j`.
            :Default: None
    [OPT] legends : (1D or 2D arrays of) string(s)
        The list of legends; the strings at index ij are used as legend
        for the ij-th figure graph. If 1D set, ignore counter `j`.
            :Default: None
    [OPT] rtexts : (1D or 2D arrays of) string(s)
        The side texts to write along the right y-axis. The ij-th string
        is used as right y-axis side text for the ij-th graph. If 1D set,
        ignore counter `j`.
            :Default: None
    [OPT] bartext : (1D or 2D arrays of) string(s)
        The texts to write at the top of the bars. These texts are typi-
        cally the (rounded) bar heights to make them easier to read.
            :Default: None

    Other Parameters
    ----------------
    [OPT] fname : str
        The image-to-save pathname (path+name+ext).
            :Default: None (do not save the figure)
    [OPT] fig_params : dict
        The figure parameters; used only if `axs` is None; in such as
        case, pass it to the Matplotlib's `subplots` function:
          - `nrows`, `ncols` (int): the number of rows/cols
          - `figsize` (2-tuple of floats): the figure dimensions
            :Default: default `subplots` parameters
    [OPT] save_params : dict
        The parameters to save the figure, passed to the Matplotlib's
        `savefig`; used only if `fname` is not empty nor None:
          - `bbox_inches` (str): the margins of the saved graphic image
          - `dpi` (int): image DPI (used if saved as a raster graphic)
            :Default: {'bbox_inches': 'tight'}
    [OPT] bar_params : dict
        The plot parameters passed to the `bar_core` function:
          - `mean` (bool): if the last values of any array of `y`
                represent the array's mean.
          - `double_bars` (bool): if the bars should be doubled or not;
                if so, any array of `y` must contain 2 values.
          - `width` (float): the width of the bars
          - `color` ((set of) strings): the bar colors; if single color,
                use it for every bar; else, cycle the colors.
          - `tick_label` (array): the manual x-ticks labels
            :Default: {'mean': False, 'double_bars': False}
    [OPT] label_params : dict
        The labels parameters passed to the `set_labels` function:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
          - `repeat_labels` (2 bools): if the xlabel and ylabel should
                be repeated on any axes, as (rep_xlabel, rep_ylabel)
            :Default: {'size': 18, 'repeat_labels': (False, False)}
    [OPT] title_params : dict
        The titles parameters passed to the `set_titles` function:
          - `size` (float): title font size
          - `pad` (float): title bounding box space pad
            :Default: {'size': 21}
    [OPT] legend_params : dict
        The legend parameters passed to the `set_legends` function:
          - `loc` (str): the position of the legend box
          - `fontsize` (float): legend font size
            :Default: {'size': 14}
    [OPT] rtext_params : dict
        The right y-axis side text parameters passed to the
        `set_right_yaxis_texts` function:
          - `size` (float): right y-axis sidetext font size
          - `labelpad` (float): right y-axis side text bounding box space pad
          - `stacked` (bool): vertically stack characters or rotate label
            :Default: default `set_right_yaxis_texts` parameters
    [OPT] bartext_params : dict
        The bar text parameters passed to the `set_bartop_text` function:
          - `size` (float): text font size
          - `ha`, `va` (str): the horizontal/vertical alignment
            :Default: {'va': 'bottom', 'ha': 'center'}
    [OPT] margin_params : dict
        The margins parameters passed to the Matplotlib's `Axis`' `margins`
        method through the `set_margins` function:
          - `x`, `y` (float): the x/y-axis margin value
            :Default: default `set_margins` parameters
    [OPT] space_params : dict
        The space between plots passed to the `remove_spaces` function:
          - `no_xspace` (bool): remove horizontal space between plots
          - `no_yspace` (bool): remove vertical space between plots
            :Default: {'no_xspace': False, 'no_yspace': False}

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt


    #--- Single bar plots examples
    # Generate dummy data & add the mean
    >>> vals1 = np.arange(1, 15)                         # 1D data
    >>> vals2 = np.arange(29, 0, -1)                     # 1D data
    >>> vals1 = np.hstack((vals1, vals1.mean()))         # Append the mean
    >>> vals2 = np.hstack((vals2, vals2.mean()))         # Append the mean

    # Generate simple x-ticks (same as those automatically generated)
    >>> xpos1 = np.arange(1, len(vals1)+1)               # +1 for the mean
    >>> xpos2 = np.arange(1, len(vals2)+1)               # +1 for the mean

    # Generate simple x-labels (optional)
    >>> tick_labels1 = np.array(np.arange(1, len(xpos1)+1), dtype=str)
    >>> tick_labels2 = np.array(np.arange(1, len(xpos2)+1), dtype=str)

    # Bar chart with optional `x` (x-values)
    >>> fig, axs = plt.subplots(2, figsize=(19.20, 10.80), layout='constrained')
    >>> fig = bar_plot(axs[0], vals1)                    # Auto xpos
    >>> fig = bar_plot(axs[1], xpos1, vals1)             # Explicit xpos
    >>> fig.show()

    # Plot a single bar chart
    >>> fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    >>> fig = bar_plot(fig.gca(),
    ...     xpos1, vals1,
    ...     labels=("Indexes", "Values"),
    ...     titles="Values against indexes",
    ...     rtexts="Values",
    ...     bartext=vals1.astype(int),
    ...     fname=None,             # Don't save the Figure in a file
    ...     bar_params={'mean': True, 'tick_label': tick_labels1, 'color': 'blue'},
    ...     label_params={'size': 20},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     rtext_params={'size': 18, 'labelpad': 12., 'stacked': False},
    ...     bartext_params={'size': 14},
    ...     margin_params={'x': 0.01, 'y': 0.01})
    >>> fig.show()

    # Plot a set of bar charts
    >>> fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    >>> fig = bar_plot(axs.T,                            # column-major
    ...     ((xpos1, xpos1),  # or just xpos1            # Left plots x-values
    ...      (xpos2, xpos2)), # or just xpos2            # Right plots x-values
    ...     ((vals1, vals1),                             # Left plots y-values
    ...      (vals2, vals2)),                            # Right plots y-values
    ...     labels=(
    ...         ("Common x-label", ("Y_11", "Y_21")),    # Left labels
    ...         (("X_11", "X_21"), "Common y-label")),   # Right labels
    ...     titles=(
    ...         ("Values 1", "Values 2"),                # Left titles
    ...         "Values 1 & 2"),                         # Right titles
    ...     legends=(
    ...         ("Leg 1", "Leg 2"),                      # Left plots legends
    ...         ("", "Leg 4")),                          # Right plots legends
    ...     rtexts=(
    ...         "",                                      # Right y-axis side texts
    ...         ("10 Hz", "50 Hz")),                     # Left y-axis side texts
    ...     bartext=(
    ...         (vals1.astype(int), vals1.astype(int)),  # Left bar's top texts
    ...         (vals2.astype(int), vals2.astype(int))), # Right bar's top texts
    ...     fname="NxM_Bar_plots.pdf",                   # Save figure (None to skip)
    ...     save_params={'bbox_inches': 'tight'},
    ...     bar_params= {'mean': True},
    ...     label_params={'size': 17,'repeat_labels': (False, True)},
    ...     title_params={'size': 20},
    ...     legend_params={'fontsize': 15, 'loc': 'best'},
    ...     rtext_params={'size': 14, 'stacked': True},
    ...     bartext_params={'size': 12},
    ...     margin_params={'x': 0.01, 'y': 0.01},
    ...     space_params={'no_xspace': False, 'no_yspace': False})
    >>> fig.show()


    #--- Double bar plots examples
    # Generate dummy data
    >>> vals1 = np.arange(1, 15).reshape(-1, 2)          # 2D data
    >>> vals1 = np.vstack((vals1, vals1.mean(0)))        # Append the mean
    >>> vals2 = np.arange(30, 0, -1).reshape(-1, 2)      # 2D data
    >>> vals2 = np.vstack((vals2, vals2.mean(0)))        # Append the mean

    # Generate simple x-ticks (same as those automatically generated)
    >>> xpos1 = np.arange(1, len(vals1)+1)               # +1 for the mean
    >>> xpos2 = np.arange(1, len(vals2)+1)               # +1 for the mean

    # Generate simple x-labels (optional)
    >>> tick_labels1 = np.array(np.arange(1, len(xpos1)+1), dtype=str)
    >>> tick_labels2 = np.array(np.arange(1, len(xpos2)+1), dtype=str)

    # Plot a single bar chart
    >>> fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    >>> fig = bar_plot(fig.gca(),
    ...     xpos1, vals1,
    ...     labels=("Indexes", "Values"),
    ...     titles="Values against indexes",
    ...     rtexts="Values",
    ...     bartext=vals1.astype(int),
    ...     fname=None,             # Don't save the Figure in a file
    ...     bar_params={'mean': True, 'double_bars': True,
    ...                 'tick_label': tick_labels1, 'color': 'blue'},
    ...     label_params={'size': 20},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     rtext_params={'size': 18, 'labelpad': 12., 'stacked': False},
    ...     bartext_params={'size': 14},
    ...     margin_params={'x': 0.01, 'y': 0.01})
    >>> fig.show()

    # Plot a set of bar charts
    >>> fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    >>> fig = bar_plot(axs.T,                            # column-major
    ...     ((xpos1, xpos1),  # or just xpos1            # Left plots x-values
    ...      (xpos2, xpos2)), # or just xpos2            # Right plots x-values
    ...     ((vals1, vals1),                             # Left plots y-values
    ...      (vals2, vals2)),                            # Right plots y-values
    ...     labels=(
    ...         ("Common x-label", ("Y_11", "Y_21")),    # Left labels
    ...         (("X_11", "X_21"), "Common y-label")),   # Right labels
    ...     titles=(
    ...         ("Values 1", "Values 2"),                # Left titles
    ...         "Values 1 & 2"),                         # Right titles
    ...     legends=(
    ...         ("Leg 1", "Leg 2"),                      # Left plots legends
    ...         ("", "Leg 4")),                          # Right plots legends
    ...     rtexts=(
    ...         "",                                      # Right y-axis side texts
    ...         ("10 Hz", "50 Hz")),                     # Left y-axis side texts
    ...     bartext=(
    ...         (vals1.astype(int), vals1.astype(int)),  # Left bar's top texts
    ...         (vals2.astype(int), vals2.astype(int))), # Right bar's top texts
    ...     fname="NxM_Bar_plots.pdf",                   # Save figure (None to skip)
    ...     save_params={'bbox_inches': 'tight'},
    ...     bar_params= {'mean': True, 'double_bars': True},
    ...     label_params={'size': 17,'repeat_labels': (False, True)},
    ...     title_params={'size': 20},
    ...     legend_params={'fontsize': 15, 'loc': 'best'},
    ...     rtext_params={'size': 14, 'stacked': True},
    ...     bartext_params={'size': 9},
    ...     margin_params={'x': 0.01, 'y': 0.01},
    ...     space_params={'no_xspace': False, 'no_yspace': False})
    >>> fig.show()
    """

    # Default parameters
    params = {'fname': None, 'fig_params': {},
              'save_params': {'bbox_inches': 'tight'},
              'bar_params': {'mean': False, 'double_bars': False},
              'label_params': {'size': 18, 'repeat_labels': (False, False)},
              'title_params': {'size': 21}, 'rtext_params': {'size': 14},
              'legend_params': {'fontsize': 14}, 'margin_params': {},
              'bartext_params': {'va': 'bottom', 'ha': 'center'},
              'space_params': {'no_xspace': False, 'no_yspace': False}}

    # Append the default parameters to those possibly provided as arguments
    params = check_keys(imgparams, params)

    # Check that a set of Axis was provided; if not, if `fig_params` was
    # provided, generate a subplots using these parameters
    if axs is None:
        fig, axs = plt.subplots(**params['fig_params'])
    else:
        # Retrieve the Figure from the axes
        fig = _retrieve_figure(axs)

    # Plot the bar values
    bar_core(axs, *x_y,
        mean=params['bar_params'].pop('mean'),
        double_bars=params['bar_params'].pop('double_bars'),
        **params['bar_params'])

    # Write the decorations (labels, titles & legends)
    set_decorations(axs, labels, titles, legends, rtexts,
        label_params=params['label_params'],
        title_params=params['title_params'],
        rtext_params=params['rtext_params'],
        legend_params=params['legend_params'])

    # Add the texts at the top of the bars, if any
    if bartext is not None:
        set_bartop_text(axs, bartext, **params['bartext_params'])

    # Reduce the space margins on the right & left of the plots
    if params['margin_params'] != {}:
        set_margins(axs, **params['margin_params'])

    # Remove the white spaces between plots (works only on subplots)
    if np.ndim(axs) > 0:
        remove_spaces(fig, **params['space_params'])

    # Save the figure in the specified file
    # If the filename contains directories, create them if they do not exist
    if params['fname'] not in (None, ''):
        fig.savefig(check_path(params['fname']), **params['save_params'])

    return fig

##############################################################################
