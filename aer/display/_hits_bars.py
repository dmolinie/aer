""" Shorthand functions to display bar plots

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2025
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=too-many-arguments, too-many-positional-arguments

__all__ = ['plot_hits']

import numpy as np
import matplotlib.pyplot as plt

from aer.tools import check_keys, check_path, make_iter
from aer.plot._core_plot import bar_core
from aer.plot._decorations import (
    set_bartop_text, set_decorations,
    set_margins, remove_spaces)


##############################################################################
##                   Classification Accuracy as Bar Chart                   ##
##############################################################################

def _plot_hits(axis, hits, double_bars, color, topline, params):
    """ Plot the bars, add the values (heights) at their top, draw a
    gray dashed line and set the ylimits between 0 and 120% `topline` """
    # Plot the bars
    bar_core(axis, hits, double_bars=double_bars, color=color, **params['bar_params'])
    # Draw a horizontal gray line at the maximal value
    if topline is not None:
        axis.axhline(topline, **params['hline_params'])
    # Set the values at the bars' top
    set_bartop_text(axis, hits.astype(int), **params['bartext_params'])
    # Set the ylimits be [0, 1.20*topline]
    axis.set_ylim([0., 1.20*topline])

def plot_hits(hits, sep_classes=False, topline=100.,
    labels=("Dataset #", "Accuracy [% Hits]"),
    titles=None,
    suptitle=None,
    legends=("Chunks", "Sequences"),
    rtexts=("Training", "Testing"),
    **imgparams):
    """ Plot the bar charts of the hits of several datasets

    Take a set of hit scores obtained on chunks (1D) or sequences (2D)
    and generate a figure composed of N set of axes (subfigures), one
    per array in `hits`; then, plot the i-th item of `hits` on the i-th
    set of axes (subfigure) as bar charts. Also draw an horizontal line
    at `topline` y-value to serve as reference (ignore it if `topline`
    is None). If the classes were separated in the hits, an additional
    dimension for the classes is expected.

    Additionally, use `titles` to entitle the subfigures (first axes),
    write `labels` along the subfigures' x- & y-axis (use the same la-
    bels for every subfigure) and `rtexts` along the right y-axis (can
    be used to specified subsets of data, e.g. 'Training' & 'Testing').

    If the classes were distinguished, a subfigure is set with K axes,
    one per class; else, use only one axis. The case when the classes
    are separated is checked as follows:
      - 1D: class-independent chunks
      - 3D: class-dependent sequences
      - 2D: either class-dependent chunks or class-independent sequences:
            if the length of the 2nd dim. is 2, consider they are class-
            independent sequences; else, they are class-dependent chunks.
    Note that the case when `hits` is a 2D array with a second dimension
    of size 2 is ambiguous and cannot be resolved; use the `sep_classes`
    argument to force consider that they represent classes on chunks.

    Note also that the above-described mechanism is applied to any hits
    arrays, so that it is possible to plot class-dependent hits on a
    figure, and class-independent hits on another one.

    The plot parameters, in particular the text, label and title para-
    meters and the bar specifications can be passed as inline keyword
    arguments (as `imgparams`; see `Other Parameters` section below).

    The figure is saved with name the `fname` keyword argument; it can
    be displayed afterwards using `plt.show()` or `fig.show()`.

    Parameters
    ----------
    hits : tuple of 1D, 2D or 3D array_like
        The accuracy scores achieved on sets of datasets; each such set
        can either be obtained when using chunks (1D) or sequences (2D),
        distinguishing the classes (+1 dimension) or not. The scores are
        expected to be NumPy 1D, 2D or 3D ndarrays, and all these arrays
        are expected to be wrapped into a tuple. The typical shape is:
            {if chunks:    ((N,) Nb_datasets, nb_classes)
            {if sequences: ((N,) Nb_datasets, 2, nb_classes)
        with `N` the number of sets. If a direct ndarray is passed ins-
        tead of a tuple, it is wrapped into a list before going forth.
    [OPT] sep_classes : bool
        Consider that the hits represent 2 classes obtained on chunks;
        used only in this specific case, to distinguish with the case
        when 2D class-independent sequences.
            :Default: False
    [OPT] topline : float
        The y-value of the horizontal line drawn on any axis (to show
        the maximal value). Use None to disable it.
            :Default: 100.
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels, organized as (x-label, y-label). The
        same labels are used for every subfigure. Use None to skip.
            :Default: ("Dataset #", "Accuracy [% Hits]")
    [OPT] titles : (tuple of) string(s)
        The subfigures' titles (set to first axis); can be left empty to
        remove the title.
            :Default: None (no title)
    [OPT] suptitle : string
        The overall figure's titles (Figure's `suptitle` method); can be
        used in addition to the subfigures' titles.
            :Default: None (no title)
    [OPT] legends : (tuple of) string(s)
        The plot legends; added only to the last axis of each subfigure;
        the same legends are used for any set of hits. Use None to skip.
            :Default: {if hits on chunks: None
                      {if on sequences: ("Chunks", "Sequences")
    [OPT] rtexts : (tuple of) string(s)
        The right y-axis side texts. Set to None to ignore.
            :Default: ("Training", "Testing")

    Other Parameters
    ----------------
    [OPT] fname : str
        The image-to-save pathname (path+name+ext).
            :Default: None (do not save the figure)
    [OPT] save_params : dict
        The parameters to save the figure, passed to the Matplotlib's
        `savefig`; used only if `fname` is not empty nor None:
          - `bbox_inches` (str): the margins of the saved graphic image
          - `dpi` (int): image DPI (used if saved as a raster graphic)
            :Default: {'bbox_inches': 'tight'}
    [OPT] fig_params : dict
        The figure parameters passed to the Matplotlib's `subplots` function:
          - `figsize` (2-tuple of floats): the figure dimensions
            :Default: {'figsize': (19.20, 10.80)}
    [OPT] subfig_params : dict
        The subfigure parameters (top & bottom figures) passed to the
        Matplotlib's `subfigures` function:
          - `hspace` (float): relative yspace between figures
          - `height_ratios` (2-tuple of ints): relative area between figures
            :Default: {'hspace': 0.1}
    [OPT] bar_params : dict
        The plot parameters passed to the `bar_core` function from the
            `plot` module:
          - `mean` (bool): if the last values of any array of `y`
                represent the array's mean.
          - `double_bars` (bool): if the bars should be doubled or not;
                if so, any array of `y` must contain 2 values.
          - `width` (float): the width of the bars
          - `color` ((set of) strings): the bar colors; if single color,
                use it for every bar; else, cycle the colors.
          - `tick_label` (array): the manual x-ticks labels
            :Default: {'mean': False}
    [OPT] hline_params : dict
        The hline parameters passed to Matplotlib's `Axis`' `axhline`:
          - `linestyle` (str): the style of the line
          - `color` (str): the color of the line
            :Default: {'linestyle': '--', 'color': 'lightgray'}
    [OPT] label_params : dict
        The labels parameters passed to the `set_labels` function from
        the `plot` module:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
          - `repeat_labels` (2 bools): if the xlabel and ylabel should
                be repeated on any axes, as (rep_xlabel, rep_ylabel)
            :Default: {'size': 18}
    [OPT] title_params : dict
        The titles parameters passed to the `set_titles` function from
        the `plot` module:
          - `size` (float): title font size
          - `pad` (float): title bounding box space pad
            :Default: {'size': 21, 'pad': 20.}
    [OPT] suptitle_params : dict
        The suptitle parameters passed to the Matplotlib's `Figure`'
        `suptitle` method:
          - `size` (float): suptitle font size
          - `x`, `y` (float): horizontal/vertical offsets
            :Default: {'size': 24, 'y': 1.05}
    [OPT] legend_params : dict
        The legend parameters passed to the `set_legends` function from
        the `plot` module:
          - `loc` (str): the position of the legend box
          - `fontsize` (float): legend font size
            :Default: {'fontsize': 15, 'loc': 'lower right'}
    [OPT] rtext_params : dict
        The right y-axis side text parameters passed to the
        `set_right_yaxis_texts` function from the `plot` module:
          - `size` (float): right y-axis sidetext font size
          - `labelpad` (float): right y-axis side text bounding box space pad
          - `stacked` (bool): vertically stack characters or rotate label
            :Default: {'size': 14, 'stacked': True}
    [OPT] bartext_params : dict
        The bar text parameters passed to the `set_bartop_text` function
        from the `plot` module:
          - `size` (float): text font size
          - `ha`, `va` (str): the horizontal/vertical alignment
            :Default: {'va': 'bottom', 'ha': 'center'}
    [OPT] margin_params : dict
        The margins parameters passed to the Matplotlib's `Axis`' `margins`
        method through the `set_margins` function:
          - `x`, `y` (float): the x/y-axis margin value
            :Default: {'x': 0.01, 'y': 0.1}
    [OPT] space_params : dict
        The space between plots passed to the `remove_spaces` function
        from the `plot` module:
          - `no_xspace` (bool): remove horizontal space between plots
          - `no_yspace` (bool): remove vertical space between plots
            :Default: {'no_xspace': False, 'no_yspace': True}

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # No-class hits on chunks
    >>> hits1 = 100. * np.linspace(0.1, 1.0, 15)
    >>> hits1 = np.hstack((hits1, hits1.mean()))
    >>> hits2 = 100. * np.linspace(1.0, 0.1, 15)
    >>> hits2 = np.hstack((hits2, hits2.mean()))
    >>> fig1 = plot_hits((hits1, hits2),
    ...     fname="No_class_chunks.pdf",
    ...     bar_params={'mean': True},
    ...     labels=("Dataset #", "Accuracy [% Hits]"),
    ...     titles="No-class hits on chunks",
    ...     rtexts=("Training", "Testing"))

    # Class hits on chunks
    >>> nb_chks = 10
    >>> nb_classes = 3
    >>> hits1 = 100. * np.linspace(0.1, 1.0, 2*nb_chks * nb_classes)
    >>> hits1 = hits1.reshape(2*nb_chks, nb_classes)
    >>> hits1 = np.vstack((hits1, hits1.mean(0)))
    >>> hits2 = 85. * np.linspace(1.0, 0.1, nb_chks * nb_classes)
    >>> hits2 = hits2.reshape(nb_chks, nb_classes)
    >>> hits2 = np.vstack((hits2, hits2.mean(0)))
    >>> fig2 = plot_hits((hits1, hits2),
    ...     bar_params={'mean': True},
    ...     labels=("Dataset #", "Accuracy [% Hits]"),
    ...     titles=(("Training datasets", "Testing datasets")),
    ...     suptitle="Class hits on chunks",
    ...     rtexts=("CLS1", "CLS2", "CLS3"),
    ...     title_params={'pad': 0.},
    ...     subfig_params={'hspace': 0.1},
    ...     suptitle_params={'y': 1.075})

    # No-class hits on sequences
    >>> nb_seqs = 4
    >>> nb_chks = 5
    >>> hits1 = 100. * np.linspace(0.1, 1.0, nb_seqs * 2*nb_chks * 2)
    >>> hits1 = hits1.reshape(nb_seqs*2*nb_chks, 2)
    >>> hits1 = np.vstack((hits1, hits1.mean(0)))
    >>> hits2 = 100. * np.linspace(1.0, 0.1, nb_seqs * nb_chks * 2)
    >>> hits2 = hits2.reshape(nb_seqs*nb_chks, 2)
    >>> hits2 = np.vstack((hits2, hits2.mean(0)))
    >>> fig3 = plot_hits((hits1, hits2),
    ...     sep_classes=False,
    ...     bar_params={'mean': True},
    ...     labels=("Dataset #", "Accuracy [% Hits]"),
    ...     suptitle="No-class hits on sequences",
    ...     rtexts=("Training", "Testing"),
    ...     suptitle_params={'y': 1.025})

    # Class hits on sequences
    >>> nb_seqs = 4
    >>> nb_chks = 5
    >>> nb_classes = 4
    >>> hits1 = 100. * np.linspace(0.1, 1.0, nb_seqs * 2*nb_chks * nb_classes * 2)
    >>> hits1 = hits1.reshape(nb_seqs*2*nb_chks, 2, nb_classes)
    >>> hits2 = 100. * np.linspace(1.0, 0.1, nb_seqs * nb_chks * nb_classes * 2)
    >>> hits2 = hits2.reshape(nb_seqs*nb_chks, 2, nb_classes)
    >>> fig4 = plot_hits((hits1, hits2),
    ...     labels=("Dataset #", "Accuracy [% Hits]"),
    ...     titles="Class hits on sequences",
    ...     rtexts=("Training", "Testing"))

    # Display the images
    >>> plt.show()
    """
    # pylint: disable=too-many-branches, too-many-locals

    # Default parameters
    params = {'fname': None, 'save_params': {'bbox_inches': 'tight'},
              'fig_params': {'figsize': (19.20, 10.80)},
              'subfig_params': {'hspace': 0.1},
              'bar_params': {'mean': False},
              'hline_params': {'linestyle': '--', 'color': 'lightgray'},
              'label_params': {'size': 18},
              'title_params': {'size': 21, 'pad': 20.},
              'suptitle_params': {'size': 24, 'y': 1.05},
              'legend_params': {'fontsize': 12, 'loc': 'lower right'},
              'rtext_params': {'size': 15, 'stacked': True},
              'bartext_params': {'va': 'bottom', 'ha': 'center'},
              'margin_params': {'x': 0.01, 'y': 0.1},
              'space_params': {'no_xspace': False, 'no_yspace': True}}

    # Append the default parameters to those possibly provided as arguments
    params = check_keys(imgparams, params)

    # Allow lone, direct np.ndarray as input
    if not isinstance(hits, (tuple, list)):
        hits = [hits]

    # Create the figure 2-graph figure
    fig = plt.figure(**params['fig_params'])
    subfigs = make_iter(fig.subfigures(len(hits), **params['subfig_params']))

    for i, hits_set in enumerate(hits):
        # Check if class-dependent hits on chunks, and get the nb of axes to build
        chk_cls = hits_set.ndim == 2 and (hits_set.shape[1] > 2 or sep_classes)
        nb_rows = hits_set.shape[-1] if hits_set.ndim == 3 or chk_cls else 1

        # Check if single (chunks) or double (sequences) bars
        double_bars = not(hits_set.ndim == 1 or chk_cls)

        # Build the axes on the subfigure (one per class)
        axs = subfigs[i].subplots(nb_rows, sharex=True)

        # 1-plot subfigure (1 class)
        if np.ndim(axs) == 0:
            _plot_hits(axs, hits_set, double_bars, 'C0', topline, params)
        # N-plot subfigure (N classes)
        elif np.ndim(axs) == 1:
            for j, (axis, hits_cls) in enumerate(zip(axs, hits_set.T)):
                _plot_hits(axis, hits_cls.T, double_bars, f'C{j}', topline, params)

        # Add some legends (use the default values only for sequences)
        if legends == ("Chunks", "Sequences"):
            if double_bars:
                subfigs[i].get_axes()[-1].legend(legends, **params['legend_params'])
        else:
            subfigs[i].get_axes()[-1].legend(legends, **params['legend_params'])

        # Check the right y-axis labels format
        if (isinstance(rtexts, (tuple, list))
            and len(rtexts) != len(subfigs[i].get_axes())):
            rtext = rtexts[i]
        else:
            rtext = rtexts

        # Write the x/y-axis labels & right y-axis side texts on the subfigure
        set_decorations(axs, labels, rtexts=rtext,
            label_params=params['label_params'],
            rtext_params=params['rtext_params'])

    # Remove the x-label of all but last (bottom) subfigures
    for subfig in subfigs[:-1]:
        for axis in subfig.get_axes():
            axis.set_xlabel('')

    # Write the figure overall title
    if suptitle is not None:
        fig.get_figure().suptitle(suptitle, **params['suptitle_params'])

    # Write the subfigures' titles
    if titles is not None:
        if isinstance(titles, str):
            titles = [titles]
        for subfig, title in zip(subfigs, titles):
            subfig.get_axes()[0].set_title(title, **params['title_params'])

    # Reduce the space margins on the right & left of the plots
    if params['margin_params'] != {}:
        for subfig in subfigs:
            set_margins(subfig.get_axes(), **params['margin_params'])

    # Remove useless white spaces
    remove_spaces(fig.get_figure(), **params['space_params'])

    # Save the figure in the specified file
    # If the filename contains directories, create them if they do not exist
    if params['fname'] not in (None, ''):
        fig.savefig(check_path(params['fname']), **params['save_params'])

    return fig.get_figure()

##############################################################################
