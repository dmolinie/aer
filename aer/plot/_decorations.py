""" Shorthand functions to decorate graph plots

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: February 2026
Last revised: April 2026

License: GPLv3
"""

__all__ = [
    'set_margins', 'remove_spaces', 'set_titles', 'set_labels',
    'set_legends', 'set_right_yaxis_texts', 'set_decorations',
    'set_bartop_text']

import numpy as np
import matplotlib.pyplot as plt

from aer.tools import check_keys, get_ndim


##############################################################################
##                          Plot Margins & Spaces                           ##
##############################################################################

#-------------------------   Set the Axes Margins   -------------------------#
def set_margins(axs, **margin_params):
    """ Set the x- and y-axis margins of (a set of) figure axes

    Take the axes of a figure and set their x & y margins. If `axs` is
    a 1D or 2D set of Axis, set the margins for any of them.

    Note that the xmargins & ymargins are the same for any axes; call
    this function specifically for every axis to set specific margins.

    N.B.: the `sharex` & `sharey` options of the Matplotlib's `subplots`
        function may overcome the margins. For instance, if `sharex` is
        set to True, any plots will share the same xmargins.

    Parameters
    ----------
    axes : (array of) Matplotlib's Axis
        The figure axes for which to set the x & y margins.

    Other Parameters
    ----------------
    **margin_params : inline keyword arguments, optional
        The parameters for the `Axis`' `margins` method; main options:
          - `x`, `y` (float): the x/y-axis margin values
        See this function for complementary explanations.

    Returns
    -------
    None : directly set the axis margins.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Define the x & ymargins
    >>> xmargins = (0.01, 0.10, 0.20)
    >>> ymargins = (0.01, 0.10, 0.20)

    # Generate dummy data
    >>> data = np.arange(10)

    # Set the same margins to any plots
    >>> fig, axs = plt.subplots(3, 1, sharex=False)
    >>> for axis in axs:
    ...     axis.plot(data)
    >>> set_margins(axs, x=xmargins[0], y=ymargins[0])

    # Set specific margins to every plot
    >>> fig, axs = plt.subplots(3, 1, sharex=False)
    >>> for axis, xmarg, ymarg in zip(axs, xmargins, ymargins):
    ...     axis.plot(data)
    ...     set_margins(axis, x=xmarg, y=ymarg)

    >>> plt.show()
    """
    if np.ndim(axs) == 0:
        axs.margins(**margin_params)
    else:
        for axis in np.ravel(axs):
            axis.margins(**margin_params)
#----------------------------------------------------------------------------#

#--------------------   Remove X/Y Space Between Axes   ---------------------#
def remove_spaces(fig,
    no_xspace=False, no_yspace=False, tight=True, **tight_params):
    """ Remove space between the axes of a figure

    Parameters
    ----------
    fig : Matplotlib's Figure
        The figure for which to remove the spaces.
    [OPT] no_xspace : bool
        If the horizontal space should be removed.
            :Default: False
    [OPT] no_yspace : bool
        If the vertical space should be removed.
            :Default: False
    [OPT] tight : bool
        If the `tight_layout` function is called or not.
            :Default: True

    Other Parameters
    ----------------
    **tight_params : inline keyword arguments, optional
        The parameters for the Matplotlib's `tight_layout` function;
        used only if `tight` is set to True; main options:
          - `pad` (float): offset between figure edge and subplots edges
          - `h_pad`, `w_pad` (float): padding between the subplots edges
        See this function for complementary explanations.

    Returns
    -------
    None : directly remove the spaces, if so.

    Examples
    --------
    >>> import matplotlib.pyplot as plt

    >>> fig, axs = plt.subplots(2, 3)
    >>> remove_spaces(fig, True, True, pad=2.0)
    >>> plt.show()
    """
    if tight:
        plt.tight_layout(**tight_params)    # Remove extra space
    if no_xspace:
        fig.subplots_adjust(wspace=0.)      # Remove horizontal space
    if no_yspace:
        fig.subplots_adjust(hspace=0.)      # Remove vertical space
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Decorations Setters                            ##
##############################################################################

#----------------------   Axis Dimension Dispatching   ----------------------#
def _set_decorators(axs, strings, function, **kwargs):
    """ Take the axes of a Figure, the string decorators and the deco-
    ration arguments and pass all of them to a decoration function; if
    `axs` is an Axis, pass the other arguments directly to `function`;
    if it is a 1D set of Axis, iterate `axs` and arrays; if it is a 2D
    set of Axis, iterate them twice. """
    # 1-graph plot
    if np.ndim(axs) == 0:
        function(axs, strings, **kwargs)
    # N-graph subplots
    elif np.ndim(axs) == 1:
        # If `strings` is a string, wrap it into a list to allow iterating it
        if get_ndim(strings) == 0:
            strings = [strings]
        # Iterate the axes and the string decorators
        for axis, string in zip(axs, strings):
            function(axis, string, **kwargs)
    # NxM-graph subplots
    elif np.ndim(axs) == 2:
        # If `strings` is a string, wrap it into a list to allow iterating it
        if get_ndim(strings) == 0:
            strings = [strings]
        for axis_row, string_row in zip(axs, strings):
            # If `string_row` is a string, wrap it into a list to allow iterating it
            if get_ndim(string_row) == 0:
                string_row = [string_row]
            # Iterate the axes and the string decorators
            for axis, string in zip(axis_row, string_row):
                function(axis, string, **kwargs)
#----------------------------------------------------------------------------#

#----------------------------   Labels Setter   -----------------------------#
def _labels_plot(axis, labels, **lab_params):
    """ Set the labels of a 1-plot figure """
    if labels[0] is not None:
        axis.set_xlabel(labels[0], **lab_params)    # X-label
    if labels[1] is not None:
        axis.set_ylabel(labels[1], **lab_params)    # Y-label

def _labels_subplot(axs, labels, commons, **lab_params):
    """ Set the labels of an N-plot figure """
    # Retrieve the number of axes in the figure (nb of rows & cols)
    # Add the x-labels
    if labels[0] is not None:
        # If there are several x-axes and x-labels
        if not isinstance(labels[0], str):
            for axis, lab in zip(axs, labels[0]):
                axis.set_xlabel(lab, **lab_params)
        # If `common-xlabels`, use the same xlabel for every axis
        elif commons[0] and isinstance(labels[0], str):
            for axis in axs:
                axis.set_xlabel(labels[0], **lab_params)
        # Else, set the xlabel on the very last axis (e.g. bottom)
        else:
            axs[-1].set_xlabel(labels[0], **lab_params)
    # Add the y-labels
    if labels[1] is not None:
        # If there are several y-axes and y-labels
        if not isinstance(labels[1], str):
            for axis, lab in zip(axs, labels[1]):
                axis.set_ylabel(lab, **lab_params)
        # If `common-texts`, use the same ylabel for every axis
        elif commons[1] and isinstance(labels[1], str):
            for axis in axs:
                axis.set_ylabel(labels[1], **lab_params)
        # Else, set the ylabel at the figure level (`supylabel`)
        else:
            axs[len(axs)//2].set_ylabel(labels[1], **lab_params)
#            axs[0].get_figure().supylabel(labels[1], size=size, x=0.)

def set_labels(axs, labels, **label_params):
    """ Write the labels on a figure (for plot & subplots)

    Take a set of figure axes, from either a single plot or subplots &
    the figure labels, and write them around the figure. The labels to
    write should be organized as (xlabel, ylabel); use `None` or empty
    string '' to skip a label (e.g. (None, `ylabel`) skips the x-label
    and writes only a label along the y-axis).

    In the case of a set of axes (e.g. `subplots`), the labels are writ-
    ten only once by default (`xlabel` as `axs[-1]` x-label and `ylabel`
    as figure y-label). The xlabels & ylabels can be written on any axes
    of `axs` by using the `repeat_labels` inline argument, which is a
    2-tuple of bools, organized as (repeat_xlabel, repeat_ylabel); set
    any of them to False to have the corresponding label written at the
    figure level only once, or to True to have it repeated for any axes.
    Notice that the repetition is done in row-major order.

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to write the labels on.
    labels : 2-array of (1D or 2D arrays of) string(s)
        The figure xlabel(s) and ylabel(s).

    Other Parameters
    ----------------
    **label_params : inline keyword arguments, optional
        The parameters for the `Axis`' `set_xlabel` & `set_ylabel` methods,
        plus the `repeat_labels` mechanism; main options:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
          - `repeat_labels` (2 bools): if the xlabel and ylabel should be
                repeated on any axes, as (rep_xlabel, rep_ylabel)
            :Default: repeat_labels=(False, False)
                + Default `Axis`' `set_xlabel` & `set_ylabel` parameters

    Returns
    -------
    None : directly write the labels on the figure.

    Examples
    --------
    >>> import matplotlib.pyplot as plt

    # 1-plot figure labels
    >>> fig = plt.figure()
    >>> set_labels(fig.gca(), ("X1", "Y1"))

    # N-plot figure labels
    >>> fig, axs = plt.subplots(2, 1)
    >>> labels = ("X", ("Y1", "Y2"))
    >>> set_labels(axs, labels)

    # Directly write the (common) labels
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> set_labels(axs[:, 0], ("X1", "Y1"), repeat_labels=(True, True))
    >>> set_labels(axs[:, 1], ("X2", "Y2"), repeat_labels=(False, False))

    # NxM-plot figure
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> set_labels(axs.T,
    ...     (("X1", ("", "Y1", "")), ("X2", "Y2")),
    ...     repeat_labels=(True, True))

    # Display the images
    >>> plt.show()
    """

    # Append the default parameters to those possibly provided as arguments
    params = check_keys(label_params, {'repeat_labels': (False, False)})
    common = params.pop('repeat_labels')

    # 1-graph plot
    if np.ndim(axs) == 0:
        _labels_plot(axs, labels, **params)
    # N-graph subplots
    elif np.ndim(axs) == 1:
        _labels_subplot(axs, labels, common, **params)
    # NxM-graph subplots
    elif np.ndim(axs) == 2:
        for axis_row, labs in zip(axs, labels):
            _labels_subplot(axis_row, labs, common, **params)
#----------------------------------------------------------------------------#

#----------------------------   Titles Setter   -----------------------------#
def _set_titles(axis, title, **title_params):
    """ Write a set of titles on a 1-plot, N-plot or NxM-plot figure """
    if title is not None:
        axis.set_title(title, **title_params)

def set_titles(axs, titles, **title_params):
    """ Write the titles on a figure (for plot & subplots)

    Take a set of figure axes, from either a single plot or subplots &
    the figure titles, and write them on the figure. If `axs` is a set
    of Axis, and that `titles` also is, write a title per axis; if it
    is a single string, write it at the top of the subplots (`axs[0]`
    or `axs[0, 0]` depending on the shape of `axs`).

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to write the title(s) on.
    titles : (1D or 2D array of) string(s)
        The figure title(s).

    Other Parameters
    ----------------
    **title_params : inline keyword arguments, optional
        The parameters for the `Axis`' `set_title` method; main options:
          - `size` (float): text font size
          - `pad` (float): text bounding box space pad
            :Default: default `Axis`' `set_title` parameters

    Returns
    -------
    None : directly write the titles on the figure.

    Examples
    --------
    >>> import matplotlib.pyplot as plt

    # Write the title of on a 1-plot figure
    >>> fig = plt.figure()
    >>> set_titles(fig.gca(), "T1", size=24)

    # Write a single title at the top of an N-plot figure
    >>> fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.80))
    >>> set_titles(axs, "T1")

    # Write the titles on an N-plot figure
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> set_titles(axs[:, 0], "T1", size=18)
    >>> set_titles(axs[:, 1], "T2", size=24, pad=12.)

    # Write the titles on an NxM-plot figure
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> set_titles(axs, (("T1", "T2"),), size=24)

    # Write the titles on an NxM-plot figure
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> set_titles(axs.T, ("T1", ("T21", "T22", "T23")), size=24)

    # Display the images
    >>> plt.show()
    """
    _set_decorators(axs, titles, _set_titles, **title_params)
#----------------------------------------------------------------------------#

#----------------------------   Legends Setter   ----------------------------#
def _set_legends(axis, legends, **legend_params):
    """ Write a set of titles on a 1-plot, N-plot or NxM-plot figure """
    if legends is not None:
        axis.legend([legends] if isinstance(legends, str) else legends,
                    **legend_params)

def set_legends(axs, legends, **legend_params):
    """ Write the legends on the axes of a figure (for plot & subplots)

    Take a set of figure axes, from either a single plot or subplots and
    the plot legends, and write them on the graphs. The legends to write
    should be organized as a tuple of tuples of strings, one string per
    colored plot, and one tuple of strings per graph (axis).

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to add the legends to.
    legends : array of (1D or 2D arrays of) string(s)
        The set of legends of the figure plots.

    Other Parameters
    ----------------
    **legend_params : inline keyword arguments, optional
        The parameters for to the `Axis`' `legend` method; main options:
          - `loc` (str): the position of the legend box
          - `fontsize` (float): legend font size
            :Default: default `legend` method's parameters.

    Returns
    -------
    None : directly write the legends on the figure plots.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Generate dummy data
    >>> tstp = np.linspace(0., 10., 100)
    >>> sin_10, cos_10 = np.sin(10.*tstp), np.cos(10.*tstp)
    >>> sin_50, cos_50 = np.sin(50.*tstp), np.cos(50.*tstp)

    # Plot the data on the figure
    >>> fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    >>> axs[0, 0].plot(tstp, sin_10)
    >>> axs[1, 0].plot(tstp, cos_10)
    >>> axs[2, 0].plot(tstp, sin_10)
    >>> axs[2, 0].plot(tstp, cos_10)
    >>> axs[0, 1].plot(tstp, sin_50)
    >>> axs[1, 1].plot(tstp, cos_50)
    >>> axs[2, 1].plot(tstp, sin_50)
    >>> axs[2, 1].plot(tstp, cos_50)

    # Set a specific legend for each plot
    >>> set_legends(axs[0, 0], "10 Hz Sine", loc='upper left')
    >>> set_legends(axs[1, 0], "10 Hz Cosine", loc='upper center')
    >>> set_legends(axs[2, 0], ("10 Hz Sine", "10 Hz Cosine"), loc='upper right')

    # Set the legends for a full set of axes at once
    >>> set_legends(axs[:, 1],
    ...     ("50 Hz Sine", "50 Hz Cosine", ("50 Hz Sine", "50 Hz Cosine")),
    ...     loc='upper right', fontsize=12)

    # Set the legends for all plots at once (row-major)
    >>> set_legends(axs,
    ...     (("10 Hz Sine", "50 Hz Cosine"),        # 1st row
    ...      ("10 Hz Cosine", "50 Hz Cosine"),      # 2nd row
    ...      (("10 Hz Sine", "10 Hz Cosine"),       # 3rd row, 1st col
    ...       ("50 Hz Sine", "50 Hz Cosine"))),     # 3rd row, 2nd col
    ...     loc='upper right', fontsize=12)

    # Set the legends for all plots at once (column-major)
    >>> set_legends(axs.T,
    ...     (("10 Hz Sine", "10 Hz Cosine", ("10 Hz Sine", "10 Hz Cosine")),  # 1st
    ...      ("50 Hz Sine", "50 Hz Cosine", ("50 Hz Sine", "50 Hz Cosine"))), # 2nd
    ...     loc='upper right', fontsize=12)

    >>> plt.show()
    """
    _set_decorators(axs, legends, _set_legends, **legend_params)
#----------------------------------------------------------------------------#

#-------------------------   Right y-axis Labels   --------------------------#
def _right_yaxis_core(axis, text, **rtext_params):
    """ Core statements for writing right y-axis side texts """
    if text is not None:
        axis_rgt = axis.twinx()
        axis_rgt.tick_params(right=False, labelright=False)
        axis_rgt.yaxis.set_label_position('right')
        axis_rgt.set_ylabel(text, **rtext_params)

def _right_yaxis_core_stacked(axis, text, **rtext_params):
    """ Core statements for writing right y-axis side stacked texts """
    esp = "\n     "
    _right_yaxis_core(axis, esp[1:]+esp.join(text), **rtext_params)

def set_right_yaxis_texts(axs, texts, stacked=False, **rtext_params):
    """ Set additional texts alongside the right y-axis

    Take the axes (graphs) and the associated labels to write alongside
    their right y-axes: hide the tick marks & texts and write the labels.

    The texts can be vertically stacked or written from left to right
    with a 90° anti-clockwise rotation; use `stacked` to force stacking.

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to write the y-axis labels on.
    texts : (1D or 2D arrays of) string(s)
        The side texts to write along the right y-axis.
    [OPT] stacked : bool
        If False, write any label with a 90° rotation (same orientation
        as a regular ylabel); else, for any label, stack its characters
        vertically and write the labels from top to bottom.
            :Default: False

    Other Parameters
    ----------------
    **rtext_params : inline keyword arguments, optional
        The parameters for the `Axis`' `set_ylabel` method; main options:
          - `size` (float): text font size
          - `rotation` (float): text orientation
          - `ha`, `va` (str): horizontal/vertical alignment
          - `labelpad` (float): text bounding box pad space
        See the Matplotlib's `Text` class properties for details.
            :Default: {rotation=0, va='center',  if 'stacked'
                      {rotation=90, va='top',    else
                + Default `Axis`' `set_ylabel` parameters

    Returns
    -------
    None : directly write the text labels on the graphs y-axes.

    Examples
    --------
    >>> import matplotlib.pyplot as plt

    # 1-plot figure
    >>> fig = plt.figure()
    >>> set_right_yaxis_texts(fig.gca(), "Label", True)

    # N-plot figure
    >>> fig, axs = plt.subplots(2, 1)
    >>> set_right_yaxis_texts(axs, (f"Img {i}" for i in range(len(axs))), True)

    # NxM-plot figure
    >>> fig, axs = plt.subplots(2, 2)
    >>> set_right_yaxis_texts(axs, (("", "Lab 01"), ("", "Lab 11")), True)
    """
    # If only one text and several Axis, set the text to the middle Axis
    if isinstance(texts, str) and np.ndim(axs) > 0:
        axs = axs[len(axs)//2]
    # Vertically stack the labels characters
    if stacked:
        params = check_keys(rtext_params, {'rotation': 0, 'va': 'center'})
        _set_decorators(axs, texts, _right_yaxis_core_stacked, **params)
    # 90°-rotate the labels to write
    else:
        params = check_keys(rtext_params, {'rotation': 90, 'va': 'top'})
        _set_decorators(axs, texts, _right_yaxis_core, **params)
#----------------------------------------------------------------------------#

#-------------------   Decoration Statements for Plots   --------------------#
def set_decorations(
    axs, labels=None, titles=None, legends=None, rtexts=None, **deco_args):
    """ Write the labels, titles and legends on the axes of a figure

    Take a set of figure axes and the labels (either regular or right y-
    axis ones), titles and legends to write, and set them on the figure.
    See the present module's `set_labels`, `set_titles`, `set_legends`
    and `set_right_yaxis_texts` functions for complementary details.

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to write the y-axis labels on.
    [OPT] labels : 2-array of (1D or 2D arrays of) string(s)
        The figure xlabels and ylabels.
            :Default: None (no label)
    [OPT] titles : (1D or 2D arrays of) string(s)
        The figure (sub)plot title(s).
            :Default: None (no title)
    [OPT] legends : array of (1D or 2D arrays of) string(s)
        The plot legends to write on the graphs.
            :Default: None (no legend)
    [OPT] rtexts : (1D or 2D arrays of) string(s)
        The side texts to write along the right y-axis, if any.
            :Default: None (no label)

    Other Parameters
    ----------------
    [OPT] label_params : dict
        The labels parameters passed to the `set_labels` function:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
          - `repeat_labels` (2 bools): if the xlabel and ylabel should be
                repeated on any axes, as (rep_xlabel, rep_ylabel)
        :Default: default `set_labels` parameters
    [OPT] title_params : dict
        The titles parameters passed to the `set_titles` function:
          - `size` (float): title font size
          - `pad` (float): title bounding box space pad
        :Default: default `set_titles` parameters
    [OPT] legend_params : dict
        The legend parameters passed to the `set_legends` parameters:
          - `loc` (str): the position of the legend box
          - `fontsize` (float): legend font size
        :Default: default `set_legends` parameters
    [OPT] rtext_params : dict
        The right y-axis side text parameters passed to the
        `set_right_yaxis_texts` function:
          - `size` (float): right y-axis sidetext font size
          - `labelpad` (float): right y-axis side text bounding box space pad
          - `stacked` (bool): vertically stack characters or rotate label
        :Default: default `set_right_yaxis_texts` parameters

    Returns
    -------
    None : directly write the labels & titles on the figure.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    # Generate dummy data
    >>> tstp = np.linspace(0., 1., 100)
    >>> sin_10, cos_10 = np.sin(10.*tstp), np.cos(10.*tstp)
    >>> sin_50, cos_50 = np.sin(50.*tstp), np.cos(50.*tstp)

    # Plot them on a single-plot figure
    >>> fig = plt.figure(figsize=(19.20, 10.80))
    >>> plt.plot(tstp, sin_10)
    >>> plt.plot(tstp, cos_10)
    >>> set_decorations(fig.gca(),
    ...     labels=("Time", "Amp"),
    ...     title="Sine & Cosine",
    ...     legends=("Sine", "Cosine"),
    ...     label_params={'size': 18, 'labelpad': 12.},
    ...     title_params={'size': 24, 'pad': 15.},
    ...     legend_params={'fontsize': 15, 'loc': 'best'})

    # Plot them on a 1D subplot figure
    >>> fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.80))
    >>> axs[0].plot(tstp, sin_10)   # Sine on 1st graph
    >>> axs[1].plot(tstp, cos_10)   # Cosine on 2nd graph
    >>> axs[2].plot(tstp, sin_50)   # Sine on 3rd graph
    >>> axs[2].plot(tstp, cos_50)   # Cosine on 3rd graph
    >>> set_decorations(axs,
    ...     labels=("Time", "Amp"),
    ...     titles="Sine & Cosine",
    ...     legends=("Sine", "Cosine", ("Sine", "Cosine")),
    ...     rtexts=("Sine", "Cosine", "Sine & Cosine"),
    ...     label_params={'size': 18, 'repeat_labels': (False, True)},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     rtext_params={'size': 14, 'labelpad': 12.})

    # Plot them on a 2D subplot figure (row-major)
    >>> fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    >>> axs[0, 0].plot(tstp, sin_10)
    >>> axs[0, 1].plot(tstp, cos_10)
    >>> axs[1, 0].plot(tstp, sin_50)
    >>> axs[1, 1].plot(tstp, cos_50)
    >>> set_decorations(axs,        # row-major
    ...     labels=(("", ("Amp 00", "Amp 01")),
    ...             (("Time 10", "Time 11"), "Amp 1X")),
    ...     titles=(("Sines", "Cosines"), ("", "")),
    ...     legends=(("10 Hz Sine", "10 Hz Cosine"), ("50 Hz Sine", "50 Hz Cosine")),
    ...     rtexts=(("", "10 Hz"), ("", "50 Hz")),
    ...     label_params={'size': 18, 'repeat_labels': (False, True)},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     legend_params={'loc': 'upper right'},
    ...     rtext_params={'stacked': False, 'size': 14, 'labelpad': 6.})

    # Plot them on a 2D subplot figure (column-major)
    >>> fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    >>> axs[0, 0].plot(tstp, sin_10)
    >>> axs[0, 1].plot(tstp, cos_10)
    >>> axs[1, 0].plot(tstp, sin_50)
    >>> axs[1, 1].plot(tstp, cos_50)
    >>> set_decorations(axs.T,      # Transpose for column-major
    ...     labels=(("Time X0", "Amp X0"), ("Time X1", "Amp X1")),
    ...     titles=("Sines", "Cosines"),
    ...     legends=(("10 Hz Sine", "50 Hz Sine"), ("10 Hz Cosine", "50 Hz Cosine")),
    ...     rtexts=("", ("10 Hz", "50 Hz")),
    ...     label_params={'size': 18, 'repeat_labels': (False, True)},
    ...     title_params={'size': 24, 'pad': 12.},
    ...     legend_params={'loc': 'upper right'},
    ...     rtext_params={'stacked': False, 'size': 14, 'labelpad': 6.})

    >>> plt.show()
    """

    # Append the default parameters to those possibly provided as arguments
#    params = {'label_params': {'repeat_labels': (False, False)},
    params = {'label_params': {}, 'title_params': {},
              'legend_params': {}, 'rtext_params': {}}
    params = check_keys(deco_args, params)

    # Set the labels, if any; 'labels' assumed to be (xlabel, ylabel)
    if labels is not None:
        set_labels(axs, labels, **params['label_params'])

    # Set the titles, if any
    if titles is not None:
        set_titles(axs, titles, **params['title_params'])

    # Set the legends, if any
    if legends is not None:
        set_legends(axs, legends, **params['legend_params'])

    # Set the side texts along the right y-axis, if any
    if rtexts is not None:
        set_right_yaxis_texts(axs, rtexts, **params['rtext_params'])
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Bar Chart Decorations                           ##
##############################################################################

def _set_bartop_text(axis, texts, **text_params):
    """ Take the axis of a bar chart figure and a set of strings & write
    them at the bars' top. The locations of the texts are computed from
    those of the bars themselves: retrieve the bars' horizontal centers
    as x-values and heights as y-values, and use these coordinates for
    the texts. The bars must have been drawn prior. """

    # 1-bar chart plot
    if np.ndim(texts) == 1:
        vals = axis.containers[0].datavalues            # Bars' heights
        bars = axis.containers[0].get_children()
        xpos = [b.get_center()[0] for b in bars]        # Bars' x locations
        # Add the texts at the bars' top
        for x, y, text in zip(xpos, vals, texts):
            axis.text(x, y, text, **text_params)
        max_x, max_y = np.max(xpos), np.max(vals)       # Max x & y coordinates
    # 2-bar chart plot
    else:
        max_x, max_y = 0., 0.
        # Run through the left bars, then the right bars
        for i, container in enumerate(axis.containers[:2]):
            vals = container.datavalues                 # Bars' heights
            bars = container.get_children()
            xpos = [b.get_center()[0] for b in bars]    # Bars' x locations
            # Add the texts at the bars' top
            for x, y, text in zip(xpos, vals, texts):
                axis.text(x, y, text[i], **text_params)
            # Max x & y coordinates among all (left & right) bars
            max_x, max_y = max(max_x, np.max(xpos)), max(max_y, np.max(vals))

    # Set an empty label at the maximum x- and y-values to get the Text object
    text = axis.text(max_x, max_y, " ", **text_params)
    # Add the Text's bbox to the axis' bbox and auto scale the plot area
    axis.autoscale_view()                                       # Mandatory (why?)
    bbox_abs = text.get_window_extent()                         # Absolute coords
    bbox_rel = bbox_abs.transformed(axis.transData.inverted())  # Relative coords
    axis.update_datalim(bbox_rel.get_points())                  # Set new limits
    axis.autoscale_view()                                       # Update view

def set_bartop_text(axs, texts, **text_params):
    """ Write texts at the top of the bars of a chart

    Take a set of figure axes, from either a single plot or subplots and
    a set of (short) texts, and write them at the bars' top. To this end,
    the texts' coordinates are retrieved from the bars themselves: their
    x-values are those of the bars' horizontal centers and their y-values
    are the bars' heights.

    Conceptually, the purpose of this function is to write the (rounded)
    values of the bars' heights just above them to make the plot clearer.

    N.B.: since the texts' coordinates are automatically retrieved from
        the bars, they must be plotted prior calling this function.

    N.B.: this function handles 1-plot, N-plot & NxM-plot figures; in
        the latter case, the axes are iterated in row-major order; to
        emulate a column-major order, simply transpose the axes prior.
        See the `Examples` section below for examples of use.

    Parameters
    ----------
    axs : (1D or 2D array of) Matplotlib's Axis
        The figure axes to write the title(s) on.
    texts : (1D or 2D array of) string(s), int(s) or float(s)
        The texts to write at the bars' top. Typically the bars' heights.

    Other Parameters
    ----------------
    **text_params : inline keyword arguments, optional
        The parameters for the Pyplot's `Text` class; main options:
          - `size` (float): text font size
          - `ha`, `va` (str): the horizontal/vertical alignment
            :Default: ha='center', va='bottom'

    Returns
    -------
    None : directly write the texts at the bars' top.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt


    ###--- Single bar plots example
    # Generate dummy data & add the mean
    >>> vals1 = np.arange(1, 31)                        # 1D data
    >>> vals2 = np.arange(30, 0, -1)                    # 1D data
    # Generate simple x-ticks
    >>> xpos1 = np.arange(1, len(vals1)+1)              # +1 for the mean
    >>> xpos2 = np.arange(1, len(vals2)+1)              # +1 for the mean

    # Plot the bar chart
    >>> fig, axs = plt.subplots(2, 1)
    >>> axs[0].bar(xpos1, vals1)
    >>> axs[1].bar(xpos2, vals2)

    # Add the values at the bars' top
    >>> vals1 = vals1.astype(int)
    >>> vals2 = vals2.astype(int)
    >>> set_bartop_text(axs, (vals1, vals2), size=8)

    # Display the figure
    >>> fig.show()


    ###--- Double bar plots example
    # Generate dummy data & add the mean
    >>> vals1 = np.arange(1, 31).reshape(-1, 2)         # 2D data
    >>> vals2 = np.arange(30, 0, -1).reshape(-1, 2)     # 2D data
    # Generate simple x-ticks
    >>> xpos1 = np.arange(1, len(vals1)+1)              # +1 for the mean
    >>> xpos2 = np.arange(1, len(vals2)+1)              # +1 for the mean

    # Plot the bar chart
    >>> fig, axs = plt.subplots(2, 1)
    >>> width = 0.4                                     # Bar width
    >>> axs[0].bar(xpos1-0.5*width, vals1[:, 0], width=width)
    >>> axs[0].bar(xpos1+0.5*width, vals1[:, 1], width=width)
    >>> axs[1].bar(xpos2-0.5*width, vals2[:, 0], width=width)
    >>> axs[1].bar(xpos2+0.5*width, vals2[:, 1], width=width)

    # Add the values at the bars' top
    >>> vals1 = vals1.astype(int)
    >>> vals2 = vals2.astype(int)
    >>> set_bartop_text(axs, (vals1, vals2), size=7)

    # Display the figure
    >>> fig.show()
    """
    params = check_keys(text_params, {'ha': 'center', 'va': 'bottom'})
    _set_decorators(axs, texts, _set_bartop_text, **params)
#----------------------------------------------------------------------------#

##############################################################################
