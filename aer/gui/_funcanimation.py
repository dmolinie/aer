""" Advanced Matplotlib's FuncAnimation classes with handling tools

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: May 2025
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=too-many-instance-attributes

__all__ = ['CLASSES', 'STEP', 'FuncAnimationOnePlot', 'FuncAnimationTwoPlots']

import time

import numpy as np
from matplotlib import animation
import sounddevice as sd
from torch import no_grad

from aer.data_tk._data_parser import get_hop_size_data

# Reference classes
CLASSES = {0: 'Background', 1: 'Glass', 2: 'Gunshot', 3: 'Screams'}

# Slicing step on display (to reduce the points)
STEP = 4


##############################################################################
##                           Core Animation Plot                            ##
##############################################################################

class _FuncAnimationBase():
    """ Base FuncAnimation class implementing the interacting functions """

    #-------------------------   Base Constructor   -------------------------#
    def __init__(self, figaxs):
        """ Base constructor

        Save the figure and its axes in the `figaxs` attribute, and de-
        clare (None) the other class attributes.

        Parameters
        ----------
        figaxs : 2-tuple as a Figure & its set of Axis
            The figure and its axes on which to plot the signals, orga-
            nized as (fig, axes).
        """

        # Plotting attributes (figure & axes)
        self.figaxs = (figaxs[0], figaxs[1])

        # Attributes that are set in the `launch` method
        self.index = 0                      # Last seen frame index
        self.pause = False                  # Animation is running or paused
        self.frate = None                   # Sampling frequency
        self.delay = None                   # Delay between two frames
        self.sizes = None                   # The full window & hop sizes
        self.frames = None                  # Nb of frames to plot per display
        self.fargs = None                   # `update` method args (the data)
        self.animation = None               # The core animation object
    #------------------------------------------------------------------------#

    #----------------------   Set Optional Arguments   ----------------------#
    def _set_params(self, delay=0.2, frames=8, frate=None):
        """ Set the semi-dynamic attributes

        Set the attributes that are constant during execution, but that
        may change between two executions. The other attributes should
        be set prior, in particular `fargs` if `frate` is not provided.

        Parameters
        ----------
        [OPT] delay : float
            The time delay (in seconds) between two animation frames.
                :Default: 0.2
        [OPT] frames : int
            The number of consecutive frames to plot per display.
                :Default: 8
        [OPT] frate : float
            The sampling rate; if None, estimate it from the chunks of
            timestamps passed as argument as `fargs[2]`.
                :Default: None
        """

        # Time delay between two plots
        self.delay = delay

        # Save the sampling frequency or retrieve it from the chunks of timestamps
        if frate is None:
            # The sampling frequency is the inverse of the time delay btw 2 frames
            self.frate = (len(self.fargs[2][1]) - 1)\
                         / (self.fargs[2][1, -1] - self.fargs[2][1, 0])
        else:
            self.frate = frate

        # Number of frames to plot per display:
        # (frames to display, current chunk index, chunks indices to plot)
        self.frames = (frames, (frames - 1) // 2, np.arange(frames))
    #------------------------------------------------------------------------#

    #------------------------   Pause/Change Frame   ------------------------#
    def toggle_pause(self):
        """ Pause or resume the animated plot

        If the animation is running, pause it, save the current frame
        number in the `frame` attribute and set the `pause` attribute
        to True; else, resume the animation and set `pause` to True.

        If it is paused and that the current frame is not that on which
        the animation was stopped (e.g. by calling the `previous_frame`
        or `next_frame` methods), then display each frame in-between by
        calling `next_frame` until going back to the originally stopped
        frame before resuming the animation.
        """
        # pylint: disable=protected-access
        if not self.pause:
            self.index = self.animation._save_seq[-1]   # Last seen frame index
            self.animation.pause()                      # Pause the animation
            self.pause = True                           # Set flag to paused
        else:
            # Resume the animation, but display anew the already seen frames
            # (one should not directly resume animation as it uses an iterator)
            while self.index < self.animation._save_seq[-1]:
                time.sleep(self.delay)                  # Delay the displays
                self._set_frame(1)                      # Update with next frame
                self.figaxs[0].canvas.flush_events()    # Empty the buffer
            self.animation.resume()                     # Resume animation
            self.pause = False                          # Set flag to running

    def previous_frame(self):
        """ If the animation is paused, go to the previous frame and
        decrease the `frame` attribute by 1 """
        if self.pause:                      # If the animation is pause,
            self._set_frame(-1)             # go to the previous frame

    def next_frame(self):
        """ If the animation is paused, go to the next frame (step the
        animation if never seen frame) and step the `frame` attribute """
        # pylint: disable=protected-access
        if self.pause:                      # If the animation is pause
            # If the current frame is not the latest seen by the iterator
            # of the `animation` object, display the next frame wrt the
            # current relative positioning (contained in self.index)
            if self.index <= self.animation._save_seq[-1]:
                self._set_frame(1)          # Go to the next frame
            # Else, increment the relative positioning, and step the
            # `animation` iterator to continue display
            else:
                self.index += 1             # Update the current positioning
                self.animation._step()      # Step the animation iterator
    #------------------------------------------------------------------------#

    #--------------------------   Save the Plot   ---------------------------#
    def save_plot(self):
        """ Save the current figure on a file named "Frame_{i}.png" """
        self.figaxs[0].savefig(f"Frame_{self.index}.pdf")
    #------------------------------------------------------------------------#

    #-------------------------   Actions on Event   -------------------------#
    def onpress(self, event):
        """ Action when clicking on a key

        Apply a specific action when clicking on either the spacebar,
        the left or right arrows, or `p`. The actions are:
          - spacebar: pause/resume the animation
          - left/right arrows: go the previous/next frame
          - `p`: save the animation figure current status as a png file

        Albeit this function is public, it should only be called by an
        event triggering/handling function, e.g.:
            fig.canvas.mpl_connect(`key_press_event`, self.onpress)
        """
        if event.key.isspace():             # Pause the animation
            self.toggle_pause()
        elif event.key == 'left':           # Go to the previous frame
            self.previous_frame()
        elif event.key == 'right':          # Go to the next frame
            self.next_frame()
        elif event.key == 'p':              # Save the current figure
            self.save_plot()
    #------------------------------------------------------------------------#

    #--------------------------   Plot Updating   ---------------------------#
    def _set_frame(self, i):
        """ Update & display the animation's frame """
        self.index = self.index + i         # Update the current frame index
        self.update(self.index)             # Update plots with the new frame
        self.figaxs[0].canvas.draw()        # Draw the new data on the graphs

    def update(self, i):
        """ The update method called by the animation to update the plot;
        virtual method here, must be reimplemented by child classes """
        raise NotImplementedError
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         1-Plot Animation Figure                          ##
##############################################################################

class FuncAnimationOnePlot(_FuncAnimationBase):
    r""" One-plot Matplotlib's FuncAnimation overload

    Take a 1-graph figure & its axes and provide some tools to instantiate
    and handle a FuncAnimation on this figure (pause, save image, etc.).
    Not a true overload, but a class built around a FuncAnimation object.

    Attributes
    ----------
    figaxs : 2-tuple as a Figure & its set of Axis
        The figure & its axes.
    index : int
        The last seen frame index.
    pause : bool
        If the animation is running or paused.
    frate : float
        The sampling frequency.
    delay : float
        The time delay between two frames.
    sizes : 2-tuple of ints
        The full window & hop sizes.
    frames : int
        The number of frames to plot per display.
    fargs : 4-tuple of np.ndarrays
        The `update` method args (the data).
    animation : FuncAnimation object
        The core animation object.

    Other Parameters
    ----------------
    [GLB] STEP : int, global variable
        The spacing between the values to display.

    Constructor
    -----------
    __init__(figaxs)

    Methods
    -------
    toggle_pause()
        Pause or resume the animated plot.
    previous_frame()
        If the animation is paused, go to the previous frame and
        decrease the `frame` attribute by 1.
    next_frame()
        If the animation is paused, go to the next frame (step the
        animation if never seen frame) and step the `frame` attribute.
    save_plot()
        Save the current figure on a file named "Frame_{i}.png".
    onpress(event)
        Action when clicking on a key.
    launch(fargs, repeat=True, **kwargs)
        Launch the animated plot.
    update(i)
        Update the plot with a new set of frames.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import aer.data_tk as dtk

    # Chunk parameters
    >>> frate = 1000                            # Sampling frequency
    >>> frm_duration = 100e-3                   # Frame duration in seconds
    >>> hop_duration = 50e-3                    # Hop duration in seconds
    >>> frm_size = int(frate * frm_duration)    # Frame size in samples
    >>> hop_size = int(frate * hop_duration)    # Hop size in samples

    # Generate example timestamps & data (50 frames)
    >>> tstp = np.linspace(0, 10, 50 * frm_size)
    >>> data = np.sin(10.*tstp)

    # Pad the timestamps & data
    >>> offset = dtk.pad_offset(len(data), frm_size, hop_size)  # Needed offset
    >>> tstp = dtk.pad(tstp, offset)                            # Pad timestamps
    >>> data = dtk.pad(data, offset)                            # Pad the data

    # Build the time & data chunks
    >>> tstp_chks = dtk.build_chunks(tstp, frm_size, hop_size)  # Time chunks
    >>> data_chks = dtk.build_chunks(data, frm_size, hop_size)  # Data chunks

    # Instantiate a new figure
    >>> fig, axs = plt.subplots(1, 1, layout='constrained', sharey=True)
    >>> axs.locator_params(axis='x', nbins=12)

    # Figure labels
    >>> axs.set_title("Sinus signal & Focused chunk", size=17)
    >>> axs.set_xlabel("Time [s]", size=15)
    >>> axs.set_ylabel("Amplitude [a.u]", size=15)

    # Build & launch the animation
    >>> ani_gui = FuncAnimationOnePlot((fig, axs))
    >>> ani_gui.launch(
    ...     (tstp, data, tstp_chks, data_chks),
    ...     repeat=True, delay=hop_size/frate, frames=8, frate=frate)
    >>> plt.show()
    """

    #------------------------   Class Constructor   -------------------------#
    def __init__(self, figaxs):
        """ Instantiate a FuncAnimationOnePlot object (constructor)

        Save the figure and its axes in the `figaxs` attribute, set its
        plots and declare (None) the other class attributes.

        Parameters
        ----------
        figaxs : 2-tuple as a Figure & its set of Axis
            The figure and its axes on which to plot the signals, orga-
            nized as (fig, axes).

        Examples
        --------
        >>>
        """

        # Set the base object attributes
        super().__init__(figaxs)

        # Instantiate the lines of the axis
        self.lines = (self.figaxs[1].plot([], [])[0],   # Main plot
                      self.figaxs[1].plot([], [])[0])   # Current chunk

#        self.figaxs[1].locator_params(axis='x', nbins=10)
        self.figaxs[1].legend(("Flux audio", "Partie en entrée"), loc='lower left')

        # Set the current chunk in red in the main graph (left)
        self.lines[1].set_color('red')
    #------------------------------------------------------------------------#

    #---------------------   Animation Initialization   ---------------------#
    def launch(self, fargs, repeat=True, **kwargs):
        """ Launch the animated plot

        Take the data to display and initialize the class attributes and
        launch the animated plot.

        Parameters
        ----------
        fargs : 4-tuple of np.ndarrays
            The data in the format of the FuncAnimation class; the data
            are passed to the `update` method in their positional order:
              - tstp (1D array): the timestamps - abscissa
              - data (1D array): the data - ordinates
              - tstp_chks (2D array): the chunks of timestamps
              - data_chks (2D array): the chunks of data
            See `update` method for detail.
        [OPT] repeat : bool
            If the animation should be looped or not.
                :Default: True

        Other Parameters
        ----------------
        **kwargs : inline keyword arguments, optional
            The optional parameters that may be changed btw executions:
              - delay (float): the time delay (in seconds) between displays
              - frames (int): the number of frames to plot per display
              - frate (float): the data sampling frequency; if not provided,
                    retrieve it from the chunks of timestamps.
                :Default: delay=0.2, frames=8, frate=None
        """

        # Reset the display parameters
        self.index = 0                              # Last seen frame index
        self.pause = False                          # Animation is running

        # Save the data in the `fargs` attribute
        self.fargs = fargs

        # Set the time delay (`delay`), sampling frequency (`frate`) &
        # number of frames to show per display (`frames`)
        self._set_params(**kwargs)

        # Set the window size (`self.frames[0]` frames) & hop size
        win_size = fargs[3].shape[-1]
        hop_size = get_hop_size_data(len(fargs[3]), win_size, len(fargs[1]))
        self.sizes = (win_size + (self.frames[0]-1)*hop_size, hop_size)

        # Set the figure vertical bounds
        self.figaxs[1].set_ylim(fargs[1].min(), fargs[1].max())

        # Set & start the animation object
        self.animation = animation.FuncAnimation(
            self.figaxs[0], self.update,            # Plot & update func
            frames=len(fargs[2])-self.frames[1],    # Total nb of frames
            interval=1000.*self.delay,              # Delay btw 2 frames
            repeat=repeat)                          # Loop the animation
    #------------------------------------------------------------------------#

    #----------------------   Animation Plot Update   -----------------------#
    def update(self, i):
        """ Update the plot with a new set of frames

        Take an integer i and, by denoting `wins` the number of frames
        to display (1st component of the `frames` attribute), plot the
        wins consecutive frames of data starting from index i*hop_size,
        where `hop_size` is the hop size (set in the `launch` method).
        The data are read from the original 1D data vector (`fargs[1]`)
        to avoid discontinuities caused by the hops between the chunks.
        Also, plot the `(wins-1)//2`-th chunk in red to highlight it.

        Use the `STEP` global variable as spacing between values to
        dynamically reduce the number of points to display.

        Note that the components of the `fargs` attribute (cf. below)
        must be organized as: (tstp, data, tstp_chks, data_chunks).

        This function is designed to be passed as a callback of a more
        general function/object using an iterator (e.g. FuncAnimation).
        It can be manually called; if so, prefer an iterative fashion.

        Parameters
        ----------
        i : int
            The current frame position. Prefer an iterator if possible.

        Other Parameters
        ----------------
        [GLB] STEP : int, global variable
            The spacing between the values to display.
        [ATR] fargs : class attribute - 4-tuple of np.ndarrays
            The data to display, organized as:
              - tstp (1D array): the timestamps used as abscissa
              - data (1D array): the data used as ordinates
              - tstp_chks (2D array): the chunks of timestamps
              - data_chks (2D array): the chunks of data

        Returns
        -------
        lines : 2-tuple of matplotlib Line2D objects
            The Line2D plots, one for the data frames (blue) and the
            second for the highlighted chunk (red).
        """

        # Plot y-margin (blank space before & after the plot)
        margins = 0.001 * self.frames[1]

        # Determine the relative positions of the data to display
        pos_sig = i*self.sizes[1]                       # Signal absolute index
        pos_chk = (self.frames[2]+i)[self.frames[1]]    # Chunks relative index

        # Extract the time stamps for both graphs
        tps_lf = self.fargs[0][pos_sig:pos_sig+self.sizes[0]:STEP]
        tps_rg = self.fargs[2][pos_chk, ::STEP]

        # Plot the group of `WIN` chunks in the left graph
        # (use the raw, original (but padded) 1D signal vector)
        self.lines[0].set_xdata(tps_lf)
        self.lines[0].set_ydata(self.fargs[1][pos_sig:pos_sig+self.sizes[0]:STEP])

        # Highlight the current chunk (in red) in the left graph
        # (use the lists of chunks to that purpose)
        self.lines[1].set_xdata(tps_rg)
        self.lines[1].set_ydata(self.fargs[3][pos_chk, ::STEP])

        # Set the graph xticks
        self.figaxs[1].set_xlim([tps_lf[0]-margins, tps_lf[-1]+margins])

        return self.lines
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         2-Plot Animation Figure                          ##
##############################################################################

class FuncAnimationTwoPlots(_FuncAnimationBase):
    r""" Two plots Matplotlib's FuncAnimation overload

    Take a 2-graph figure & its axes and provide some tools to instantiate
    and handle a FuncAnimation on this figure. The main plot is displayed
    on the first graph and the focused chunk is displayed on both graphs.
    Not a true overload, but a class built around a FuncAnimation object.

    Attributes
    ----------
    figaxs : 2-tuple as a Figure & its set of Axis
        The figure & its axes.
    index : int
        The last seen frame index.
    pause : bool
        If the animation is running or paused.
    frate : float
        The sampling frequency.
    delay : float
        The time delay between two frames.
    sizes : 2-tuple of ints
        The full window & hop sizes.
    frames : int
        The number of frames to plot per display.
    fargs : 6-tuple of arrays
        The `update` method args (the data).
    animation : FuncAnimation object
        The core animation object.

    Other Parameters
    ----------------
    [GLB] STEP : int, global variable
        The spacing between the values to display.
    [GLB] CLASSES : dict, global variable
        The equivalence between a class number and its text.

    Constructor
    -----------
    __init__(figaxs)

    Methods
    -------
    toggle_pause()
        Pause or resume the animated plot.
    previous_frame()
        If the animation is paused, go to the previous frame and
        decrease the `frame` attribute by 1.
    next_frame()
        If the animation is paused, go to the next frame (step the
        animation if never seen frame) and step the `frame` attribute.
    save_plot()
        Save the current figure on a file named "Frame_{i}.png".
    onpress(event)
        Action when clicking on a key.
    launch(fargs, repeat=True, **kwargs)
        Launch the animated plot.
    update(i)
        Update the plot with a new set of frames.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import torch
    >>> import aer.data_tk as dtk

    >>> class Label():
    ...     ''' Emulate a very simple Tkinter Label '''
    ...     def __init__(self, end='\n'):
    ...         self.end = end
    ...     def config(self, text, **kwargs):
    ...         print(text, end=self.end)

    # Chunk parameters
    >>> frate = 1000                            # Sampling frequency
    >>> frm_duration = 100e-3                   # Frame duration in seconds
    >>> hop_duration = 50e-3                    # Hop duration in seconds
    >>> frm_size = int(frate * frm_duration)    # Frame size in samples
    >>> hop_size = int(frate * hop_duration)    # Hop size in samples

    # Generate example timestamps & data (50 frames)
    >>> tstp = np.linspace(0, 10, 50 * frm_size)
    >>> data = np.sin(10.*tstp)

    # Pad the timestamps & data
    >>> offset = dtk.pad_offset(len(data), frm_size, hop_size)  # Needed offset
    >>> tstp = dtk.pad(tstp, offset)                            # Pad timestamps
    >>> data = dtk.pad(data, offset)                            # Pad the data

    # Build the time & data chunks
    >>> tstp_chks = dtk.build_chunks(tstp, frm_size, hop_size)  # Time chunks
    >>> data_chks = dtk.build_chunks(data, frm_size, hop_size)  # Data chunks

    # Convert the data into Torch tensors
    >>> tens_chks = torch.Tensor(data_chks)                     # Torch tensors
    >>> tens_chks = tens_chks.unsqueeze(1)                      # NCW, with C=1

    # Generate the random classes (one per chunk)
    >>> classes = np.random.random((len(data_chks), 4))

    # Generate some (3) events (delineated by their start & ending times)
    >>> events = np.array(((1.1, 1.5), (2.3, 4.9), (7.3, 9.5)))
    >>> events = np.array(events*frate, int)

    # Build a dummy Torch-like model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(frm_size, 4),
    ...     torch.nn.Softmax(1))

    # Build two empty Tk Labels
    >>> labels_ref = {'number': Label(' '), 'class': Label('\n')}
    >>> labels_est = {'number': Label(' '), 'class': Label('\n')}

    # Instantiate a new figure
    >>> fig, axs = plt.subplots(1, 2, layout='constrained', sharey=True)
    >>> axs[0].locator_params(axis='x', nbins=10)
    >>> axs[1].locator_params(axis='x', nbins=5)

    # Figure labels
    >>> axs[0].set_title("Sinus signal & Focused chunk", size=15)
    >>> axs[1].set_title("Chunk used as model input", size=15)
    >>> axs[0].set_xlabel("Time [s]", size=14)
    >>> axs[1].set_xlabel("Time [s]", size=14)
    >>> axs[0].set_ylabel("Amplitude [a.u.]", size=14)

    # Build & launch the animation
    >>> ani_gui = FuncAnimationTwoPlots((fig, axs), labels_ref, labels_est)
    >>> ani_gui.launch(
    ...     (tstp, data, tstp_chks, tens_chks, events, classes), model,
    ...     repeat=True, delay=hop_size/frate, frames=8, frate=frate)
    >>> plt.show()
    """

    #------------------------   Class Constructor   -------------------------#
    def __init__(self, figaxs, labels_ref, labels_est):
        """ Instantiate a FuncAnimationTwoPlots object (constructor)

        Save the figure and its axes in the `figaxs` attribute, set its
        plots and declare (None) the other class attributes.

        Parameters
        ----------
        figaxs : 2-tuple as a Figure & its set of Axis
            The figure and its axes on which to plot the signals, orga-
            nized as (fig, axes).
        labels_ref : 2-dict of 2 Tkinter Label objects
            The Tk Labels on which to display the reference class for
            every chunk, as a dictionary of at least 2 keys:
              - number (Tk Label): used to display the class number
              - class (Tk Label): used to display the class name
        labels_est : 2-dict of 2 Tkinter Label objects
            Same as `labels_est` but for the estimated classes outputted
            by the Torch model.
        """

        # Set the base object attributes
        super().__init__(figaxs)

        # Torch model used to predict the classes (set in the 'launch' method)
        self.model = None

        # Instantiate the lines of the axis
        self.lines = (self.figaxs[1][0].plot([], [])[0],    # Main plot
                      self.figaxs[1][0].plot([], [])[0],    # Current chunk
                      self.figaxs[1][1].plot([], [])[0])    # Right plot (chunk)

#        self.figaxs[1][0].locator_params(axis='x', nbins=10)
#        self.figaxs[1][1].locator_params(axis='x', nbins=5)
        self.figaxs[1][0].legend(("Flux audio", "Partie en entrée"), loc='lower left')

        # Set the current chunk in red in the main graph (left)
        self.lines[1].set_color('red')

        # Tkinter's Labels to be updated
        self.labels_ref = labels_ref
        self.labels_est = labels_est
    #------------------------------------------------------------------------#

    #---------------------   Animation Initialization   ---------------------#
    def launch(self, fargs, model, repeat=True, **kwargs):
        """ Launch the animated plot

        Take the data to display, the Torch model and the sampling rate,
        and initialize the class attributes and launch the animated plot.

        If the data sampling rate is not provided, retrieve it from the
        chunks of timestamps passed as argument in `fargs`.

        Parameters
        ----------
        fargs : 6-tuple of np.ndarrays
            The data in the format of the FuncAnimation class; the data
            are passed to the `update` method in their positional order:
              - tstp (1D array): the timestamps - abscissa
              - data (1D array): the data - ordinates
              - tstp_chks (2D array): the chunks of timestamps
              - data_chks (3D tensor): the chunks of data
              - events (2D array): the events boundering indexes
              - classes (2D array): the chunks of classes
            See `update` method for detail.
        model : Torch model
            The model used to predict the class of the every chunk.
        [OPT] repeat : bool
            If the animation should be looped or not.
                :Default: True

        Other Parameters
        ----------------
        **kwargs : inline keyword arguments, optional
            The optional parameters that may be changed btw executions:
              - delay (float): the time delay (in seconds) between displays
              - frames (int): the number of frames to plot per display
              - frate (float): the data sampling frequency; if not provided,
                    retrieve it from the chunks of timestamps.
                :Default: delay=0.2, frames=8, frate=None
        """

        # Reset the display parameters
        self.index = 0                              # Last seen frame index
        self.pause = False                          # Animation is running

        # Set the data & model in local attributes
        self.fargs = fargs                          # Data to display
        self.model = model                          # (Trained) Torch model

        # Set the time delay (`delay`), sampling frequency (`frate`) &
        # number of frames to show per display (`frames`)
        self._set_params(**kwargs)

        # Set the window size (`self.frames[0]` frames) & hop size
        win_size = fargs[3].shape[-1]
        hop_size = get_hop_size_data(len(fargs[3]), win_size, len(fargs[1]))
        self.sizes = (win_size + (self.frames[0]-1)*hop_size, hop_size)

        # Set the figure vertical bounds
        limits = (fargs[1].min(), fargs[1].max())
        self.figaxs[1][0].set_ylim(limits)
        self.figaxs[1][1].set_ylim(limits)

        # Set & start the animation object
        self.animation = animation.FuncAnimation(
            self.figaxs[0], self.update,            # Plot & update func
            frames=len(fargs[2])-self.frames[1],    # Total nb of frames
            interval=1000.*self.delay,              # Delay btw 2 frames
            repeat=repeat)                          # Loop the animation
    #------------------------------------------------------------------------#

    #----------------------   Animation Plot Update   -----------------------#
    def update(self, i):
        """ Update the plot with a new set of frames

        Assuming the figure (`figaxs` attribute) has two graphs (axes),
        plot several frames of data and the current chunk in the first
        graph, and a zoom on the chunk in the second graph.

        Also, assuming the Tkinter Labels are set, use `labels_ref` to
        display the reference class number and name; additionally, pass
        the current chunk to the Torch model (`model` attribute) to get
        the estimated class and use `labels_est` to display this number
        and class name. The vector of the reference classes is assumed
        to be the 6-th (last) component of the `fargs` attribute.

        For the first graph, by denoting `wins` the number of frames
        to display (1st component of the `frames` attribute), plot the
        wins consecutive frames of data starting from index i*hop_size,
        where `hop_size` is the hop size (set in the `launch` method).
        The data are read from the original 1D data vector (`fargs[1]`)
        to avoid discontinuities caused by the hops between the chunks.
        Also, plot the `(wins-1)//2`-th chunk in red to highlight it.
        Additionally, plot the current chunk on the second graph.

        Use the `STEP` global variable as spacing between values to
        dynamically reduce the number of points to display.

        Finally, for every frame, check if it is the start of an event;
        if so, extract the data spanning all across this event and read
        (audio speakers) this event while displaying its data.

        Note that the components of the `fargs` attribute (cf. below)
        must be organized as:
            (tstp, data, tstp_chks, data_chunks, events, classes).

        This function is designed to be passed as a callback of a more
        general function/object using an iterator (e.g. FuncAnimation).
        It can be manually called; if so, prefer an iterative fashion.

        Parameters
        ----------
        i : int
            The current frame position. Prefer an iterator if possible.

        Other Parameters
        ----------------
        [GLB] STEP : int, global variable
            The spacing between the values to display.
        [GLB] CLASSES : dict, global variable
            The equivalence between a class number and its text.
        [ATR] fargs : class attribute - 6-tuple of np.ndarrays
            The data to display, organized as:
              - tstp (1D array): the timestamps used as abscissa
              - data (1D array): the data used as ordinates
              - tstp_chks (2D array): the chunks of timestamps
              - data_chks (3D tensor): the chunks of data
              - events (2D array): the events boundering indexes
              - classes (2D array): the chunks of classes

        Returns
        -------
        lines : 3-tuple of matplotlib Line2D objects
            The Line2D plots, one for the data frames (blue) and the
            second for the highlighted chunk (red), and the last one
            for the right graph plot.
        """

        # Plot vertical margins
        marg_rgt = 0.001                                # Right graph margins
        marg_lft = self.frames[1] * marg_rgt            # Left graph margins

        # Determine the relative positions of the data to display
        pos_sig = i*self.sizes[1]                       # Signal absolute index
        pos_chk = (self.frames[2]+i)[self.frames[1]]    # Chunks relative index

        # Check if the current frame time is between the times of any events
        pos = np.where((pos_sig <= self.fargs[4][:, 0])
                       & (self.fargs[4][:, 0] < pos_sig+self.sizes[1]))[0]

        # If the frame is the start of an event, read this event
        if len(pos) > 0:
            sd.play(self.fargs[1][self.fargs[4][pos[-1], 0]:\
                                  self.fargs[4][pos[-1], 1]], self.frate)

        # Extract the time stamps for both graphs
        tps_lf = self.fargs[0][pos_sig:pos_sig+self.sizes[0]:STEP]
        tps_rg = self.fargs[2][pos_chk, ::STEP]

        # Plot the group of `WIN` chunks in the left graph
        # (use the raw, original (but padded) 1D signal vector)
        self.lines[0].set_xdata(tps_lf)
        self.lines[0].set_ydata(self.fargs[1][pos_sig:pos_sig+self.sizes[0]:STEP])

        # Highlight the current chunk (in red) in the left graph
        # (use the lists of chunks to that purpose)
        self.lines[1].set_xdata(tps_rg)
        self.lines[1].set_ydata(self.fargs[3][pos_chk, 0, ::STEP])  # NCW

        # Plot (focus) the current chunk alone in the right graph
        # (use the lists of chunks to that purpose)
        self.lines[2].set_xdata(tps_rg)
        self.lines[2].set_ydata(self.fargs[3][pos_chk, 0, ::STEP])  # NCW

        # Set the xticks of both graphs
        self.figaxs[1][0].set_xlim([tps_lf[0]-marg_lft, tps_lf[-1]+marg_lft])
        self.figaxs[1][1].set_xlim([tps_rg[0]-marg_rgt, tps_rg[-1]+marg_rgt])

        # Use the Torch model to predict the class of the actual chunk
        with no_grad():
            pred = self.model(self.fargs[3][pos_chk].unsqueeze(0)) # Forward
            class_ref = self.fargs[5][pos_chk].argmax()   # True class
            class_est = int(pred.argmax())                # Estimated class

        # Set the color to green if the prediction is right, else set it to red
        color = 'green' if class_ref == class_est else 'red'

        # Display the true chunk class number and name in the left labels
        self.labels_ref['number'].config(text=str(class_ref))
        self.labels_ref['class'].config(text=CLASSES[class_ref])

        # Display the estimated chunk class number and name in the right labels
        self.labels_est['number'].config(text=str(class_est), fg=color)
        self.labels_est['class'].config(text=CLASSES[class_est], fg=color)

        return self.lines
    #------------------------------------------------------------------------#

##############################################################################
