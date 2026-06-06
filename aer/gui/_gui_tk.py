""" Tkinter-based Graphical User Interface

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: May 2025
Last revised: April 2026

License: GPLv3
"""

__all__ = ['TkInterfaceOnePlot', 'TkInterfaceTwoPlots']

import tkinter as tk
from PIL import ImageGrab

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aer.gui._funcanimation import (
    CLASSES, FuncAnimationOnePlot, FuncAnimationTwoPlots)


##############################################################################
##                            Tkinter-based GUI                             ##
##############################################################################

class _TkInterfaceBase():
    """ Tkinter-based GUI to embed an animation """

    #-------------------------   Base Constructor   -------------------------#
    def __init__(self, fig, axs):
        """ Base constructor of the GUI

        Save the figure and its axes in the `figaxs` attribute, instan-
        tiate the Tkinter object, place the figure canvas at the top of
        the interface and declare (None) the other class attributes.

        Parameters
        ----------
        fig : Matplotlib Figure object
            The figure on which to plot the signals.
        axs : Matplotlib Axes object or array of such objects
            The axes of the figure.
        """

        # Instantiate a Tkinter object
        self.root = tk.Tk()

        # Get the figure and its axes
        self.figaxs = (fig, axs)

        # Set a drawing area at the interface top and place the figure in it
        self.figure = FigureCanvasTkAgg(self.figaxs[0], master=self.root)
        self.figure.draw()
        self.figure.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Declare the buttons (interface bottom); set by `_set_buttons`
        self.buttons = {}

        # Declare the interface frames (set in the child class)
        self.frames = None

        # Declare the animation (interface canvas); set by `build_animation`
        self.animation = None
    #------------------------------------------------------------------------#

    #-----------------------   GUI Actions on Event   -----------------------#
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
            self.animation.toggle_pause()
        elif event.key == 'left':           # Go to the previous frame
            self.animation.previous_frame()
        elif event.key == 'right':          # Go to the next frame
            self.animation.next_frame()
        elif event.key == 'p':              # Save the GUI display
            self.save_plot()

    def save_plot(self):
        """ Take a "screenshot" (using PIL) of the screen area occupied
        by the interface and save it as a png image ("Frame_{i}.png") """
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        image = ImageGrab.grab(bbox=(x, y, width+x, height+y))
        image.save(f"Frame_{self.animation.index}.png")
    #------------------------------------------------------------------------#

    #-------------   Buttons Instantiation & Action bounding   --------------#
    def _set_buttons(self):
        """ Instantiate the buttons (`buttons`) and bound the actions on
        clicking; the animation (`animation`) must be instantiated """
        self.buttons = {
            'prev': tk.Button(self.frames['but'], text="Previous",
                              command=self.animation.previous_frame),
            'stop': tk.Button(self.frames['but'], text="Pause/Resume",
                              command=self.animation.toggle_pause),
            'next': tk.Button(self.frames['but'], text="Next",
                              command=self.animation.next_frame),
            'save': tk.Button(self.frames['but'], text="Save plot",
                              command=self.save_plot),
            'quit': tk.Button(self.frames['but'], text="Quit",
                              command=self.root.destroy)}

    def _place_buttons(self):
        """ Place the buttons at the interface bottom """
        self.frames['but'].pack(side=tk.BOTTOM, fill=tk.BOTH)
        self.buttons['prev'].pack(side=tk.LEFT)
        self.buttons['stop'].pack(side=tk.LEFT)
        self.buttons['next'].pack(side=tk.LEFT)
        self.buttons['quit'].pack(side=tk.RIGHT)
        self.buttons['save'].pack(side=tk.RIGHT)
    #------------------------------------------------------------------------#

    #--------------------------   Start the GUI   ---------------------------#
    def start(self):
        """ Start the interface (Tkinter `mainloop` wrapper) """
        tk.mainloop()
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           1-Plot Animated GUI                            ##
##############################################################################

class TkInterfaceOnePlot(_TkInterfaceBase):
    """ Tkinter-based GUI to embed a 1-graph animation

    Somehow a Tkinter-based wrapper for a FuncAnimationOnePlot, which
    adds a set of interactive, control buttons on the interface.

    Attributes
    ----------
    root : Tkinter object
        The base Tkinter object.
    figaxs : 2-tuple as a Figure & its set of Axis
        The figure & its axes.
    figure : Matplotlib FigureCanvasTkAgg
        The drawing area.
    frames : Tkinter Frames
        The interface frames (buttons only).
    buttons : dict of Tkinter Buttons
        The interface buttons.
    animation : Matplotlib FuncAnimation
        The animation object.

    Other Parameters
    ----------------
    [GLB] STEP : int, global variable from `funcanimation`
        The spacing between the values to display.

    Constructor
    -----------
    __init__(fig, axs)

    Methods
    -------
    onpress(event)
        Action when clicking on a key.
    build_interface()
        Instantiate the animation, bound the actions on clicking
        and build the Tk interface (place the frames).
    launch_animation(fargs, repeat=True, **kwargs)
        Start the animation in the GUI.
    start()
        Start the interface (Tkinter `mainloop` wrapper).

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

    # Build the animation & the interface
    >>> ani_gui = TkInterfaceOnePlot(fig, axs)
    >>> ani_gui.build_interface()

    # Launch the animation
    >>> ani_gui.launch_animation(
    ...     (tstp, data, tstp_chks, data_chks),
    ...     repeat=True, delay=hop_size/frate, frames=8, frate=frate)

    # Start the interface
    >>> ani_gui.start()
    """

    #------------------------   Class Constructor   -------------------------#
    def __init__(self, fig, axs):
        """ Instantiate a TkInterfaceOnePlot object (constructor)

        Save the figure and its axe in the `figaxs` attribute, instan-
        tiate the Tkinter object, add a buttons frame to it, place the
        figure at the interface top and declare the other attributes.

        Parameters
        ----------
        fig : Matplotlib Figure object
            The figure on which to plot the signals.
        axs : Matplotlib Axes object or array of such objects
            The axes of the figure.
        """

        # Build the base interface
        super().__init__(fig, axs)

        # Set the interface name
        self.root.wm_title("Audio Event Recognition Simulator")

        # Build the interface button frame
        self.frames = {'but': tk.Frame(self.root, borderwidth=2, relief='groove')}
    #------------------------------------------------------------------------#

    #-------------------   Place the Frames on the GUI   --------------------#
    def _place_frames(self):
        """ Place the interface frames (only the buttons here) """
        self._place_buttons()
    #------------------------------------------------------------------------#

    #-----------------------   Build the Animation   ------------------------#
    def build_interface(self):
        """ Instantiate the animation, bound the actions on clicking
        and build the Tk interface (place the frames) """

        # Build the animation object
        self.animation = FuncAnimationOnePlot(self.figaxs)

        # Bind the keyboard keys to the animation
        self.figaxs[0].canvas.mpl_connect('key_press_event', self.onpress)

        # Set the interface buttons
        self._set_buttons()

        # Place the frames on the interface (buttons & labels)
        self._place_frames()
    #------------------------------------------------------------------------#

    #-------------------------   Animation Launch   -------------------------#
    def launch_animation(self, fargs, repeat=True, **kwargs):
        """ Start the animation in the GUI

        Once the animation & interface are built, launch the animation;
        simply calls the FuncAnimationOnePlot `launch` method.

        Note that the `start` method must be called afterwards.

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
            The optional parameters that may be change btw executions:
              - delay (float): the time delay (in seconds) between displays
              - frames (int): the number of frames to plot per display
              - frate (float): the data sampling frequency; if not provided,
                    retrieve it from the chunks of timestamps.
                :Default: delay=0.2, frames=8, frate=None
        """
        self.animation.launch(fargs, repeat, **kwargs)
#        self.animation.animation.save("Exemple_Son.mp4")
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           2-Plot Animated GUI                            ##
##############################################################################

class TkInterfaceTwoPlots(_TkInterfaceBase):
    """ Tkinter-based GUI to embed a 2-graphs animation

    Somehow a Tkinter-based wrapper for a FuncAnimationTwoPlots, which
    adds a set of interactive buttons on the interface and some labels
    used to display the reference and estimated class numbers and names.

    Attributes
    ----------
    root : Tkinter object
        The base Tkinter object.
    figaxs : 2-tuple as a Figure & its set of Axis
        The figure & its axes.
    figure : Matplotlib FigureCanvasTkAgg
        The drawing area at the interface top.
    frames : Tkinter Frames
        The interface frames (labels & buttons).
    buttons : dict of Tkinter Buttons
        The interface buttons.
    labels_ref : dict of Tkinter Labels
        The reference class labels.
    labels_est : dict of Tkinter Labels
        The estimated class labels.
    animation : Matplotlib FuncAnimation
        The animation object.

    Other Parameters
    ----------------
    [GLB] STEP : int, global variable from `funcanimation`
        The spacing between the values to display.
    [GLB] CLASSES : dict, global variable from `funcanimation`
        The equivalence between a class number and its text.

    Constructor
    -----------
    __init__(fig, axs)

    Methods
    -------
    onpress(event)
        Action when clicking on a key.
    build_interface()
        Instantiate the animation, bound the actions on clicking
        and build the Tk interface (place the frames).
    launch_animation(fargs, model, repeat=True, **kwargs)
        Start the animation in the GUI.
    start()
        Start the interface (Tkinter `mainloop` wrapper).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import torch
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

    # Generate the random classes (one per chunk)
    >>> classes = np.random.random((len(data_chks), 4))

    # Generate some (3) events (delineated by their start & ending times)
    >>> events = np.array(((1.1, 1.5), (2.3, 4.9), (7.3, 9.5)))
    >>> events = np.array(events*frate, int)

    # Convert the data into Torch tensors
    >>> tens_chks = torch.Tensor(data_chks)                     # Torch tensors
    >>> tens_chks = tens_chks.unsqueeze(1)                      # NCW, with C=1

    # Build a dummy Torch-like model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(frm_size, 4),
    ...     torch.nn.Softmax(1))

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

    # Build the animation & the interface
    >>> ani_gui = TkInterfaceTwoPlots(fig, axs)
    >>> ani_gui.build_interface()

    # Launch the animation
    >>> ani_gui.launch_animation(
    ...     (tstp, data, tstp_chks, tens_chks, events, classes), model,
    ...     repeat=True, delay=hop_size/frate, frames=8, frate=frate)

    # Start the interface
    >>> ani_gui.start()
    """

    #------------------------   Class Constructor   -------------------------#
    def __init__(self, fig, axs):
        """ Instantiate a TkInterfaceTwoPlots object (constructor)

        Save the figure and its axes in the `figaxs` attribute, instan-
        tiate the Tkinter object, add the frames (labels) on which to
        display the class number and names and the buttons, place the
        figure at the interface top and declare the other attributes.

        Parameters
        ----------
        fig : Matplotlib Figure object
            The figure on which to plot the signals.
        axs : Matplotlib Axes object or array of such objects
            The axes of the figure.

        Other Parameters
        ----------------
        [GLB] CLASSES : dict, global variable from `funcanimation`
            The equivalence between a class number and its text.
        """

        # Build the base interface
        super().__init__(fig, axs)

        # Set the interface name
        self.root.wm_title("Audio Event Recognition Simulator")

        # Build the frames of the interface (two for the class labels,
        # and another one for the functionality buttons)
        self.frames = {
            'ref': tk.Frame(self.root, borderwidth=2, relief='groove'),
            'est': tk.Frame(self.root, borderwidth=2, relief='groove'),
            'but': tk.Frame(self.root, borderwidth=2, relief='groove')}

        fonts = (('Helvetica', 32), ('Helvetica', 40))

        # Build the labels of the reference classes (interface left side)
        self.labels_ref = {
            'title': tk.Label(
                self.frames['ref'], text="Classe Objective", font=fonts[0]),
            'number': tk.Label(
                self.frames['ref'], text="0", font=fonts[1]),
            'class': tk.Label(
                self.frames['ref'], text=CLASSES[0], font=fonts[0])}

        # Build the labels of the estimated classes (interface right side)
        self.labels_est = {
            'title': tk.Label(
                self.frames['est'], text="Classe Estimée", font=fonts[0]),
            'number': tk.Label(
                self.frames['est'], text="0", font=fonts[1]),
            'class': tk.Label(
                self.frames['est'], text=CLASSES[0], font=fonts[0])}
    #------------------------------------------------------------------------#

    #-------------------   Place the Frames on the GUI   --------------------#
    def _place_frames(self):
        """ Place the interface frames (labels & buttons) """

        # Place the buttons at the interface bottom
        self._place_buttons()

        # Place the labels of the reference classes at the left
        self.frames['ref'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.labels_ref['title'].pack(padx=10, pady=5)
        self.labels_ref['number'].pack(padx=5, pady=5)
        self.labels_ref['class'].pack(padx=5, pady=5)

        # Place the labels of the estimated classes at the right
        self.frames['est'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.labels_est['title'].pack(padx=10, pady=5)
        self.labels_est['number'].pack(padx=5, pady=5)
        self.labels_est['class'].pack(padx=5, pady=5)
    #------------------------------------------------------------------------#

    #-----------------------   Build the Animation   ------------------------#
    def build_interface(self):
        """ Instantiate the animation, bound the actions on clicking
        and build the Tk interface (place the frames) """

        # Build the animation object
        self.animation = FuncAnimationTwoPlots(
            self.figaxs, self.labels_ref, self.labels_est)

        # Bind the keyboard keys to the animation
        self.figaxs[0].canvas.mpl_connect('key_press_event', self.onpress)

        # Set the interface buttons
        self._set_buttons()

        # Place the frames on the interface (buttons & labels)
        self._place_frames()
    #------------------------------------------------------------------------#

    #-------------------------   Animation Launch   -------------------------#
    def launch_animation(self, fargs, model, repeat=True, **kwargs):
        """ Start the animation in the GUI

        Once the animation & interface are built, launch the animation;
        simply calls the FuncAnimationTwoPlots `launch` method.

        Note that the `start` method must be called afterwards.

        Parameters
        ----------
        fargs : 6-tuple of np.ndarrays
            The data in the format of the FuncAnimation class; the data
            are passed to the `update` method in their positional order:
              - tstp (1D array): the timestamps - abscissa
              - data (1D array): the data - ordinates
              - tstp_chks (2D array): the chunks of timestamps
              - data_chks (2D array): the chunks of data
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
            The optional parameters that may be change btw executions:
              - delay (float): the time delay (in seconds) between displays
              - frames (int): the number of frames to plot per display
              - frate (float): the data sampling frequency; if not provided,
                    retrieve it from the chunks of timestamps.
                :Default: delay=0.2, frames=8, frate=None
        """
        self.animation.launch(fargs, model, repeat, **kwargs)
    #------------------------------------------------------------------------#

##############################################################################
