import numpy as np
import matplotlib.pyplot as plt
import aer.data_tk as dtk
from aer.gui._gui_tk import *


def test_TkInterfaceOnePlot():
    """ Tkinter-based GUI to embed a 1-graph animation """

    # Chunk parameters
    frate = 1000                            # Sampling frequency
    frm_duration = 100e-3                   # Frame duration in seconds
    hop_duration = 50e-3                    # Hop duration in seconds
    frm_size = int(frate * frm_duration)    # Frame size in samples
    hop_size = int(frate * hop_duration)    # Hop size in samples

    # Generate example timestamps & data (50 frames)
    tstp = np.linspace(0, 10, 50 * frm_size)
    data = np.sin(10.*tstp)

    # Pad the timestamps & data
    offset = dtk.pad_offset(len(data), frm_size, hop_size)  # Needed offset
    tstp = dtk.pad(tstp, offset)                            # Pad timestamps
    data = dtk.pad(data, offset)                            # Pad the data

    # Build the time & data chunks
    tstp_chks = dtk.build_chunks(tstp, frm_size, hop_size)  # Time chunks
    data_chks = dtk.build_chunks(data, frm_size, hop_size)  # Data chunks

    # Instantiate a new figure
    fig, axs = plt.subplots(1, 1, layout='constrained', sharey=True)
    axs.locator_params(axis='x', nbins=12)

    # Figure labels
    axs.set_title("Sinus signal & Focused chunk", size=17)
    axs.set_xlabel("Time [s]", size=15)
    axs.set_ylabel("Amplitude [a.u]", size=15)

    # Build the animation & the interface
    ani_gui = TkInterfaceOnePlot(fig, axs)
    ani_gui.build_interface()

    # Launch the animation
    ani_gui.launch_animation(
        (tstp, data, tstp_chks, data_chks),
        repeat=True, delay=hop_size/frate, frames=8, frate=frate)

    # Start the interface
    ani_gui.start()

def test_TkInterfaceTwoPlots():
    """ Tkinter-based GUI to embed a 2-graphs animation """
    import torch

    # Chunk parameters
    frate = 1000                            # Sampling frequency
    frm_duration = 100e-3                   # Frame duration in seconds
    hop_duration = 50e-3                    # Hop duration in seconds
    frm_size = int(frate * frm_duration)    # Frame size in samples
    hop_size = int(frate * hop_duration)    # Hop size in samples

    # Generate example timestamps & data (50 frames)
    tstp = np.linspace(0, 10, 50 * frm_size)
    data = np.sin(10.*tstp)

    # Pad the timestamps & data
    offset = dtk.pad_offset(len(data), frm_size, hop_size)  # Needed offset
    tstp = dtk.pad(tstp, offset)                            # Pad timestamps
    data = dtk.pad(data, offset)                            # Pad the data

    # Build the time & data chunks
    tstp_chks = dtk.build_chunks(tstp, frm_size, hop_size)  # Time chunks
    data_chks = dtk.build_chunks(data, frm_size, hop_size)  # Data chunks

    # Generate the random classes (one per chunk)
    classes = np.random.random((len(data_chks), 4))

    # Generate some (3) events (delineated by their start & ending times)
    events = np.array(((1.1, 1.5), (2.3, 4.9), (7.3, 9.5)))
    events = np.array(events*frate, int)

    # Convert the data into Torch tensors
    tens_chks = torch.Tensor(data_chks)                     # Torch tensors
    tens_chks = tens_chks.unsqueeze(1)                      # NCW, with C=1

    # Build a dummy Torch-like model
    model = torch.nn.Sequential(
        torch.nn.Linear(frm_size, 4),
        torch.nn.Softmax(1))

    # Instantiate a new figure
    fig, axs = plt.subplots(1, 2, layout='constrained', sharey=True)
    axs[0].locator_params(axis='x', nbins=10)
    axs[1].locator_params(axis='x', nbins=5)

    # Figure labels
    axs[0].set_title("Sinus signal & Focused chunk", size=15)
    axs[1].set_title("Chunk used as model input", size=15)
    axs[0].set_xlabel("Time [s]", size=14)
    axs[1].set_xlabel("Time [s]", size=14)
    axs[0].set_ylabel("Amplitude [a.u.]", size=14)

    # Build the animation & the interface
    ani_gui = TkInterfaceTwoPlots(fig, axs)
    ani_gui.build_interface()

    # Launch the animation
    ani_gui.launch_animation(
        (tstp, data, tstp_chks, tens_chks, events, classes), model,
        repeat=True, delay=hop_size/frate, frames=8, frate=frate)

    # Start the interface
    ani_gui.start()



# Launch test/example functions
test_TkInterfaceOnePlot()

test_TkInterfaceTwoPlots()

