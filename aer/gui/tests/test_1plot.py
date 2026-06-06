""" TkInterfaceOnePlot example on real data """

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    import aer
    import aer.data_tk as dtk
    from aer.datasets import mivia_loader as loader
    from aer.gui import TkInterfaceOnePlot

    # Data folder locations
#    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    ROOT = "/home/dm281323/Sources/data/MIVIA_AE/training/"
    XMLFOLDER = ROOT                            # Events XML files folder
    SNDFOLDER = ROOT + "sounds/"                # Audio WAV files folder

    # Set the dataset index and SNR (only values available in the example set)
    IND, SNR = 15, 20

    # Chunk parameters
    NB_CLASSES = 4                              # Nb of possible classes
    FRM_DURATION = 100e-3                       # Frame duration in seconds
    HOP_DURATION = 50e-3                        # Hop duration in seconds
    chk_params = (FRM_DURATION, HOP_DURATION)   # Chunks settings

    # Load the raw data directly from the WAV file
    frate, data = loader.read_signal(SNDFOLDER, IND, SNR)

    frm_size = int(frate * FRM_DURATION)
    hop_size = int(frate * HOP_DURATION)

    # Generate the timestamps (must be done first) & pad the data
    offset = dtk.pad_offset(len(data), frm_size, hop_size)  # Data offset
    tstp = dtk.pad(np.arange(0, len(data)) / frate, offset) # Generate timestamps
    data = dtk.pad(data, offset)                            # Pad the data

    # Load the data and build the chunks
    specs, data_chks, classes = loader.build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR),
        NB_CLASSES, chk_params)

    # Generate the timestamps
    tstp_chks = dtk.data_chunks(
        np.arange(0, specs[1]) / specs[0],
        (int(specs[0]*chk_params[0]), int(specs[0]*chk_params[1])))

    # Instantiate a new figure
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(
        1, 1, figsize=(19., 9.00), layout='constrained', sharey=True)

    # Figure labels
    plt.rcParams.update({'font.size': 15})
    axs.set_title("Flux audio & Partie étudiée", size=26)
    axs.set_xlabel("Temps [s]", size=22)
    axs.set_ylabel("Amplitude [u.a.]", size=22)

    # Build the interface
    ani_gui = TkInterfaceOnePlot(fig, axs)
    ani_gui.build_interface()

    # Launch the animation
    tempo = 1.0
    ani_gui.launch_animation(
        (tstp, data, tstp_chks, data_chks),
        repeat=True, delay=tempo*hop_size/frate, frames=5, frate=frate)

    # Start the interface & animation
    ani_gui.start()

