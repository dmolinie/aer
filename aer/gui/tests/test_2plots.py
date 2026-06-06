""" TkInterfaceTwoPlots example on real data and a Torch model """

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    import aer
    import aer.data_tk as dtk
    from aer.datasets import mivia_loader as loader
    from aer.gui import TkInterfaceTwoPlots
    from aer.models import SincNet

    # Data folder locations
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT                            # Events XML files folder
    SNDFOLDER = ROOT + "sounds/"                # Audio WAV files folder

    # Set the dataset index and SNR (only values available in the example set)
    IND, SNR = 66, 15

    # Chunk parameters
    NB_CLASSES = 4                              # Nb of possible classes
    FRM_DURATION = 100e-3                       # Frame duration in seconds
    HOP_DURATION = 50e-3                        # Hop duration in seconds
    chk_params = (FRM_DURATION, HOP_DURATION)   # Chunks settings

    # Load the raw data directly from the WAV file
    frate, data = loader.read_signal(SNDFOLDER, IND, SNR)

    events = loader.read_events(XMLFOLDER, IND)
    events = np.array(events[1]*frate, int)

    frm_size = int(frate * FRM_DURATION)
    hop_size = int(frate * HOP_DURATION)

    # Generate the timestamps (must be done first) & pad the data
    offset = dtk.pad_offset(len(data), frm_size, hop_size)
    tstp = dtk.pad(np.arange(0, len(data)) / frate, offset) # Generate timestamps
    data = dtk.pad(data, offset)                            # Pad the data

    # Load the data and build the chunks
    specs, data_chks, classes = loader.build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR),
        NB_CLASSES, chk_params)
    input_shape = (1, data_chks.shape[-1])

    # Convert the data into Torch float32 tensor
    data_chks = torch.as_tensor(data_chks, dtype=torch.float32)
    data_chks = data_chks.unsqueeze(1)      # NCW format, with C=1 here

    # Generate the timestamps
    tstp_chks = dtk.data_chunks(
        np.arange(0, specs[1]) / specs[0],
        (int(specs[0]*chk_params[0]), int(specs[0]*chk_params[1])))

    # Create the Torch model
    # N.B.: results are random as the model is not trained
    SCL_PARAMS = {'nb_filters': 40, 'filter_length': 251, 'padding': 'valid',
                  'frate': specs[0], 'bandwidth': (50., specs[0]/2)}
    model = SincNet(
        input_shape, NB_CLASSES, SCL_PARAMS,
        reg_linear=0.3, reg_conv=(3, 0.1), rec_cell='NoRec')
    model.eval()

    # Instantiate a new figure
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(
        1, 2, figsize=(19., 6.50), layout='constrained', sharey=True)

    # Figure labels
    axs[0].set_title("Flux audio & Partie étudiée", size=15)
    axs[1].set_title("Partie du signal en entrée du modèle", size=15)

    axs[0].set_xlabel("Temps [s]", size=14)
    axs[1].set_xlabel("Temps [s]", size=14)
    axs[0].set_ylabel("Amplitude [u.a.]", size=14)

    # Build the interface
    ani_gui = TkInterfaceTwoPlots(fig, axs)
    ani_gui.build_interface()

    # Launch the animation
    tempo = 2.5
    ani_gui.launch_animation(
        (tstp, data, tstp_chks, data_chks, events, classes), model,
        repeat=True, delay=tempo*hop_size/frate, frames=8, frate=frate)

    # Start the interface & animation
    ani_gui.start()

