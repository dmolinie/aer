import numpy as np
import matplotlib.pyplot as plt
from aer.display._hits_bars import *


def test_plot_hits():
    """ Plot the bar charts of the hits of several datasets """

    # No-class hits on chunks
    hits1 = 100. * np.linspace(0.1, 1.0, 15)
    hits1 = np.hstack((hits1, hits1.mean()))
    hits2 = 100. * np.linspace(1.0, 0.1, 15)
    hits2 = np.hstack((hits2, hits2.mean()))
    fig1 = plot_hits((hits1, hits2),
        fname="No_class_chunks.pdf",
        bar_params={'mean': True},
        labels=("Dataset #", "Accuracy [% Hits]"),
        title="No-class hits on chunks",
        rtexts=("Training", "Testing"))

    # Class hits on chunks
    nb_chks = 10
    nb_classes = 3
    hits1 = 100. * np.linspace(0.1, 1.0, 2*nb_chks * nb_classes)
    hits1 = hits1.reshape(2*nb_chks, nb_classes)
    hits1 = np.vstack((hits1, hits1.mean(0)))
    hits2 = 85. * np.linspace(1.0, 0.1, nb_chks * nb_classes)
    hits2 = hits2.reshape(nb_chks, nb_classes)
    hits2 = np.vstack((hits2, hits2.mean(0)))
    fig2 = plot_hits((hits1, hits2),
        bar_params={'mean': True},
        labels=("Dataset #", "Accuracy [% Hits]"),
        titles=(("Training datasets", "Testing datasets")),
        suptitle="Class hits on chunks",
        rtexts=("CLS1", "CLS2", "CLS3"),
        title_params={'pad': 0.},
        subfig_params={'hspace': 0.1},
        suptitle_params={'y': 1.075})

    # No-class hits on sequences
    nb_seqs = 4
    nb_chks = 5
    hits1 = 100. * np.linspace(0.1, 1.0, nb_seqs * 2*nb_chks * 2)
    hits1 = hits1.reshape(nb_seqs*2*nb_chks, 2)
    hits1 = np.vstack((hits1, hits1.mean(0)))
    hits2 = 100. * np.linspace(1.0, 0.1, nb_seqs * nb_chks * 2)
    hits2 = hits2.reshape(nb_seqs*nb_chks, 2)
    hits2 = np.vstack((hits2, hits2.mean(0)))
    fig3 = plot_hits((hits1, hits2),
        sep_classes=False,
        bar_params={'mean': True},
        labels=("Dataset #", "Accuracy [% Hits]"),
        suptitle="No-class hits on sequences",
        rtexts=("Training", "Testing"),
        suptitle_params={'y': 1.025})

    # Class hits on sequences
    nb_seqs = 4
    nb_chks = 5
    nb_classes = 4
    hits1 = 100. * np.linspace(0.1, 1.0, nb_seqs * 2*nb_chks * nb_classes * 2)
    hits1 = hits1.reshape(nb_seqs*2*nb_chks, 2, nb_classes)
    hits2 = 100. * np.linspace(1.0, 0.1, nb_seqs * nb_chks * nb_classes * 2)
    hits2 = hits2.reshape(nb_seqs*nb_chks, 2, nb_classes)
    fig4 = plot_hits((hits1, hits2),
        labels=("Dataset #", "Accuracy [% Hits]"),
        title="Class hits on sequences",
        rtexts=("Training", "Testing"))

    # Display the images
    plt.show()



# Launch test/example functions
test_plot_hits()

