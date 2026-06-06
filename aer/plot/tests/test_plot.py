import numpy as np
import matplotlib.pyplot as plt
from aer.plot._plot import *


def test_plot():
    """ Plot data against stamps on 1-graph figure & save it """
    # Generate dummy data
    tstp1 = np.linspace(0., 1., 100)    # From 0s to 1s -- 100 points
    tstp2 = np.linspace(0., 1., 200)    # From 0s to 1s -- 200 points
    sin_10 = np.sin(2*np.pi*10.*tstp1)  # 10 Hz sine (100 pts)
    sin_50 = np.sin(2*np.pi*50.*tstp1)  # 50 Hz sine (100 pts)
    cos_10 = np.cos(2*np.pi*10.*tstp1)  # 10 Hz cosine (100 pts)
    cos_50 = np.cos(2*np.pi*50.*tstp2)  # 50 Hz cosine (200 pts)
    # Simple plot with optional x-values and line style
    fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    fig = plot(fig.gca(), sin_10)                   # Auto xvals
    fig = plot(fig.gca(), 100.*tstp1, sin_10+1.)    # Explicit xvals
    fig.show()
    # Plot a single data array on a 1-plot figure
    fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    fig = plot(fig.gca(),
        tstp1, sin_10,
        labels=('Time', 'Amplitude'),
        titles='Signal amplitude against time',
        rtexts='Time domain',
        fname=None,             # Don't save the Figure in a file
        plot_params={'fmt': 'b--', 'linewidth': 2},
        label_params={'size': 20},
        title_params={'size': 24, 'pad': 12.},
        rtext_params={'size': 18, 'labelpad': 12., 'stacked': False},
        margin_params={'x': 0.01, 'y': 0.01})
    fig.show()
    # Plot several data on the same 1-plot figure
    fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    fig = plot(fig.gca(),
        (tstp1, tstp2), (sin_10, cos_50),
        labels=('Time', 'Amplitude'),
        titles='Signal amplitudes against time',
        legends=('10 Hz Sine', '50 Hz Cosine'),
        fname="Sine_Cosine.pdf",    # Save the figure in a file
        save_params={'bbox_inches': 'tight'},
        plot_params={'fmt': '--', 'linewidth': 2},
        label_params={'size': 20},
        title_params={'size': 24, 'pad': 12.},
        legend_params={'fontsize': 14, 'loc': 'upper center'},
        margin_params={'x': 0.01, 'y': 0.01})
    fig.show()
    # Plot data arrays on an N-plot figure (one set of arrays per plot)
    # Here, generate the Figure in the function with the 'fig_params' dict. arg.
    fig = plot(None,                # Generate the Figure in the function
        tstp1,
        (sin_10, cos_10, (sin_10, cos_10)),
        labels=('Time', ('Amp. Sine', 'Amp. Cosine', 'Amp. Sine & Cosine')),
        titles='Signal amplitudes against time',
        rtexts=('Sine', 'Cosine', 'Sine & Cosine'),
        legends=('', '', ('Sine', 'Cosine')),
        fname="Sine_Cosine.pdf",    # Save the figure; set to None to ignore
        fig_params={'nrows': 3, 'ncols': 1, 'figsize': (19.20, 10.80)},
        save_params={'bbox_inches': 'tight'},
        plot_params={'linewidth': 2},
        label_params={'size': 18, 'repeat_labels': (True, False)},
        title_params={'size': 24},
        legend_params={'loc': 'upper right'},
        margin_params={'x': 0.01, 'y': 0.1},
        space_params={'no_xspace': False, 'no_yspace': False})
    fig.show()
    # Plot data arrays on an N-plot figure (one set of arrays per plot)
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    fig = plot(axs.T,                               # column-major
        (tstp1,                                     # Left plots x-values
         (tstp1, tstp2, (tstp1, tstp2))),           # Right plots x-values
        ((sin_10, sin_50, (sin_10, sin_50)),        # Left plots y-values
         (cos_10, cos_50, (cos_10, cos_50))),       # Right plots y-values
        labels=(
            ('Common x-label', ('Y_11', 'Y_21', 'Y_31')),     # Left labels
            (('X_11', 'X_21', 'X_31'), 'Common y-label')),    # Right labels
        titles=(
            ('10 Hz Sine', '50 Hz Sine', '10 & 50 Hz Sines'), # Left titles
            '10 & 50 Hz Cosines'),                            # Right titles
        legends=(
            ('10 Hz', '50 Hz', ''),                 # Left plots legends
            ('', '', ('10 Hz', '50 Hz'))),          # Right plots legends
        rtexts=(
            '',                                     # Right y-axis side texts
            ('10 Hz', '50 Hz', '10 & 50 Hz')),      # Left y-axis side texts
        fname="Sines_Cosines.pdf",                  # Save figure (None to skip)
        save_params={'bbox_inches': 'tight'},
        plot_params= {'fmt': '--', 'linewidth': 2},
        label_params={'size': 17,'repeat_labels': (False, True)},
        title_params={'size': 20},
        legend_params={'fontsize': 15, 'loc': 'upper right'},
        rtext_params={'size': 14, 'stacked': True},
        margin_params={'x': 0.01, 'y': 0.1},
        space_params={'no_xspace': False, 'no_yspace': False})
    fig.show()

def test_bar_plot():
    """ Plot the 'yvals' as bar charts on an NxM-graph figure """
    #--- Single bar plots examples
    # Generate dummy data & add the mean
    vals1 = np.arange(1, 15)                         # 1D data
    vals2 = np.arange(29, 0, -1)                     # 1D data
    vals1 = np.hstack((vals1, vals1.mean()))         # Append the mean
    vals2 = np.hstack((vals2, vals2.mean()))         # Append the mean
    # Generate simple x-ticks (same as those automatically generated)
    xpos1 = np.arange(1, len(vals1)+1)               # +1 for the mean
    xpos2 = np.arange(1, len(vals2)+1)               # +1 for the mean
    # Generate simple x-labels (optional)
    tick_labels1 = np.array(np.arange(1, len(xpos1)+1), dtype=str)
    tick_labels2 = np.array(np.arange(1, len(xpos2)+1), dtype=str)
    # Bar chart with optional `x` (x-values)
    fig, axs = plt.subplots(2, figsize=(19.20, 10.80), layout='constrained')
    fig = bar_plot(axs[0], vals1)                    # Auto xpos
    fig = bar_plot(axs[1], xpos1, vals1)             # Explicit xpos
    fig.show()
    # Plot a single bar chart
    fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    fig = bar_plot(fig.gca(),
        xpos1, vals1,
        labels=('Indexes', 'Values'),
        titles='Values against indexes',
        rtexts='Values',
        bartext=vals1.astype(int),
        fname=None,             # Don't save the Figure in a file
        bar_params={'mean': True, 'tick_label': tick_labels1, 'color': 'blue'},
        label_params={'size': 20},
        title_params={'size': 24, 'pad': 12.},
        rtext_params={'size': 18, 'labelpad': 12., 'stacked': False},
        bartext_params={'size': 14},
        margin_params={'x': 0.01, 'y': 0.01})
    fig.show()
    # Plot a set of bar charts
    fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    fig = bar_plot(axs.T,                            # column-major
        ((xpos1, xpos1),  # or just xpos1            # Left plots x-values
         (xpos2, xpos2)), # or just xpos2            # Right plots x-values
        ((vals1, vals1),                             # Left plots y-values
         (vals2, vals2)),                            # Right plots y-values
        labels=(
            ('Common x-label', ('Y_11', 'Y_21')),    # Left labels
            (('X_11', 'X_21'), 'Common y-label')),   # Right labels
        titles=(
            ('Values 1', 'Values 2'),                # Left titles
            'Values 1 & 2'),                         # Right titles
        legends=(
            ('Leg 1', 'Leg 2'),                      # Left plots legends
            ('', 'Leg 4')),                          # Right plots legends
        rtexts=(
            '',                                      # Right y-axis side texts
            ('10 Hz', '50 Hz')),                     # Left y-axis side texts
        bartext=(
            (vals1.astype(int), vals1.astype(int)),  # Left bar's top texts
            (vals2.astype(int), vals2.astype(int))), # Right bar's top texts
        fname="NxM_Bar_plots.pdf",                   # Save figure (None to skip)
        save_params={'bbox_inches': 'tight'},
        bar_params= {'mean': True},
        label_params={'size': 17,'repeat_labels': (False, True)},
        title_params={'size': 20},
        legend_params={'fontsize': 15, 'loc': 'best'},
        rtext_params={'size': 14, 'stacked': True},
        bartext_params={'size': 12},
        margin_params={'x': 0.01, 'y': 0.01},
        space_params={'no_xspace': False, 'no_yspace': False})
    fig.show()
    #--- Double bar plots examples
    # Generate dummy data
    vals1 = np.arange(1, 15).reshape(-1, 2)          # 2D data
    vals1 = np.vstack((vals1, vals1.mean(0)))        # Append the mean
    vals2 = np.arange(30, 0, -1).reshape(-1, 2)      # 2D data
    vals2 = np.vstack((vals2, vals2.mean(0)))        # Append the mean
    # Generate simple x-ticks (same as those automatically generated)
    xpos1 = np.arange(1, len(vals1)+1)               # +1 for the mean
    xpos2 = np.arange(1, len(vals2)+1)               # +1 for the mean
    # Generate simple x-labels (optional)
    tick_labels1 = np.array(np.arange(1, len(xpos1)+1), dtype=str)
    tick_labels2 = np.array(np.arange(1, len(xpos2)+1), dtype=str)
    # Plot a single bar chart
    fig = plt.figure(figsize=(19.20, 10.80), layout='constrained')
    fig = bar_plot(fig.gca(),
        xpos1, vals1,
        labels=('Indexes', 'Values'),
        titles='Values against indexes',
        rtexts='Values',
        bartext=vals1.astype(int),
        fname=None,             # Don't save the Figure in a file
        bar_params={'mean': True, 'double_bars': True,
                    'tick_label': tick_labels1, 'color': 'blue'},
        label_params={'size': 20},
        title_params={'size': 24, 'pad': 12.},
        rtext_params={'size': 18, 'labelpad': 12., 'stacked': False},
        bartext_params={'size': 14},
        margin_params={'x': 0.01, 'y': 0.01})
    fig.show()
    # Plot a set of bar charts
    fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    fig = bar_plot(axs.T,                            # column-major
        ((xpos1, xpos1),  # or just xpos1            # Left plots x-values
         (xpos2, xpos2)), # or just xpos2            # Right plots x-values
        ((vals1, vals1),                             # Left plots y-values
         (vals2, vals2)),                            # Right plots y-values
        labels=(
            ('Common x-label', ('Y_11', 'Y_21')),    # Left labels
            (('X_11', 'X_21'), 'Common y-label')),   # Right labels
        titles=(
            ('Values 1', 'Values 2'),                # Left titles
            'Values 1 & 2'),                         # Right titles
        legends=(
            ('Leg 1', 'Leg 2'),                      # Left plots legends
            ('', 'Leg 4')),                          # Right plots legends
        rtexts=(
            '',                                      # Right y-axis side texts
            ('10 Hz', '50 Hz')),                     # Left y-axis side texts
        bartext=(
            (vals1.astype(int), vals1.astype(int)),  # Left bar's top texts
            (vals2.astype(int), vals2.astype(int))), # Right bar's top texts
        fname="NxM_Bar_plots.pdf",                   # Save figure (None to skip)
        save_params={'bbox_inches': 'tight'},
        bar_params= {'mean': True, 'double_bars': True},
        label_params={'size': 17,'repeat_labels': (False, True)},
        title_params={'size': 20},
        legend_params={'fontsize': 15, 'loc': 'best'},
        rtext_params={'size': 14, 'stacked': True},
        bartext_params={'size': 9},
        margin_params={'x': 0.01, 'y': 0.01},
        space_params={'no_xspace': False, 'no_yspace': False})
    fig.show()



# Launch test/example functions
test_plot()

test_bar_plot()

