import numpy as np
import matplotlib.pyplot as plt
from aer.plot._decorations import *


def test_set_margins():
    """ Set the x- and y-axis margins of (a set of) figure axes """
    # Define the x & ymargins
    xmargins = (0.01, 0.10, 0.20)
    ymargins = (0.01, 0.10, 0.20)
    # Generate dummy data
    data = np.arange(10)
    # Set the same margins to any plots
    fig, axs = plt.subplots(3, 1, sharex=False)
    for axis in axs:
        axis.plot(data)
    set_margins(axs, x=xmargins[0], y=ymargins[0])
    # Set specific margins to every plot
    fig, axs = plt.subplots(3, 1, sharex=False)
    for axis, xmarg, ymarg in zip(axs, xmargins, ymargins):
        axis.plot(data)
        set_margins(axis, x=xmarg, y=ymarg)
    plt.show()

def test_remove_spaces():
    """ Remove space between the axes of a figure """
    fig, axs = plt.subplots(2, 3)
    remove_spaces(fig, True, True, pad=2.0)
    plt.show()

def test_set_labels():
    """ Write the labels on a figure (for plot & subplots) """
    # 1-plot figure labels
    fig = plt.figure()
    set_labels(fig.gca(), ("X1", "Y1"))
    # N-plot figure labels
    fig, axs = plt.subplots(2, 1)
    labels = ("X", ("Y1", "Y2"))
    set_labels(axs, labels)
    # Directly write the (common) labels
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    set_labels(axs[:, 0], ("X1", "Y1"), repeat_labels=(True, True))
    set_labels(axs[:, 1], ("X2", "Y2"), repeat_labels=(False, False))
    # NxM-plot figure
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    set_labels(axs.T,
        (("X1", ("", "Y1", "")), ("X2", "Y2")),
        repeat_labels=(True, True))
    # Display the images
    plt.show()

def test_set_titles():
    """ Write the titles on a figure (for plot & subplots) """
    # Write the title of on a 1-plot figure
    fig = plt.figure()
    set_titles(fig.gca(), "T1", size=24)
    # Write a single title at the top of an N-plot figure
    fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.80))
    set_titles(axs, "T1")
    # Write the titles on an N-plot figure
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    set_titles(axs[:, 0], "T1", size=18)
    set_titles(axs[:, 1], "T2", size=24, pad=12.)
    # Write the titles on an NxM-plot figure
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    set_titles(axs, (("T1", "T2"),), size=24)
    # Write the titles on an NxM-plot figure
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    set_titles(axs.T, ("T1", ("T21", "T22", "T23")), size=24)
    # Display the images
    plt.show()

def test_set_legends():
    """ Write the legends on the axes of a figure (for plot & subplots) """
    # Generate dummy data
    tstp = np.linspace(0., 10., 100)
    sin_10, cos_10 = np.sin(10.*tstp), np.cos(10.*tstp)
    sin_50, cos_50 = np.sin(50.*tstp), np.cos(50.*tstp)
    # Plot the data on the figure
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    axs[0, 0].plot(tstp, sin_10)
    axs[1, 0].plot(tstp, cos_10)
    axs[2, 0].plot(tstp, sin_10)
    axs[2, 0].plot(tstp, cos_10)
    axs[0, 1].plot(tstp, sin_50)
    axs[1, 1].plot(tstp, cos_50)
    axs[2, 1].plot(tstp, sin_50)
    axs[2, 1].plot(tstp, cos_50)
    # Set a specific legend for each plot
    set_legends(axs[0, 0], "10 Hz Sine", loc='upper left')
    set_legends(axs[1, 0], "10 Hz Cosine", loc='upper center')
    set_legends(axs[2, 0], ("10 Hz Sine", "10 Hz Cosine"), loc='upper right')
    # Set the legends for a full set of axes at once
    set_legends(axs[:, 1],
        ("50 Hz Sine", "50 Hz Cosine", ("50 Hz Sine", "50 Hz Cosine")),
        loc='upper right', fontsize=12)
    # Set the legends for all plots at once (row-major)
    set_legends(axs,
        (("10 Hz Sine", "50 Hz Cosine"),        # 1st row
         ("10 Hz Cosine", "50 Hz Cosine"),      # 2nd row
         (("10 Hz Sine", "10 Hz Cosine"),       # 3rd row, 1st col
          ("50 Hz Sine", "50 Hz Cosine"))),     # 3rd row, 2nd col
        loc='upper right', fontsize=12)
    # Set the legends for all plots at once (column-major)
    set_legends(axs.T,
        (("10 Hz Sine", "10 Hz Cosine", ("10 Hz Sine", "10 Hz Cosine")),  # 1st
         ("50 Hz Sine", "50 Hz Cosine", ("50 Hz Sine", "50 Hz Cosine"))), # 2nd
        loc='upper right', fontsize=12)
    plt.show()

def test_set_right_yaxis_texts():
    """ Set additional ylabels alongside the right y-axis axis """
    # 1-plot figure
    fig = plt.figure()
    set_right_yaxis_texts(fig.gca(), "Label", True)
    # N-plot figure
    fig, axs = plt.subplots(2, 1)
    set_right_yaxis_texts(axs, (f"Img {i}" for i in range(len(axs))), True)
    # NxM-plot figure
    fig, axs = plt.subplots(2, 2)
    set_right_yaxis_texts(axs, (("", "Lab 01"), ("", "Lab 11")), True)

def test_set_decorations():
    """ Write the labels, titles and legends on the axes of a figure """
    # Generate dummy data
    tstp = np.linspace(0., 1., 100)
    sin_10, cos_10 = np.sin(10.*tstp), np.cos(10.*tstp)
    sin_50, cos_50 = np.sin(50.*tstp), np.cos(50.*tstp)
    # Plot them on a single-plot figure
    fig = plt.figure(figsize=(19.20, 10.80))
    plt.plot(tstp, sin_10)
    plt.plot(tstp, cos_10)
    set_decorations(fig.gca(),
        labels=("Time", "Amp"),
        title="Sine & Cosine",
        legends=("Sine", "Cosine"),
        label_params={'size': 18, 'labelpad': 12.},
        title_params={'size': 24, 'pad': 15.},
        legend_params={'fontsize': 15, 'loc': 'best'})
    # Plot them on a 1D subplot figure
    fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.80))
    axs[0].plot(tstp, sin_10)   # Sine on 1st graph
    axs[1].plot(tstp, cos_10)   # Cosine on 2nd graph
    axs[2].plot(tstp, sin_50)   # Sine on 3rd graph
    axs[2].plot(tstp, cos_50)   # Cosine on 3rd graph
    set_decorations(axs,
        labels=("Time", "Amp"),
        titles="Sine & Cosine",
        legends=("Sine", "Cosine", ("Sine", "Cosine")),
        rtexts=("Sine", "Cosine", "Sine & Cosine"),
        label_params={'size': 18, 'repeat_labels': (False, True)},
        title_params={'size': 24, 'pad': 12.},
        rtext_params={'size': 14, 'labelpad': 12.})
    # Plot them on a 2D subplot figure (row-major)
    fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    axs[0, 0].plot(tstp, sin_10)
    axs[0, 1].plot(tstp, cos_10)
    axs[1, 0].plot(tstp, sin_50)
    axs[1, 1].plot(tstp, cos_50)
    set_decorations(axs,        # row-major
        labels=(("", ("Amp 00", "Amp 01")),
                (("Time 10", "Time 11"), "Amp 1X")),
        titles=(("Sines", "Cosines"), ("", "")),
        legends=(("10 Hz Sine", "10 Hz Cosine"), ("50 Hz Sine", "50 Hz Cosine")),
        rtexts=(("", "10 Hz"), ("", "50 Hz")),
        label_params={'size': 18, 'repeat_labels': (False, True)},
        title_params={'size': 24, 'pad': 12.},
        legend_params={'loc': 'upper right'},
        rtext_params={'stacked': False, 'size': 14, 'labelpad': 6.})
    # Plot them on a 2D subplot figure (column-major)
    fig, axs = plt.subplots(2, 2, figsize=(19.20, 10.80))
    axs[0, 0].plot(tstp, sin_10)
    axs[0, 1].plot(tstp, cos_10)
    axs[1, 0].plot(tstp, sin_50)
    axs[1, 1].plot(tstp, cos_50)
    set_decorations(axs.T,      # Transpose for column-major
        labels=(("Time X0", "Amp X0"), ("Time X1", "Amp X1")),
        titles=("Sines", "Cosines"),
        legends=(("10 Hz Sine", "50 Hz Sine"), ("10 Hz Cosine", "50 Hz Cosine")),
        rtexts=("", ("10 Hz", "50 Hz")),
        label_params={'size': 18, 'repeat_labels': (False, True)},
        title_params={'size': 24, 'pad': 12.},
        legend_params={'loc': 'upper right'},
        rtext_params={'stacked': False, 'size': 14, 'labelpad': 6.})
    plt.show()

def test_set_bartop_text():
    """ Write texts at the top of the bars of a chart """
    ###--- Single bar plots example
    # Generate dummy data & add the mean
    vals1 = np.arange(1, 31)                        # 1D data
    vals2 = np.arange(30, 0, -1)                    # 1D data
    # Generate simple x-ticks
    xpos1 = np.arange(1, len(vals1)+1)              # +1 for the mean
    xpos2 = np.arange(1, len(vals2)+1)              # +1 for the mean
    # Plot the bar chart
    fig, axs = plt.subplots(2, 1)
    axs[0].bar(xpos1, vals1)
    axs[1].bar(xpos2, vals2)
    # Add the values at the bars' top
    vals1 = vals1.astype(int)
    vals2 = vals2.astype(int)
    set_bartop_text(axs, (vals1, vals2), size=8)
    # Display the figure
    fig.show()
    ###--- Double bar plots example
    # Generate dummy data & add the mean
    vals1 = np.arange(1, 31).reshape(-1, 2)         # 2D data
    vals2 = np.arange(30, 0, -1).reshape(-1, 2)     # 2D data
    # Generate simple x-ticks
    xpos1 = np.arange(1, len(vals1)+1)              # +1 for the mean
    xpos2 = np.arange(1, len(vals2)+1)              # +1 for the mean
    # Plot the bar chart
    fig, axs = plt.subplots(2, 1)
    width = 0.4                                     # Bar width
    axs[0].bar(xpos1-0.5*width, vals1[:, 0], width=width)
    axs[0].bar(xpos1+0.5*width, vals1[:, 1], width=width)
    axs[1].bar(xpos2-0.5*width, vals2[:, 0], width=width)
    axs[1].bar(xpos2+0.5*width, vals2[:, 1], width=width)
    # Add the values at the bars' top
    vals1 = vals1.astype(int)
    vals2 = vals2.astype(int)
    set_bartop_text(axs, (vals1, vals2), size=7)
    # Display the figure
    fig.show()


# Launch test/example functions
test_set_margins()

test_remove_spaces()

test_set_labels()

test_set_titles()

test_set_right_yaxis_texts()

test_set_legends()

test_set_decorations()

test_set_bartop_text()

