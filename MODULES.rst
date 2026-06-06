AER package's modules, functions & classes
++++++++++++++++++++++++++++++++++++++++++

This file aims to summarize all the objects implemented in the ``aer`` package; to this purpose, it briefly introduces the modules of the package, as well as their respective functions and classes.

.. Contents::


accuracy
========
Tools to build confusion matrices, compute accuracy scores (True/False Positives/Negatives) and some statistics on the classification issued by a model.

..
  _metrics.py

* ``round_classes``  
		For a 2D array, set the max of a row to True and the rest to False

* ``accuracy_items``  
		Classifications accuracy items on 2D arrays (class by class)

* ``confusion_matrix``  
		Confusion matrix between two sets of classes

* ``conf_mat_to_acc_items``  
		Compute the amounts of TPs, FPs, FNs & TNs from a confusion matrix

* ``accuracy_chunks``  
		Classification accuracy on data chunks

* ``accuracy_sequences``  
		Classification accuracy on sequences of chunks

* ``compute_avgs``  
		Rates of TPs per category with the mean at the bottom

* ``sensitivity``
        Compute the "sensitivity" of a set of classifications

* ``specificity``
        Compute the "specificity" of a set of classifications

..
  _comparison.py

* ``euclidean``  
		Compute the 1D Euclidean distance between 2 arrays

* ``compute_dists_chks``  
		Compute the Euclidean distance between the accuracy stats vector obtained on a dataset (chunks) and the objective accuracy vector

* ``compute_dists_seqs``  
		Compute the Euclidean distance between the accuracy stats vector obtained on a dataset (sequences) and the objective accuracy vector

* ``find_best``  
		Find the common value with the lowest rank in two arrays

..
  _save.py

* ``save``  
		Save accuracy on sequences (confusion matrix or accuracy items)

* ``load``  
		Classifications accuracy items on 2D arrays (class by class)


aidge
=====
Wrapper for the Aidge functions & tools, allowing to load an ONNX graph, manipulate it using the Aidge tools and export it into C++ sources. Requires Aidge to be installed on the system.

..
  _aidge_tk.py

* ``load_onnx``  
		Load the ONNX model into an Aidge graph

* ``export_onnx``  
		Export an Aidge graph into an ONNX file

* ``get_backend``  
		Get the backend of a graph, if any

* ``get_node_dtype``  
		Get the operator's data type of the operator of a node

* ``get_input_node``  
		Find the input node of a GraphView

* ``get_output``  
		Get the values contained in a given channel of a given output node

* ``forward``  
		Propagate some data in an Aidge model using a scheduler

* ``add_node2graph``  
		Add a node to an Aidge graph

* ``graph_check_backend``  
		Take an Aidge graph and check that its backend is fully set

* ``improve_graph_fc``  
		Take an Aidge graph and improve its FC-oriented operations

* ``conv_tiling``  
		Perform tiling on an Aidge graph model

* ``rescale_tensors``  
		Rescale a set of tensors

* ``reset_graph_dims``  
		Get the input nodes' dimensions and forward them to the graph

..
  _aidge_io.py

* ``save_mmd``  
		Export the an Aidge model graph as a Mermaid instructions file

* ``load_mmd``  
		Load a Mermaid file as text (string)

* ``plot_mmd_as_html``  
		Generate an HTML file to visualize an Aidge graph

* ``plot_mmd_as_jpg``  
		Visualize the graph of a Mermaid file and save it as a jpg file

* ``improve_graph_cpp``  
		Improve a graph for C++ export

* ``nchw2nhwc4export``  
		Change the data format of an Aidge graph

* ``export_cpp``  
		Export the model graph in a C++ source project

* ``export_array_cpp``  
		Export a (list of) data array(s) in C++


datasets
========
Dataset loader. For now, only a loader for the MIVIA dataset is available. Also, provides examples of sound files for testing purpose.

..
  mivia_loader.py

* ``parse_xml``  
		Parse an XML file from the MIVIA AE or RAE databases

* ``read_events``  
		Read & parse a MIVIA XML events file

* ``read_signal``  
		Read & load a MIVIA database audio signal

* ``build_dataset``  
		Build the data & class chunks or sequences


data_tk
=======
Functions to manipulate data, in particular to normalize/denormalize them, to pad them and to parse a signal into chunks or sequence of chunks. Also, provides simple windowing functions.

..
  _data_parser.py

* ``pad_offset``  
		Number of samples to be added to the vector

* ``pad``  
		Distribute the offset samples to both sides of the data vector

* ``get_hop_size``  
		Retrieve the hop size from a set of chunks

* ``get_hop_size_data``  
		Retrieve the hop size from the data vector & chunks shapes

* ``build_chunks``  
		Build the chunks using a unit sliding window

* ``chunk_classes``  
		Build the vector of the classes of the signal chunks

* ``data_chunks``  
		Build 2D chunks of 1D vectors

* ``class_chunks``  
		Build the chunk classes in a one-hot encoding fashion

* ``data_sequences``  
		Build 3D sequences of 2D chunks from a 1D data vector

* ``class_sequences``  
		Build the sequences of classes in a one-hot encoding fashion

..
  _scaling.py

* ``extrema``  
		Return the min and max values within the data

* ``extrema_nd``  
		Get the min & max values among several arrays

* ``rescale``  
    Data rescaling between bounds

* ``hzscale``  
    Build a set of linearly spaced samples in the Hertz scale spanning from a minimal & maximal frequencies (in Hertz)

* ``melscale``  
		Build a set of linearly spaced samples in the Mel scale spanning from a minimal & maximal frequencies (in Hertz)

* ``barkscale``  
		Build a set of linearly spaced samples in the Bark scale spanning from a minimal & maximal frequencies (in Hertz)

* ``hz2mel``  
		Convert Hertz frequency into Mel frequency

* ``mel2hz``  
		Convert Mel frequency into Hertz frequency

* ``hz2bark``  
		Convert Hertz frequency into Bark frequency

* ``bark2hz``  
		Convert Bark frequency into Hertz frequency

* ``sub_bands``  
		Split a bandwidth interval into sub-bands on a given scale

..
  _windows.py

* ``cosine_sum``  
		General Cosine-sum window

* ``hann``  
		Hann window function

* ``hamming``  
		Hamming window function

* ``blackman``  
		Blackman window function


display
=======
Functions to display a signal directly; the functions typically instantiate a figure and plot the data passed to them as arguments; many optional parameters allow to adjust the figure dynamically.

..
  _signals.py

* ``freqs2str``  
		Convert a list of scalar frequencies as a literal

* ``signal``  
		Plot a time-domain signal against time

* ``spectrum``  
		Plot the spectrum (Fourier Transform) of a signal

* ``cepstrum``  
		Plot the cepstrum of a signal

* ``spectralpower``  
		Plot the spectral power density (periodogram) of a signal

* ``cepstralpower``  
		Plot the cepstral power density of a signal

* ``spectrogram``  
		Plot the spectrogram of a signal

* ``filterbank_spectrum``  
		Plot the frequency response of the bank's filters

* ``filterbank``  
		Plot the temporal response of the bank's filters

..
  _hits_bars.py

* ``plot_hits``  
		Plot the bar charts of the hits of several datasets


filters
=======
Simple time- and frequency-domain filters (rectangular, triangular, Gamma). Also, provides the FilterBank class that allows to build a bank of filters based on implemented filters.

..
  _filters.py

* ``flip``  
		Mirror a vector and stack the mirrored and original data

* ``convolve``  
		Operate the convolution two vectors

* ``rectangular_time``  
		Rectangular-shaped filter time impulse response

* ``triangular_time``  
		Triangular-shaped filter time impulse response

* ``gammatone_time``  
		Gammatone filter time impulse response

* ``rectangular_freq``  
		Frequency-domain rectangular-shaped filter

* ``triangular_freq``  
		Frequency-domain triangular-shaped filter

..
  _filterbank.py

* ``build_bands``  
		Split the bandwidth attribute into sub-bands on a given scale

* ``build_filters``  
		Compute the time impulse response of the filters

* ``filter_signal``  
		Filter the signal data with the filters of a bank

* ``filter_signal_chunks``  
		Filter a set of signal chunks

* ``spectrum``  
		Build the spectrum of the filters with the NumPy FFT


gui
===
Simple Graphical User Interface that can be used to dynamically display a signal and the classes outputted on its chunks by a Torch model.

..
  _funcanimation.py

* ``FuncAnimationOnePlot``  
		One plot Matplotlib's FuncAnimation overload

* ``FuncAnimationTwoPlots``  
		Two plots Matplotlib's FuncAnimation overload

..
  _gui_tk.py

* ``TkInterfaceOnePlot``  
		Tkinter-based GUI to embed a 1-graph animation

* ``TkInterfaceTwoPlots``  
		Tkinter-based GUI to embed a 2-graphs animation


layers
======
Additional custom Torch layers. In particular, wraps several Torch functions into layers (``modules``) that can be used inside larger models.

..
  _layers.py

* ``conv_output_length``  
		Compute the output length of a convolution given an input length

* ``Reshape``  
		Reshape a tensor on the flow, and return the it

* ``SwapDims``  
        Swap two dimensions of a tensor on the flow

* ``Squeeze``  
		Squeeze (remove 1-item axis) an on-the-flow tensor along a dimension

* ``Unsqueeze``  
		Unsqueeze an on-the-flow tensor along a dimension

* ``PrintShape``  
		Print the shape of a tensor on the flow, and return the tensor;

* ``ExtractTensor``  
		Extract a sub-tensor from an on-the-flow tensor

* ``LayerNorm``  
		Torch Layer Normalization (LayerNorm)

..
  _sinclayer.py

* ``SincLayer``  
		Sinc Filterbank Layer classs

..
  _delayer.py

* ``DELayer``  
		Torch-based Denoising-Enhancement Layer class


models
======
Provides the SincNet & DENet models, plus a simpler standard layers-only variant of it (i.e. using no custom layer), essentially consisting of the same architecture than SincNet, but in which SincLayer is replaced with a standard Convolution layer. The models accept both chunks and sequences of chunks as inputs.

..
  _convnet.py

* ``ConvNet``  
    	Standard layers-only network based on the SincNet architecture

..
  _sincnet.py

* ``SincNet``  
    	Original Sinc Filterbank Network

..
  _denet.py

* ``DENet``  
    	Original Denoising-Enhancement Network


models_tk
=========
Tools to semi-automatize the training & testing of the models, by providing functions that automatically read and load the data from a set of datasets, parse them into chunks or sequences of chunks and pass them to the models for training or testing (loss & accuracy).

..
  _models_tk.py

* ``numpy_to_torch_dtype``  
		Convert a NumPy datatype into a Torch datatype

* ``torch_to_numpy_dtype``  
		Convert a NumPy datatype into a Torch datatype

* ``build_model_name``  
		Build the name of the file to save

* ``parse_modname``  
		Parse the model name into more readable strings (for display)

* ``torch_dataset``  
		Wrap a data array and its labels into a Torch Dataset

* ``train_batch``  
		Train a Torch model on a batched dataset

* ``train_epoch``  
		Train a Torch model on a set of batched datasets (1 epoch)

* ``train_model``  
		Train a Torch model on a set of batched Torch datasets (several epochs)

* ``test_model_loss``  
		Loss of a model on a set of batched Torch datasets

* ``test_model_accuracy``  
		Classification accuracy of a model over a set of batched Torch datasets

..
  _models_mivia.py

* ``train_model_mivia``  
        Train a model on a set of files from the MIVIA AE dataset

* ``test_model_loss_mivia``  
        Evaluate model loss on a set of files from the MIVIA AE dataset

* ``test_model_accuracy_mivia``  
        Model classification accuracy on a set of files from the MIVIA AE dataset


onnx
====
Tools to export Torch models to ONNX computation graphs (using the ``torch`` & ``onnx`` modules), and to operate inference with the ``onnxruntime`` Python module.

..
  _onnx.py

* ``torch2onnx``  
		Convert and export a Torch model into an ONNX graph

* ``load_onnx``  
		Load an ONNX graph model into a Session object (onnxruntime)

* ``run_onnx``  
		Use an ONNX graph model (onnxruntime)


plot
====
Functions that wrap the main statements of any plot; in particular, given a figure, these functions allow to plot data passed as arguments and decorate the figure (labels, titles, etc.). These functions are useful to swiftly plot data without paying attention to the main plotting functions (e.g. bar charts) and alleviate scripts.

..
  _core_plot.py

* ``plot_core``  
		Plot (a set of) data on (a set of) figure' axes

* ``bar_core``  
		Plot data bars on a figure

..
  _decorations.py

* ``set_margins``  
		Set the x- and y-axis margins of (a set of) figure axes

* ``remove_spaces``  
		Remove space between the axes of a figure

* ``set_labels``  
		Write the labels on a figure (for plot & subplots)

* ``set_titles``  
		Write the titles on a figure (for plot & subplots)

* ``set_legends``  
		Write the legends on the axes of a figure (for plot & subplots)

* ``set_right_yaxis_texts``  
		Set additional ylabels alongside the right y-axis axis

* ``set_decorations``  
		Write the labels, titles and legends on the axes of a figure

* ``set_bartop_text``  
		Write texts at the top of the bars of a chart

..
  _plot.py

* ``plot``  
		Plot data against stamps on 1-graph figure & save it

* ``bar_plot``  
		Plot the yvalues as bar charts on an NxM-graph figure


spectrum
========
Classes allowing to build the Spectrum and Cepstrum of a signal, plus the Short-Time Fourier Transform (STFT) that is used to build the spectrogram of a signal.

..
  _spectrum.py

* ``onesidedfft``  
		Given a two-sided FFT, return the corresponding one-sided FFT

* ``twosidedfft``  
		Rebuild the whole FFT (neg & pos freqs) from a one-sided FFT

* ``Spectrum``  
		Spectrum class

* ``spec_freq``  
		Build the frequency scale for an FFT

* ``spec_sfft``  
		Compute the FFT components of the data provided as input

* ``spectrum``  
		Compute the FFT of an input signal

* ``spectral_power``  
		Spectral Power Density estimation (periodogram)

..
  _cepstrum.py

* ``Cepstrum``  
		Cepstrum class

* ``ceps_quef``  
		Build the frequency scale for an FFT

* ``ceps_sfft``  
		Compute the FFT components of the data provided as input

* ``cepstrum``  
		Cepstrum transformation of a given signal from its FT

* ``cepstral_power``  
		Cepstral power density

..
  _stft.py

* ``ShortTimeFT``  
		Short-Time Fourier Transform class

* ``stft_chunks``  
		Apply the STFT to a set of signal chunks

* ``stft_time``  
		Compute the Short-Time Fourier Transform

* ``spectrogram``  
		Build the spectrogram amplitude of an STFT


tools
=====
Simple I/O tools, data format checkers, etc.

..
  _tools.py

* ``exec_file``  
		Execute a Python script file

* ``save_as_csv``  
		Save a data set in a csv file 

* ``load_as_csv``  
		Load the data contained in a CSV file

* ``save_ws``  
		Save a workspace as a shelve

* ``load_ws``  
		Restore a shelved workspace

* ``get_ndim``  
		Get the depth (number of dimensions) of an array

* ``flat_list``  
		Flatten a 2D list

* ``make_iter``  
		Check if an object is scalar, and wrap it into a tuple if so

* ``check_ext``  
		Check the file name extension

* ``check_folder``  
		Check the validity of folder name

* ``check_path``  
		Check the path of a file (folders + extension)

* ``check_keys``  
		Dictionary check function

