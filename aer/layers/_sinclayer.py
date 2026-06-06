""" Layer consisting of a filterbank of sinc functions

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: February 2025
Last revised: March 2026

License: GPLv3

The SincLayer was proposed by Mirco Ravanelli and Yoshua Bengio in their
paper "Speaker recognition from raw waveform with SincNet", in 2018 IEEE
Spoken Language Technology Workshop (SLT), pages 1021-1028, 2018, freely
accessible at https://arxiv.org/abs/1808.00158.

The implementation below is a direct application of their concept, which
is greatly inspired from the original authors' own implementation, freely
accessible at https://github.com/mravanelli/SincNet.

The proposed implementation is essentially a code porting from PyTorch
to Torch, with homemade, NumPy-based optimized filters and windows.

This implementation runs with Torch 2.9.1 in Python 3.12, and runs on
Ubuntu 24.04.
"""

__all__ = ['SincLayer']

import numpy as np
import torch

from aer import data_tk as dtk

SCALE = 'Hz'            # Frequency scale to use
FGATE = True            # Rectangular (T) or triangular (F) filters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################################################
##                          Sinc Filterbank Layer                           ##
##############################################################################

class SincLayer(torch.nn.Module):
    """ Sinc Filterbank Layer class

    Provide a class that is a filterbank of pass-band filters, which are
    sinc-based filters whose cutoff frequencies are trainable variables.
    Two types of filters are available: rectangular & triangular-shaped
    filters, as static methods; see the equivalent `rectangular_time`
    and `triangular_time` functions from the `filters` module.

    This layer inherits from the Torch base `Module` class, and is int-
    ended to operate with NumPy and Torch-based mathematical operators.
    It instantiates the filterbank, whose filters summed bandwidths co-
    ver a bandwidth specified in the constructor. Then, if the layer is
    used in a Torch model, the two cutoff frequencies of the pass-band
    filters are passed to the network optimizer as trainable weights.

    Attributes
    ----------
    nb_filters : int, getter only
        The number of filters in the bank.
    filter_length : int, getter only
        The bank's filter's length, in samples.
    frate : float, getter only
        The sampling frequency.
    bandwidth : 2-tuple of floats, getter only
        The filterbank bandwidth.
    low_co : Torch Tensor, getter only
        The bank's filters' lower cutoff frequencies.
    upp_co : Torch Tensor, getter only
        The bank's filters' upper cutoff frequencies.
    padding : str, getter & setter
        The padding strategy for the convolutions ('same' and 'valid')

    Other Parameters
    ----------------
    [GLB] SCALE : str, global variable
        Frequency scale to use, among {'Hz', 'Mel', 'Bark'}.
    [GLB] FGATE : bool, global variable
        Use rectangular (T) or triangular (F) filters.

    Constructor
    -----------
    __init__(nb_filters, filter_length, frate,
             bandwidth=(None, None), padding='same')

    Methods
    -------
    rescale(freqs)
        Rescale the frequencies provided as argument.
    forward(x)
        Apply filtering to an input signal.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from aer.layers import SwapDims
    >>> from aer.models_tk import torch_dataset, train_batch

    # Dataset parameters (chunks)
    >>> batch_size = 15         # Number of datasets
    >>> nb_frames = 10          # Number of signal frames (chunks)
    >>> len_frames = 400        # Number of samples in each chunk
    >>> nb_out_classes = 5      # Number of objective classes

    # SincLayer parameters
    >>> nb_filters = 12         # Number of filters in the bank
    >>> filt_lgt = 251          # Length (in samples) of the filters
    >>> frate = 16000           # Sampling frequency


    #----- Chunk variant (1D input data)
    # Shape of the input (NCW) -- Exactly 1 channel for SincLayer
    >>> input_shape = (batch_size, channels:=1, len_frames)
    # Shape of the output
    >>> output_shape = (batch_size, nb_out_classes)

    # Instantiate the model with SincLayer
    >>> model = torch.nn.Sequential(
    ...     SincLayer(nb_filters, filt_lgt, frate),
    ...     torch.nn.Flatten(),
    ...     torch.nn.Linear(nb_filters*len_frames, nb_out_classes))
    #-------------------------

    #----- Sequence variant (2D input data)
    # Shape of the input (NCHW) -- Exactly 1 channel for SincLayer
    >>> input_shape = (batch_size, channels:=1, nb_frames, len_frames)
    # Shape of the output
    >>> output_shape = (batch_size, nb_frames, nb_out_classes)

    # Instantiate the model with SincLayer
    >>> model = torch.nn.Sequential(
    ...     SincLayer(nb_filters, filt_lgt, frate),
    ...     SwapDims(1, 2),
    ...     torch.nn.Flatten(2, 3),
    ...     torch.nn.Linear(nb_filters*len_frames, nb_out_classes))
    #-------------------------


    # Define the loss function & and model optimizer
    >>> loss_fct = torch.nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.SGD(model.parameters())

    # Generate the data samples (NCW data format)
    >>> inputs = torch.arange(np.prod(input_shape), dtype=torch.float32)
    >>> inputs = inputs.reshape(input_shape)

    # Generate the objective data
    >>> outputs = torch.arange(np.prod(output_shape), dtype=torch.float32)
    >>> outputs = outputs.reshape(output_shape)

    # Wrap the inputs & outputs into a Torch dataset
    >>> dataset = torch_dataset(inputs, outputs, batch_size=1, shuffle=True)

    # Train the model
    >>> train_batch(dataset, model, loss_fct, optimizer)

    # Use the model for prediction
    >>> pred_out = model.forward(inputs)
    """

    #---------------------------   Constructor   ----------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(self, nb_filters, filter_length, frate,
                 bandwidth=(None, None), padding='same'):
        """ Instantiate a SincLayer object (constructor) and add the
        cutoff frequencies to its trainable parameters

        Parameters
        ----------
        nb_filters : int
            The number of filters in the bank.
        filter_length : int
            The length (in samples) of the filters.
        frate : int or float
            The sampling frequency; casted into float anyway.
        [OPT] bandwidth : 2-tuple of ints or floats
            The filterbank total bandwidth, as a 2-tuple organized as
            (freq_min, freq_max). If `(None, None)` or not provided,
            use 0 as minimal bound and half the sampling frequency as
            maximal bound, i.e. the tuple (0, `frate`/2) as default.
                :Default: (0, frate/2) (dynamically set)
        [OPT] padding : str
            The convolution padding strategy; should be one among:
            {'same', 'valid'}.
                :Default: 'same'

        Returns
        -------
        sinclayer : SincLayer class object
            The instantiated SincLayer object.

        Examples
        --------
        >>> sinclayer = SincLayer(10, 251, 16000)
        >>> sinclayer = SincLayer(10, 251, 16000, (0., 100.))
        """

        super().__init__()

        # Filterbank parameters
        self._fshape = (nb_filters, filter_length)      # Filter parameters
        self._frate = float(frate)                      # Sampling rate

        # Check the parity of the filter length
        if filter_length % 2 == 1:                      # Odd number
            self._parity = filter_length // 2, [1.]
        else:                                           # Even number
            self._parity = filter_length // 2 - 1, [1., 1.]

        # Filterbank total bandwidth
        self._bandwidth =\
            (0., 0.5*frate) if bandwidth==(None, None) else bandwidth

        # Additional trainable weights (cutoff frequencies)
        self._low_co = None                             # Lower cutoff freqs
        self._upp_co = None                             # Upper cutoff freqs

        # Convolution padding strategy
        self._padding = padding

        # Initialize the cutoff frequencies and add them to the training params
        self._set_cutoff_freqs()
    #------------------------------------------------------------------------#

    #----------------------------   Properties   ----------------------------#
    @property
    def nb_filters(self):
        """ Get the number of filters in the bank """
        return self._fshape[0]

    @property
    def filter_length(self):
        """ Get the bank's filter's length (samples) """
        return self._fshape[1]

    @property
    def frate(self):
        """ Get the sampling frequency """
        return self._frate

    @property
    def bandwidth(self):
        """ Get the filterbank bandwidth """
        return self._bandwidth

    @property
    def padding(self):
        """ Get the padding strategy for the convolutions """
        return self._padding

    @padding.setter
    def padding(self, padding):
        """ Set the padding strategy for the convolutions """
        if padding in ('valid', 'same'):
            self._padding = padding
        else:
            print(f"Invalid `{padding}` padding; use 'same' or 'valid'")

    @property
    def low_co(self):
        """ Get the bank's filters' lower cutoff frequencies """
        return self._low_co

    @property
    def upp_co(self):
        """ Get the bank's filters' upper cutoff frequencies """
        return self._upp_co
    #------------------------------------------------------------------------#

    #-------------------------   Filter Functions   -------------------------#
    @staticmethod
    def _flip(vals, cnt):
        """ Mirror a vector and stack the mirrored and original data """
        return torch.concatenate(
            [torch.flip(vals, [0]), torch.ones(len(cnt), device=DEVICE), vals])

    @staticmethod
    def _gate(tstp, fmin, fmax):
        """ Rectangle-shape filter core -- See `_gate` in `filters` """
        return (torch.sin(fmax*tstp) - torch.sin(fmin*tstp)) / ((fmax-fmin)*tstp)

    @staticmethod
    def _trig(tstp, fmin, fmax):
        """ Triangle-shape filter core -- See `_triangle` in `filters` """
        fcnt = 0.5 * (fmax + fmin)                          # Central freq
        flw, fmd, fup = fmax-fcnt, fcnt+fmin, fmax+fcnt     # Low, Peak, Upp
        return (torch.sin(flw*tstp) * torch.sin(fup*tstp)
                - torch.sin(flw*tstp) * torch.sin(fmd*tstp)) / (flw*(fup-fmd)*tstp**2)
    #------------------------------------------------------------------------#

    #-----------------------   Layer Handling Tools   -----------------------#
    def rescale(self, freqs):
        """ Rescale the frequencies provided as argument

        Parameters
        ----------
        freqs : array or tensor
            The frequencies to rescale.

        Returns
        -------
        freqs_rescale : same type as `freqs` argument
            The rescaled frequencies.

        Examples
        --------
        >>> nb_frames, len_frames = 100, 400                    # Input params
        >>> nb_filters, filt_lgt, frate = 10, 251, 16000        # Layer params
        >>> input_shape = (nb_frames, channels:=1, len_frames)  # Input shape

        >>> sinclayer = SincLayer(nb_filters, filt_lgt, frate)

        >>> low_co = sinclayer.rescale(sinclayer.low_co)
        >>> upp_co = sinclayer.rescale(sinclayer.upp_co)
        """
        return 0.5 * self._frate * freqs
    #------------------------------------------------------------------------#

    #-------------------------   Build the Layer   --------------------------#
    def _set_cutoff_freqs(self):
        """ Initialize the trainable weights

        Initialize the SincLayer convolutional layer specific weights
        (the pairs of cutoff frequencies), add them to the list of trai-
        nable weights, and set the layer topology.

        The SincLayer specific trainable weights are the pairs of cutoff
        frequencies of the pass-band filters. The lower and upper cutoff
        frequencies are the `low_co` and `upp_co` vectors, respectively.
        The weights are divided by half the sampling rate (frate/2).

        N.B.: to initialize the cutoff frequencies, by denoting M the
            number of filters, generate a set of M+1 linearly-spaced
            frequencies, and set the weights to these values; use the
            first M ones (0-M) for the lower cutoff frequencies, and
            the M last ones (1-M+1) for the upper cutoff frequencies.
            The scale on which the frequencies are linearly generated
            is given by the `SCALE` global variable; if it is not the
            standard Hertz scale, the frequencies are linearly genera-
            ted on this specific scale (Melody or Bark), before being
            sent back to the Hertz scale. Set the `SCALE` global var.
            to specify the scale; the non case-sensitive choices are:
              - Melody scale: 'm', 'mel' or 'melody';
              - Bark scale: 'b', 'bk' or 'bark';
              - Hertz scale: any other.

        Other Parameters
        ----------------
        [GLB] SCALE : string, global variable
            The scale on which to generate the initial weights.
                :Default: 'Hz' (Hertz scale)

        Returns
        -------
        None : directly initialize and add the `low_co` and `upp_co`
        vectors to the network trainable weights.
        """

        # Generate a set of frequencies linearly spaced on the `SCALE`
        # scale (global var) and project them back onto the Hertz scale
        # Use 1 additional filter for rolling (N+1 filters for N bands)
        if SCALE.lower() in ('m', 'mel', 'melody'):     # Mel scale
            freq = dtk.mel2hz(
                dtk.melscale(*self._bandwidth, self._fshape[0]+1))
        elif SCALE.lower() in ('b', 'bk', 'bark'):      # Bark scale
            freq = dtk.bark2hz(
                dtk.barkscale(*self._bandwidth, self._fshape[0]+1))
        else:                                           # Hertz scale
            freq = dtk.hzscale(*self._bandwidth, self._fshape[0]+1)

        # Rescale the frequencies (divide them by frate/2)
        freq *= 2./self._frate

        # Add the filters' two cutoff frequencies to the trainable weights
        self._low_co = torch.nn.Parameter(torch.Tensor(freq[:-1]).to(DEVICE))
        self._upp_co = torch.nn.Parameter(torch.Tensor(freq[1:]).to(DEVICE))
    #------------------------------------------------------------------------#

    #------------------   Cutoff Frequencies Constraints   ------------------#
    def _weights2freqs(self, coef):
        """ Apply the training constraints to the cutoff frequencies,
        and return the frequencies as Torch tensors """
        # Apply the training constraints to the frequencies
        fbeg = torch.abs(self._low_co)                           # flow >= 0
        fend = fbeg + torch.abs(self._upp_co - self._low_co)     # fupp >= flow
        return coef*fbeg, coef*fend
    #------------------------------------------------------------------------#

    #------------------   Chunk/Sequence `call` Variants   ------------------#
    def _call_chks(self, x, filters):
        """ Torch-based filterbank for chunks data """
        # Filter the signal by operating the time-domain convolution
        filters = torch.reshape(filters, (self._fshape[0], 1, self._fshape[1]))
#        return torch.nn.functional.conv1d(x, filters, padding=self._padding)
        return torch.conv1d(x, filters, padding=self._padding)

    def _call_seqs(self, x, filters):
        """ Torch-based filterbank for sequences data """
        # Filter the signal by operating the time-domain convolution
        filters = torch.reshape(filters, (self._fshape[0], 1, 1, self._fshape[1]))
        return torch.conv2d(x, filters, padding=self._padding)
    #------------------------------------------------------------------------#

    #-------------------------   Model Forwarding   -------------------------#
    def forward(self, x):
        """ Apply filtering to an input signal

        Take a tensor `x` of shape (batchsize, in_channels, in_width),
        build the time response of the bank's filters and apply a convo-
        lution to `x` using the stacked filters' responses as kernel.

        The pass-band filters are masked by the Hamming window before
        convolution. The convolution kernel has shape (out_channels,
        in_channels, f_length), where f_length is the filter length
        (in samples) and stored in the `filter_length` attribute, and
        out_channels is the number of filters of the bank, that also
        defines the number of channels of the layer's output.

        IMP: `SincLayer` accepts input tensors with 1 channel only,
            meaning that `in_channels` must always be 1 here.

        N.B.: both rectangle and triangle-shaped pass-band filters can
            be used; use the `FGATE` boolean global variable to use
            either the rectangle (True) or triangle (False) shape.

        Parameters
        ----------
        x : Torch tensor
            The input tensor of shape (batch, in_channels=1, in_width),
            where `batch` is the batch size, `in_width` is the width of
            the input frames and `in_channels` is the number of channels;
            `in_channels` must (and is assumed to) be 1, as SincLayer
            operates on 1D time signal only.

        Other Parameters
        ----------------
        [GLB] FGATE : bool, global variable
            The pass-band filters to use; set to True to use rectangle-
            shape filters, and to False to use triangle-shape filters.
                :Default: True (use rectangular pass-band filters)

        Returns
        -------
        x_filt : Torch tensor
            The layer output, as the convolution between the input `x`
            and the bank's filters time response. The output shape is
            (batchsize, nb_filters, nb_samples), where `nb_samples`
            depends on the padding strategy used for the convolution,
            and contained in the `padding` attribute.

        Examples
        --------
        # Dataset parameters (chunks)
        >>> nb_frames = 100             # Number of signal frames (chunks)
        >>> len_frames = 400            # Number of samples in each chunk

        # Shape of the input (NCW) -- Exactly 1 channel for SincLayer
        >>> inp_shape = (nb_frames, channels:=1, len_frames)

        # SincLayer parameters
        >>> nb_filters = 10             # Number of filters in the bank
        >>> filt_lgt = 251              # Length (in samples) of the filters
        >>> frate = 16000               # Sampling frequency

        # Generate the data samples (NCW data format)
        >>> inputs = torch.arange(np.prod(inp_shape), dtype=torch.float32)
        >>> inputs = inputs.reshape(inp_shape)

        # Instantiate and use the SincLayer filter
        >>> sinclayer = SincLayer(nb_filters, filt_lgt, frate).to('cpu')
        >>> outputs = sinclayer.forward(inputs)

        #----- Check the effect of the padding strategy

        # Default padding strategy ('same')
        >>> sinclayer.padding = 'same'
        >>> sinclayer.forward(inputs).shape
        torch.Size([100, 10, 400])

        # Change the padding strategy ('same' as default)
        >>> sinclayer.padding = 'valid'
        >>> sinclayer.forward(inputs).shape
        torch.Size([100, 10, 150])
        """

        # Filter the data with the filterbank
#        dtype = models_tk.torch_to_numpy_dtype(x.dtype)
        dtype = np.dtype(str(x.dtype).removeprefix('torch.'))   # Datatype
        lgt, cnt = self._parity                     # Filter parameters

        # Select the type of filter to use (rectangular or triangular)
        if FGATE:
            coef = np.pi*self._frate                # 2pi * frate/2
            fct = self._gate                        # Rectangular filter
        else:
            coef = 0.5*np.pi*self._frate            # pi * frate/2
            fct = self._trig                        # Triangular filter

        # Get the cutoff frequencies from the weights
        flow, fupp = self._weights2freqs(coef)

        # Generate the filtering window (Hamming here)
        window = torch.Tensor(
            dtk.hamming(self._fshape[1], dtype)).to(x.dtype).to(DEVICE)

        # Generate the time scale: tstp = nTe, n \in [[1, (N-1)/2]]
        tstp = np.linspace(1, lgt, lgt, dtype=dtype) / self._frate
        tstp = torch.Tensor(tstp).to(x.dtype).to(DEVICE)

        # Compute the filters
        filters = [self._flip(fct(tstp, flow[i], fupp[i]), cnt) * window
                   for i in range(self._fshape[0])]

        # Stack and reshape the responses to build the convolution kernel
        filters = torch.stack(filters)

        # Check if data are chunks or sequences and call the right method
        if x.ndim == 4:
            return self._call_seqs(x, filters)
        return self._call_chks(x, filters)
    #------------------------------------------------------------------------#

##############################################################################
