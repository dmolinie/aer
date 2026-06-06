""" Shorthand tools to manipulate ONNX files

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: August 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = ['torch2onnx', 'load_onnx', 'run_onnx']

import onnxruntime

from aer.tools import check_ext, check_path


##############################################################################
##                            Torch Model Export                            ##
##############################################################################

#----------------------   Torch Model to ONNX Graph   -----------------------#
def torch2onnx(pathname, model, ex_data, dynamic_batch=True, **kwargs):
    """ Convert and export a Torch model into an ONNX graph

    Take a Torch model, convert it into an ONNX graph and save it as an
    ONNX file with name `pathname` (any intermediate directory is auto-
    matically created).

    Automatically pass the model to `eval` mode before exporting it to
    ONNX, and reset it to its original mode after (training or eval).

    N.B.: this function requires both `onnx` and `onnxscript` modules.

    Parameters
    ----------
    pathname : string
        The name for the export; used only if `save` is True.
    model : Torch model
        The Torch model to convert into an ONNX graph.
    ex_data : Torch Tensor
        The example data to export with the ONNX; must be a valid input
        for `model`.
    [OPT] dynamic_batch : bool
        If True, set the first dimension (batch) to a dynamic value; used
        only if no `dynamic_shapes` is in the optional keyword arguments.
        Note that the `dynamo` optional argument should be True (default)
        (argument for the `torch.onnx.export` function, cf. `kwargs`).
            :Default: True

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The optional keyword arguments for the `torch.onnx.export` function;
        see this function for details.

    Returns
    -------
    model_nx : ONNX graph
        The model converted into an ONNX graph.

    Examples
    --------
    >>> import torch

    >>> batch = 10      # Batch size
    >>> c_inp = 10      # Input channels
    >>> c_out = 5       # Output channels

    # Generate dummy data
    >>> inputs = torch.arange(batch*c_inp, dtype=torch.float32)
    >>> inputs = inputs.reshape(batch, c_inp)

    # Define a dummy model
    >>> model = torch.nn.Sequential(torch.nn.Linear(c_inp, c_out))

    # Convert the model into an ONNX graph, and save it
    >>> model_nx = torch2onnx("Models/model.onnx", model, inputs, dynamo=True)
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from torch.export import Dim

    # Simplify the model before export (prune the layers that are needed
    # only for training, and not for inference)
    training = model.training       # Keep the original model mode
    model.eval()                    # Pass the model to eval mode

    # If `model` is a `Module` class object, retrieve its `Sequential` model
    if hasattr(model, 'model'):
        model = model.model

    # Set the first dimension to a dynamic value
    if dynamic_batch and 'dynamic_shapes' not in kwargs:
        static_dims = (Dim.STATIC for _ in range(ex_data.ndim-1))
        kwargs['dynamic_shapes'] = [(Dim("batch", min=1), *static_dims)]

    # Convert the Torch model into an ONNX graph
    model_nx = torch.onnx.export(model, ex_data, **kwargs)

    # Export the model to ONNX
    model_nx.save(check_path(pathname))

    # Reset the model to training, if it was originally on this mode
    if training:
        model.train()

    return model_nx
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                      Load & Run ONNX (onnxruntime)                       ##
##############################################################################

#---------------------------   Load ONNX Graph   ----------------------------#
def load_onnx(pathname, **kwargs):
    """ Load an ONNX graph model into a Session object (for inference)

    N.B.: this function requires the `onnxruntime` module.

    Parameters
    ----------
    pathname : string
        The pathname to the ONNX file to load containing the model graph;
        it should end with '.onnx'; this extension is appended else.

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The `onnxruntime` `InferenceSession` class constructor arguments.
        See this class for the details.

    Returns
    -------
    onnx_session : ONNX Session object (`onnxruntime` module)
        The ONNX Session object that can be used for inference.

    Examples
    --------
    # Let assume that an ONNX file is saved with name `model.onnx` in
    # folder `Models`, e.g. using the `torch2onnx` function example.
    >>> onnx_session = load_onnx("Models/model.onnx")
    """
    return onnxruntime.InferenceSession(check_ext(pathname, '.onnx'), **kwargs)
#----------------------------------------------------------------------------#

#------------------------   ONNX Session Inference   ------------------------#
def run_onnx(onnx_session, input_data):
    """ Use an ONNX graph model for inference

    N.B.: this function requires the `onnxruntime` module.

    Parameters
    ----------
    onnx_session : ONNX Session object
        The ONNX Session model (cf. function `load_onnx`).
    input_data : array_like
        The data to pass to the model for inference.

    Returns
    -------
    pred : either np.ndarray, sparse tensor, list or dictionary.
        The ONNX model prediction in response to `input_data`.

    Examples
    --------
    # Let assume that an ONNX file is saved with name `model.onnx` in
    # folder `Models`, e.g. using the `torch2onnx` function example.

    >>> import numpy as np

    # Load the ONNX model and use it for inference
    >>> onnx_session = load_onnx("Models/model.onnx")

    # Retrieve the input data shape and generate data accordingly
    >>> batch = 10
    >>> input_shape = onnx_session.get_inputs()[0].shape[1:]
    >>> input_data = np.arange(batch*np.prod(input_shape), dtype=np.float32)
    >>> input_data = input_data.reshape(batch, *input_shape)

    # Run inference on the input data
    >>> pred = run_onnx(onnx_session, input_data)
    """
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    return onnx_session.run([output_name], {input_name: input_data})
#----------------------------------------------------------------------------#

##############################################################################
