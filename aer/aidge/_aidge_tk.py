""" Shorthand tools to manipulate Aidge graphs

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: August 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'load_onnx', 'export_onnx',
    'get_backend', 'get_node_dtype', 'graph_check_backend',
    'get_input_node', 'get_output', 'forward',
    'add_node2graph', 'improve_graph_fc', 'conv_tiling',
    'rescale_tensors', 'reset_graph_dims']

import numpy as np

import aidge_core
from aidge_core.aidge_core import Tensor
#import aidge_backend_cpu   # Should be loaded in the main script
import aidge_onnx

from aer.tools._tools import check_ext


##############################################################################
##                           Aidge ONNX Handling                            ##
##############################################################################

#--------------------   Load ONNX Model as Aidge Graph   --------------------#
def load_onnx(pathname, verbose=False):
    """ Load the ONNX model into an Aidge graph

    Parameters
    ----------
    pathname : str
        The pathname to the ONNX graph model to load.
    [OPT] verbose : bool
        If the `load_onnx` function from the `aidge_core` module should
        be verbose or not.
            :Default: False

    Returns
    -------
    graph : aidge_core.GraphView object
        The ONNX model converted into an Aidge graph.

    Examples
    --------
    >>> import aidge_core
    >>> import aidge_backend_cpu
    >>> import aidge_onnx

    # Create an Aidge graph
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Producer([10, 256], name='dataProvider'),
    ...     aidge_core.FC(256, 5, name='fc')])

    # Set the graph implementation (backend)
    >>> graph.set_datatype(aidge_core.dtype.float32)
    >>> graph.set_backend('cpu')

    # Save the graph into an ONNX file
    >>> aidge_onnx.export_onnx(graph, "aidge_graph.onnx")

    # (Re)load the model graph from the ONNX file
    >>> graph_onnx = load_onnx("aidge_graph.onnx")
    """
    # Check that the pathname ends with 'onnx' and load the ONNX model
    return aidge_onnx.load_onnx(check_ext(pathname, '.onnx'), verbose)
#----------------------------------------------------------------------------#

#-----------------------   Save Aidge Graph as ONNX   -----------------------#
def export_onnx(pathname, graph, **kwargs):
    """ Export an Aidge graph into an ONNX file

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The Aidge graph model to export into ONNX.
    pathname : str
        The path name where to save the ONNX file.

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments
        The `aidge_onnx`'s `export_onnx` function additional arguments.
          - inputs_dims (list): input dimensions of the network
          - outputs_dims (list): output dimensions of the network
          - enable_custom_op (bool): if True, export will not fail
          - opset (int): version of the ONNX opset generated
          - ir_version (int): version of the ONNX representation
        :Default: inputs_dims=None, outputs_dims=None,
                  enable_custom_op=False, opset=None, ir_version=None

    Returns
    -------
    None : save the ONNX file in memory.

    Examples
    --------
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Create an Aidge graph
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Producer([10, 256], name='dataProvider'),
    ...     aidge_core.FC(256, 5, name='fc')])

    # Set the graph implementation (backend)
    >>> graph.set_datatype(aidge_core.dtype.float32)
    >>> graph.set_backend('cpu')

    # Save the graph into an ONNX file
    >>> export_onnx("aidge_graph.onnx", graph)
    """
    # Check that the pathname ends with 'onnx' and save the ONNX model
    aidge_onnx.export_onnx(graph, check_ext(pathname, '.onnx'), **kwargs)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                      Graph Attributes/Data Getters                       ##
##############################################################################

#------------------------   Retrieve Graph Backend   ------------------------#
def get_backend(graph):
    """ Get the backend of a graph, if any

    Parameters
    ----------
    graph : aidge_core.GraphView
        The graph for which to retrieve the backend, is any.

    Returns
    -------
    backend : str
        The graph backend; return an empty string '' if no backend.

    Examples
    --------
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Create an Aidge graph
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Producer([10, 256], name='dataProvider'),
    ...     aidge_core.FC(256, 5, name='fc')])

    # Set the graph implementation (backend)
    >>> graph.set_backend('cpu')

    # Retrieve the graph backend
    >>> get_backend(graph)
    'cpu'
    """
    backend, i = '', 0
    nodes = graph.get_ordered_nodes()
    while backend == '' and i < len(nodes):
        backend = nodes[i].get_operator().backend()
        i += 1
    return backend
#----------------------------------------------------------------------------#

#----------------------   Node's Operator Data Type   -----------------------#
def get_node_dtype(node):
    """ Get the operator's data type of the operator of a node

    Parameters
    ----------
    node : aidge_core.Node object
        The graph's node for which to get the operator's data type.

    Returns
    -------
    dtype : aidge_core.dtype data type
        The data type of the node's operator.

    Examples
    --------
    >>> import aidge_core

    # Define a simple Aidge graph and retrieve its nodes
    >>> graph = aidge_core.sequential([
    ...     aidge_core.FC(256, 5, name='fc'),
    ...     aidge_core.Flatten(name='flat')])
    >>> nodes = graph.get_ordered_nodes()

    # Get the default data type
    >>> get_node_dtype(nodes[0])
    <dtype.float32: 1>

    # Set a data type to the graph and retrieve it thereafter
    >>> graph.set_datatype(aidge_core.dtype.int32)
    >>> get_node_dtype(nodes[0])
    <dtype.int32: 25>
    """
    return node.get_operator().get_raw_output(0).dtype
#----------------------------------------------------------------------------#

#----------------------------   Get Input Node   ----------------------------#
def get_input_node(graph):
    """ Find the input node of a GraphView

    Take an Aidge graph and return the input node with no parent node.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph for which to find the input node.

    Returns
    -------
    input_node : aidge_core.Node object
        The graph's input node (~entry point).

    Examples
    --------
    >>> import aidge_core

    >>> graph = aidge_core.sequential([
    ...    aidge_core.Producer([16, 3, 512, 512], name='dataProvider'),
    ...    aidge_core.Conv2D(3, 4, [5, 5], name='conv'),
    ...    aidge_core.ReLU(name='relu')])

    >>> get_input_node(graph)
    Node(name='dataProvider', optype='Producer', children: [[1]])
    """
    for node in graph.get_ordered_inputs():
        if node[0].get_parent(0) is None:
            return node[0]
    return graph.get_ordered_nodes()[0]
#----------------------------------------------------------------------------#

#------------------------   Get Aidge Graph Output   ------------------------#
def get_output(graph, out_node=0, out_idx=0):
    """ Get the values contained in a given channel of a given output node

    Take an Aidge graph and return the data contained in the `out_idx`-th
    output of the operator of the `out_node`-th output node.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph for which to retrieve the output data.
    [OPT] out_node : int
        The index of the output node from which to retrieve the values.
            :Default: 0 (first output node)
    [OPT] out_idx : int
        The index of the output of the `out_node`-th node's operator.
            :Default: 0 (first operator's output)

    Returns
    -------
    output : aidge_core.Tensor object
        The output (data values) of the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Generate some example data
    >>> data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)

    # Build a simple Aidge graph
    >>> graph = aidge_core.sequential([aidge_core.FC(256, 5, name='fc')])
    >>> graph.set_datatype(aidge_core.dtype.float32)
    >>> graph.set_backend('cpu')

    # Instantiate a scheduler and forward the 'data' to the model
    >>> scheduler = aidge_core.SequentialScheduler(graph)
    >>> scheduler.forward(True, data=[aidge_core.Tensor(data)])

    # Retrieve the graph's output
    >>> output = get_output(graph)
    >>> np.shape(output)
    (10, 5)
    """
    output_node = list(graph.get_output_nodes())[out_node]
    return output_node.get_operator().get_output(out_idx)#.clone()
#----------------------------------------------------------------------------#

#------------------------   Forward Data to Graph   -------------------------#
def forward(graph, scheduler, data, **kw_outnode):
    """ Propagate some data in an Aidge model using a scheduler

    Take an Aidge graph and its scheduler, and forward the data to it;
    the data can either be a NumPy array or an Aidge Tensor. Retrieve
    the graph output and return it as an Aidge Tensor.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph to which pass the data for forwarding.
    scheduler : aidge_core.SequentialScheduler object
        The scheduler (for forwarding data) on the model graph.
    data : np.ndarray or aidge_core.Tensor
        The data to pass to the graph as input for forwarding.

    Other Parameters
    ----------------
    **kw_outnode : inline keyword arguments
        The arguments for the `get_output` function:
          - out_node (int): The output node index.
          - out_idx (int): The `out_node`-th node's operator output index.

    Returns
    -------
    output : aidge_core.Tensor object
        The output (data values) of the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Generate some example data
    >>> data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)

    # Build a simple Aidge graph
    >>> graph = aidge_core.sequential([aidge_core.FC(256, 5, name='fc')])
    >>> graph.set_datatype(aidge_core.dtype.float32)
    >>> graph.set_backend('cpu')

    # Instantiate a scheduler
    >>> scheduler = aidge_core.SequentialScheduler(graph)

    # Forward the data to the model and retrieve the graph's output
    >>> output = forward(graph, scheduler, data)
    >>> np.shape(output)
    (10, 5)
    """
    if isinstance(data, Tensor):
        scheduler.forward(True, data=[data])
    else:
        scheduler.forward(True, data=[Tensor(data)])
    return get_output(graph, **kw_outnode)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         Graph Operators Mutators                         ##
##############################################################################

# Additional improvement functions:
#   - aidge_core.constant_folding : compute constant part of the graph
#         and replace them by pre-computed values.

#--------------------------   Add Node to Graph   ---------------------------#
def add_node2graph(graph, node, graph_node, node_out_id=0, gnode_in_id=0):
    """ Add a node to an Aidge graph

    Take a node and link to a graph as a child of its `graph_node` node.

    Parameters
    ----------
    graph : aidge_core.GraphView
        The graph for which to add `node`.
    node :  aidge_core.Node
        The node to add to the graph, as a parent of `graph_node`.
    graph_node : string or aidge_core.Node
        The graph node to which connect `node`, as its child. If it is
        a string, search and use the node with this name in the graph.
    [OPT] node_out_id :
        The output index of the `node` to add (as a parent).
            :Default: 0 (first output)
    [OPT] gnode_in_id :
        The input index of the `graph_onnx` (child) to which link `node`.
            :Default: 0 (first input)

    Returns
    -------
    None : dynamically modify the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core

    # Define a simple, 1-layer graph
    >>> graph = aidge_core.sequential([aidge_core.Add(name='add')])

    # Define a `Producer` node to be added before the graph's `Add` node
    >>> bias_add = np.linspace(0., 1., 100, dtype=np.float32).reshape(10, 10)
    >>> producer = aidge_core.Producer(aidge_core.Tensor(bias_add), name='add_bias')

    # Show the graph's operators before modification
    >>> [print(node.get_operator().type()) for node in graph.get_ordered_nodes()]
    Add

    # Add the Producer node to the graph, as bias of the Add operator
    # The `Producer` 1st output (0) is connected to the `Add` 2nd input (1)
    >>> add_node2graph(graph, producer, 'add', 0, 1)

    # Show the graph's operators after modification
    >>> [print(node.get_operator().type()) for node in graph.get_ordered_nodes()]
    Producer
    Add
    """
    # If `graph_node` is a node' name, retrieve the node from the graph
    if isinstance(graph_node, str):
        gnode = graph.get_node(graph_node)
        if gnode is None:
            print(f"No node found with name '{graph_node}', doing nothing")
            return
        graph_node = gnode

    # Set `graph_node` as a child for `node`
    node.add_child(graph_node, node_out_id, gnode_in_id)
    # Link `node` to `graph` (as a parent of `graph_node`)
    graph.add(node)
#----------------------------------------------------------------------------#

#-------------------------   Adapt Graph Backend   --------------------------#
def graph_check_backend(graph, adapt_backend=False, adapt_fc=False):
    """ Take an Aidge graph and check that its backend is fully set

    Possibly apply the following functions from the `aidge_core` module:
      - `adapt_to_backend`: if needed, add Transpose nodes to a graph to
            match the lastly set backend input/output data format; if the
            data format is already correct, do nothing.
      - `adapt_fc_params_format`: adapt the format of the parameters of
            a FC layer to be compatible with the input format, e.g. if
            the input format is NHWC, weights will be adapted to NHWC.

    N.B.: since Aidge provides only NCHW implementations for its opera-
        tors in Python, changing the data format of an Aidge graph will
        break its use for forwarding data via a scheduler (in Python).
        As a consequence, changing the data format should only be done
        before a C++ export using Aidge, since the C++ implementations
        of the Aidge operators use a NHWC data format.

    N.B.: to change the data format of a graph from NCWH to NHWC, prefer
        using the `nchw2nhwc4export` function from the `aidge_io` module.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph for which to optimize the operators.
    [OPT] adapt_backend : bool or str
        If True, adapt the graph to its backend; if non-empty string,
        set the graph backend to `adapt_backend` before adapting it.
            :Default: False
    [OPT] constant_folding : bool
        Compute graph constant parts and replace them with constant values.
            :Default: False

    Returns
    -------
    None : directly prune the object.

    Examples
    --------
    >>> import aidge_core
    >>> import aidge_backend_cpu
    >>> import aidge_export_cpp

    >>> def show_nodes(graph):
    ...     for node in graph.get_ordered_nodes():
    ...         print(node.get_operator().type())

    # Build a simple Aidge graph model
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Conv1D(32, 16, [5], name='conv')])
    >>> graph.set_backend('cpu')
    >>> graph.set_datatype(aidge_core.dtype.float32)

    # List the graph's nodes
    >>> show_nodes(graph)
    Producer
    Producer
    Conv1D

    # Set the graph backend to NHWC (for C++ export)
    >>> graph_check_backend(graph, aidge_export_cpp.ExportLibCpp._name)

    # List the graph's nodes
    >>> show_nodes(graph)
    Producer
    Transpose
    Producer
    Conv1D
    Transpose
    """
    # Set the graph backend to 'adapt_backend' if it is a string
    if isinstance(adapt_backend, str) and adapt_backend != '':
        graph.set_backend(adapt_backend)
    # Adapt the graph backend by adding Transpose nodes to it, if needed
    if adapt_backend:
        aidge_core.adapt_to_backend(graph)
    # Adapt the parameters of a 'FC' node, if so
    if adapt_fc:
        aidge_core.adapt_fc_params_format(graph)
#----------------------------------------------------------------------------#

#----------------------   Improve Graph FC Operators   ----------------------#
def improve_graph_fc(graph,
    rm_flatten=False, fuse_batchnorm=False, matmul_to_fc=False):
    """ Take an Aidge graph and improve its FC-oriented operations

    Apply some simplification functions on an Aidge graph, essentially
    regarding the FC operators:
      - `remove_flatten`: remove a Flatten operator if it is followed
            by FC or MatMul; can remove multiple Flatten operators
            if they are one after the other.
      - `matmul_to_fc`: fuse MatMul and Add operator into a FC operator;
            incompatible with TensorFlow/Keras `Dense` layer.
      - `fuse_batchnorm`: fuse BatchNorm with Conv or FC operator.

    If a new operator is set (FC or BatchNorm), check if the graph has
    a backend, and, if any, set the new operators to this backend too.

    IMPORTANT: not compatible with TensorFlow/Keras, as data must not be
        flattened before a `Dense` layer, whereas they must be before a
        PyTorch `Linear` or Aidge `FC` layer. As this last layer contains
        a hidden `Flatten` operation, a `FC` operator used with a Keras
        model leads to an error in dimensions, prohibiting the use of the
        `matmul_to_fc` function, and making `fuse_batchnorm` useless too;
        additionally, since no `Flatten` operator is added before a Keras
        `Dense` layer, the `remove_flatten` function has no effect here.
        Also, the `Producer` attached to the `MatMul` operator' bias has
        only `C` data when adapting a Keras `Dense` layer, whilst `WxC`
        for an Aidge `FC` node, prohibiting the adaptation of it to `FC`.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph for which to optimize the FC operators.
    [OPT] rm_flatten : bool
        Remove Flatten if it is before a FC or MatMul operator.
            :Default: False
    [OPT] fuse_batchnorm : bool
        Fuse BatchNorm with Conv or FC operator.
            :Default: False
    [OPT] matmul_to_fc : bool
        Merge MatMul and Add operator as a FC operator. Be careful as
        this option may cause segmentation faults (core dumped).
            :Default: False

    Returns
    -------
    None : directly prune the object.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core

    >>> def show_nodes(graph):
    ...     for node in graph.get_ordered_nodes():
    ...         print(node.get_operator().type())

    # Build a simple graph
    >>> graph = aidge_core.sequential([
    ...    aidge_core.Flatten(name='flat1'),
    ...    aidge_core.MatMul(name='matmul'),
    ...    aidge_core.Add(name='add'),
    ...    aidge_core.Flatten(name='flat2')])

    # Example data (can be used in a scheduler thereafter)
    >>> shape = (2, 100, 5)
    >>> nb_inps = np.prod(shape[1:])
    >>> nb_outs = 10
    >>> data = np.linspace(0., 1., np.prod(shape), dtype=np.float32)
    >>> data = data.reshape(shape)

    # Add a Producer for the `MatMul` bias input (no backend required)
    >>> bias_mat = np.linspace(0., 1., nb_inps*nb_outs, dtype=np.float32)
    >>> bias_mat = bias_mat.reshape(nb_inps, nb_outs)
    >>> prod_mat = aidge_core.Producer(
    ...     aidge_core.Tensor(bias_mat), name='matmul_bias')
    >>> add_node2graph(graph, prod_mat, 'matmul', 0, 1)

    # Add a Producer for the `Add` second input (no backend required)
    >>> bias_add = np.linspace(0., 1., nb_outs, dtype=np.float32)
    >>> bias_add = bias_add.reshape(nb_outs)
    >>> prod_add = aidge_core.Producer(
    ...     aidge_core.Tensor(bias_add), name='add_bias')
    >>> add_node2graph(graph, prod_add, 'add', 0, 1)

    # List the nodes before modification
    >>> show_nodes(graph)
    Flatten
    Producer
    MatMul
    Producer
    Add
    Flatten

    # Remove the redundant Flatten nodes and fuse the MatMul & Add nodes
    >>> improve_graph_fc(graph, rm_flatten=True, matmul_to_fc=True)
    >>> show_nodes(graph)
    Producer
    Producer
    FC
    Flatten
    """

    # Get and save the backend in case the graph will be changed (new op)
    backend = get_backend(graph) if matmul_to_fc or fuse_batchnorm else ''

    # Perform optimization on the MatMul/Add nodes
    if rm_flatten:      # Useless with TensorFlow/Keras
        aidge_core.remove_flatten(graph)
    if fuse_batchnorm:
        aidge_core.fuse_batchnorm(graph)
    if matmul_to_fc:    # Incompatible with TensorFlow/Keras
        aidge_core.matmul_to_fc(graph)

    # Set the backend in case the graph has changed (new op)
    if (matmul_to_fc or fuse_batchnorm) and backend != '':
        graph.set_backend(backend)
#----------------------------------------------------------------------------#

#--------------------------   Convolution Tiling   --------------------------#
def conv_tiling(graph, conv_name, axis=1, nb_slices=2):
    """ Perform tiling on an Aidge graph model

     Split a convolution operator into `nb_slices` smaller convolutions
     preceded by a Slice operator. All `nb_slices` results are finally
     concatenated into a single Tensor as output.

    Parameters
    ----------
    graph : aidge_core.GraphView
        The graph on which perform tiling.
    conv_name : str
        The name of the convolution operator to tile.
    [OPT] axis : int
        The axis along which perform tiling.
            :Default: 1
    [OPT] nb_slices : int
        The number of smaller convs in which decompose the conv node.
            :Default: 2

    Returns
    -------
    None : directly modified the graph, and set anew its backend, if any.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    >>> def show_nodes(graph):
    ...     for node in graph.get_ordered_nodes():
    ...         print(node.get_operator().type(), '---', node.name())

    # Generate example data
    >>> data = np.linspace(0., 10., 2*32*500, dtype=np.float32)

    # Build a simple Aidge graph model
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Identity(name='identity'),
    ...     aidge_core.Conv1D(32, 16, [5], name='conv1'),
    ...     aidge_core.Conv1D(16, 8, [5], name='conv2')])
    >>> graph.set_backend('cpu')
    >>> graph.set_datatype(aidge_core.dtype.float32)

    # List the graph's nodes
    >>> show_nodes(graph)
    Identity --- identity
    Producer --- conv1_w
    Producer --- conv1_b
    Conv1D --- conv1
    Producer --- conv2_w
    Producer --- conv2_b
    Conv1D --- conv2

    # Perform tiling on 'conv1', but leave 'conv2' unchanged
    >>> conv_tiling(graph, 'conv1', 1, 2)

    # List the graph's nodes
    >>> show_nodes(graph)
    Identity --- identity
    Producer --- conv1_Slice_0_1
    Producer --- conv1_Slice_0_2
    Producer --- conv1_Slice_0_3
    Producer --- conv1_Slice_0_4
    Slice --- conv1_Slice_0
    Producer --- conv1_w
    Producer --- conv1_b
    Conv1D --- conv1_0
    Producer --- conv1_Slice_1_1
    Producer --- conv1_Slice_1_2
    Producer --- conv1_Slice_1_3
    Producer --- conv1_Slice_1_4
    Slice --- conv1_Slice_1
    Conv1D --- conv1_8
    Concat --- Concat
    Producer --- conv2_w
    Producer --- conv2_b
    Conv1D --- conv2
    """

    # Get and save the current graph backend, if any
    backend = get_backend(graph)

    # Apply tiling on the conv node (split into 'nb_slices' convs)
    tiled_conv = aidge_core.get_conv_horizontal_tiling(
        node=graph.get_node(conv_name), axis=axis, nb_slices=nb_slices)

    # Replace the conv and its 2 parents (weights & bias) by the slices
    node_to_replace = {graph.get_node(conv_name),
                       graph.get_node(conv_name).get_parent(1),
                       graph.get_node(conv_name).get_parent(2)}
    aidge_core.GraphView.replace(node_to_replace, tiled_conv)

    # Set back the backend (if any), since the graph has been modified
    if backend != '':
        graph.set_backend(backend)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                        Post Training Quantization                        ##
##############################################################################

#---------------------------   Rescale Tensors   ----------------------------#
def rescale_tensors(tensors, nb_bits, dtype, backend='cpu'):
    """ Rescale a set of tensors

    Take a (set of) tensor(s), and rescale them to fit the PTQ format,
    that is every data is multiplied by 2**(NB_BITS-1)-1. The `tensors`
    can either be a lone or set of either NumPy arrays or Aidge Tensors;
    if `tensors` is a single array, the returned object will also be an
    Aidge Tensor; if it was a set (list/tuple) of arrays, the returned
    object will also be a set of Aidge Tensors.

    See the `quantize_network` function from the `aidge_quantization`
    module for detail on the parameters.

    IMPORTANT: the data must be normalized between 0 and 1 before.

    Parameters
    ----------
    tensors : (list/tuple of) np.ndarray or aidge_core.Tensor
        The tensors to rescale; if it is a set of arrays, they must all
        be of the same type.
    nb_bits : int
        The size in bits of the PTQ target data type.
    dtype : aidge_core.dtype data type
        The PTQ target data type.
    [OPT] backend : str
        The backend to which to set the rescaled tensors to.
            :Default: 'cpu'

    Returns
    -------
    tensors_res : (list/tuple of) aidge_core.Tensor
        The rescaled Aidge Tensors; if `tensors` is a set of Tensors,
        `tensors_res` is also a set of Tensors; if it is a single Tensor,
        `tensors_res` is also a single Tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    >>> nb_bits = 8
    >>> dtype = aidge_core.dtype.int32

    # Generate example data
    >>> data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)

    # Perform rescaling
    >>> tensors_res_np = rescale_tensors(data, nb_bits, dtype)
    >>> tensors_res_np_set = rescale_tensors([data, data+1.], nb_bits, dtype)
    >>> tensors_res = rescale_tensors(aidge_core.Tensor(data), nb_bits, dtype)
    >>> tensors_res_set = rescale_tensors(
    ...     [aidge_core.Tensor(data), aidge_core.Tensor(data+1.)], nb_bits, dtype)
    """

    # Set `tensors` into a set (list, tuple), if needed
    if not isinstance(tensors, (list, tuple)):
        flag = True             # `tensors` was originally a Tensor
        tensors = [tensors]
    else:
        flag = False            # `tensors` was originally a set of Tensors

    # Rescale the tensors data
    coef = float(2**(nb_bits-1) - 1)
    # If the target type is int-based, round the data and cast them before
    if 'int' in str(dtype):
        if isinstance(tensors[0], np.ndarray):
            tensors_res = [Tensor(np.round(coef*tensor).astype(int))
                           for tensor in tensors]
        else:
            tensors_res = [Tensor(np.round(coef*np.array(tensor)).astype(int))
                           for tensor in tensors]
    # Else, if it is float-based, rescale the tensors directly
    else:
        if isinstance(tensors[0], np.ndarray):
            tensors_res = [Tensor(coef*tensor) for tensor in tensors]
        else:
            tensors_res = [Tensor(coef*np.array(tensor)) for tensor in tensors]

    # Set data type & backend
    for tensor in tensors_res:
        tensor.set_datatype(dtype)      # Set tensor data type
        tensor.set_backend(backend)     # Set tensor backend

    if flag:
        return tensors_res[0]           # Return a lone Tensor
    return tensors_res                  # Return a set of Tensors
#----------------------------------------------------------------------------#

#------------------------   Reset Graph Dimensions   ------------------------#
def reset_graph_dims(graph):
    """ Get the input nodes' dimensions and forward them to the graph

    Should only be used to forward the dimensions of the graph after a
    change in its layers. Data must be forwarded prior (see `propagate`),
    so that the dimensions can be extracted from the data themselves.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph for which to reset the dimensions forwarding.

    Returns
    -------
    None : directly forward the dimensions to the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Generate example data
    >>> data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)

    # Build a simple Aidge graph model
    >>> graph = aidge_core.sequential([
    ...     aidge_core.FC(256, 128, name='fc1'),
    ...     aidge_core.FC(128, 5, name='fc2')])
    >>> graph.set_backend('cpu')
    >>> graph.set_datatype(aidge_core.dtype.float32)

    # Forward the data to the model to set its dimensions
    >>> scheduler = aidge_core.SequentialScheduler(graph)
    >>> scheduler.forward(True, data=[aidge_core.Tensor(data)])

    # Reset the graph dimensions
    >>> reset_graph_dims(graph)
    """
    # Retrieve the new dimensions of the graph input
    dims = [node[0].get_operator().get_input(0).dims
            for node in graph.get_ordered_inputs()]
    # Pass the input dimensions to the graph
    graph.forward_dims(dims=dims, allow_data_dependency=True)
#----------------------------------------------------------------------------#

##############################################################################
