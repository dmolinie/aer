import numpy as np
import aidge_core
import aidge_backend_cpu
from aer.aidge._aidge_tk import *


def test_load_onnx():
    """ Load the ONNX model into an Aidge graph """
    # Create an Aidge graph
    import aidge_onnx
    graph = aidge_core.sequential([
        aidge_core.Producer([10, 256], name='dataProvider'),
        aidge_core.FC(256, 5, name='fc')])
    # Set the graph implementation (backend)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend('cpu')
    # Save the graph into an ONNX file
    aidge_onnx.export_onnx(graph, "aidge_graph.onnx")
    # (Re)load the model graph from the ONNX file
    graph_onnx = load_onnx("aidge_graph.onnx")

def test_export_onnx():
    """ Export an Aidge graph into an ONNX file """
    # Create an Aidge graph
    graph = aidge_core.sequential([
        aidge_core.Producer([10, 256], name='dataProvider'),
        aidge_core.FC(256, 5, name='fc')])
    # Set the graph implementation (backend)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend('cpu')
    # Save the graph into an ONNX file
    export_onnx("aidge_graph.onnx", graph)

def test_get_backend():
    """ Get the backend of a graph, if any """
    # Create an Aidge graph
    graph = aidge_core.sequential([
        aidge_core.Producer([10, 256], name='dataProvider'),
        aidge_core.FC(256, 5, name='fc')])
    # Set the graph implementation (backend)
    graph.set_backend('cpu')
    # Retrieve the graph backend
    print(get_backend(graph))

def test_get_node_dtype():
    """ Get the operator's data type of the operator of a node """
    # Define a simple Aidge graph and retrieve its nodes
    graph = aidge_core.sequential([
        aidge_core.FC(256, 5, name='fc'),
        aidge_core.Flatten(name='flat')])
    nodes = graph.get_ordered_nodes()
    # Get the default data type
    print(get_node_dtype(nodes[0]))
    # Set a data type to the graph and retrieve it thereafter
    graph.set_datatype(aidge_core.dtype.int32)
    print(get_node_dtype(nodes[0]))

def test_get_input_node():
    """ Find the input node of a GraphView """
    graph = aidge_core.sequential([
       aidge_core.Producer([16, 3, 512, 512], name='dataProvider'),
       aidge_core.Conv2D(3, 4, [5, 5], name='conv'),
       aidge_core.ReLU(name='relu')])
    print(get_input_node(graph))

def test_get_output():
    """ Get the values contained in a given channel of a given output node """
    # Generate some example data
    data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)
    # Build a simple Aidge graph
    graph = aidge_core.sequential([aidge_core.FC(256, 5, name='fc')])
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend('cpu')
    # Instantiate a scheduler and forward the `data` to the model
    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.forward(True, data=[aidge_core.Tensor(data)])
    # Retrieve the graph's output
    output = get_output(graph)
    print(np.shape(output))

def test_forward():
    """ Propagate some data in an Aidge model using a scheduler """
    # Generate some example data
    data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)
    # Build a simple Aidge graph
    graph = aidge_core.sequential([aidge_core.FC(256, 5, name='fc')])
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend('cpu')
    # Instantiate a scheduler
    scheduler = aidge_core.SequentialScheduler(graph)
    # Forward the data to the model and retrieve the graph's output
    output = forward(graph, scheduler, data)
    print(np.shape(output))

def test_add_node2graph():
    """ Add a node to an Aidge graph """
    # Define a simple, 1-layer graph
    graph = aidge_core.sequential([aidge_core.Add(name='add')])
    # Define a `Producer` node to be added before the graph's `Add` node
    bias_add = np.linspace(0., 1., 100, dtype=np.float32).reshape(10, 10)
    producer = aidge_core.Producer(aidge_core.Tensor(bias_add), name='add_bias')
    # Show the graph's operators before modification
    [print(node.get_operator().type()) for node in graph.get_ordered_nodes()]
    # Add the Producer node to the graph, as bias of the Add operator
    # The `Producer` 1st output (0) is connected to the `Add` 2nd input (1)
    add_node2graph(graph, producer, 'add', 0, 1)
    # Show the graph's operators after modification
    [print(node.get_operator().type()) for node in graph.get_ordered_nodes()]

def test_graph_check_backend():
    """ Take an Aidge graph and check that its backend is fully set """
    import aidge_export_cpp
    def show_nodes(graph):
        for node in graph.get_ordered_nodes():
            print(node.get_operator().type())
    # Build a simple Aidge graph model
    graph = aidge_core.sequential([
        aidge_core.Conv1D(32, 16, [5], name='conv')])
    graph.set_backend('cpu')
    graph.set_datatype(aidge_core.dtype.float32)
    # List the graph's nodes
    show_nodes(graph)
    # Set the graph backend to NHWC (for C++ export)
    graph_check_backend(graph, aidge_export_cpp.ExportLibCpp._name)
    # List the graph's nodes
    show_nodes(graph)

def test_improve_graph_fc():
    """ Take an Aidge graph and improve its FC-oriented operations """
    def show_nodes(graph):
        for node in graph.get_ordered_nodes():
            print(node.get_operator().type())
    # Build a simple graph
    graph = aidge_core.sequential([
       aidge_core.Flatten(name='flat1'),
       aidge_core.MatMul(name='matmul'),
       aidge_core.Add(name='add'),
       aidge_core.Flatten(name='flat2')])
    # Example data (can be used in a scheduler thereafter)
    shape = (2, 100, 5)
    nb_inps = np.prod(shape[1:])
    nb_outs = 10
    data = np.linspace(0., 1., np.prod(shape), dtype=np.float32)
    data = data.reshape(shape)
    # Add a Producer for the `MatMul` bias input (no backend required)
    bias_mat = np.linspace(0., 1., nb_inps*nb_outs, dtype=np.float32)
    bias_mat = bias_mat.reshape(nb_inps, nb_outs)
    prod_mat = aidge_core.Producer(
        aidge_core.Tensor(bias_mat), name='matmul_bias')
    add_node2graph(graph, prod_mat, 'matmul', 0, 1)
    # Add a Producer for the `Add` second input (no backend required)
    bias_add = np.linspace(0., 1., nb_outs, dtype=np.float32)
    bias_add = bias_add.reshape(nb_outs)
    prod_add = aidge_core.Producer(
        aidge_core.Tensor(bias_add), name='add_bias')
    add_node2graph(graph, prod_add, 'add', 0, 1)
    # List the nodes before modification
    show_nodes(graph)
    # Remove the redundant Flatten nodes and fuse the MatMul & Add nodes
    improve_graph_fc(graph, rm_flatten=True, matmul_to_fc=True)
    show_nodes(graph)

def test_conv_tiling():
    """ Perform tiling on an Aidge graph model """
    def show_nodes(graph):
        for node in graph.get_ordered_nodes():
            print(node.get_operator().type(), '---', node.name())
    # Generate example data
    data = np.linspace(0., 10., 2*32*500, dtype=np.float32)
    # Build a simple Aidge graph model
    graph = aidge_core.sequential([
        aidge_core.Identity(name='identity'),
        aidge_core.Conv1D(32, 16, [5], name='conv1'),
        aidge_core.Conv1D(16, 8, [5], name='conv2')])
    graph.set_backend('cpu')
    graph.set_datatype(aidge_core.dtype.float32)
    # List the graph's nodes
    show_nodes(graph)
    # Perform tiling on `conv1`, but leave `conv2` unchanged
    conv_tiling(graph, 'conv1', 1, 2)
    # List the graph's nodes
    show_nodes(graph)

def test_rescale_tensors():
    """ Rescale a set of tensors """
    nb_bits = 8
    dtype = aidge_core.dtype.int32
    # Generate example data
    data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)
    # Perform rescaling
    tensors_res_np = rescale_tensors(data, nb_bits, dtype)
    tensors_res_np_set = rescale_tensors([data, data+1.], nb_bits, dtype)
    tensors_res = rescale_tensors(aidge_core.Tensor(data), nb_bits, dtype)
    tensors_res_set = rescale_tensors(
        [aidge_core.Tensor(data), aidge_core.Tensor(data+1.)], nb_bits, dtype)

def test_reset_graph_dims():
    """ Get the input nodes' dimensions and forward them to the graph """
    # Generate example data
    data = np.linspace(0., 10., 2560, dtype=np.float32).reshape(10, 256)
    # Build a simple Aidge graph model
    graph = aidge_core.sequential([
        aidge_core.FC(256, 128, name='fc1'),
        aidge_core.FC(128, 5, name='fc2')])
    graph.set_backend('cpu')
    graph.set_datatype(aidge_core.dtype.float32)
    # Forward the data to the model to set its dimensions
    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.forward(True, data=[aidge_core.Tensor(data)])
    # Reset the graph dimensions
    reset_graph_dims(graph)



# Launch test/example functions
test_load_onnx()

test_export_onnx()

test_get_backend()

test_get_node_dtype()

test_get_input_node()

test_get_output()

test_forward()

test_add_node2graph()

test_graph_check_backend()

test_improve_graph_fc()

#test_conv_tiling() # Segfault in new Aidge version (> V0.8.1)

test_rescale_tensors()

test_reset_graph_dims()

