import numpy as np
import aidge_core
import aidge_backend_cpu
from aer.aidge._aidge_io import *


def test_save_mmd():
    """ Export the an Aidge model graph as a Mermaid instructions file """
    # Create the Aidge graph to save
    graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])
    # Save the model graph as Mermaid instructions
    save_mmd("Dir/graph.mmd", graph)

def test_load_mmd():
    """ Load a Mermaid file as text (str) """
    # Create the Aidge graph to save
    graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])
    # Save the model graph as Mermaid instructions
    save_mmd("graph.mmd", graph)
    # Load the just saved mermaid instructions (as text)
    graph_mmd = load_mmd("graph.mmd")

def test_plot_mmd_as_html():
    """ Generate an HTML file to visualize an Aidge graph """
    # Create the Aidge graph to save
    graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])
    # Save the model graph as Mermaid instructions
    save_mmd("graph.mmd", graph)
    # Load the just saved mermaid instructions (as text)
    graph_mmd = plot_mmd_as_html("graph.mmd")

def test_plot_mmd_as_jpg():
    """ Visualize the graph of a Mermaid file and save it as a jpg file """
    # Create the Aidge graph to save
    graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])
    # Save the model graph as Mermaid instructions
    save_mmd("graph.mmd", graph)
    # Load the just saved mermaid instructions (as text)
    graph_mmd = plot_mmd_as_jpg("graph.mmd", min_size_px=500)

def test_improve_graph_cpp():
    """ Improve a graph for C++ export """
    import aer.aidge as aidge_tk
    # Build a simple graph
    graph = aidge_core.sequential([
       aidge_core.Mul(name='mul'),
       aidge_core.Add(name='add')])
    # Example data (can be used in a scheduler thereafter)
    shape = (2, 100, 5)
    nb_inps = np.prod(shape[1:])
    data = np.linspace(0., 1., np.prod(shape), dtype=np.float32)
    data = data.reshape(shape)
    # Add a Producer for the `MatMul` bias input (no backend required)
    bias_mat = np.linspace(0., 1., 1000, dtype=np.float32)
    bias_mat = bias_mat.reshape(2, 100, 5)
    prod_mat = aidge_core.Producer(
        aidge_core.Tensor(bias_mat), name='mul_bias')
    aidge_tk.add_node2graph(graph, prod_mat, 'mul', 0, 1)
    # Add a Producer for the `Add` second input (no backend required)
    bias_add = np.linspace(0., 1., 5, dtype=np.float32)
    bias_add = bias_add.reshape(5)
    prod_add = aidge_core.Producer(
        aidge_core.Tensor(bias_add), name='add_bias')
    aidge_tk.add_node2graph(graph, prod_add, 'add', 0, 1)
    # Set the graph backend, forward a batch of data and export it to C++
    graph.set_backend('cpu')
    scheduler = aidge_core.SequentialScheduler(graph)
    output_org = aidge_tk.forward(graph, scheduler, aidge_core.Tensor(data))
    # Export the original graph in C++
    export_cpp(graph, scheduler, folder="Mul_CPP")
    # Improve the export by removing the unnecessary `Mul` producer
    improve_graph_cpp(graph, ex_prods=True)
    # Reset the backend & scheduler and forward anew a batch of data
    graph.set_backend('cpu')
    scheduler.reset_scheduling()
    scheduler.generate_scheduling()
    output_imp = aidge_tk.forward(graph, scheduler, aidge_core.Tensor(data))
    # Export the corrected graph in C++
    export_cpp(graph, scheduler, folder="Mul_CPP_Imp")

def test_nchw2nhwc4export():
    """ Change the data format of an Aidge graph """
    # Generate example data
    shape = (15, 5, 300)                # NCW (3D data)
    data = np.linspace(0., 10., np.prod(shape), dtype=np.float32)
    data = data.reshape(shape)
    # Build a simple Aidge graph model
    graph = aidge_core.sequential([
        aidge_core.Conv1D(5, 3, [10], name='conv'),
        aidge_core.ReLU(name='relu')])
    graph.set_backend('cpu')
    graph.set_datatype(aidge_core.dtype.float32)
    # Instantiate a scheduler to forward the data to the graph
    scheduler = aidge_core.SequentialScheduler(graph)
    pred = scheduler.forward(True, data=[aidge_core.Tensor(data)])
    # Change graph data format to NHWC and pass reshaped data to scheduler
    nchw2nhwc4export(graph, 'ncw')      # Required for Conv1D
    # C++ Export part, cf. the `export_cpp` function
    export_cpp(graph, scheduler, folder="export_format")

def test_export_cpp():
    """ Export the model graph in a C++ source project """
    # Generate some test data (required for C++ export)
    shape = (15, 3, 28, 28)             # NCHW (Aidge default in Python)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    # Generate a simple model to export
    graph = aidge_core.sequential([
        aidge_core.Conv2D(3, 3, [5, 5], name='conv'),
        aidge_core.ReLU(name='relu')])
    # Set the model graph implementation & datatype
    graph.set_backend('cpu')
    graph.set_datatype(aidge_core.dtype.float32)
    # Build a scheduler for forwarding data to the graph
    scheduler = aidge_core.SequentialScheduler(graph)
    pred = scheduler.forward(data=[aidge_core.Tensor(data)])
    # Set the graph's data format to NHWC (default for C++ export)
    nchw2nhwc4export(graph, 'nchw')     # Required for `Conv2D`
    # Perform the export of the Aidge model graph to a C++ project
    export_cpp(graph, scheduler, folder="export_cpp")

def test_export_array_cpp():
    """ Export a (list of) data array(s) in C++ """
    # Generate the data to export
    N = 10                  # Batch size
    shape = (256, 3)        # Input shape
    data = np.arange(N*np.prod(shape), dtype=np.float32).reshape(N, *shape)
    # Perform the export
    export_array_cpp(data, "test_export/data", False)



# Launch test/example functions
test_save_mmd()

test_load_mmd()

test_plot_mmd_as_html()

test_plot_mmd_as_jpg()

test_improve_graph_cpp()

test_nchw2nhwc4export()

test_export_cpp()

test_export_array_cpp()

