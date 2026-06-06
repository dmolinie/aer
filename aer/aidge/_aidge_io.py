""" Shorthand tools to manipulate ONNX files in Aidge

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: September 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'save_mmd', 'load_mmd', 'plot_mmd_as_html', 'plot_mmd_as_jpg',
    'improve_graph_cpp', 'nchw2nhwc4export', 'export_cpp', 'export_array_cpp']

import os
import io
import base64

import requests
from PIL import Image
import matplotlib.pyplot as plt

import aidge_core
#import aidge_backend_cpu   # Should be loaded in the main script
import aidge_export_cpp

from aer.tools._tools import check_ext, check_folder, check_path
from aer.aidge import _aidge_tk as aidge_tk


##############################################################################
##                            Mermaid Files I/O                             ##
##############################################################################

def _check_file_ext(fname, refname, ext):
    """ Check a filename, or take the reference one if not provided; if
    it has no path, take that of the reference name; check filename ext """
    # If no filename provided, take the reference one without its extension
    if fname is None:
        fname, _ = os.path.splitext(refname)
    # If provided but with no path (dirs), take that of the reference one
    elif os.path.split(fname)[0] == '':
        fname = os.path.join(os.path.split(refname)[0], fname)
    # Check the extension of the filename and return it
    return check_ext(fname, ext)

#------------------------   Save Graph as Mermaid   -------------------------#
def save_mmd(pathname, graph):
    """ Export the an Aidge model graph as a Mermaid instructions file

    Parameters
    ----------
    pathname : str
        The pathname of the Mermaid file in which to save the model; if
        it contains directories, create them if they do not exist; add
        the '.mmd' extension to the file if required.
    graph : aidge_core.GraphView object
        The Aidge graph to save as Mermaid instructions.

    Returns
    -------
    None : directly create the folders (if needed) and write the file.

    Examples
    --------
    >>> import aidge_core

    # Create the Aidge graph to save
    >>> graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])

    # Save the model graph as Mermaid instructions
    >>> save_mmd("Dir/graph.mmd", graph)
    """
    # Check the folder & file name
    pathname = check_path(pathname)             # Create the folders if needed
    pathname, _ = os.path.splitext(pathname)    # Remove the extension
    # Save the graph in the mmd file (`save` method adds the '.mmd' ext)
    graph.save(pathname)                        # Save the graph as Mermaid
#----------------------------------------------------------------------------#

#--------------------------   Load Mermaid Graph   --------------------------#
def load_mmd(pathname):
    """ Load a Mermaid file as text (str)

    Parameters
    ----------
    pathname : str
        The pathname of the Mermaid file to read; should end with '.mmd',
        but this extension is automatically appended if needed.

    Returns
    -------
    graph_mmd : str
        The (non-formatted) Mermaid instructions of the Aidge graph.

    Examples
    --------
    >>> import aidge_core

    # Create the Aidge graph to save
    >>> graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])

    # Save the model graph as Mermaid instructions
    >>> save_mmd("graph.mmd", graph)

    # Load the just saved mermaid instructions (as text)
    >>> graph_mmd = load_mmd("graph.mmd")
    """
    pathname = check_ext(pathname, '.mmd')
    # Check that the mmd file exists; raise an error else
    if not os.path.exists(pathname):
        raise FileNotFoundError(f"File {pathname} not found; save the graph prior")
    # Load the graph as a Mermaid set of instructions
    with open(pathname, 'r', encoding='utf-8') as file:
        graph_mmd = file.read()
    return graph_mmd
#----------------------------------------------------------------------------#

#---------------------   Save Graphical Graph as HTML   ---------------------#
def plot_mmd_as_html(mmdname, htmlname=None):
    """ Generate an HTML file to visualize an Aidge graph

    Take the pathname of an mmd file and load the pointed file as a text
    of Mermaid-compliant instructions; then, generate a simple HTML file
    which includes these instructions, as well as a call to the online
    Mermaid API that will be used to generate the graph corresponding
    to the instructions contained in the mmd file.

    N.B.: the HTML file can be loaded using a Web Browser thereafter;
        this requires an active Internet connection as the HTML needs
        to access the online Mermaid API.

    N.B.: the main interest of this function is that the graph can be
        visualized as a vector image, contrary to the `plot_mmd_as_jpg`
        function which saves the image in a graphical format directly.

    Parameters
    ----------
    mmdname : str
        The pathname in which is saved the model Mermaid graph; the
        `mmdname` filename is expected to have an '.mmd' extension.
    [OPT] htmlname : str
        The pathname of the HTML file in which to save the call to the
        Mermaid API. If None, it is the mmd filename and path which are
        used instead, just replacing the 'mmd' extension with 'html'.
            :Default: None

    Returns
    -------
    None : directly save the HTML file.

    Examples
    --------
    >>> import aidge_core

    # Create the Aidge graph to save
    >>> graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])

    # Save the model graph as Mermaid instructions
    >>> save_mmd("graph.mmd", graph)

    # Load the just saved mermaid instructions (as text)
    >>> graph_mmd = plot_mmd_as_html("graph.mmd")
    """
    # Load the Mermaid graph
    graph_mmd = load_mmd(check_ext(mmdname, '.mmd'))
    # Check the file name (no folder verification)
    htmlname = _check_file_ext(htmlname, mmdname, '.html')
    # Write the graph instructions in a Mermaid-compliant HTML format
    url = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"
    with open(htmlname, 'w', encoding='utf-8') as file:
        file.write('<html>\n\t<body>\n\t\t<pre class="mermaid">\n')
        file.write(graph_mmd)
        file.write('\t\t</pre>\n')
        file.write('\t\t<script type="module">\n')
        file.write(f'\t\t\timport mermaid from "{url}";\n')
        file.write('\t\t\tmermaid.initialize({ startOnLoad: true });\n')
        file.write('\t\t</script>\n\t</body>\n</html>')
#----------------------------------------------------------------------------#

#---------------------   Save Graphical Graph as JPG   ----------------------#
def plot_mmd_as_jpg(mmdname, imgname=None, min_size_px=1000, show=False):
    """ Visualize the graph of a Mermaid file and save it as a jpg file

    Take the pathname of an mmd file and load the pointed file as a text
    of Mermaid-compliant instructions; send these instructions to the
    Mermaid server and retrieve the image that it generated and returned
    in response to the mmd instructions. Then, wrap the returned image
    into a Matplotlib figure and save it as a jpg file. If `show` is set
    to True, show the image; close it before returning else.

    As the image returned by the Mermaid server is graphical, its sizes
    are fixed and cannot be changed; however, Matplotlib can adjust the
    sizes of the wrapping figure. Use the `min_size_px` argument to set
    the sizes of the resulting, wrapped figure: this argument indicates
    to Matplotlib to set the smallest size of the original image to its
    value (in pixels), while enlarging the other size accordingly to keep
    the original image ratio. If the `min_size_px` value is smaller than
    the smallest size of the original image, leave it unchanged.

    N.B.: this function requires an active Internet connection to access
        the online Mermaid server. The maximal latency is set to 10 secs.

    Parameters
    ----------
    mmdname : str
        The pathname in which is saved the model Mermaid graph to plot;
        the `mmdname` filename is expected to have an '.mmd' extension.
    [OPT] imgname : str
        The pathname of the file in which to save the figure; the image
        is saved as a jpg file. If None, it is the mmd filename and path
        which are used instead, just replacing the 'mmd' ext with 'jpg'.
            :Default: None
    [OPT] min_size_px : int
        The minimal side size (in pixels) of the graph representation.
            :Default: 1000 (pixels)
    [OPT] show : bool
        Whether to show or close the image after saving it.
            :Default: False (free the memory after saving)

    Returns
    -------
    None : directly save (and possibly show prior) the graph image.

    Examples
    --------
    >>> import aidge_core

    # Create the Aidge graph to save
    >>> graph = aidge_core.sequential([aidge_core.FC(768, 5, name='fc')])

    # Save the model graph as Mermaid instructions
    >>> save_mmd("graph.mmd", graph)

    # Load the just saved mermaid instructions (as text)
    >>> graph_mmd = plot_mmd_as_jpg("graph.mmd", min_size_px=500)
    """

    # Load the Mermaid graph
    graph_mmd = load_mmd(check_ext(mmdname, '.mmd'))
    # Check the file name (no folder verification)
    imgname = _check_file_ext(imgname, mmdname, '.jpg')

    # Retrieve the image from the Mermaid server
    graphbytes = graph_mmd.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    url = 'https://mermaid.ink/img/' + base64_bytes.decode('ascii')
    img = Image.open(io.BytesIO(requests.get(url, timeout=10).content)) # 10 secs

    # Plot the figure retrieve from the Mermaid server
    plt.figure(layout='constrained')
    plt.imshow(img)     # Insert the image in a figure
    plt.axis('off')     # Hide the figure axes
    # Ensure that the smallest side (width or height) is 1000 px minimum
    size = img.getdata().size
    if size[1] > size[0]:
        # If the height is greater than the width, use the fixed height (491 px)
        # to estimate the future width after 'bbox_inches'
        ratio = size[1] / (size[0]*491)
    else:
        # Else, if the width is greater, use the fixed width (651 px) to estimate
        # the future image height
        ratio = size[0] / (size[1]*651)
    # Set the smallest side (H/W) to 'min_size_px' px while keeping the image ratio
    min_side = ratio * min_size_px
    # Multiply the above quantity by 100 (default) to convert Pixels into Inches
    plt.savefig(imgname, bbox_inches='tight', dpi=100*min_side)
    # Either show or close the figure (both return None)
    return plt.show() if show else plt.close()
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           C++ Export Functions                           ##
##############################################################################

#---------------------   Improve Graph For C++ Export   ---------------------#
def improve_graph_cpp(graph, ex_prods=False, fuse_metaops=False):#, set_dtypes=False):
    """ Improve a graph for C++ export

    Take an Aidge graph and apply the following functions to it, is so:
      - `exclude_unwanted_producers`: exclude the producers useless for
            C++ export, e.g. those of the Mul and BitShift nodes, as the
            data they contain are exported as arguments to the C++ ops.
      - `cpp_fuse_to_metaops`: fuse nodes into metaops adapted for the
            C++ export.
#      - `set_nodes_datatypes`: set the C++ operators datatype to that of
#            the graph. Can not be used on Conv2D and MatMul/FC nodes directly
#            as the bias datatype is generally specific.

    Parameters
    ----------
    [OPT] ex_prods : bool
        If useless producers should be excluded.
            :Default: False
    [OPT] fuse_metaops : bool
        If operators should be fused into metaops for export.
            :Default: False
#    [OPT] set_datatype : bool
#        If the operators datatype(s) should be updated before export.
#            :Default: False

    Returns
    -------
    None : directly apply the function to the graph.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu
    >>> import aer.aidge as aidge_tk

    # Build a simple graph
    >>> graph = aidge_core.sequential([
    ...    aidge_core.Mul(name='mul'),
    ...    aidge_core.Add(name='add')])

    # Example data (can be used in a scheduler thereafter)
    >>> shape = (2, 100, 5)
    >>> nb_inps = np.prod(shape[1:])
    >>> data = np.linspace(0., 1., np.prod(shape), dtype=np.float32)
    >>> data = data.reshape(shape)

    # Add a Producer for the `MatMul` bias input (no backend required)
    >>> bias_mat = np.linspace(0., 1., 1000, dtype=np.float32)
    >>> bias_mat = bias_mat.reshape(2, 100, 5)
    >>> prod_mat = aidge_core.Producer(
    ...     aidge_core.Tensor(bias_mat), name='mul_bias')
    >>> aidge_tk.add_node2graph(graph, prod_mat, 'mul', 0, 1)

    # Add a Producer for the `Add` second input (no backend required)
    >>> bias_add = np.linspace(0., 1., 5, dtype=np.float32)
    >>> bias_add = bias_add.reshape(5)
    >>> prod_add = aidge_core.Producer(
    ...     aidge_core.Tensor(bias_add), name='add_bias')
    >>> aidge_tk.add_node2graph(graph, prod_add, 'add', 0, 1)

    # Set the graph backend, forward a batch of data and export it to C++
    >>> graph.set_backend('cpu')
    >>> scheduler = aidge_core.SequentialScheduler(graph)
    >>> output_org = aidge_tk.forward(graph, scheduler, aidge_core.Tensor(data))

    # Export the original graph in C++
    >>> export_cpp(graph, scheduler, folder="Mul_CPP")
    gen : Mul_CPP/data/mul_input_0.h

    # Improve the export by removing the unnecessary `Mul` producer
    >>> improve_graph_cpp(graph, ex_prods=True)

    # Reset the backend & scheduler and forward anew a batch of data
    >>> graph.set_backend('cpu')
    >>> scheduler.reset_scheduling()
    >>> scheduler.generate_scheduling()
    >>> output_imp = aidge_tk.forward(graph, scheduler, aidge_core.Tensor(data))

    # Export the corrected graph in C++
    >>> export_cpp(graph, scheduler, folder="Mul_CPP_Imp")
    ## Here, the 'Mul''s producer data files are not exported anymore:
    ##   - 'dnn/include/layers/mul_bias.h'
    ##   - 'dnn/include/parameters/mul_bias.h"
    gen : Mul_CPP_Imp/data/mul_input_0.h
    """
    if ex_prods:
        aidge_export_cpp.exclude_unwanted_producers(graph)
    if fuse_metaops:
        aidge_export_cpp.cpp_fuse_to_metaops(graph)
#    if set_dtypes:
#        flag = False
#        for node in graph.get_ordered_nodes():
#            op_type = node.get_operator().type()
#            flag = flag or op_type in ('MatMul', 'FC') or 'Conv' in op_type
#            if flag:
#                break
#        if flag:
#            print("[WARN] Graph contains either MatMul/FC or Conv operator,",
#                  "cannot apply 'set_nodes_datatypes' function")
#        else:
#        aidge_export_cpp.set_nodes_datatypes(graph)
#----------------------------------------------------------------------------#

#-----------------------------   NCHW to NHWC   -----------------------------#
def nchw2nhwc4export(graph, dformat=None):
    """ Change the data format of an Aidge graph

    Take an Aidge graph and set the data format of its operators and the
    tensors they contain to NHWC.

    N.B.: if a scheduler is attached to the graph, it should be reset;
        also, data forwarding using it would probably fail, as this
        function is designed for C++ export only.

    IMPORTANT: before calling this function, make sure that:
      - a backend has been given to the graph (and tensors);
      - a batch of data has been forwarded to the graph st the nodes'
            operators are not empty.

    IMPORTANT: this function must be used for C++ export only, as the
        dimensions of the graph's operators are not properly changed.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph for which to reset the dimensions forwarding.
    [OPT] dformat : string or None
        The original data format of the model; can be `ncw` or `nchw`.
        If None, try to deduce it from the data contained in the input
        node of the graph, if any; note that, although this should al-
        ways work when data were previously forwarded to the graph, the
        Aidge C++ export function may require 4D data, especially with
        models generated with Torch and exported in ONNX; in such a case,
        it may require `nchw` even though models are actually `ncw`.
            :Default: None

    Returns
    -------
    None : directly perform the dimensions swap & graph dimensions reset
        if a correct data format has been provided; return None else.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Generate example data
    >>> shape = (15, 5, 300)                # NCW (3D data)
    >>> data = np.linspace(0., 10., np.prod(shape), dtype=np.float32)
    >>> data = data.reshape(shape)

    # Build a simple Aidge graph model
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Conv1D(5, 3, [10], name='conv1'),
    ...     aidge_core.ReLU(name='relu')])
    >>> graph.set_backend('cpu')
    >>> graph.set_datatype(aidge_core.dtype.float32)

    # Instantiate a scheduler to forward the data to the graph
    >>> scheduler = aidge_core.SequentialScheduler(graph)
    >>> pred = scheduler.forward(True, data=[aidge_core.Tensor(data)])

    # Change graph data format to NHWC and pass reshaped data to scheduler
    >>> nchw2nhwc4export(graph, format='ncw')

    # C++ Export part, cf. the `export_cpp` function
    >>> export_cpp(graph, scheduler, folder="test_export")
    """

    # Default data format (suppose 4D)
    if dformat in ('ncw', 'nwc'):
        dformat_old = aidge_core.dformat.ncw
        dformat_new = aidge_core.dformat.nwc
    elif dformat in ('nchw', 'nhwc'):
        dformat_old = aidge_core.dformat.nchw
        dformat_new = aidge_core.dformat.nhwc
    elif dformat is None:
        # Check that input are 3D or 4D (if there is an 'H' dimension)
        for node in graph.get_ordered_inputs():
            data = node[0].get_operator().get_input(0)
            if data is not None:
                if len(data.dims) == 3:     # 3D data
                    dformat_old = aidge_core.dformat.ncw
                    dformat_new = aidge_core.dformat.nwc
                else:                       # 4D data
                    dformat_old = aidge_core.dformat.nchw
                    dformat_new = aidge_core.dformat.nhwc
                break
    else:
        print("Error in `nchw2nhwc4export`,",
            f"`{dformat}` is not a valid data format;",
            "please provide a valid data format for conversion")
        return

    # Set the graph's data format to NHWC (None as default)
    graph.set_dataformat(dformat_new)

    # Set the graph's tensors data format to NHWC
    for node in graph.get_ordered_inputs():
        data = node[0].get_operator().get_input(0)
        if data is not None:
            data_cpy = data.clone()
            # Set the data format to NCHW to tell Aidge how to swap axes
            data_cpy.set_data_format(dformat_old)
            # Change the data format to NHWC
            data_cpy.set_data_format(dformat_new)
            # Set the node's data to the correctly formatted tensor
            node[0].get_operator().set_input(0, data_cpy)
#----------------------------------------------------------------------------#

#-------------------------   Export Graph to C++   --------------------------#
def export_cpp(graph, scheduler, folder='', **kw_maincpp):
    """ Export the model graph in a C++ source project

    Take an Aidge graph, its scheduler and the input dimensions, and
    export them all as a C++ language project, as well as a makefile
    to compile the source codes.

    This function is essentially a wrapper of the elementary instructions
    required to operate the export: compilation of the graph, export in
    C++ with a memory consumption profiling alongside and the generation
    of a main file as entry point and the makefile for compilation.

    The export_scheduler function will generate:
      - "dnn/include/forward.hpp" : API functions to use the export;
      - "dnn/include/kernels" : folders for kernels;
      - "dnn/include/layers" : layers configuration;
      - "dnn/include/parameters" : folder with parameters;
      - "dnn/src/forward.cpp" : forward function that calls kernels;
      - "Makefile" to compile the project and the "main.cpp" file;
    The export also operates a memory analysis of the operators; this
    analysis is saved into the "stats" folder.

    Also, a "main.cpp" file is generated using the `generate_main_cpp`
    function from `aidge_core.export_utils` module, which is used as
    entry point to the project.

    Parameters
    ----------
    graph : aidge_core.GraphView object
        The graph to export in C++.
    scheduler : aidge_core.SequentialScheduler object
        The scheduler (for forwarding data) on the model graph.
    [OPT] folder : str
        The folder where to operate the export. It is checked before
        operating export, and created if it does not exist.
            :Default: '' (current folder)

    Other Parameters
    ----------------
    **kw_maincpp : inline keyword arguments
        The optional arguments for the `generate_main_cpp` function:
          - inputs_tensor (Aidge Tensor): external tensors to use in the
                main function. By default, the graph inputs are exported.
          - labels (Aidge Tensor): label tensors to generate and use in
                the main function.

    Returns
    -------
    None : directly operate the export and write the files in memory.

    Examples
    --------
    >>> import numpy as np
    >>> import aidge_core
    >>> import aidge_backend_cpu

    # Generate some test data (required for C++ export)
    >>> shape = (15, 3, 28, 28)             # NCHW (Aidge default in Python)
    >>> data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    # Generate a simple model to export
    >>> graph = aidge_core.sequential([
    ...     aidge_core.Conv2D(3, 3, [5, 5], name='conv'),
    ...     aidge_core.ReLU(name='relu')])

    # Set the model graph implementation & datatype
    >>> graph.set_backend('cpu')
    >>> graph.set_datatype(aidge_core.dtype.float32)

    # Build a scheduler for forwarding data to the graph
    >>> scheduler = aidge_core.SequentialScheduler(graph)
    >>> pred = scheduler.forward(data=[aidge_core.Tensor(data)])

    # Set the graph's data format to NHWC (default for C++ export)
    >>> nchw2nhwc4export(graph)         # Required for 'Conv2D'

    # Perform the export of the Aidge model graph to a C++ project
    >>> export_cpp(graph, scheduler, folder="test_export")
    """

    # Set the backend to `export_cpp` and adapt the graph accordingly
#    aidge_tk.graph_check_backend(graph, aidge_export_cpp.ExportLibCpp._name, True)
    aidge_tk.graph_check_backend(graph, 'export_cpp', True)

    # Check that the folder path is valid and create it if necessary
    folder = check_folder(folder)
#    aidge_core.export_utils.scheduler_export(  # Both work
    aidge_export_cpp.scheduler_export(
        scheduler,
        folder,
        aidge_export_cpp.ExportLibCpp,
        memory_manager=aidge_core.mem_info.generate_optimized_memory_info,
        memory_manager_args={'stats_folder': folder+"stats", 'wrapping': False},
        dev_mode=False)     # Deactivate the developer mode
    # Generate main file
#    aidge_core.export_utils.generate_main_cpp(folder, graph)
    aidge_export_cpp.generate_main_cpp(folder, graph, **kw_maincpp)
#----------------------------------------------------------------------------#

#----------------------   Export Data Arrays in C++   -----------------------#
def _export_array(array, i, folder):
    """ Export an array in C++; convert it as an Aidge Tensor prior if
    it not already in this format """
    # Convert the array in Aidge Tensor if needed
    if not isinstance(array, aidge_core.Tensor):
        array = aidge_core.Tensor(array)
    # Export the array in C++
    aidge_core.export_utils.generate_input_file(
        export_folder=folder, array_name=f"data_{i}", tensor=array)

def export_array_cpp(data, folder="data", sep_fst_dim=False):
    """ Export a (list of) data array(s) in C++

    Take a (list of) data array(s) and convert them into C++ files using
    the `generate_input_file` func. from `aidge_core.export_utils` module.
    The data are stored into `folder`.

    Notice that the `generate_main_cpp` already exports the data array
    set as input to the model graph when creating the `main.cpp` file.

    Parameters
    ----------
    data : (list of) NumPy ndarrays or Aidge Tensors
        The data to export. Can either be a single array or a list of
        arrays. The data should be Aidge Tensors; if not, they are con-
        verted prior export.
    [OPT] folder : str
        The folder where to save the data to; it is recommended to use
        a dedicated folder to store every input data array. It is also
        recommended to store these arrays in a folder `data` inside the
        C++ project folder (where the Aidge model graph is exported).
        The folder is checked and created if it does not exist yet.
            :Default: "data"
    [OPT] sep_fst_dim : bool
        If True, see the ND array as a list of (N-1)D arrays, and export
        them separately; ignore this option if `data` is already a list.
            :Default: False

    Returns
    -------
    None : directly perform the export.

    Examples
    --------
    >>> import numpy as np

    # Generate the data to export
    >>> N = 10                  # Batch size
    >>> shape = (256, 3)        # Input shape
    >>> data = np.arange(N*np.prod(shape), dtype=np.float32).reshape(N, *shape)

    # Perform the export
    >>> export_array_cpp(data, "test_export/data", False)
    """
    # Check that the folder path is valid and create it if necessary
    folder = check_folder(folder)
    # Export a list of data arrays in C++
    if isinstance(data, (list, tuple)) or sep_fst_dim:
        for i, arr in enumerate(data):
            _export_array(arr, i, folder)
    # Export a single data array in C++
    else:
        _export_array(data, 0, folder)
#----------------------------------------------------------------------------#

##############################################################################
