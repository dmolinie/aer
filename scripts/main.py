""" Generate ONNX from test Torch model

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: July 2025

License: GPLv3

# %magic IPython
Tip: You can use `files = !ls *.png`

Passer du mode d'affichage de IPython à Python
%doctest_mode
Tip: Run your doctests from within IPython for development and debugging. The special %doctest_mode command toggles a mode where the prompt, output and exceptions display matches as closely as possible that of the default Python interpreter.

Tip: You can use LaTeX or Unicode completion, `\alpha<tab>` will insert the α symbol.
Tip: You can use `%hist` to view history, see the options with `%history?`
Tip: Use `F2` or %edit with no arguments to open an empty editor with a temporary file.
Tip: The `%timeit` magic has a `-o` flag, which returns the results, making it easy to plot. See `%timeit?`.
Tip: Use the IPython.lib.demo.Demo class to load any Python script as an interactive demo.

#print(x'{var}') --> x in {b, f, r, u}

eIQ®Toolkit
https://www.nxp.com/design/design-center/training/TIP-ML-AND-AI-SERIES-EIQ-TOOLKIT

Non-breaking space: ' '

#YOLOv8
"""

import time
import psutil

import numpy as np
import torch

import aidge_core
import aidge_onnx
import aidge_backend_cpu
from aidge_export_cpp.export_utils import set_nodes_names

import aer.tools as tls
from aer.datasets.mivia_loader import build_dataset, LIMITS
import aer.onnx as oxtk
import aer.aidge as aidge_tk

tls.exec_file('.pystartup.py', globals(), root='~/')
aidge_core.Log.set_console_level(aidge_core.aidge_core.Level(4)) # ERROR & FATAL

# Data folders
ROOT = "data/MIVIA_AE/"
XMLFOLDER_TRN = ROOT + "training/"          # Events XML files folder
SNDFOLDER_TRN = ROOT + "training/sounds/"   # Audio WAV files folder
NB_CLASSES = 4

# Benchmark ID (for the model path)
BENCH = 2

# Signal-to-Noise Ratio of the model & dataset (for the path & model name)
SNR = 20

# Data type (prefer using chunks)
SEQUENCES = False

# Set the data format extension
DTYPE = 'Sequences' if SEQUENCES else 'Chunks'
dtype = 'seqs' if SEQUENCES else 'chks'

# Model name & folder

# Bench_0:
#   Models/Bench_0/Models/20/Chunks/SincNet_NoRec/
#       SincNet_NoRec_chks_20.pth
#BASE_NAME = "SincNet_NoRec"

# Bench_1:
#   Models/Bench_1/Models/20/Chunks/ConvNet_NoRec_2CV_2FC/
#       ConvNet_NoRec_2CV_2FC_chks_20.pth
#BASE_NAME = "ConvNet_NoRec_2CV_3FC"

# Bench_2:
#   Models/Bench_2/Models/20/Chunks/ConvNet_NoRec_20Filters_1024Neurons/
#      ConvNet_NoRec_20Filters_1024Neurons_chks_20.pth'
BASE_NAME = "ConvNet_NoRec_20Filters_1024Neurons"

MODEL_NAME = BASE_NAME + f"_{dtype}_{SNR}"
FOLDER_MODEL = f"Models/Bench_{BENCH}/Models/{SNR}/{DTYPE}/{BASE_NAME}/"

print(MODEL_NAME)


#------------------------   Load Calibration Data   -------------------------#
# Set the frame & hop durations
FRM_DURATION = 100e-3                       # Frame duration in seconds
HOP_DURATION = 50e-3                        # Hop duration in seconds
chk_params = (FRM_DURATION, HOP_DURATION)   # Chunks settings

# Data rescaling bounds
BOUNDS = (-1., +1.)

# Set the frame & hop durations
specs, data, classes = build_dataset(
    (XMLFOLDER_TRN, SNDFOLDER_TRN), (66, SNR),
    NB_CLASSES, chk_params,
        limits=LIMITS, bounds=BOUNDS)
input_shape = (1, data.shape[1])            # Data format: NCW

## Extract only some chunks to shorten testing
#data = data[:100]

# Cast the data & classes to float32
data = data.astype(np.float32)
#classes = classes.astype(np.float32)

# Set a Channel of 1 to the data
data = data.reshape(len(data), 1, -1)
tensors = torch.Tensor(data)
#classes = classes.reshape(len(classes), 1, -1)

# Extract only the first batch (so that the ONNX graph will use only 1 batch)
# Use slice [:1] to keep the dimensions
data_calib = data[:1]       # Numpy array
tensor_calib = tensors[:1]  # Torch tensor
#class_calib = classes[:1]   # Numpy array
#----------------------------------------------------------------------------#


##############################################################################
##                         Torch Model ONNX Export                          ##
##############################################################################

import torch

import aer.tools as tls
import aer.onnx as oxtk

def compare_res(session, model, data):
    """ Compare the accuracy of a Torch model and its ONNX counterpart """
    with torch.no_grad():
        # Forward data to the ONNX graph
        pred_onx = oxtk.run_onnx(session, data.detach().numpy())[0]
        # Forward data to the Torch model
        pred_pth = model.forward(data).detach().numpy()
        # Compare the ONNX & Torch models predictions
        print("Torch and ONNX close:", np.allclose(pred_onx, pred_pth, atol=1e-1))
    return pred_pth-pred_onx


# Load the Torch model & export it to ONNX
model = torch.load(FOLDER_MODEL+MODEL_NAME+'.pth', weights_only=False)

# Beware 'dynamic_batch', as it adds additional layers to the ONNX export
print(f"Exporting model `{MODEL_NAME}` to ONNX graph in folder `{FOLDER_MODEL}`")
model_onnx = oxtk.torch2onnx(
    FOLDER_MODEL+MODEL_NAME+'.onnx', model, tensor_calib,
    dynamic_batch=False, dynamo=True, verbose=False)

# Compare Torch & ONNX models accuracy
# (ONNX must be reloaded as it is wrapped into an onnxruntime session object)
print("Checking accuracy of the model (Torch vs ONNX")
error = compare_res(
    oxtk.load_onnx(FOLDER_MODEL+MODEL_NAME+'.onnx'), model, tensor_calib)

# List and save the model operators (ONNX version)
ops = [node.op_type for node in model_onnx.model.graph.all_nodes()]
tls.save_as_csv(FOLDER_MODEL+MODEL_NAME+'.csv', ops)

##############################################################################



##############################################################################
##                            Aidge C++ Export                            ##
##############################################################################

import aidge_core
import aidge_onnx
import aidge_backend_cpu
from aidge_export_cpp.export_utils import set_nodes_names

import aer.aidge as aidge_tk

aidge_core.Log.set_console_level(aidge_core.aidge_core.Level(4)) # ERROR & FATAL


# Load the ONNX model in Aidge
graph = aidge_tk.load_onnx(FOLDER_MODEL+MODEL_NAME+'.onnx', False)

# Set the graph backend (~implementation)
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend('cpu')

# Create scheduler
scheduler = aidge_core.SequentialScheduler(graph)

# Run inference as the scheduler should be empty for export
pred_aidge = aidge_tk.forward(graph, scheduler, data_calib)

# Simplify the MatMul-Add nodes as 'FC' (only for export)
aidge_tk.improve_graph_fc(graph, True, True)

# Improve the graph for C++ export
aidge_tk.improve_graph_cpp(graph, True, True)

# Set nodes names (optional)
set_nodes_names(scheduler)

# Graph data format from NCHW (Aidge) to NHWC (Export C++)
aidge_tk.nchw2nhwc4export(graph, 'nchw')

# Adapt the graph backend (also done in 'aidge_tk.export_cpp')
aidge_tk.graph_check_backend(graph, 'export_cpp', True)

# Reset graph input dimensions to comply with NHWC data format
aidge_tk.reset_graph_dims(graph)

# Reset the scheduler to take into account new data format
scheduler.reset_scheduling()
scheduler.generate_scheduling()

folder_exp = FOLDER_MODEL + "CPP/"
# Export the model into C++ source codes
# N.B.: if the model was used for prediction ('forward'),
# the data array used is also exported.
print(f"Exporting model '{MODEL_NAME}' to C++ sources in folder '{folder_exp}'")
aidge_tk.export_cpp(graph, scheduler, folder_exp)

## Transcribe the data of a numpy array into a C++ file
## For C++ export, data should be NHWC
#aidge_tk.export_array_cpp(data_calib, folder_exp+"data")
#print()

vals = np.array(pred_aidge)

##############################################################################



##############################################################################
##                          Frameworks Comparison                           ##
##############################################################################

## This part requires the Torch model to be exported into ONNX graph prior


#def ram_usage(ram_aft, ram_bfr):
#    """ Difference of active RAM between two measures """
#    return (ram_aft.active - ram_bfr.active)

#batch_shape = (1, *data.shape[1:])
#memory = np.zeros(3, dtype=float)
#times = np.zeros((3, len(tensors)), dtype=float)

## Load the Torch model and get its memory footprint
#ram_bfr = psutil.virtual_memory()
#model = torch.load(FOLDER_MODEL+MODEL_NAME+'.pth', weights_only=False)
#ram_aft = psutil.virtual_memory()
#memory[0] = ram_usage(ram_aft, ram_bfr)
## Pass every data chunk to the model and retrieve the inference time
#pred_pth = []
#with torch.no_grad():
#    for i, tensor in enumerate(tensors):
#        time_bfr = time.time()
#        pred_pth.append(model.forward(tensor.unsqueeze(0)))
#        time_aft = time.time()
#        times[0, i] = time_aft - time_bfr

## Load the ONNX model graph (onnxruntime) and get its memory footprint
#ram_bfr = psutil.virtual_memory()
#session = oxtk.load_onnx(FOLDER_MODEL+MODEL_NAME+'.onnx')
#ram_aft = psutil.virtual_memory()
#memory[1] = ram_usage(ram_aft, ram_bfr)
## Pass every data chunk to the model and retrieve the inference time
#pred_onx = []
#for i, arr in enumerate(data):
#    time_bfr = time.time()
#    pred_onx.append(oxtk.run_onnx(session, arr.reshape(batch_shape))[0])
#    time_aft = time.time()
#    times[1, i] = time_aft - time_bfr

## Load the ONNX model graph in Aidge and get its memory footprint
#ram_bfr = psutil.virtual_memory()
#graph = aidge_tk.load_onnx(FOLDER_MODEL+MODEL_NAME+'.onnx', False)
##ops = aidge_onnx.native_coverage_report(graph)     # List Aidge operators
## Set the graph backend (~implementation)
#graph.set_datatype(aidge_core.dtype.float32)
#graph.set_backend('cpu')
## Create scheduler
#scheduler = aidge_core.SequentialScheduler(graph)
#ram_aft = psutil.virtual_memory()
#memory[2] = ram_usage(ram_aft, ram_bfr)
## Pass every data chunk to the model and retrieve the inference time
#pred_adg = []
#for i, arr in enumerate(data):
#    time_bfr = time.time()
#    scheduler.forward(data=[aidge_core.Tensor(arr.reshape(batch_shape))])
#    outNode = list(graph.get_output_nodes())[0]
#    pred_adg.append(np.array(outNode.get_operator().get_output(0)))
#    time_aft = time.time()
#    times[2, i] = time_aft - time_bfr


#print("Predictions comparison between frameworks:")
#print("Torch & Onnxruntime close:\t", np.allclose(pred_pth, pred_onx, atol=5e-2))
#print("Torch & Aidge close:\t\t", np.allclose(pred_pth, pred_adg, atol=5e-2))
#print("Onnxruntime & Aidge close:\t", np.allclose(pred_onx, pred_adg, atol=5e-2))

#times *= 1e3    # Times in ms
#print("Mean execution times per chunk:")
#for fwork, tps in zip(
#    ("Torch\t\t", "Onnxruntime\t", "Aidge (Python)\t"),
#    times.mean(1).round(3)):
#    print(fwork, tps, "ms")

#memory *= 1e-6  # RAM in Mo
#print("Model RAM consumption by framework:")
#for fwork, mem in zip(
#    ("Torch\t\t", "Onnxruntime\t", "Aidge (Python)\t"),
#    memory.round(3)):
#    print(fwork, mem, "Mo")

##############################################################################
