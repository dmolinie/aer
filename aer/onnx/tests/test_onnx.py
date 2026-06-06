from aer.onnx._onnx import *


def test_torch2onnx():
    """ Convert and export a Torch model into an ONNX graph """
    import torch
    batch = 10      # Batch size
    c_inp = 10      # Input channels
    c_out = 5       # Output channels
    # Generate dummy data
    inputs = torch.arange(batch*c_inp, dtype=torch.float32)
    inputs = inputs.reshape(batch, c_inp)
    # Define a dummy model
    model = torch.nn.Sequential(torch.nn.Linear(c_inp, c_out))
    # Convert the model into an ONNX graph, and save it
    model_nx = torch2onnx("Models/model.onnx", model, inputs, dynamo=True)

def test_load_onnx():
    """ Load an ONNX graph model into a Session object (for inference) """
    onnx_session = load_onnx("Models/model.onnx")

def test_run_onnx():
    """ Use an ONNX graph model for inference """
    import numpy as np
    # Load the ONNX model and use it for inference
    onnx_session = load_onnx("Models/model.onnx")
    # Retrieve the input data shape and generate data accordingly
    batch = 10
    input_shape = onnx_session.get_inputs()[0].shape[1:]
    input_data = np.arange(batch*np.prod(input_shape), dtype=np.float32)
    input_data = input_data.reshape(batch, *input_shape)
    # Run inference on the input data
    pred = run_onnx(onnx_session, input_data)



# Launch test/example functions
test_torch2onnx()

test_load_onnx()

test_run_onnx()

