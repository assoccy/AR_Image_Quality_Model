#!/usr/bin/python3
# coding=utf-8
import onnx

# Load the ONNX model
onnx_model_path = 'mobilenetv3.onnx'

model = onnx.load(onnx_model_path)

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
