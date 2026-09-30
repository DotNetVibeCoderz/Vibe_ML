"""Tiny ONNX graphs for the GraviOptimum tests. Run from this folder: python make_fixtures.py

half.onnx    y = x * w + b, all float16, with w and b stored as float16 initializers.
random.onnx  y = x + RandomNormalLike(x), with the node's scale left at its default of 1 -
             the shape of a Stable Diffusion VAE encoder's in-graph sampling.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

w = np.array([0.5, -1.25, 3.0], dtype=np.float16)
b = np.array([0.1, 0.2, 0.3], dtype=np.float16)
half = helper.make_graph(
    [helper.make_node("Mul", ["x", "w"], ["xw"], name="/scale/Mul"),
     helper.make_node("Add", ["xw", "b"], ["y"], name="/shift/Add")],
    "half",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["n", 3])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["n", 3])],
    [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
onnx.save(helper.make_model(half, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), "half.onnx")

rand = helper.make_graph(
    [helper.make_node("RandomNormalLike", ["x"], ["noise"], name="/sample/RandomNormalLike"),
     helper.make_node("Add", ["x", "noise"], ["y"], name="/sample/Add")],
    "random",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])])
onnx.save(helper.make_model(rand, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), "random.onnx")

import onnxruntime as ort
x = np.array([[1, 2, 3], [-4, 0.5, 8]], dtype=np.float16)
print("half:", ort.InferenceSession("half.onnx").run(None, {"x": x})[0].astype(np.float64).tolist())
