"""
Turns each .tflite model into files that tflite_numpy.py can run with numpy alone:

  models_numpy/<task>.json   the graph: operators, their inputs/outputs and options
  models_numpy/<task>.npz    every constant tensor (weights, biases, shape constants)

Needed because TensorFlow Lite can't run in a web browser, but numpy can (Pyodide).

    pip install tflite numpy
    python export_numpy_models.py /path/to/main/checkout
"""
import json
import os
import sys

import numpy as np
import tflite

SRC = sys.argv[1] if len(sys.argv) > 1 else "../Capstone-Project-Illuminado-Retinal--Image-Classification"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_numpy")
os.makedirs(OUT, exist_ok=True)

OP_NAMES = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}
ACT = {v: k for k, v in tflite.ActivationFunctionType.__dict__.items() if not k.startswith("_")}
PAD = {v: k for k, v in tflite.Padding.__dict__.items() if not k.startswith("_")}
NP_TYPE = {tflite.TensorType.FLOAT32: np.float32, tflite.TensorType.INT32: np.int32,
           tflite.TensorType.INT64: np.int64}


def options(op, kind):
    """The few options each operator needs, as plain JSON."""
    o = op.BuiltinOptions()
    if kind == "CONV_2D":
        x = tflite.Conv2DOptions(); x.Init(o.Bytes, o.Pos)
        return dict(padding=PAD[x.Padding()], stride=[x.StrideH(), x.StrideW()],
                    dilation=[x.DilationHFactor(), x.DilationWFactor()], act=ACT[x.FusedActivationFunction()])
    if kind == "DEPTHWISE_CONV_2D":
        x = tflite.DepthwiseConv2DOptions(); x.Init(o.Bytes, o.Pos)
        return dict(padding=PAD[x.Padding()], stride=[x.StrideH(), x.StrideW()],
                    dilation=[x.DilationHFactor(), x.DilationWFactor()], act=ACT[x.FusedActivationFunction()],
                    depth_multiplier=x.DepthMultiplier())
    if kind in ("ADD", "MUL", "SUB"):
        cls = {"ADD": tflite.AddOptions, "MUL": tflite.MulOptions, "SUB": tflite.SubOptions}[kind]
        if o is None:
            return dict(act="NONE")
        x = cls(); x.Init(o.Bytes, o.Pos)
        return dict(act=ACT[x.FusedActivationFunction()])
    if kind in ("MEAN", "REDUCE_PROD"):
        x = tflite.ReducerOptions(); x.Init(o.Bytes, o.Pos)
        return dict(keep_dims=bool(x.KeepDims()))
    if kind == "FULLY_CONNECTED":
        x = tflite.FullyConnectedOptions(); x.Init(o.Bytes, o.Pos)
        return dict(act=ACT[x.FusedActivationFunction()], keep_num_dims=bool(x.KeepNumDims()))
    if kind == "STRIDED_SLICE":
        x = tflite.StridedSliceOptions(); x.Init(o.Bytes, o.Pos)
        for flag in ("EllipsisMask", "NewAxisMask"):
            if getattr(x, flag)():
                sys.exit(f"STRIDED_SLICE with {flag} is not supported")
        return dict(begin_mask=x.BeginMask(), end_mask=x.EndMask(), shrink_axis_mask=x.ShrinkAxisMask())
    if kind == "PACK":
        x = tflite.PackOptions(); x.Init(o.Bytes, o.Pos)
        return dict(axis=x.Axis())
    if kind == "CONCATENATION":
        x = tflite.ConcatenationOptions(); x.Init(o.Bytes, o.Pos)
        return dict(axis=x.Axis(), act=ACT[x.FusedActivationFunction()])
    if kind == "GATHER":
        x = tflite.GatherOptions(); x.Init(o.Bytes, o.Pos)
        return dict(axis=x.Axis(), batch_dims=x.BatchDims())
    return {}


def export(task, tflite_path):
    buf = open(tflite_path, "rb").read()
    model = tflite.Model.GetRootAsModel(buf, 0)
    codes = []
    for i in range(model.OperatorCodesLength()):
        c = model.OperatorCodes(i)
        codes.append(OP_NAMES[max(c.BuiltinCode(), c.DeprecatedBuiltinCode())])
    g = model.Subgraphs(0)

    consts, tensors = {}, {}
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        tensors[i] = dict(name=t.Name().decode(), shape=t.ShapeAsNumpy().tolist() if t.ShapeLength() else [])
        data = model.Buffers(t.Buffer()).DataAsNumpy()
        if not isinstance(data, int) and data.size:
            consts[f"t{i}"] = np.frombuffer(data.tobytes(), dtype=NP_TYPE[t.Type()]).reshape(tensors[i]["shape"])

    ops = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        kind = codes[op.OpcodeIndex()]
        ops.append(dict(op=kind, inputs=op.InputsAsNumpy().tolist(), outputs=op.OutputsAsNumpy().tolist(),
                        options=options(op, kind)))

    graph = dict(inputs=g.InputsAsNumpy().tolist(), outputs=g.OutputsAsNumpy().tolist(),
                 output_names={str(i): tensors[i]["name"] for i in g.OutputsAsNumpy().tolist()},
                 ops=ops)
    with open(os.path.join(OUT, f"{task}.json"), "w") as f:
        json.dump(graph, f, separators=(",", ":"))
    np.savez_compressed(os.path.join(OUT, f"{task}.npz"), **consts)
    size = os.path.getsize(os.path.join(OUT, f"{task}.npz")) / 1e6
    print(f"{task}: {len(ops)} ops, {len(consts)} constants, {size:.1f} MB")


for task in ("binary", "multilabel"):
    cfg = json.load(open(os.path.join(SRC, "models", f"{task}_ensemble.json")))
    export(task, os.path.join(SRC, "models", cfg["tflite"]))
