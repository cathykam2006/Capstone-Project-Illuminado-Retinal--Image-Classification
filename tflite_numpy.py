"""
A tiny numpy runner for the two Iluminado TensorFlow Lite models.

TensorFlow Lite can't run inside a web browser, but numpy can (Pyodide), so the
browser version of the app runs the exported graph (export_numpy_models.py) here.
It implements only the operators these float32 EfficientNet-B0 models use, with
TensorFlow Lite's rules for padding and broadcasting. The results were checked
against TensorFlow Lite on the sample images (see check_numpy_models.py).
"""
from __future__ import annotations

import json

import numpy as np


def _act(x, act):
    if act == "NONE":
        return x
    if act == "RELU":
        return np.maximum(x, 0)
    if act == "RELU6":
        return np.clip(x, 0, 6)
    raise NotImplementedError(f"activation {act}")


def _pad_amounts(size, k, stride, dilation, padding):
    """TensorFlow 'SAME' / 'VALID' padding along one axis: (before, after, out_size)."""
    eff = (k - 1) * dilation + 1
    if padding == "VALID":
        return 0, 0, (size - eff) // stride + 1
    out = -(-size // stride)
    total = max((out - 1) * stride + eff - size, 0)
    return total // 2, total - total // 2, out


def _windows(x, kh, kw, opt):
    """Padded input and a function giving the strided slice for each kernel tap."""
    sh, sw = opt["stride"]
    dh, dw = opt["dilation"]
    _, h, w, _ = x.shape
    t, b, oh = _pad_amounts(h, kh, sh, dh, opt["padding"])
    l, r, ow = _pad_amounts(w, kw, sw, dw, opt["padding"])
    if t or b or l or r:
        x = np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)))

    def tap(i, j):
        return x[:, i * dh: i * dh + (oh - 1) * sh + 1: sh, j * dw: j * dw + (ow - 1) * sw + 1: sw, :]
    return tap, oh, ow


def conv2d(x, w, bias, opt):
    """w: (out, kh, kw, in), NHWC input."""
    out_c, kh, kw, in_c = w.shape
    if kh == 1 and kw == 1 and opt["stride"] == [1, 1]:
        y = x @ w.reshape(out_c, in_c).T
    else:
        tap, oh, ow = _windows(x, kh, kw, opt)
        cols = np.concatenate([tap(i, j) for i in range(kh) for j in range(kw)], axis=-1)
        y = cols @ w.transpose(1, 2, 3, 0).reshape(kh * kw * in_c, out_c)
    if bias is not None:
        y = y + bias
    return _act(y.astype(np.float32, copy=False), opt["act"])


def depthwise_conv2d(x, w, bias, opt):
    """w: (1, kh, kw, channels); depth multiplier 1."""
    if opt.get("depth_multiplier", 1) not in (0, 1):
        raise NotImplementedError("depth multiplier > 1")
    _, kh, kw, c = w.shape
    tap, oh, ow = _windows(x, kh, kw, opt)
    y = np.zeros((x.shape[0], oh, ow, c), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            y += tap(i, j) * w[0, i, j]
    if bias is not None:
        y += bias
    return _act(y, opt["act"])


def strided_slice(x, begin, end, strides, opt):
    x = np.asarray(x)
    idx = []
    for d in range(len(begin)):
        if opt["shrink_axis_mask"] >> d & 1:
            idx.append(int(begin[d]))
            continue
        b = None if opt["begin_mask"] >> d & 1 else int(begin[d])
        e = None if opt["end_mask"] >> d & 1 else int(end[d])
        idx.append(slice(b, e, int(strides[d])))
    return np.asarray(x[tuple(idx)])


class Model:
    """Load models_numpy/<task>.json + .npz and run it on one or more images."""

    def __init__(self, graph_path, weights_path):
        with open(graph_path) as f:
            self.graph = json.load(f)
        w = np.load(weights_path)
        self.consts = {int(k[1:]): w[k] for k in w.files}

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        v = dict(self.consts)
        v[self.graph["inputs"][0]] = np.asarray(image, dtype=np.float32)

        def get(i):
            return None if i < 0 else v[i]

        for op in self.graph["ops"]:
            kind, ins, opt = op["op"], [get(i) for i in op["inputs"]], op["options"]
            if kind == "CONV_2D":
                out = conv2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None, opt)
            elif kind == "DEPTHWISE_CONV_2D":
                out = depthwise_conv2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None, opt)
            elif kind == "ADD":
                out = _act(ins[0] + ins[1], opt["act"])
            elif kind == "SUB":
                out = _act(ins[0] - ins[1], opt["act"])
            elif kind == "MUL":
                out = _act(ins[0] * ins[1], opt["act"])
            elif kind == "LOGISTIC":
                x = ins[0]
                out = (1.0 / (1.0 + np.exp(-x))).astype(x.dtype, copy=False)
            elif kind == "MEAN":
                out = ins[0].mean(axis=tuple(np.atleast_1d(ins[1]).tolist()), keepdims=opt["keep_dims"])
                out = out.astype(ins[0].dtype, copy=False)
            elif kind == "REDUCE_PROD":
                out = np.prod(ins[0], axis=tuple(np.atleast_1d(ins[1]).tolist()), keepdims=opt["keep_dims"])
                out = np.asarray(out, dtype=ins[0].dtype)
            elif kind == "FULLY_CONNECTED":
                x, w = ins[0], ins[1]
                y = x.reshape(-1, w.shape[1]) @ w.T
                if len(ins) > 2 and ins[2] is not None:
                    y = y + ins[2]
                if opt["keep_num_dims"]:
                    y = y.reshape(*x.shape[:-1], w.shape[0])
                out = _act(y.astype(np.float32, copy=False), opt["act"])
            elif kind == "SHAPE":
                out = np.array(ins[0].shape, dtype=np.int32)
            elif kind == "STRIDED_SLICE":
                out = strided_slice(ins[0], ins[1], ins[2], ins[3], opt)
            elif kind == "PACK":
                out = np.stack([np.asarray(a) for a in ins], axis=opt["axis"])
            elif kind == "RESHAPE":
                shape = ins[1] if len(ins) > 1 and ins[1] is not None else None
                if shape is None:
                    raise NotImplementedError("RESHAPE without a shape tensor")
                out = ins[0].reshape([int(s) for s in np.atleast_1d(shape)])
            elif kind == "GATHER":
                out = np.take(np.asarray(ins[0]), np.asarray(ins[1]), axis=opt["axis"])
            elif kind == "CONCATENATION":
                out = _act(np.concatenate([np.atleast_1d(a) for a in ins], axis=opt["axis"]), opt["act"])
            else:
                raise NotImplementedError(f"operator {kind}")
            v[op["outputs"][0]] = out
        return {self.graph["output_names"][str(i)]: v[i] for i in self.graph["outputs"]}
