#!/usr/bin/env python3
"""Run the released PyTorch layer class directly on a Gaugefields fixture."""

import argparse
import importlib.util

import numpy as np
import torch


def _load_layers(path):
    spec = importlib.util.spec_from_file_location("official_lge_cnn_layers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_upstream_tensor(values, dtype):
    # Gaugefields fixture: (channel, row, column, x, y, ...)
    # Upstream: (batch, flattened_site, channel, row, column, re/im)
    dimension = values.ndim - 3
    axes = (*range(3, 3 + dimension), 0, 1, 2)
    lattice_first = values.transpose(axes)
    flattened = lattice_first.reshape((-1, *values.shape[:3]))
    split = np.stack((flattened.real, flattened.imag), axis=-1)
    return torch.as_tensor(split, dtype=dtype).unsqueeze(0)


def _from_upstream_tensor(values, lattice_shape):
    array = values.detach().cpu().numpy()[0]
    complex_array = array[..., 0] + 1j * array[..., 1]
    lattice_first = complex_array.reshape((*lattice_shape, *complex_array.shape[1:]))
    dimension = len(lattice_shape)
    axes = (dimension, dimension + 1, dimension + 2, *range(dimension))
    return lattice_first.transpose(axes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", help="official lge_cnn/nn/layers.py")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--convention", choices=("favoni_arxiv", "favoni_prl"), required=True
    )
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--dilation", type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    layers = _load_layers(args.layers)
    data = np.load(args.input)
    lattice_shape = tuple(data["features"].shape[3:])
    dimension = len(lattice_shape)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    dilation = args.dilation
    if dilation is None:
        dilation = 0 if args.convention == "favoni_arxiv" else 1

    links = _to_upstream_tensor(data["links"], dtype)
    features = _to_upstream_tensor(data["features"], dtype)
    packed = torch.cat((links, features), dim=2)
    common = dict(
        dims=list(lattice_shape),
        kernel_size=args.kernel_size,
        dilation=dilation,
        n_in=features.shape[2],
        n_out=data["lcb_weight"].shape[0],
        nc=features.shape[3],
        init_w=1.0,
        use_unit_elements=True,
    )
    if args.convention == "favoni_arxiv":
        layer = layers.GCMConv(**common)
        trace_layer = layers.GCTrace(list(lattice_shape))
    else:
        layer = layers.LConvBilin(**common, use_symmetric=False)
        trace_layer = layers.LTrace(list(lattice_shape))
    layer = layer.to(dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(data["lcb_weight"], dtype=dtype))
        packed_output = layer(packed)
        traced = trace_layer(packed_output)
        flattened = traced.reshape((-1, traced.shape[2] * traced.shape[3]))
        readout_weight = torch.as_tensor(data["readout_weight"], dtype=dtype)
        readout_bias = torch.as_tensor(data["readout_bias"], dtype=dtype)
        local_prediction = flattened @ readout_weight + readout_bias[0]
        scalar = local_prediction.mean()

    _, output_features = layers.unpack_x(packed_output, dimension)
    np.savez(
        args.output,
        features=_from_upstream_tensor(output_features, lattice_shape),
        scalar=np.asarray(scalar.detach().cpu().item()),
    )


if __name__ == "__main__":
    main()
