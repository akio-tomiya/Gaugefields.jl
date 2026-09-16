#!/usr/bin/env python3
"""Export an LGE-CNN PyTorch state_dict to Gaugefields' portable NPZ schema.

The recommended use is from the Python process that already owns the model:

    export_lge_checkpoint(
        model.state_dict(), "model.npz", convention="favoni_prl",
        dimension=2, colors=2, conv_channels=[2], kernel_size=[2],
        dilation=[1], symmetric=False, global_average=False,
        linear_sizes=[],
    )

Strings are stored as UTF-8 UInt8 arrays because NPZ.jl deliberately supports
numeric NPY element types only.
"""

import argparse
import re

import numpy as np


FORMAT_VERSION = 1


def _bytes(value):
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _numbered_keys(state_dict, pattern):
    matches = []
    regex = re.compile(pattern)
    for key in state_dict:
        match = regex.fullmatch(key)
        if match:
            matches.append((int(match.group(1)), key))
    return [key for _, key in sorted(matches)]


def export_lge_checkpoint(
    state_dict,
    output,
    *,
    convention,
    dimension,
    colors,
    conv_channels,
    kernel_size,
    dilation,
    symmetric=False,
    use_unit_elements=True,
    global_average=False,
    linear_sizes=(),
    source_commit="unknown",
    test_mse=None,
):
    if convention not in ("favoni_arxiv", "favoni_prl"):
        raise ValueError("convention must be favoni_arxiv or favoni_prl")
    conv_keys = _numbered_keys(state_dict, r"convs\.(\d+)\.weight")
    if not conv_keys:
        raise KeyError("state_dict contains no convs.<index>.weight tensors")
    if len(conv_keys) != len(conv_channels):
        raise ValueError("conv_channels does not match the state_dict layer count")
    if len(kernel_size) != len(conv_keys) or len(dilation) != len(conv_keys):
        raise ValueError("kernel_size and dilation must contain one value per layer")

    linear_weight_keys = _numbered_keys(state_dict, r"linears\.(\d+)\.weight")
    linear_bias_keys = _numbered_keys(state_dict, r"linears\.(\d+)\.bias")
    if len(linear_weight_keys) != 1 or len(linear_bias_keys) != 1:
        raise ValueError(
            "Gaugefields' current LCNN readout supports exactly one PyTorch "
            "Linear layer; hidden linear layers must be implemented first"
        )

    arrays = {
        "meta.format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
        "meta.source": _bytes("openpixi/lge-cnn"),
        "meta.source_commit": _bytes(source_commit),
        "meta.convention": _bytes(convention),
        "meta.dimension": np.asarray(dimension, dtype=np.int64),
        "meta.colors": np.asarray(colors, dtype=np.int64),
        "meta.layer_count": np.asarray(len(conv_keys), dtype=np.int64),
        "meta.conv_channels": np.asarray(conv_channels, dtype=np.int64),
        "meta.kernel_size": np.asarray(kernel_size, dtype=np.int64),
        "meta.dilation": np.asarray(dilation, dtype=np.int64),
        "meta.symmetric": np.asarray(symmetric, dtype=np.bool_),
        "meta.use_unit_elements": np.asarray(use_unit_elements, dtype=np.bool_),
        "meta.global_average": np.asarray(global_average, dtype=np.bool_),
        "meta.complex_component_order": _bytes("channel_re_im"),
    }
    if linear_sizes:
        arrays["meta.linear_sizes"] = np.asarray(linear_sizes, dtype=np.int64)
    for index, key in enumerate(conv_keys, start=1):
        arrays[f"layers.{index}.weight"] = _array(state_dict[key])
    arrays["readout.weight"] = _array(state_dict[linear_weight_keys[0]]).reshape(-1)
    arrays["readout.bias"] = _array(state_dict[linear_bias_keys[0]]).reshape(-1)
    if test_mse is not None:
        arrays["meta.test_mse"] = np.asarray(test_mse)
    np.savez(output, **arrays)


def main():
    parser = argparse.ArgumentParser(
        description="Export a trusted PyTorch checkpoint/state_dict to LCNN NPZ",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument(
        "--convention", required=True,
        choices=("favoni_arxiv", "favoni_prl"),
    )
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--colors", required=True, type=int)
    parser.add_argument("--conv-channels", required=True, type=int, nargs="+")
    parser.add_argument("--kernel-size", required=True, type=int, nargs="+")
    parser.add_argument("--dilation", required=True, type=int, nargs="+")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--global-average", action="store_true")
    parser.add_argument("--linear-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()

    import torch

    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
    export_lge_checkpoint(
        state_dict,
        args.output,
        convention=args.convention,
        dimension=args.dimension,
        colors=args.colors,
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        symmetric=args.symmetric,
        global_average=args.global_average,
        linear_sizes=args.linear_sizes,
        source_commit=args.source_commit,
    )


if __name__ == "__main__":
    main()
