#!/usr/bin/env python3
"""Extract Favoni et al. result-pickle tensors without requiring PyTorch.

The published pickle embeds legacy FloatStorage byte streams.  This loader
only reconstructs those arrays; it does not execute model code.
"""

import argparse
import pickle
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_lge_checkpoint import export_lge_checkpoint


def _load_storage(payload):
    return payload


def _rebuild_tensor(storage, offset, size, stride, requires_grad, hooks):
    del requires_grad, hooks
    count = int(np.prod(size))
    values = np.frombuffer(storage[-4 * (offset + count):], dtype="<f4")
    values = values[offset:offset + count].copy()
    # Published tensors are contiguous. Refuse a surprising layout rather
    # than silently importing the wrong coefficient ordering.
    expected_stride = []
    running = 1
    for extent in reversed(size):
        expected_stride.append(running)
        running *= extent
    if tuple(reversed(expected_stride)) != tuple(stride):
        raise ValueError("non-contiguous checkpoint tensor is unsupported")
    return values.reshape(size)


class _LightningStub:
    _arg_default = None


class _TorchFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _load_storage
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module.startswith("pytorch_lightning"):
            return _LightningStub
        if module == "torch" and name.endswith("Storage"):
            return _LightningStub
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle")
    parser.add_argument("output")
    parser.add_argument("--model", type=int, default=0)
    args = parser.parse_args()

    with open(args.pickle, "rb") as stream:
        results = _TorchFreeUnpickler(stream).load()
    entry = results[args.model]
    state = entry["state_dict"]
    hparams = entry["hparams"]
    export_lge_checkpoint(
        state,
        args.output,
        convention="favoni_prl",
        dimension=len(hparams.dims),
        colors=hparams.nc,
        conv_channels=list(hparams.conv_channels),
        kernel_size=list(hparams.conv_kernel_size),
        dilation=list(hparams.conv_dilation),
        symmetric=bool(getattr(hparams, "symmetric", False)),
        global_average=bool(hparams.global_average),
        linear_sizes=list(hparams.linear_sizes),
        source_commit="prl_2022",
        test_mse=entry["test_mse"],
    )
    print("LCB shape:", state["convs.0.weight"].shape)
    print("stored test MSE:", float(entry["test_mse"]))


if __name__ == "__main__":
    main()
