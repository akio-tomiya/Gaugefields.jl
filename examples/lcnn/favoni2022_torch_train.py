#!/usr/bin/env python3
"""Tiny official-layer SGD run used for cross-framework training parity."""

import argparse

import numpy as np
import torch

from favoni2022_torch_reference import _load_layers, _to_upstream_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layers")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--convention", choices=("favoni_arxiv", "favoni_prl"), required=True
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()

    official = _load_layers(args.layers)
    data = np.load(args.input)
    lattice_shape = tuple(data["features"].shape[3:])
    dtype = torch.float64
    links = _to_upstream_tensor(data["links"], dtype)
    features = _to_upstream_tensor(data["features"], dtype)
    packed = torch.cat((links, features), dim=2)
    common = dict(
        dims=list(lattice_shape), kernel_size=2,
        dilation=0 if args.convention == "favoni_arxiv" else 1,
        n_in=1, n_out=2, nc=2, init_w=1.0, use_unit_elements=True,
    )
    if args.convention == "favoni_arxiv":
        layer = official.GCMConv(**common).to(dtype=dtype)
        trace_layer = official.GCTrace(list(lattice_shape))
    else:
        layer = official.LConvBilin(
            **common, use_symmetric=False,
        ).to(dtype=dtype)
        trace_layer = official.LTrace(list(lattice_shape))
    linear = torch.nn.Linear(4, 1, bias=True, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(data["lcb_weight"], dtype=dtype))
        linear.weight.copy_(
            torch.as_tensor(data["readout_weight"], dtype=dtype).reshape(1, 4)
        )
        linear.bias.copy_(torch.as_tensor(data["readout_bias"], dtype=dtype))

    def predict():
        packed_output = layer(packed)
        traced = trace_layer(packed_output)
        flattened = traced.reshape((-1, 4))
        # Mean of site-local predictions. This equals Linear(mean(trace)) and
        # therefore matches LCNNAction(reduction=:mean).
        return linear(flattened).mean()

    target = torch.as_tensor(float(data["target"]), dtype=dtype)
    optimizer = torch.optim.SGD(
        list(layer.parameters()) + list(linear.parameters()),
        lr=args.learning_rate,
    )
    initial_prediction = predict().detach().item()
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict()
        loss = (prediction - target) ** 2
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    final_prediction = predict().detach().item()
    np.savez(
        args.output,
        initial_prediction=np.asarray(initial_prediction),
        losses=np.asarray(losses),
        final_prediction=np.asarray(final_prediction),
        lcb_weight=layer.weight.detach().cpu().numpy(),
        readout_weight=linear.weight.detach().cpu().numpy().reshape(-1),
        readout_bias=linear.bias.detach().cpu().numpy().reshape(-1),
    )


if __name__ == "__main__":
    main()
