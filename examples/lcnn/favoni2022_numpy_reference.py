#!/usr/bin/env python3
"""Independent NumPy evaluation of the released GCMConv/LConvBilin formula."""

import argparse
import numpy as np


def transport(links, feature, direction, step):
    """T_step(feature), arrays laid out as (channel,color,color,x,...)."""
    current = feature
    orientation = int(np.sign(step))
    lattice_shape = feature.shape[3:]
    for _ in range(abs(step)):
        result = np.empty_like(current)
        for site in np.ndindex(lattice_shape):
            source = list(site)
            source[direction] = (source[direction] + orientation) % lattice_shape[direction]
            source = tuple(source)
            link_site = site if orientation > 0 else source
            link = links[(direction, slice(None), slice(None), *link_site)]
            for channel in range(feature.shape[0]):
                value = current[(channel, slice(None), slice(None), *source)]
                if orientation > 0:
                    result[(channel, slice(None), slice(None), *site)] = (
                        link @ value @ link.conj().T
                    )
                else:
                    result[(channel, slice(None), slice(None), *site)] = (
                        link.conj().T @ value @ link
                    )
        current = result
    return current


def lcb(links, feature, weight, convention, kernel_size, dilation):
    dimension = len(feature.shape) - 3
    if convention == "favoni_arxiv":
        spacing = dilation + 1
        raw = [
            transport(links, feature, axis, distance * spacing)
            for axis in range(dimension)
            for distance in range(1, kernel_size)
        ]
    elif convention == "favoni_prl":
        spacing = dilation
        raw = [feature.copy()] + [
            transport(links, feature, axis, distance * spacing)
            for axis in range(dimension)
            for distance in range(1, kernel_size)
        ]
    else:
        raise ValueError("convention must be favoni_arxiv or favoni_prl")

    colors = feature.shape[1]
    lattice_shape = feature.shape[3:]
    identity = np.zeros((1, colors, colors, *lattice_shape), dtype=feature.dtype)
    diagonal = np.arange(colors)
    identity[(0, diagonal, diagonal, *([slice(None)] * dimension))] = 1

    local = np.concatenate((feature, feature.swapaxes(1, 2).conj(), identity), axis=0)
    transported_raw = np.concatenate(raw, axis=0)
    transported = np.concatenate(
        (transported_raw, transported_raw.swapaxes(1, 2).conj(), identity),
        axis=0,
    )
    # This is the released complex_einsum followed by einsum('uvw,...').
    products = np.einsum("vij...,wjk...->vwik...", local, transported)
    return np.einsum("uvw,vwik...->uik...", weight, products)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--convention", choices=("favoni_arxiv", "favoni_prl"), required=True
    )
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--dilation", type=int)
    args = parser.parse_args()
    dilation = args.dilation
    if dilation is None:
        dilation = 0 if args.convention == "favoni_arxiv" else 1

    data = np.load(args.input)
    output = lcb(
        data["links"], data["features"], data["lcb_weight"],
        args.convention, args.kernel_size, dilation,
    )
    traces = np.trace(output, axis1=1, axis2=2)
    # Upstream view(..., -1) order is re(W1), im(W1), re(W2), im(W2), ... .
    invariants = np.stack((traces.real, traces.imag), axis=1)
    axes = tuple(range(2, invariants.ndim))
    invariants = invariants.mean(axis=axes).reshape(-1)
    scalar = data["readout_weight"] @ invariants + data["readout_bias"][0]
    np.savez(args.output, features=output, scalar=np.asarray(scalar))


if __name__ == "__main__":
    main()
