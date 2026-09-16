# Frozen Favoni et al. PyTorch outputs

`favoni2022_pytorch_outputs.npz` contains only the layer outputs and final
scalars produced by the official PyTorch implementations. Inputs and model
parameters are reconstructed deterministically by `test/lcnn_reference.jl`.
PyTorch is therefore not installed or executed by the test suite.

The fixture was generated in Float64 from:

- `arxiv_v1`, commit `e5e0d4894502b320696d6c099596be442f3a3be5`;
- `prl_2022`, commit `186f599d750506fd607cd7db63117f816c845734`.

To regenerate it manually:

```sh
export LCNN_TORCH_PYTHON=/path/to/python
export LCNN_ARXIV_LAYERS=/path/to/arxiv_v1/lge_cnn/nn/layers.py
export LCNN_PRL_LAYERS=/path/to/prl_2022/lge_cnn/nn/layers.py

julia --project examples/lcnn/generate_favoni2022_pytorch_golden.jl \
    test/data/lcnn/favoni2022_pytorch_outputs.npz
```

The generated arrays cover both public conventions and two independently
generated SU(2) gauge configurations with seeds `0x4c434e4e` and
`0x554c434e`.
