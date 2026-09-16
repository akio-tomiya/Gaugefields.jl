using Gaugefields
using NPZ

const LCNN = Gaugefields.LCNN

function dense_fields(fields, lattice)
    colors = first(fields).NC
    output = Array{ComplexF64}(undef, length(fields), colors, colors, lattice...)
    for channel in eachindex(fields), site_index in CartesianIndices(lattice)
        site = Tuple(site_index)
        for column in 1:colors, row in 1:colors
            output[channel, row, column, site...] =
                fields[channel][row, column, site...]
        end
    end
    return output
end

function generate_output(convention, seed, layers_path, python, directory)
    lattice = (4, 4)
    U = gauge_configuration(
        lattice;
        colors=2,
        halo=1,
        start=:hot,
        seed,
        process_grid=(1, 1),
        verbose=0,
    )
    action = LCNN.favoni2022_wilson_1x2_small(U; convention)
    workspace = LCNN.ReferenceWorkspace(U)
    input = LCNN.plaquette_features(LCNN.Plaq(), U, workspace)
    shape = LCNN.parameter_shapes(action).layers[1].weight
    lcb_weight = reshape(
        collect(range(-0.09f0, 0.07f0; length=prod(shape))), shape,
    )

    input_path = joinpath(directory, "$(convention)-$(seed)-input.npz")
    output_path = joinpath(directory, "$(convention)-$(seed)-output.npz")
    npzwrite(input_path, Dict(
        "links" => dense_fields(U, lattice),
        "features" => dense_fields(input.channels, lattice),
        "lcb_weight" => lcb_weight,
        "readout_weight" => Float32[0.4, -0.3, 0.2, 0.1],
        "readout_bias" => Float32[-0.05],
    ))
    run(`$python $(joinpath(@__DIR__, "favoni2022_torch_reference.py")) $layers_path $input_path $output_path --convention $(String(convention)) --dtype float64`)
    return npzread(output_path)
end

function main(output_path)
    python = get(ENV, "LCNN_TORCH_PYTHON", "python3")
    layer_paths = (
        favoni_arxiv=ENV["LCNN_ARXIV_LAYERS"],
        favoni_prl=ENV["LCNN_PRL_LAYERS"],
    )
    arrays = Dict{String,Any}()
    mktempdir() do directory
        for convention in (:favoni_arxiv, :favoni_prl)
            for seed in (0x4c434e4e, 0x554c434e)
                reference = generate_output(
                    convention,
                    seed,
                    getproperty(layer_paths, convention),
                    python,
                    directory,
                )
                prefix = "$(convention).$(string(seed; base=16))"
                arrays["$prefix.features"] = reference["features"]
                arrays["$prefix.scalar"] = reference["scalar"]
            end
        end
    end
    mkpath(dirname(output_path))
    npzwrite(output_path, arrays)
    println("wrote official PyTorch outputs to $output_path")
end

length(ARGS) == 1 || error(
    "usage: julia --project generate_favoni2022_pytorch_golden.jl OUTPUT.npz",
)
main(only(ARGS))
