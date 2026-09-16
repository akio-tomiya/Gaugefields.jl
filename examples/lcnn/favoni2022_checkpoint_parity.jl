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

function compare_with_numpy(convention::Symbol; checkpoint=nothing, seed)
    convention in (:favoni_arxiv, :favoni_prl) || error("unknown convention $convention")
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

    mktempdir() do directory
        if checkpoint === nothing
            shape = LCNN.parameter_shapes(action).layers[1].weight
            lcb_weight = reshape(
                collect(range(-0.09f0, 0.07f0; length=prod(shape))), shape,
            )
            readout_weight = Float32[0.4, -0.3, 0.2, 0.1]
            readout_bias = Float32[-0.05]
        else
            convention === :favoni_prl || error(
                "released result files use convention=:favoni_prl",
            )
            extracted = joinpath(directory, "checkpoint.npz")
            run(`python3 $(joinpath(@__DIR__, "extract_favoni_checkpoint.py")) $checkpoint $extracted`)
            arrays = npzread(extracted)
            lcb_weight = arrays["layers.1.weight"]
            readout_weight = vec(arrays["readout.weight"])
            readout_bias = vec(arrays["readout.bias"])
        end
        parameters = (
            layers=((weight=lcb_weight,),),
            readout=(weight=readout_weight, bias=readout_bias),
        )
        LCNN.validate_parameters(action, parameters)

        input_path = joinpath(directory, "input.npz")
        reference_path = joinpath(directory, "reference.npz")
        npzwrite(input_path, Dict(
            "links" => dense_fields(U, lattice),
            "features" => dense_fields(input.channels, lattice),
            "lcb_weight" => lcb_weight,
            "readout_weight" => readout_weight,
            "readout_bias" => readout_bias,
        ))
        run(`python3 $(joinpath(@__DIR__, "favoni2022_numpy_reference.py")) $input_path $reference_path --convention $(String(convention))`)
        reference = npzread(reference_path)

        julia_features = LCNN.forward_features(
            action.feature_model, U, (layers=parameters.layers,), workspace,
        )
        julia_dense = dense_fields(julia_features.channels, lattice)
        julia_scalar = action(U, parameters, workspace)
        feature_error = maximum(abs, julia_dense .- reference["features"])
        scalar_error = abs(julia_scalar - only(reference["scalar"]))
        println(convention, ": seed=0x", string(seed; base=16),
                ", parameters=", LCNN.parameter_count(action),
                ", max feature error=", feature_error,
                ", scalar error=", scalar_error,
                ", prediction=", julia_scalar)
        @assert feature_error < 2e-10
        @assert scalar_error < 2e-10

        if haskey(ENV, "LCNN_TORCH_PYTHON") &&
           haskey(ENV, "LCNN_UPSTREAM_LAYERS")
            torch_path = joinpath(directory, "torch-reference.npz")
            torch_dtype = get(ENV, "LCNN_TORCH_DTYPE", "float64")
            run(`$(ENV["LCNN_TORCH_PYTHON"]) $(joinpath(@__DIR__, "favoni2022_torch_reference.py")) $(ENV["LCNN_UPSTREAM_LAYERS"]) $input_path $torch_path --convention $(String(convention)) --dtype $torch_dtype`)
            torch_reference = npzread(torch_path)
            torch_feature_error = maximum(
                abs, julia_dense .- torch_reference["features"],
            )
            torch_scalar_error = abs(
                julia_scalar - only(torch_reference["scalar"]),
            )
            println("official PyTorch: max feature error=",
                    torch_feature_error, ", scalar error=", torch_scalar_error)
            torch_tolerance = torch_dtype == "float32" ? 2e-5 : 2e-10
            @assert torch_feature_error < torch_tolerance
            @assert torch_scalar_error < torch_tolerance
        end
        return julia_scalar
    end
end

convention = isempty(ARGS) ? :favoni_prl : Symbol(ARGS[1])
checkpoint = length(ARGS) >= 2 ? ARGS[2] : nothing
predictions = [
    compare_with_numpy(convention; checkpoint, seed)
    for seed in (0x4c434e4e, 0x554c434e)
]
println("difference between independent configurations=",
        abs(predictions[1] - predictions[2]))
