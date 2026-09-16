using Enzyme
using Gaugefields
using LinearAlgebra
using NPZ
import Wilsonloop: Wilsonline

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

function loss(parameters, action, U, workspace, target)
    residual = action(U, parameters, workspace) - target
    return residual * residual
end

function clear_parameter_gradient!(gradient)
    for group in gradient.layers
        fill!(group.weight, 0)
    end
    fill!(gradient.readout.weight, 0)
    fill!(gradient.readout.bias, 0)
    return gradient
end

function sgd_step!(parameters, gradient, learning_rate)
    for index in eachindex(parameters.layers)
        parameters.layers[index].weight .-=
            learning_rate .* gradient.layers[index].weight
    end
    parameters.readout.weight .-=
        learning_rate .* gradient.readout.weight
    parameters.readout.bias .-=
        learning_rate .* gradient.readout.bias
    return parameters
end

function main(; convention=:favoni_prl, steps=3, learning_rate=1.0e-3)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    learning_rate >= 0 || throw(ArgumentError("learning_rate must be nonnegative"))
    python = get(ENV, "LCNN_TORCH_PYTHON", "python3")
    layers_path = get(ENV, "LCNN_UPSTREAM_LAYERS", "")
    isempty(layers_path) && error("set LCNN_UPSTREAM_LAYERS to official layers.py")

    lattice = (4, 4)
    U = gauge_configuration(
        lattice; colors=2, halo=1, start=:hot, seed=0x54524149,
        process_grid=(1, 1), verbose=0,
    )
    action = LCNN.favoni2022_wilson_1x2_small(U; convention)
    workspace = LCNN.ModelWorkspace(action, U)
    input = LCNN.plaquette_features(
        LCNN.Plaq(), U, workspace.reference,
    )
    shape = LCNN.parameter_shapes(action).layers[1].weight
    parameters = (
        layers=((weight=reshape(
            collect(range(-0.08, 0.06; length=prod(shape))), shape,
        ),),),
        readout=(weight=[0.35, -0.25, 0.15, 0.05], bias=[-0.04]),
    )
    println(
        "tiny training parity: convention=", convention,
        ", lattice=", lattice,
        ", colors=2, samples=1, steps=", steps,
        ", learning_rate=", learning_rate,
        ", parameters=", LCNN.parameter_count(action),
    )

    rectangle = similar(first(U))
    path = Wilsonline([(1, 1), (2, 2), (1, -1), (2, -2)]; Dim=2)
    Gaugefields.evaluate_gaugelinks!(
        rectangle, path, U, workspace.reference.path_temps,
    )
    target = real(tr(rectangle)) / prod(lattice)

    mktempdir() do directory
        fixture_path = joinpath(directory, "training-input.npz")
        torch_path = joinpath(directory, "torch-training.npz")
        npzwrite(fixture_path, Dict(
            "links" => dense_fields(U, lattice),
            "features" => dense_fields(input.channels, lattice),
            "lcb_weight" => parameters.layers[1].weight,
            "readout_weight" => parameters.readout.weight,
            "readout_bias" => parameters.readout.bias,
            "target" => target,
        ))
        run(`$python $(joinpath(@__DIR__, "favoni2022_torch_train.py")) $layers_path $fixture_path $torch_path --convention $(String(convention)) --steps $steps --learning-rate $learning_rate`)
        torch_result = npzread(torch_path)

        initial_prediction = action(U, parameters, workspace)
        gradient = Enzyme.make_zero(parameters)
        shadow_workspace = Enzyme.make_zero(workspace)
        losses = Float64[]
        for _ in 1:steps
            clear_parameter_gradient!(gradient)
            Enzyme.remake_zero!(shadow_workspace)
            push!(losses, loss(parameters, action, U, workspace, target))
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(loss),
                Enzyme.Active,
                Enzyme.Duplicated(parameters, gradient),
                Enzyme.Const(action),
                Enzyme.Const(U),
                Enzyme.Duplicated(workspace, shadow_workspace),
                Enzyme.Const(target),
            )
            sgd_step!(parameters, gradient, learning_rate)
        end
        final_prediction = action(U, parameters, workspace)

        println("target = ", target)
        println("Julia losses = ", losses)
        println("PyTorch losses = ", torch_result["losses"])
        println("initial prediction error = ",
                abs(initial_prediction - only(torch_result["initial_prediction"])))
        println("loss trajectory error = ",
                maximum(abs, losses .- torch_result["losses"]))
        println("final prediction error = ",
                abs(final_prediction - only(torch_result["final_prediction"])))
        println("LCB weight error = ", maximum(
            abs, parameters.layers[1].weight .- torch_result["lcb_weight"],
        ))
        println("readout weight error = ", maximum(
            abs, parameters.readout.weight .- torch_result["readout_weight"],
        ))
        println("readout bias error = ", maximum(
            abs, parameters.readout.bias .- torch_result["readout_bias"],
        ))

        @assert initial_prediction ≈ only(torch_result["initial_prediction"]) atol=2e-12 rtol=2e-12
        @assert losses ≈ torch_result["losses"] atol=2e-11 rtol=2e-11
        @assert final_prediction ≈ only(torch_result["final_prediction"]) atol=2e-11 rtol=2e-11
        @assert parameters.layers[1].weight ≈ torch_result["lcb_weight"] atol=2e-11 rtol=2e-11
        @assert parameters.readout.weight ≈ torch_result["readout_weight"] atol=2e-11 rtol=2e-11
        @assert parameters.readout.bias ≈ torch_result["readout_bias"] atol=2e-11 rtol=2e-11
    end
end

length(ARGS) <= 3 || error(
    "usage: julia --project favoni2022_training_parity.jl " *
    "[favoni_arxiv|favoni_prl] [steps] [learning_rate]",
)
convention = isempty(ARGS) ? :favoni_prl : Symbol(ARGS[1])
steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
learning_rate = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0e-3
main(; convention, steps, learning_rate)
