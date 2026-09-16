import JACC
JACC.@init_backend

using Enzyme
using Gaugefields
using LinearAlgebra
using Random
using Test

const BackendLCNN = Gaugefields.LCNN
const USE_MPI = lowercase(get(ENV, "LCNN_USE_MPI", "false")) in
    ("1", "true", "yes")
const TEST_PARAMETER_GRADIENT = lowercase(get(
    ENV, "LCNN_TEST_PARAMETER_GRADIENT", "true",
)) in ("1", "true", "yes")
const TEST_LINK_GRADIENT = lowercase(get(
    ENV, "LCNN_TEST_LINK_GRADIENT", "true",
)) in ("1", "true", "yes")
const TEST_LINK_MODEL_PULLBACK = lowercase(get(
    ENV, "LCNN_TEST_LINK_MODEL_PULLBACK", "true",
)) in ("1", "true", "yes")
const TEST_LINK_MODEL_PARAMETERS = lowercase(get(
    ENV, "LCNN_TEST_LINK_MODEL_PARAMETERS", "true",
)) in ("1", "true", "yes")
const TEST_CASES = Set(filter(value -> !isempty(value), split(
    lowercase(get(ENV, "LCNN_TEST_CASES", "2d,4d")), ',',
)))

if USE_MPI
    @eval using MPI
    MPI.Initialized() || MPI.Init()
end

const COMM = USE_MPI ? MPI.COMM_WORLD : nothing
const NRANKS = USE_MPI ? MPI.Comm_size(COMM) : 1
const RANK = USE_MPI ? MPI.Comm_rank(COMM) : 0

@inline function scalar_consensus(value)
    USE_MPI || return value, value
    return (
        MPI.Allreduce(value, MPI.MIN, COMM),
        MPI.Allreduce(value, MPI.MAX, COMM),
    )
end

function parameter_gradient(action, U, parameters, workspace)
    return BackendLCNN.parameter_gradient(
        action, U, parameters; workspace,
    )
end

function parameter_norm(parameters)
    layer_norm = sum(
        sum(abs2, group.weight) for group in parameters.layers;
        init=0.0,
    )
    return sqrt(
        layer_norm + sum(abs2, parameters.readout.weight) +
        sum(abs2, parameters.readout.bias),
    )
end

function link_parameter_norm(parameters)
    layer_norm = sum(
        sum(abs2, group.weight) for group in parameters.layers;
        init=0.0,
    )
    return sqrt(layer_norm + sum(abs2, parameters.lexp.weight))
end

function field_norm(fields)
    return sqrt(sum(real(dot(field.U, field.U)) for field in fields))
end

function make_configuration(lattice, colors, seed)
    process_grid = if length(lattice) == 2
        (1, NRANKS)
    else
        (1, 1, 1, NRANKS)
    end
    common = (
        colors=colors,
        halo=1,
        start=:hot,
        seed=seed,
        process_grid=process_grid,
        verbose=0,
    )
    return USE_MPI ?
        gauge_configuration(lattice; common..., comm=COMM) :
        gauge_configuration(lattice; common...)
end

function run_case(label, U, action, parameter_seed, expected)
    parameters = BackendLCNN.initial_parameters(
        MersenneTwister(parameter_seed), action, Float64,
    )
    workspace = BackendLCNN.ModelWorkspace(action, U)
    value = action(U, parameters, workspace)
    value_min, value_max = scalar_consensus(value)

    @test isfinite(value)
    @test value_min ≈ value_max atol=2e-12 rtol=2e-12
    @test value ≈ expected.value atol=2e-12 rtol=2e-11
    if JACC.backend == "cuda"
        @test occursin("CuArray", string(typeof(first(U).U.A)))
    end

    weight_norm = NaN
    if TEST_PARAMETER_GRADIENT
        gradient = parameter_gradient(action, U, parameters, workspace)
        weight_norm = parameter_norm(gradient)
        norm_min, norm_max = scalar_consensus(weight_norm)
        @test isfinite(weight_norm)
        @test weight_norm > 0
        @test norm_min ≈ norm_max atol=2e-11 rtol=2e-11
        @test weight_norm ≈ expected.dtheta_norm atol=2e-10 rtol=2e-10
    end

    link_norm = NaN
    if TEST_LINK_GRADIENT
        link_gradient = BackendLCNN.dSdu(
            action, U, parameters; workspace,
        )
        link_norm = field_norm(link_gradient)
        norm_min, norm_max = scalar_consensus(link_norm)
        @test isfinite(link_norm)
        @test link_norm > 0
        @test norm_min ≈ norm_max atol=2e-11 rtol=2e-11
        @test link_norm ≈ expected.dU_norm atol=2e-10 rtol=2e-10
    end


    link_model_dU_norm = NaN
    link_model_dtheta_norm = NaN
    if TEST_LINK_MODEL_PULLBACK
        link_model = BackendLCNN.LCNNLinkModel(action.feature_model)
        link_parameters = (
            layers=parameters.layers,
            lexp=(weight=fill(
                0.003,
                length(U),
                BackendLCNN.output_channels(action.feature_model),
            ),),
        )
        link_workspace = BackendLCNN.LinkModelWorkspace(link_model, U)
        if TEST_LINK_MODEL_PARAMETERS
            pullback = BackendLCNN.link_model_pullback(
                U, link_model, U, link_parameters; workspace=link_workspace,
            )
            link_model_dU_norm = field_norm(pullback.links)
            link_model_dtheta_norm = link_parameter_norm(pullback.parameters)
        else
            specification = lcnn_smearing(link_model, link_parameters)
            recorded = smear(U, specification; record=true, calcdSdU=true)
            link_model_dU_norm = field_norm(recorded.derivative(U))
        end
        values = TEST_LINK_MODEL_PARAMETERS ?
            (link_model_dU_norm, link_model_dtheta_norm) :
            (link_model_dU_norm,)
        for value in values
            value_min, value_max = scalar_consensus(value)
            @test isfinite(value)
            @test value > 0
            @test value_min ≈ value_max atol=2e-10 rtol=2e-10
        end
    end

    if RANK == 0
        println(
            label,
            " backend=", JACC.backend,
            " ranks=", NRANKS,
            " storage=", typeof(first(U).U.A),
            " value=", value,
            " dtheta_norm=", weight_norm,
            " dU_norm=", link_norm,
            " link_model_dtheta_norm=", link_model_dtheta_norm,
            " link_model_dU_norm=", link_model_dU_norm,
        )
    end
    return nothing
end

@testset "LCNN JACC backend smoke" begin
    if "2d" in TEST_CASES
        U2 = make_configuration((4, 4), 2, 0x475055)
        action2 = BackendLCNN.favoni2022_wilson_1x2_small(
            U2; convention=:favoni_prl,
        )
        run_case("2D SU(2)", U2, action2, 7, (
            value=0.6821891823496806,
            dtheta_norm=4.37749335859341,
            dU_norm=1.2399091050309203,
        ))
    end

    if "4d" in TEST_CASES
        U4 = make_configuration((4, 4, 4, 4), 3, 0x4d5049)
        # A compact but genuine LCB keeps the opt-in accelerator smoke test
        # short while exercising all four transport directions for SU(3).
        layer4 = BackendLCNN.LCB(
            6 => 1;
            convention=:favoni_arxiv,
            kernel_size=2,
            identity=false,
            adjoints=false,
        )
        model4 = BackendLCNN.LCNNFeatureModel(U4, (layer4,))
        action4 = BackendLCNN.LCNNAction(
            model4; component=:real, reduction=:mean,
        )
        run_case("4D SU(3)", U4, action4, 11, (
            value=-0.00022776219406983998,
            dtheta_norm=1.0183172007682137,
            dU_norm=0.128162891377158,
        ))
    end
end

if USE_MPI
    MPI.Barrier(COMM)
    MPI.Finalize()
end
