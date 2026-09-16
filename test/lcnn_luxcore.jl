using LuxCore

const LuxLCNNCore = Gaugefields.LCNN

@testset "LCNN LuxCore adapter" begin
    U = gauge_configuration(
        (3, 4);
        colors=2,
        halo=1,
        start=:hot,
        seed=0x4c555843,
        process_grid=(1, 1),
        verbose=0,
    )
    feature_model = LuxLCNNCore.LCNNFeatureModel(
        Val(2), 2;
        shifts=:positive,
        identity=false,
        adjoints=false,
    )
    action = LuxLCNNCore.LCNNAction(feature_model; reduction=:mean)
    layer = LuxLCNNCore.lux_layer(action; parameter_type=Float64)
    @test layer isa LuxLCNNCore.LuxLCNN
    parameters, state = LuxCore.setup(MersenneTwister(0x4c5558), layer)

    @test parameters |> LuxCore.parameterlength == 9
    @test eltype(parameters.layers[1].weight) === Float64
    @test state == NamedTuple()

    value, next_state = layer(U, parameters, state)
    reference = action(U, parameters, LuxLCNNCore.ModelWorkspace(action, U))
    @test value isa Float64
    @test value ≈ reference atol=2e-12 rtol=2e-12
    @test next_state === state

    workspace = LuxLCNNCore.ModelWorkspace(action, U)
    workspace_value, workspace_state = layer((U, workspace), parameters, state)
    @test workspace_value ≈ reference atol=2e-12 rtol=2e-12
    @test workspace_state === state
end
