using Gaugefields
using LinearAlgebra
using Test

function _g2_identity_links()
    u1 = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    return [identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0) for _ in 1:4]
end

@testset "G2 cold plaquette" begin
    U = _g2_identity_links()
    temp1 = similar(U[1])
    temp2 = similar(U[1])

    plaq = calculate_Plaquette(U, temp1, temp2)
    factor = 1 / (6 * U[1].NV * U[1].NC)
    @test real(plaq * factor) ≈ 1.0 atol = 1.0e-12
    @test abs(imag(plaq)) < 1.0e-12
end

@testset "G2 GaugeAction cold path" begin
    U = _g2_identity_links()
    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    push!(gauge_action, 1.0, plaqloop)

    untraced = evaluate_GaugeAction_untraced(gauge_action, U)
    value = evaluate_GaugeAction(gauge_action, U)

    @test untraced isa G2Gaugefields_4D_wing
    @test value == tr(untraced)
    @test isfinite(real(value))
    @test abs(imag(value)) < 1.0e-12
end

@testset "G2 GaugeAction random path" begin
    U = [
        randomG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0, randomnumber = "Reproducible", scale = 0.03)
        for _ in 1:4
    ]
    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    push!(gauge_action, 0.7, plaqloop)

    value = evaluate_GaugeAction(gauge_action, U)
    @test isfinite(real(value))
    @test isfinite(imag(value))

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    plaq = calculate_Plaquette(U, temp1, temp2)
    @test isfinite(real(plaq))
    @test isfinite(imag(plaq))
end
