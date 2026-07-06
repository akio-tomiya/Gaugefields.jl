using Gaugefields
using LinearAlgebra
using Test

function _g2_force_identity_links()
    return [identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0) for _ in 1:4]
end

function _g2_force_random_links()
    return [
        randomG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0, randomnumber = "Reproducible", scale = 0.02)
        for _ in 1:4
    ]
end

function _g2_force_action(U)
    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, adjoint(plaqloop))
    push!(gauge_action, 1.0, plaqloop)
    return gauge_action
end

function _g2_force_action_value(gauge_action, U)
    return real(-evaluate_GaugeAction(gauge_action, U) / U[1].NC)
end

function _g2_force_copy_links(U)
    V = similar(U)
    substitute_U!(V, U)
    return V
end

function _g2_force_perturbed_links(U, μ, eps, coeffs; ix = 1, iy = 1, iz = 1, it = 1)
    V = _g2_force_copy_links(U)
    update = exp(Float64(eps) .* g2_matrix(coeffs))
    oldlink = Matrix{ComplexF64}(undef, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    @inbounds for j in 1:G2_FUNDAMENTAL_DIM
        for i in 1:G2_FUNDAMENTAL_DIM
            oldlink[i, j] = U[μ][i, j, ix, iy, iz, it]
        end
    end
    newlink = update * oldlink
    @test is_g2_link(newlink; atol = 5.0e-12)
    @inbounds for j in 1:G2_FUNDAMENTAL_DIM
        for i in 1:G2_FUNDAMENTAL_DIM
            V[μ][i, j, ix, iy, iz, it] = newlink[i, j]
        end
    end
    set_wing_U!(V[μ])
    return V
end

function _g2_force_directional_projection(F, μ, coeffs; ix = 1, iy = 1, iz = 1, it = 1)
    projection = 0.0
    @inbounds for k in 1:G2_ALGEBRA_DIM
        projection += coeffs[k] * F[μ][k, ix, iy, iz, it]
    end
    return projection
end

function _g2_force_field(U)
    F = initialize_TA_Gaugefields(U)
    temps = Temporalfields(U[1]; num = 10)
    clear_U!(F)
    add_force!(F, U, temps; plaqonly = true, factor = 1 / U[1].NC)
    return F
end

@testset "G2 cold plaquette force" begin
    U = _g2_force_identity_links()
    F = _g2_force_field(U)
    @test F * F < 1.0e-24
end

@testset "G2 plaquette force finite difference" begin
    U = _g2_force_random_links()
    gauge_action = _g2_force_action(U)
    F = _g2_force_field(U)
    μ = 1
    coeffs = [0.01 * sin(0.3 * k) for k in 1:G2_ALGEBRA_DIM]
    force_projection = _g2_force_directional_projection(F, μ, coeffs)
    s0 = _g2_force_action_value(gauge_action, U)

    eps_values = (1.0e-4, 3.0e-5, 1.0e-5)
    finite_differences = Float64[]
    for eps in eps_values
        plus = _g2_force_perturbed_links(U, μ, eps, coeffs)
        minus = _g2_force_perturbed_links(U, μ, -eps, coeffs)
        push!(
            finite_differences,
            (_g2_force_action_value(gauge_action, plus) - _g2_force_action_value(gauge_action, minus)) / (2 * eps),
        )
    end

    @test isfinite(force_projection)
    @test all(isfinite, finite_differences)
    @test abs(s0) > 0
    @test maximum(abs.(finite_differences .- finite_differences[1])) < 1.0e-8
    @test sign(force_projection) == sign(finite_differences[1])
    @test isapprox(force_projection, finite_differences[1]; rtol = 2.0e-2, atol = 5.0e-6)
end
