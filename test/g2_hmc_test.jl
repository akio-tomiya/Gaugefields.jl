using Gaugefields
using LinearAlgebra
using Random
using Test

function _g2_hmc_links()
    return [
        randomG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0, randomnumber = "Reproducible", scale = 0.01)
        for _ in 1:4
    ]
end

function _g2_hmc_action(U; beta = 0.4)
    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, adjoint(plaqloop))
    push!(gauge_action, beta / 2, plaqloop)
    return gauge_action
end

function _g2_hmc_set_momentum!(p; scale = 0.01)
    clear_U!(p)
    for μ in 1:4
        pμ = p[μ]
        @inbounds for it in 1:pμ.NT
            for iz in 1:pμ.NZ
                for iy in 1:pμ.NY
                    for ix in 1:pμ.NX
                        for k in 1:G2_ALGEBRA_DIM
                            phase = 0.17 * (k + 2ix + 3iy + 5iz + 7it + 11μ)
                            pμ[k, ix, iy, iz, it] = scale * cos(phase)
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function _g2_hmc_hamiltonian(gauge_action, U, p)
    gauge = -evaluate_GaugeAction(gauge_action, U) / U[1].NC
    kinetic = p * p / 2
    return real(gauge + kinetic)
end

function _g2_hmc_update_links!(gauge_action, U, p, step_size)
    temps = get_temporary_gaugefields(gauge_action)
    expU = temps[3]
    work = temps[4]
    for μ in 1:4
        exptU!(expU, step_size, p[μ], temps[1:2])
        mul!(work, expU, U[μ])
        substitute_U!(U[μ], work)
    end
    set_wing_U!(U)
    return nothing
end

function _g2_hmc_update_momenta!(gauge_action, U, p, step_size)
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[4]
    factor = -step_size / NC
    for μ in 1:4
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temps[1], U[μ], dSdUμ)
        Traceless_antihermitian_add!(p[μ], factor, temps[1])
    end
    return nothing
end

function _g2_hmc_leapfrog!(gauge_action, U, p; step_size = 0.01, md_steps = 2)
    for _ in 1:md_steps
        _g2_hmc_update_links!(gauge_action, U, p, step_size / 2)
        _g2_hmc_update_momenta!(gauge_action, U, p, step_size)
        _g2_hmc_update_links!(gauge_action, U, p, step_size / 2)
    end
    return nothing
end

function _g2_hmc_acceptance_probability(old_hamiltonian, new_hamiltonian)
    if new_hamiltonian <= old_hamiltonian
        return 1.0
    end
    return exp(old_hamiltonian - new_hamiltonian)
end

function _g2_hmc_proposal!(gauge_action, U, p, rng; step_size = 0.01, md_steps = 2)
    Uold = similar(U)
    substitute_U!(Uold, U)
    old_hamiltonian = _g2_hmc_hamiltonian(gauge_action, U, p)
    _g2_hmc_leapfrog!(gauge_action, U, p; step_size = step_size, md_steps = md_steps)
    new_hamiltonian = _g2_hmc_hamiltonian(gauge_action, U, p)
    probability = _g2_hmc_acceptance_probability(old_hamiltonian, new_hamiltonian)
    accepted = rand(rng) <= probability
    if !accepted
        substitute_U!(U, Uold)
    end
    return (
        accepted = accepted,
        old_hamiltonian = old_hamiltonian,
        new_hamiltonian = new_hamiltonian,
        delta_h = new_hamiltonian - old_hamiltonian,
        acceptance_probability = probability,
    )
end

function _g2_hmc_site_matrix(Uμ, ix, iy, iz, it)
    matrix = Matrix{ComplexF64}(undef, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    @inbounds for j in 1:G2_FUNDAMENTAL_DIM
        for i in 1:G2_FUNDAMENTAL_DIM
            matrix[i, j] = Uμ[i, j, ix, iy, iz, it]
        end
    end
    return matrix
end

function _g2_hmc_max_link_defect(U)
    maxdefect = 0.0
    for μ in 1:4
        Uμ = U[μ]
        @inbounds for it in 1:Uμ.NT
            for iz in 1:Uμ.NZ
                for iy in 1:Uμ.NY
                    for ix in 1:Uμ.NX
                        defects = g2_link_defects(_g2_hmc_site_matrix(Uμ, ix, iy, iz, it))
                        maxdefect = max(
                            maxdefect,
                            defects.imaginary,
                            defects.orthogonal,
                            defects.determinant,
                            defects.algebra,
                        )
                    end
                end
            end
        end
    end
    return maxdefect
end

function _g2_hmc_plaquette(U)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    return calculate_Plaquette(U, temp1, temp2) / (6 * U[1].NV * U[1].NC)
end

@testset "G2 HMC acceptance probability" begin
    @test _g2_hmc_acceptance_probability(1.0, 0.5) == 1.0
    @test 0 < _g2_hmc_acceptance_probability(0.5, 1.0) < 1
end

@testset "G2 quenched HMC smoke" begin
    U = _g2_hmc_links()
    gauge_action = _g2_hmc_action(U)
    p = initialize_TA_Gaugefields(U)
    _g2_hmc_set_momentum!(p; scale = 0.01)
    rng = MersenneTwister(20260702)

    initial_plaquette = _g2_hmc_plaquette(U)
    proposal = _g2_hmc_proposal!(gauge_action, U, p, rng; step_size = 0.01, md_steps = 2)
    final_plaquette = _g2_hmc_plaquette(U)

    @test proposal.accepted isa Bool
    @test isfinite(proposal.old_hamiltonian)
    @test isfinite(proposal.new_hamiltonian)
    @test isfinite(proposal.delta_h)
    @test abs(proposal.delta_h) < 1.0e-1
    @test 0 <= proposal.acceptance_probability <= 1
    @test isfinite(real(initial_plaquette))
    @test isfinite(imag(initial_plaquette))
    @test isfinite(real(final_plaquette))
    @test isfinite(imag(final_plaquette))
    @test _g2_hmc_max_link_defect(U) < 5.0e-9
end
