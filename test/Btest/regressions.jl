using Gaugefields
using LinearAlgebra
using Random
using Test

_breg_matrix(field, site) =
    [field[i, j, site...] for i in 1:field.NC, j in 1:field.NC]

# Independent site-by-site definition, without the staple/path evaluators.
function _breg_plaquette(U, B, lattice)
    value = 0.0
    for index in CartesianIndices(lattice)
        site = Tuple(index)
        for mu in 1:3, nu in (mu + 1):4
            forward_mu = ntuple(d -> d == mu ? mod1(site[d] + 1, lattice[d]) : site[d], 4)
            forward_nu = ntuple(d -> d == nu ? mod1(site[d] + 1, lattice[d]) : site[d], 4)
            loop = _breg_matrix(U[mu], site) * _breg_matrix(U[nu], forward_mu) *
                   _breg_matrix(U[mu], forward_nu)' * _breg_matrix(U[nu], site)'
            value += real(tr(_breg_matrix(B[mu, nu], site) * loop))
        end
    end
    return value
end

@testset "B-dependent plaquette gradient flow" begin
    for nc in (2, 3), ndw in (0, 1)
        U = Initialize_Gaugefields(nc, ndw, 2, 2, 2, 2;
            condition="hot", randomnumber="Reproducible", verbose_level=0)
        B = Initialize_Bfields(nc, [1, 0, 1, 0, 1, 0], ndw, 2, 2, 2, 2; verbose_level=0)
        reference = deepcopy(U)
        ordinary = Gradientflow(U; eps=0.005, Nflow=2)
        general = Gradientflow_general_Bfields(reference, B, ["plaquette"], [1.0];
            eps=0.005, Nflow=2)
        B_before = Dict((mu, nu) => copy(B[mu, nu].U)
            for mu in 1:4 for nu in 1:4 if mu != nu)
        flow!(U, B, ordinary)
        flow!(reference, B, general)
        @test all(isapprox(U[d].U, reference[d].U; atol=2e-12, rtol=2e-12) for d in 1:4)
        @test all(B[mu, nu].U == old for ((mu, nu), old) in B_before)

        zero_B = Initialize_Bfields(nc, zeros(Int, 6), ndw, 2, 2, 2, 2; verbose_level=0)
        substitute_U!(B, zero_B)
        substitute_U!(reference, U)
        flow!(U, B, ordinary)
        flow!(reference, Gradientflow(reference; eps=0.005, Nflow=2))
        @test all(isapprox(U[d].U, reference[d].U; atol=2e-12, rtol=2e-12) for d in 1:4)

        add_force! = Gaugefields.AbstractGaugefields_module.add_force!
        force = initialize_TA_Gaugefields(U)
        work = [similar(U[1]) for _ in 1:7]
        @test_throws ArgumentError add_force!(force, U, B, work)
        @test_throws ArgumentError add_force!(force, U, B, work[1:6]; plaqonly=true)
        add_force!(force, U, B, work; plaqonly=true)
        first_force = [copy(f.a) for f in force]
        add_force!(force, U, B, work; plaqonly=true)
        @test all(isapprox(force[d].a, 2first_force[d]; atol=2e-12) for d in 1:4)
    end
end

function _breg_action(U, B)
    loops = make_loops_fromname("plaquette"; Dim=4)
    append!(loops, loops')
    action = GaugeAction(U, B)
    push!(action, 1.0, loops)
    return action
end

@testset "B plaquette measurement and serial wing constructors" begin
    lattice = (2, 3, 2, 2)
    for nc in (2, 3), ndw in (0, 1), condition in ("tflux", "tloop")
        U = Initialize_Gaugefields(nc, ndw, lattice...;
            condition="hot", randomnumber="Reproducible", verbose_level=0)
        B = Initialize_Bfields(nc, [1, 0, 1, 0, 1, 0], ndw, lattice...;
            condition, verbose_level=0)
        U_before = [copy(link.U) for link in U]
        B_before = Dict((mu, nu) => copy(B[mu, nu].U)
            for mu in 1:4 for nu in 1:4 if mu != nu)
        reference = _breg_plaquette(U, B, lattice)
        action = _breg_action(U, B)
        @test real(evaluate_GaugeAction(action, U, B)) / 2 ≈ reference atol=2e-12
        temps = [similar(U[1]), similar(U[1])]
        for _ in 1:2
            @test calculate_Plaquette(U, B, temps) ≈ reference atol=2e-12
            @test all(U[d].U == U_before[d] for d in 1:4)
            @test all(B[mu, nu].U == old for ((mu, nu), old) in B_before)
        end
        if ndw == 1
            B_nowing = Initialize_Bfields(nc, [1, 0, 1, 0, 1, 0], 0, lattice...;
                condition, verbose_level=0)
            @test all(_breg_matrix(B[mu, nu], Tuple(site)) ==
                      _breg_matrix(B_nowing[mu, nu], Tuple(site))
                for mu in 1:4 for nu in 1:4 if mu != nu
                for site in CartesianIndices(lattice))
        end
        # With identity links, a cold measurement is just the sum of phases.
        cold = Initialize_Gaugefields(nc, ndw, lattice...; condition="cold", verbose_level=0)
        expected = sum(real(tr(_breg_matrix(B[mu, nu], Tuple(site))))
            for mu in 1:3 for nu in (mu + 1):4 for site in CartesianIndices(lattice))
        @test calculate_Plaquette(cold, B, temps) ≈ expected atol=2e-12
        zero_B = Initialize_Bfields(nc, zeros(Int, 6), ndw, lattice...; verbose_level=0)
        @test calculate_Plaquette(U, zero_B, temps) ≈ calculate_Plaquette(U, temps...) atol=2e-12
    end
end

@testset "B-dependent molecular dynamics" begin
    Random.seed!(9127)
    lattice = (2, 2, 2, 2)
    for nc in (2, 3), ndw in (0, 1)
        U = Initialize_Gaugefields(nc, ndw, lattice...;
            condition="hot", randomnumber="Reproducible", verbose_level=0)
        B = Initialize_Bfields(nc, [1, 0, 1, 0, 1, 0], ndw, lattice...; verbose_level=0)
        action = _breg_action(U, B)
        bound = BfieldGaugeAction(action, B)
        workspace = md_action_workspace(bound, U)
        reusable_driver = md_driver(U, action, B; steps=2)
        zero_momenta = initialize_TA_Gaugefields(U)
        force = initialize_TA_Gaugefields(U)
        reference_force = initialize_TA_Gaugefields(U)
        derivative, product = similar(U[1]), similar(U[1])
        initial_value = md_potential(bound, U, workspace)
        @test initial_value ≈ -2 * _breg_plaquette(U, B, lattice) / nc atol=2e-12
        @test !isapprox(initial_value, md_potential(action, U, workspace); atol=1e-8)
        for change_B in (false, true)
            if change_B
                Bnew = Initialize_Bfields(nc, [0, 1, 0, 1, 0, 1], ndw, lattice...; verbose_level=0)
                substitute_U!(B, Bnew)
                @test !isapprox(md_potential(bound, U, workspace), initial_value; atol=1e-8)
            end
            @test md_potential(bound, U, workspace) ≈
                  -real(evaluate_GaugeAction(action, U, B)) / nc atol=2e-12
            @test md_hamiltonian(U, zero_momenta, reusable_driver) ≈
                  md_potential(bound, U, workspace) atol=2e-12
            md_force!(force, bound, U, workspace)
            for mu in 1:4
                calc_dSdUμ!(derivative, action, mu, U, B)
                mul!(product, U[mu], derivative)
                clear_U!(reference_force[mu])
                Traceless_antihermitian_add!(reference_force[mu], -1 / nc, product)
                @test force[mu].a ≈ reference_force[mu].a atol=2e-12
            end
        end
        for integrator in (PQP(), QPQ())
            p = initialize_TA_Gaugefields(U)
            gauss_distribution!(p)
            initial_U = [copy(link.U) for link in U]
            initial_p = [copy(momentum.a) for momentum in p]
            forward = md_driver(U, action, B; steps=2, trajectory_length=0.04, integrator)
            backward = md_driver(U, bound; steps=2, trajectory_length=-0.04, integrator)
            @test forward.action.B === B
            @test backward.action.B === B
            result = md_trajectory!(U, p, forward)
            @test isfinite(result.delta_hamiltonian)
            md_trajectory!(U, p, backward; diagnostics=false)
            @test all(isapprox(U[d].U, initial_U[d]; atol=3e-12, rtol=3e-12) for d in 1:4)
            @test all(isapprox(p[d].a, initial_p[d]; atol=3e-12, rtol=3e-12) for d in 1:4)
        end
    end
end
