import JACC
JACC.@init_backend
using Gaugefields, LatticeMatrices, MPI, Test
MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices center-valued B fast path" begin
    @test isdefined(LatticeMatrices, :ScaledIdentityLattice)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    lattice = (2nprocs, 2, 2, 2)
    grid = (nprocs, 1, 1, 1)
    for nc in (2, 3, 4), nw in (0, 1)
        U = Initialize_Gaugefields(nc, nw, lattice...;
            condition="hot", randomnumber="Reproducible",
            isMPILattice=true, PEs=grid, verbose_level=0)
        B = Initialize_Bfields(nc, [1, 0, 1, 1, 0, 1], nw, lattice...;
            isMPILattice=true, PEs=grid, verbose_level=0)
        full_B = Initialize_Bfields(nc, [1, 0, 1, 1, 0, 1], nw, lattice...;
            isMPILattice=true, PEs=grid, verbose_level=0, use_center_fastpath=false)
        legacy_B = Initialize_Bfields(nc, [1, 0, 1, 1, 0, 1], nw, lattice...;
            isMPILattice=true, PEs=grid, verbose_level=0, bfield_evaluation=:legacy)
        @test Gaugefields.Bfield_module._supports_center_phase(U[1])
        plaquettes = make_loops_fromname("plaquette"; Dim=4)
        append!(plaquettes, plaquettes')
        rectangles = make_loops_fromname("rectangular"; Dim=4)
        append!(rectangles, rectangles')
        action = GaugeAction(U, B)
        push!(action, 0.7, plaquettes)
        push!(action, -0.13, rectangles)
        paths = vcat(plaquettes, rectangles)
        temp = [similar(U[1]) for _ in 1:4]
        fast, reference = similar(U[1]), similar(U[1])
        counts = nothing
        for change_B in (false, true)
            if change_B
                updated = Initialize_Bfields(nc, [0, 1, 0, 0, 1, 0], nw, lattice...;
                    isMPILattice=true, PEs=grid, verbose_level=0)
                substitute_U!(B, updated)
                substitute_U!(full_B, updated)
                substitute_U!(legacy_B, updated)
            end
            for path in paths
                evaluate_Bplaquettes!(fast, path, B, temp)
                evaluate_Bplaquettes!(reference, path, full_B, temp)
                @test maximum(abs, gather_and_bcast_matrix(fast.U) - gather_and_bcast_matrix(reference.U)) < 2e-12
            end
            value = evaluate_GaugeAction(action, U, B)
            @test value ≈ evaluate_GaugeAction(action, U, full_B) atol=2e-10 rtol=2e-12
            @test value ≈ evaluate_GaugeAction(action, U, legacy_B) atol=2e-10 rtol=2e-12
            for mu in 1:4
                calc_dSdUμ!(fast, action, mu, U, B)
                calc_dSdUμ!(reference, action, mu, U, full_B)
                @test maximum(abs, gather_and_bcast_matrix(fast.U) - gather_and_bcast_matrix(reference.U)) < 2e-11
            end
            @test isempty(legacy_B.pathplans)
            if change_B
                @test length(B.pathplans) == counts
            else
                counts = length(B.pathplans)
            end
        end
    end
end

@testset "wing B force agrees with halo-free reference" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    lattice, grid = (2nprocs, 2, 2, 2), (nprocs, 1, 1, 1)
    args = (; isMPILattice=true, PEs=grid, verbose_level=0)
    flux = [1, 0, 1, 1, 0, 1]
    for nc in (2, 3, 4)
        U0 = Initialize_Gaugefields(nc, 0, lattice...; args...,
            condition="hot", randomnumber="Reproducible")
        U1 = Initialize_Gaugefields(nc, 1, lattice...; args...,
            condition="hot", randomnumber="Reproducible")
        for mu in 1:4
            @test gather_and_bcast_matrix(U0[mu].U) == gather_and_bcast_matrix(U1[mu].U)
        end
        B0 = Initialize_Bfields(nc, flux, 0, lattice...; args..., bfield_storage=:matrix)
        loops = make_loops_fromname("rectangular"; Dim=4)
        append!(loops, loops')
        action0 = GaugeAction(U0, B0)
        push!(action0, 0.7, loops)
        value0 = evaluate_GaugeAction(action0, U0, B0)
        force0 = map(1:4) do mu
            out = similar(U0[1])
            calc_dSdUμ!(out, action0, mu, U0, B0)
            gather_and_bcast_matrix(out.U)
        end
        for storage in (:matrix, :scalar)
            B1 = Initialize_Bfields(nc, flux, 1, lattice...; args..., bfield_storage=storage)
            action1 = GaugeAction(U1, B1)
            push!(action1, 0.7, loops)
            @test evaluate_GaugeAction(action1, U1, B1) ≈ value0 atol=2e-10 rtol=2e-12
            for mu in 1:4
                out = similar(U1[1])
                calc_dSdUμ!(out, action1, mu, U1, B1)
                @test maximum(abs, gather_and_bcast_matrix(out.U) - force0[mu]) < 2e-11
            end
        end
    end
end
