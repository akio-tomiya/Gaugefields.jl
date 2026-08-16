import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

function _maximum_unitarity_error(U, nc)
    maximum_error = 0.0
    for field in U
        storage = JACC.to_host(field.U.A)
        for site in CartesianIndices(field.U.PN)
            array_indices = ntuple(
                d -> site[d] + field.U.nw,
                length(field.U.PN),
            )
            matrix = [
                storage[ic, jc, array_indices...]
                for ic in 1:nc, jc in 1:nc
            ]
            maximum_error = max(
                maximum_error,
                norm(adjoint(matrix) * matrix - I),
            )
        end
    end
    return maximum_error
end

function _plaquette_action(U, dim)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette"; Dim=dim)
    append!(plaquettes, adjoint(plaquettes))
    push!(action, 1.0, plaquettes)
    return action
end

@testset "MPILattice general SU(N) heatbath and overrelaxation" begin
    nc = 4
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs <= 2 || error("this regression test supports at most two MPI ranks")

    for (global_size, process_grid) in (
        ((2 * nprocs, 2), (nprocs, 1)),
        ((2 * nprocs, 2, 2, 2), (nprocs, 1, 1, 1)),
    )
        dim = length(global_size)
        U = Initialize_Gaugefields(
            nc,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )

        updater = Heatbath(U, 2.0; seed=0x243f6a8885a308d3)
        heatbath!(U, updater)
        overrelaxation!(U, updater)
        @test updater.sweep == 1
        @test updater.overrelaxation_sweep == 1
        @test _maximum_unitarity_error(U, nc) < 2e-11

        action = _plaquette_action(U, dim)
        general_updater = Gaugefields.heatbath_module.Heatbath_update(
            U,
            action;
            seed=0x13198a2e03707344,
        )
        heatbath!(U, general_updater)
        overrelaxation!(U, general_updater)
        @test general_updater.sweep == 1
        @test general_updater.overrelaxation_sweep == 1
        @test _maximum_unitarity_error(U, nc) < 2e-11
    end
end
