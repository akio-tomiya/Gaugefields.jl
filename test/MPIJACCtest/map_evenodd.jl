import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Random
using Test

MPI.Initialized() || MPI.Init()

function deterministic_checkerboard_map!(U, V)
    for jc in 1:size(U, 2)
        for ic in 1:size(U, 1)
            U[ic, jc] = 2 * U[ic, jc] - conj(V[jc, ic])
        end
    end
    return nothing
end

function allocationfree_su2_map!(U, V)
    T = eltype(U)
    temps = (
        LatticeMatrices.MMatrix{2,2,T}(undef),
        LatticeMatrices.MMatrix{2,2,T}(undef),
    )
    Gaugefields.SU2update_KP_allocationfree!(U, V, 5.7, 2, temps, 100_000)
    return nothing
end

@testset "MPILattice checkerboard site map" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (3 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    shift = (1, 0, 0, 0)
    nc = 2
    nvalues = nc * nc * prod(global_size)
    original_u = reshape(
        [complex(Float64(i), Float64(2i + 1)) for i in 1:nvalues],
        nc,
        nc,
        global_size...,
    )
    original_v = reshape(
        [complex(Float64(3i - 1), Float64(-i)) for i in 1:nvalues],
        nc,
        nc,
        global_size...,
    )

    for target_even in (true, false)
        fields = Initialize_Gaugefields(
            nc,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        U = fields[1]
        V = fields[2]
        reference_fields = Initialize_Gaugefields(
            nc,
            0,
            global_size...;
            condition="cold",
            verbose_level=0,
        )
        U_reference = reference_fields[1]
        V_reference = reference_fields[2]

        for site in CartesianIndices(U.U.PN)
            local_indices = Tuple(site)
            global_indices = ntuple(
                d -> U.U.coords[d] * U.U.PN[d] + local_indices[d],
                4,
            )
            for jc in 1:nc
                for ic in 1:nc
                    U[ic, jc, local_indices...] = original_u[ic, jc, global_indices...]
                    V[ic, jc, local_indices...] = original_v[ic, jc, global_indices...]
                end
            end
        end
        for site in CartesianIndices(global_size)
            global_indices = Tuple(site)
            for jc in 1:nc
                for ic in 1:nc
                    U_reference[ic, jc, global_indices...] =
                        original_u[ic, jc, global_indices...]
                    V_reference[ic, jc, global_indices...] =
                        original_v[ic, jc, global_indices...]
                end
            end
        end

        Gaugefields.set_wing_U!(U)
        Gaugefields.set_wing_U!(V)
        Gaugefields.map_U!(U, deterministic_checkerboard_map!, V, target_even)
        Gaugefields.map_U!(
            U_reference,
            deterministic_checkerboard_map!,
            V_reference,
            target_even,
        )

        shifted = similar(U)
        Gaugefields.substitute_U!(shifted, Gaugefields.shift_U(U, shift))
        shifted_reference = Gaugefields.shift_U(U_reference, shift)

        for site in CartesianIndices(U.U.PN)
            local_indices = Tuple(site)
            global_indices = ntuple(
                d -> U.U.coords[d] * U.U.PN[d] + local_indices[d],
                4,
            )
            for jc in 1:nc
                for ic in 1:nc
                    @test U[ic, jc, local_indices...] ==
                          U_reference[ic, jc, global_indices...]
                    @test shifted[ic, jc, local_indices...] ==
                          shifted_reference[ic, jc, global_indices...]
                end
            end
        end
    end
end

@testset "MPILattice map invokes Gaugefields SU(2) local update" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (3 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)

    for target_even in (true, false)
        fields = Initialize_Gaugefields(
            2,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        U = fields[1]
        V = fields[2]
        Random.seed!(0x51a7 + MPI.Comm_rank(MPI.COMM_WORLD))
        Gaugefields.map_U!(U, allocationfree_su2_map!, V, target_even)

        for site in CartesianIndices(U.U.PN)
            local_indices = Tuple(site)
            global_indices = ntuple(
                d -> U.U.coords[d] * U.U.PN[d] + local_indices[d],
                4,
            )
            site_is_even = iseven(sum(global_indices))
            matrix = [U[ic, jc, local_indices...] for ic in 1:2, jc in 1:2]

            if site_is_even == target_even
                @test all(isfinite, matrix)
                @test matrix' * matrix ≈ Matrix{ComplexF64}(I, 2, 2) atol = 1e-13
                @test det(matrix) ≈ 1 atol = 1e-13
            else
                @test matrix == Matrix{ComplexF64}(I, 2, 2)
            end
        end
    end
end
