import JACC
JACC.@init_backend

using Gaugefields
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "MPILattice checkerboard shifted multiplication" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (3 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    sentinel = complex(-17.0, 5.0)
    nc = 2
    nvalues = nc * nc * prod(global_size)
    original_a = reshape(
        [complex(Float64(i), Float64(2i + 1)) for i in 1:nvalues],
        nc,
        nc,
        global_size...,
    )
    original_b = reshape(
        [complex(Float64(3i - 1), Float64(-i)) for i in 1:nvalues],
        nc,
        nc,
        global_size...,
    )
    shift_a = (1, 0, 0, 0)
    shift_b = (-1, 0, 0, 0)

    U = Initialize_Gaugefields(
        nc,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    A = U[1]
    B = U[2]
    U_reference = Initialize_Gaugefields(
        nc,
        0,
        global_size...;
        condition="cold",
        verbose_level=0,
    )
    A_reference = U_reference[1]
    B_reference = U_reference[2]

    for site in CartesianIndices(A.U.PN)
        local_indices = Tuple(site)
        global_indices = ntuple(
            d -> A.U.coords[d] * A.U.PN[d] + local_indices[d],
            4,
        )
        for jc in 1:nc
            for ic in 1:nc
                A[ic, jc, local_indices...] = original_a[ic, jc, global_indices...]
                B[ic, jc, local_indices...] = original_b[ic, jc, global_indices...]
            end
        end
    end
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        for jc in 1:nc
            for ic in 1:nc
                A_reference[ic, jc, global_indices...] =
                    original_a[ic, jc, global_indices...]
                B_reference[ic, jc, global_indices...] =
                    original_b[ic, jc, global_indices...]
            end
        end
    end
    Gaugefields.set_wing_U!(A)
    Gaugefields.set_wing_U!(B)

    shifted_a = Gaugefields.shift_U(A, shift_a)
    shifted_b = Gaugefields.shift_U(B, shift_b)
    shifted_a_reference = Gaugefields.shift_U(A_reference, shift_a)
    shifted_b_reference = Gaugefields.shift_U(B_reference, shift_b)
    adjoint_pairs = ((false, false), (true, false), (false, true), (true, true))

    for (adjoint_a, adjoint_b) in adjoint_pairs
        operand_a = adjoint_a ? shifted_a' : shifted_a
        operand_b = adjoint_b ? shifted_b' : shifted_b
        reference_operand_a = adjoint_a ? shifted_a_reference' : shifted_a_reference
        reference_operand_b = adjoint_b ? shifted_b_reference' : shifted_b_reference

        for target_even in (true, false)
            C = similar(A)
            C_reference = similar(A_reference)
            for site in CartesianIndices(C.U.PN)
                local_indices = Tuple(site)
                for jc in 1:C.NC
                    for ic in 1:C.NC
                        C[ic, jc, local_indices...] = sentinel
                    end
                end
            end
            for site in CartesianIndices(global_size)
                global_indices = Tuple(site)
                for jc in 1:nc
                    for ic in 1:nc
                        C_reference[ic, jc, global_indices...] = sentinel
                    end
                end
            end

            mul!(C, operand_a, operand_b, target_even)
            mul!(C_reference, reference_operand_a, reference_operand_b, target_even)

            for site in CartesianIndices(C.U.PN)
                local_indices = Tuple(site)
                global_indices = ntuple(
                    d -> C.U.coords[d] * C.U.PN[d] + local_indices[d],
                    4,
                )

                for jc in 1:C.NC
                    for ic in 1:C.NC
                        @test C[ic, jc, local_indices...] ==
                              C_reference[ic, jc, global_indices...]
                    end
                end
            end
        end
    end
end
