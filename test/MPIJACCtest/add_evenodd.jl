import JACC
JACC.@init_backend

using Gaugefields
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "MPILattice checkerboard addition" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (3 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    source_shift = (1, 0, 0, 0)
    sentinel = complex(-17.0, 5.0)
    nc = 2
    nvalues = nc * nc * prod(global_size)
    original_a = reshape(
        [complex(Float64(2i + 1), Float64(-i)) for i in 1:nvalues],
        nc,
        nc,
        global_size...,
    )

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
    U_reference = Initialize_Gaugefields(
        nc,
        0,
        global_size...;
        condition="cold",
        verbose_level=0,
    )
    A_reference = U_reference[1]

    for site in CartesianIndices(A.U.PN)
        local_indices = Tuple(site)
        global_indices = ntuple(
            d -> A.U.coords[d] * A.U.PN[d] + local_indices[d],
            4,
        )
        for jc in 1:nc
            for ic in 1:nc
                A[ic, jc, local_indices...] = original_a[ic, jc, global_indices...]
            end
        end
    end
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        for jc in 1:nc
            for ic in 1:nc
                A_reference[ic, jc, global_indices...] =
                    original_a[ic, jc, global_indices...]
            end
        end
    end
    Gaugefields.set_wing_U!(A)

    shifted_a = Gaugefields.shift_U(A, source_shift)
    shifted_a_reference = Gaugefields.shift_U(A_reference, source_shift)
    operand_specs = (
        (A, A_reference),
        (shifted_a, shifted_a_reference),
        (A', A_reference'),
        (shifted_a', shifted_a_reference'),
    )

    for (operand, reference_operand) in operand_specs
        for target_even in (true, false)
            for α in (1.0, -0.5)
                C = similar(A)
                C_reference = similar(A_reference)
                for site in CartesianIndices(C.U.PN)
                    local_indices = Tuple(site)
                    for jc in 1:nc
                        for ic in 1:nc
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

                if α == 1
                    Gaugefields.add_U!(C, operand, target_even)
                    Gaugefields.add_U!(C_reference, reference_operand, target_even)
                else
                    Gaugefields.add_U!(C, α, operand, target_even)
                    scaled_reference_operand = similar(A_reference)
                    for site in CartesianIndices(global_size)
                        global_indices = Tuple(site)
                        for jc in 1:nc
                            for ic in 1:nc
                                scaled_reference_operand[ic, jc, global_indices...] =
                                    α * reference_operand[ic, jc, global_indices...]
                            end
                        end
                    end
                    Gaugefields.add_U!(
                        C_reference,
                        scaled_reference_operand,
                        target_even,
                    )
                end

                for site in CartesianIndices(C.U.PN)
                    local_indices = Tuple(site)
                    global_indices = ntuple(
                        d -> C.U.coords[d] * C.U.PN[d] + local_indices[d],
                        4,
                    )

                    for jc in 1:nc
                        for ic in 1:nc
                            @test C[ic, jc, local_indices...] ==
                                  C_reference[ic, jc, global_indices...]
                        end
                    end
                end
            end
        end
    end
end
