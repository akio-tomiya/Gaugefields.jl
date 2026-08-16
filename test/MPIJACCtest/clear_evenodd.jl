import JACC
JACC.@init_backend

using Gaugefields
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "MPILattice checkerboard clear" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (3 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)

    for target_even in (true, false)
        U = Initialize_Gaugefields(
            2,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        field = U[1]
        U_reference = Initialize_Gaugefields(
            2,
            0,
            global_size...;
            condition="cold",
            verbose_level=0,
        )
        field_reference = U_reference[1]

        Gaugefields.clear_U!(field, target_even)
        Gaugefields.clear_U!(field_reference, target_even)

        for site in CartesianIndices(field.U.PN)
            local_indices = Tuple(site)
            global_indices = ntuple(
                d -> field.U.coords[d] * field.U.PN[d] + local_indices[d],
                4,
            )

            for jc in 1:field.NC
                for ic in 1:field.NC
                    @test field[ic, jc, local_indices...] ==
                          field_reference[ic, jc, global_indices...]
                end
            end
        end
    end
end
