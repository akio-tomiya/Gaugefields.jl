import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices lambda_k_mul!" begin
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)

    for NC in (2, 3, 4)
        U = Initialize_Gaugefields(
            NC,
            1,
            global_size...;
            condition="hot",
            randomnumber="Reproducible",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )[1]
        output = similar(U)
        generator = Gaugefields.Generator(NC)

        for k in (1, length(generator))
            Gaugefields.lambda_k_mul!(output, U, k, generator)
            original_global = gather_and_bcast_matrix(U.U)
            output_global = gather_and_bcast_matrix(output.U)
            expected = similar(original_global)
            matrix = generator.generator[k] / 2
            for site in CartesianIndices(global_size)
                coordinates = Tuple(site)
                expected[:, :, coordinates...] =
                    matrix * original_global[:, :, coordinates...]
            end
            @test maximum(abs, output_global .- expected) < 1e-12
        end

        factor = 0.25im
        Gaugefields.AbstractGaugefields_module.Antihermitian!(
            output, U; factor,
        )
        original_global = gather_and_bcast_matrix(U.U)
        output_global = gather_and_bcast_matrix(output.U)
        expected = similar(original_global)
        for site in CartesianIndices(global_size)
            coordinates = Tuple(site)
            matrix = original_global[:, :, coordinates...]
            expected[:, :, coordinates...] = factor * (matrix - matrix')
        end
        @test maximum(abs, output_global .- expected) < 1e-12
    end
end
