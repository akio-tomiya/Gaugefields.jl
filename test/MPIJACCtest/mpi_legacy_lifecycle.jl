import JACC
JACC.@init_backend

using MPI
using Test

@test !MPI.Initialized()

using Gaugefields

@testset "Deprecated MPI constructor lifecycle" begin
    @test Base.get_extension(Gaugefields, :GaugefieldsMPIExt) !== nothing

    field = Gaugefields.AbstractGaugefields_module.Gaugefields_4D_nowing_mpi(
        2,
        2,
        2,
        2,
        2,
        (1, 1, 1, 1);
        mpiinit=false,
        verbose_level=0,
    )

    @test MPI.Initialized()
    @test field.mpi
end
