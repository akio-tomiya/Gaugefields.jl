import JACC
JACC.@init_backend

using Gaugefields
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices B-field workflow" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    flux = [1, 0, 1, 0, 1, 0]

    U = Initialize_Gaugefields(
        2,
        1,
        global_size...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    B = Initialize_Bfields(
        2,
        flux,
        1,
        global_size...;
        condition="tflux",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )

    Bcopy = similar(B)
    substitute_U!(Bcopy, B)
    substitute_U!(Bcopy, B, false)
    @test typeof(Bcopy[1, 2]) == typeof(B[1, 2])

    plaquettes = make_loops_fromname("plaquette"; Dim=4)
    flow = Gradientflow_general_Bfields(
        U,
        B,
        [plaquettes],
        [1 + 0im];
        eps=0.01,
    )
    flow!(U, B, flow)
    plaquette = calculate_Plaquette(U, similar(U[1]), similar(U[1]))
    @test isfinite(real(plaquette))
    @test isfinite(imag(plaquette))
end
