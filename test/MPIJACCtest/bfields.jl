import JACC
JACC.@init_backend

using Gaugefields
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices B-field workflow" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nc = 2
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    flux = [1, 0, 1, 0, 1, 0]

    U = Initialize_Gaugefields(
        nc,
        1,
        global_size...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    B = Initialize_Bfields(
        nc,
        flux,
        1,
        global_size...;
        condition="tflux",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )

    cold_U = Initialize_Gaugefields(
        nc,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    plaquette_loops = make_loops_fromname("plaquette"; Dim=4)
    append!(plaquette_loops, plaquette_loops')
    physical_action = GaugeAction(cold_U, B)
    push!(physical_action, 1.0, plaquette_loops)

    measured_action = real(evaluate_GaugeAction(physical_action, cold_U, B))
    planes = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    volume = prod(global_size)
    expected_action = 2nc * (
        6volume - sum(
            volume ÷ (global_size[mu] * global_size[nu]) *
            (1 - real(exp(-2pi * im * flux[flux_index] / nc)))
            for (flux_index, (mu, nu)) in enumerate(planes)
        )
    )
    @test measured_action ≈ expected_action atol=2e-10 rtol=2e-12

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
