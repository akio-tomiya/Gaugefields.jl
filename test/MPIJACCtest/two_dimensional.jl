import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "2D LatticeMatrices with MPI" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1)
    NX = 2 * nprocs
    NY = 4

    U = Initialize_Gaugefields(
        2,
        1,
        NX,
        NY;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )

    flow = Gradientflow(U; eps=0.01)
    flow!(U, flow)

    network = CovNeuralnet(U)
    push!(network, STOUT_Layer(["plaquette"], [0.1], U))
    smeared, history, derivative = calc_smearedU(U, network)
    @test length(history) == 1
    @test derivative === nothing

    updater = Heatbath(U, 2.0; seed=17)
    heatbath!(U, updater)
    overrelaxation!(U, updater)
    @test updater.sweep == 1
    @test updater.overrelaxation_sweep == 1

    plaquette = calculate_Plaquette(
        smeared,
        similar(smeared[1]),
        similar(smeared[1]),
    )
    @test isfinite(real(plaquette))

    legacy = Oneinstanton(2, 0, NX, NY; verbose_level=0)
    instanton = Oneinstanton(
        2,
        1,
        NX,
        NY;
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    instanton_host = similar(legacy)
    substitute_U!(instanton_host, instanton)
    @test maximum(
        maximum(abs.(instanton_host[mu].U .- legacy[mu].U)) for mu = 1:2
    ) < 1e-12
end
