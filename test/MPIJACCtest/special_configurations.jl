import JACC
JACC.@init_backend

using Gaugefields
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices special configurations" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)

    instanton = Oneinstanton(
        2,
        1,
        global_size...;
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    plaquette = calculate_Plaquette(
        instanton,
        similar(instanton[1]),
        similar(instanton[1]),
    )
    @test isfinite(real(plaquette))
    @test isfinite(imag(plaquette))
    if nprocs > 1
        @test_throws ArgumentError topological_charge(instanton)
        @test_throws ArgumentError topological_charge_density(instanton)
    end

    embedded = Oneinstanton_SUN_embedded(
        4,
        global_size...;
        NDW=1,
        isMPILattice=true,
        PEs=process_grid,
        elementtype=ComplexF32,
        verbose_level=0,
    )
    @test eltype(embedded[1]) == ComplexF32
    embedded_plaquette = calculate_Plaquette(
        embedded,
        similar(embedded[1]),
        similar(embedded[1]),
    )
    @test isfinite(real(embedded_plaquette))
    @test isfinite(imag(embedded_plaquette))
end
