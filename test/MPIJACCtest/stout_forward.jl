import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "4D LatticeMatrices stout forward with MPI" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    U = Initialize_Gaugefields(
        3,
        1,
        2 * nprocs,
        2,
        2,
        2;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )

    momentum = initialize_TA_Gaugefields(U[1])
    momentum_copy = similar(momentum)
    @test momentum_copy.a.dims == (nprocs, 1, 1, 1)
    @test momentum_copy.a.phases == momentum.a.phases
    @test momentum_copy.a.comm == momentum.a.comm

    nn = CovNeuralnet(U)
    push!(nn, STOUT_Layer(["plaquette"], [0.1], U))
    smeared, history, derivative = calc_smearedU(U, nn)
    plaquette = calculate_Plaquette(
        smeared,
        similar(smeared[1]),
        similar(smeared[1]),
    )

    @test length(history) == 1
    @test derivative === nothing
    @test isfinite(real(plaquette))

    cotangent = similar(U)
    substitute_U!(cotangent, U)
    gradient = back_prop(cotangent, nn, history, U)
    global_gradient = gather_and_bcast_matrix.(getproperty.(gradient, :U))
    @test all(field -> all(isfinite, field), global_gradient)
end
