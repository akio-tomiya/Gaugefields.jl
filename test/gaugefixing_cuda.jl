# Optional direct-CUDA regression test.  Run this file from an environment
# containing Gaugefields, LatticeMatrices, JACC, MPI, Test, and CUDA.  It is
# intentionally separate from runtests.jl because CUDA is not a hard package
# dependency and CI hosts need not have an NVIDIA GPU.
import JACC
JACC.@init_backend

using CUDA
using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()
CUDA.device!(parse(Int, get(ENV, "CUDA_TEST_DEVICE", "0")))

const GF = Gaugefields.AbstractGaugefields_module
const GFix = Gaugefields.AbstractGaugefixing_module

function direct_cuda_gaugefix_case(D_fix)
    dims = (4, 4, 4, 4)
    U_lm = gauge_configuration(
        dims;
        colors=3,
        start=:hot,
        seed=UInt64(0x155 + D_fix),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U_serial = Initialize_Gaugefields(
        3, 0, dims...;
        condition="cold",
        isMPILattice=false,
        verbose_level=0,
    )
    substitute_U!(U_serial, U_lm)

    constructor = () -> GF.identityGaugefields_4D_accelerator(
        3,
        dims...,
        (2, 2, 2, 2);
        verbose_level=0,
        accelerator="cuda",
    )
    U_cuda = [constructor() for _ in 1:4]
    substitute_U!(U_cuda, U_serial)

    g_lm = similar(U_lm[1])
    g_cuda = similar(U_cuda[1])
    temps_lm = [similar(U_lm[1]) for _ in 1:6]
    temps_cuda = [similar(U_cuda[1]) for _ in 1:6]

    @test GFix.gaugefixing_backend_supported(g_cuda)
    gaugefixing!(
        U_lm, g_lm, 1.5, 2, 1.5, 2, 0.0, 155, temps_lm; D_fix,
    )
    gaugefixing!(
        U_cuda, g_cuda, 1.5, 2, 1.5, 2, 0.0, 155, temps_cuda; D_fix,
    )
    CUDA.synchronize()

    U_cuda_host = Initialize_Gaugefields(
        3, 0, dims...;
        condition="cold",
        isMPILattice=false,
        verbose_level=0,
    )
    g_cuda_host = similar(U_cuda_host[1])
    substitute_U!(U_cuda_host, U_cuda)
    substitute_U!(g_cuda_host, g_cuda)

    for mu in eachindex(U_lm)
        @test isapprox(
            gather_and_bcast_matrix(U_lm[mu].U), U_cuda_host[mu].U;
            rtol=5e-12, atol=5e-12,
        )
    end
    @test isapprox(
        gather_and_bcast_matrix(g_lm.U), g_cuda_host.U;
        rtol=5e-12, atol=5e-12,
    )

    g_host = Array(g_cuda.U)
    for r in axes(g_host, 4), b in axes(g_host, 3)
        matrix = @view g_host[:, :, b, r]
        @test matrix * matrix' ≈ I rtol=5e-12 atol=5e-12
        @test det(matrix) ≈ 1 rtol=5e-12 atol=5e-12
    end
end

@testset "direct CUDA gauge fixing on $(CUDA.name(CUDA.device()))" begin
    direct_cuda_gaugefix_case(3)
    direct_cuda_gaugefix_case(4)
end
