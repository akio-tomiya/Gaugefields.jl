# Optional direct-CUDA regression test for the legacy STOUT compatibility API.
# Run this file from an environment containing Gaugefields, CUDA, and Test.
# It is separate from runtests.jl because CUDA is not a hard dependency and
# regular CI hosts need not have an NVIDIA GPU.

using CUDA
using Gaugefields
using LinearAlgebra
using Random
using Test

CUDA.device!(parse(Int, get(ENV, "CUDA_TEST_DEVICE", "0")))

const GF = Gaugefields.AbstractGaugefields_module
const Smearing = Gaugefields.Abstractsmearing_module

function accelerator_field(accelerator, ::Type{T}) where {T<:Complex}
    return GF.identityGaugefields_4D_accelerator(
        3,
        2,
        2,
        2,
        2,
        (2, 2, 2, 2);
        accelerator,
        singleprecision=T === ComplexF32,
        verbose_level=0,
    )
end

function traceless_antihermitian_data(rng, ::Type{T}, dims; scale) where {T<:Complex}
    data = zeros(T, dims)
    for r in axes(data, 4), b in axes(data, 3)
        matrix = scale * (randn(rng, T, 3, 3) - randn(rng, T, 3, 3)')
        matrix = (matrix - matrix') / 2
        trace_part = tr(matrix) / 3
        for i in axes(matrix, 1)
            matrix[i, i] -= trace_part
        end
        data[:, :, b, r] .= matrix
    end
    return data
end

@testset "direct CUDA STOUT compatibility on $(CUDA.name(CUDA.device()))" begin
    @test CUDA.functional()

    for T in (ComplexF64, ComplexF32)
        @testset "$T" begin
            cpu = () -> accelerator_field("none", T)
            gpu = () -> accelerator_field("cuda", T)

            u_cpu = cpu()
            delta_cpu = cpu()
            Q_cpu = cpu()
            pullback_cpu = cpu()
            lambda_cpu = cpu()

            u_gpu = gpu()
            delta_gpu = gpu()
            Q_gpu = gpu()
            pullback_gpu = gpu()
            lambda_gpu = gpu()

            rng = MersenneTwister(0x167)
            delta_data = randn(rng, T, size(delta_cpu.U))
            delta_cpu.U .= delta_data
            copyto!(delta_gpu.U, delta_data)

            tolerance = T === ComplexF32 ? 5e-5 : 5e-12

            # At Q = 0 the Frechet derivative of exp is the identity map.
            clear_U!(Q_cpu)
            clear_U!(Q_gpu)
            Smearing.CdexpQdQ!(pullback_cpu, delta_cpu, Q_cpu)
            Smearing.CdexpQdQ!(pullback_gpu, delta_gpu, Q_gpu)
            CUDA.synchronize()
            @test Array(pullback_gpu.U) ≈ pullback_cpu.U rtol=tolerance atol=tolerance
            @test Array(pullback_gpu.U) ≈ delta_data rtol=tolerance atol=tolerance

            construct_Λmatrix_forSTOUT!(lambda_cpu, delta_cpu, Q_cpu, u_cpu)
            construct_Λmatrix_forSTOUT!(lambda_gpu, delta_gpu, Q_gpu, u_gpu)
            CUDA.synchronize()
            @test Array(lambda_gpu.U) ≈ lambda_cpu.U rtol=tolerance atol=tolerance

            # Exercise the full nonzero-Q coefficient and pullback path.
            Q_data = traceless_antihermitian_data(
                rng, T, size(Q_cpu.U); scale=convert(real(T), 0.05),
            )
            Q_cpu.U .= Q_data
            copyto!(Q_gpu.U, Q_data)

            Smearing.CdexpQdQ!(pullback_cpu, delta_cpu, Q_cpu)
            Smearing.CdexpQdQ!(pullback_gpu, delta_gpu, Q_gpu)
            CUDA.synchronize()
            pullback_gpu_host = Array(pullback_gpu.U)
            @test all(isfinite, pullback_gpu_host)
            @test pullback_gpu_host ≈ pullback_cpu.U rtol=tolerance atol=tolerance

            construct_Λmatrix_forSTOUT!(lambda_cpu, delta_cpu, Q_cpu, u_cpu)
            construct_Λmatrix_forSTOUT!(lambda_gpu, delta_gpu, Q_gpu, u_gpu)
            CUDA.synchronize()
            lambda_gpu_host = Array(lambda_gpu.U)
            @test all(isfinite, lambda_gpu_host)
            @test lambda_gpu_host ≈ lambda_cpu.U rtol=tolerance atol=tolerance
        end
    end
end
