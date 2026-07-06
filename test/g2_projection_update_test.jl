using Gaugefields
using LinearAlgebra
using Test

function _site_matrix(U, ix, iy, iz, it)
    return [U[i, j, ix, iy, iz, it] for i in 1:7, j in 1:7]
end

@testset "G2 projection to momentum" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    raw = similar(U)
    clear_U!(raw)

    for j in 1:7
        for i in 1:7
            raw[i, j, 1, 1, 1, 1] = sin(0.17 * i + 0.31 * j) + 0.2im * cos(0.13 * i * j)
        end
    end
    set_wing_U!(raw)

    p = initialize_TA_Gaugefields(U)
    Traceless_antihermitian!(p, raw)
    expected = project_to_g2_coefficients(_site_matrix(raw, 1, 1, 1, 1))

    @test p isa G2TA_Gaugefields_4D_serial
    @test norm([p[k, 1, 1, 1, 1] for k in 1:14] - expected) < 1.0e-12

    q = similar(p)
    Traceless_antihermitian_add!(q, 2.0, raw)
    @test norm([q[k, 1, 1, 1, 1] for k in 1:14] - 2.0 .* expected) < 1.0e-12
end

@testset "G2 exptU" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    p = initialize_TA_Gaugefields(U)
    coeffs = [0.03 * sin(0.2 * k) for k in 1:14]
    for k in 1:14
        p[k, 1, 1, 1, 1] = coeffs[k]
    end

    expU = similar(U)
    temp1 = similar(U)
    temp2 = similar(U)
    exptU!(expU, 0.7, p, [temp1, temp2])

    expected = exp(0.7 .* g2_matrix(coeffs))
    @test norm(_site_matrix(expU, 1, 1, 1, 1) - expected) < 1.0e-12
    @test is_g2_link(_site_matrix(expU, 1, 1, 1, 1); atol = 1.0e-10)

    for it in 1:2
        for iz in 1:2
            for iy in 1:2
                for ix in 1:2
                    if (ix, iy, iz, it) != (1, 1, 1, 1)
                        @test _site_matrix(expU, ix, iy, iz, it) == Matrix{ComplexF64}(I, 7, 7)
                    end
                end
            end
        end
    end
end

@testset "G2 update times link" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    p = initialize_TA_Gaugefields(U)
    for k in 1:14
        p[k, 1, 1, 1, 1] = 0.02 * cos(0.1 * k)
    end

    expU = similar(U)
    W = similar(U)
    exptU!(expU, 1.0, p, [similar(U), similar(U)])
    mul!(W, expU, U)

    @test is_g2_link(_site_matrix(W, 1, 1, 1, 1); atol = 1.0e-10)
    @test norm(_site_matrix(W, 1, 1, 1, 1) - _site_matrix(expU, 1, 1, 1, 1)) < 1.0e-12
end
