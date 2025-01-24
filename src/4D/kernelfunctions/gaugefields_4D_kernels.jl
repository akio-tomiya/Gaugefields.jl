function kernel_identityGaugefields!(U, NC,b,r)
    @inbounds for ic = 1:NC
        U[ic, ic, b, r] = 1
    end
end

function kernel_randomGaugefields!(U, NC,b,r)
    @inbounds for ic = 1:NC
        for jc = 1:NC
            U[jc, ic, b, r] = rand() - 0.5 + im * (rand() - 0.5)
        end
    end
end

function kernel_normalize_U_NC2!(u,b,r)
    α = u[1, 1, b, r]
    β = u[2, 1, b, r]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, b, r] = α / detU
    u[2, 1, b, r] = β / detU
    u[1, 2, b, r] = -conj(β) / detU
    u[2, 2, b, r] = conj(α) / detU
    return
end

function kernel_normalize_U_NC3!(u,b,r)
    #b = Int64(CUDA.threadIdx().x)
    #r = Int64(CUDA.blockIdx().x)
    w1 = 0
    w2 = 0
    @inbounds for ic = 1:3
        w1 += u[2, ic, b, r] * conj(u[1, ic, b, r])
        w2 += u[1, ic, b, r] * conj(u[1, ic, b, r])
    end
    zerock2 = w2
    w1 = -w1 / w2

    x4 = (u[2, 1, b, r]) + w1 * u[1, 1, b, r]
    x5 = (u[2, 2, b, r]) + w1 * u[1, 2, b, r]
    x6 = (u[2, 3, b, r]) + w1 * u[1, 3, b, r]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3

    u[2, 1, b, r] = x4
    u[2, 2, b, r] = x5
    u[2, 3, b, r] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1, b, r] = u[1, 1, b, r] * w2
    u[1, 2, b, r] = u[1, 2, b, r] * w2
    u[1, 3, b, r] = u[1, 3, b, r] * w2
    u[2, 1, b, r] = u[2, 1, b, r] * w3
    u[2, 2, b, r] = u[2, 2, b, r] * w3
    u[2, 3, b, r] = u[2, 3, b, r] * w3

    aa1 = real(u[1, 1, b, r])
    aa2 = imag(u[1, 1, b, r])
    aa3 = real(u[1, 2, b, r])
    aa4 = imag(u[1, 2, b, r])
    aa5 = real(u[1, 3, b, r])
    aa6 = imag(u[1, 3, b, r])
    aa7 = real(u[2, 1, b, r])
    aa8 = imag(u[2, 1, b, r])
    aa9 = real(u[2, 2, b, r])
    aa10 = imag(u[2, 2, b, r])
    aa11 = real(u[2, 3, b, r])
    aa12 = imag(u[2, 3, b, r])

    aa13 =
        aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
    aa14 =
        aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
    aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
    aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
    aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
    aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

    u[3, 1, b, r] = aa13 + im * aa14
    u[3, 2, b, r] = aa15 + im * aa16
    u[3, 3, b, r] = aa17 + im * aa18

    return

end


function kernel_mul_NC!(C, A, B, NC,b,r)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = 0
            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    A[k1, k3, b, r] * B[k3, k2, b, r]
            end
        end
    end
end

function kernel_mul_NC!(C, A, B, α, β, NC,b,r)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * B[k3, k2, b, r]
            end
        end
    end
end

function kernel_mul_NC3!(C, A, B,b,r)
    a11 = A[1, 1, b, r]
    a21 = A[2, 1, b, r]
    a31 = A[3, 1, b, r]
    a12 = A[1, 2, b, r]
    a22 = A[2, 2, b, r]
    a32 = A[3, 2, b, r]
    a13 = A[1, 3, b, r]
    a23 = A[2, 3, b, r]
    a33 = A[3, 3, b, r]
    b11 = B[1, 1, b, r]
    b21 = B[2, 1, b, r]
    b31 = B[3, 1, b, r]
    b12 = B[1, 2, b, r]
    b22 = B[2, 2, b, r]
    b32 = B[3, 2, b, r]
    b13 = B[1, 3, b, r]
    b23 = B[2, 3, b, r]
    b33 = B[3, 3, b, r]
    C[1, 1, b, r] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, b, r] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, b, r] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, b, r] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, b, r] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, b, r] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, b, r] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, b, r] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, b, r] = a31 * b13 + a32 * b23 + a33 * b33

    return

end

function kernel_mul_NC3!(C, A, B, α, β,b,r)
    a11 = α * A[1, 1, b, r]
    a21 = α * A[2, 1, b, r]
    a31 = α * A[3, 1, b, r]
    a12 = α * A[1, 2, b, r]
    a22 = α * A[2, 2, b, r]
    a32 = α * A[3, 2, b, r]
    a13 = α * A[1, 3, b, r]
    a23 = α * A[2, 3, b, r]
    a33 = α * A[3, 3, b, r]
    b11 = B[1, 1, b, r]
    b21 = B[2, 1, b, r]
    b31 = B[3, 1, b, r]
    b12 = B[1, 2, b, r]
    b22 = B[2, 2, b, r]
    b32 = B[3, 2, b, r]
    b13 = B[1, 3, b, r]
    b23 = B[2, 3, b, r]
    b33 = B[3, 3, b, r]
    C[1, 1, b, r] = β * C[1, 1, b, r] + a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, b, r] = β * C[2, 1, b, r] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, b, r] = β * C[3, 1, b, r] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, b, r] = β * C[1, 2, b, r] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, b, r] = β * C[2, 2, b, r] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, b, r] = β * C[3, 2, b, r] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, b, r] = β * C[1, 3, b, r] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, b, r] = β * C[2, 3, b, r] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, b, r] = β * C[3, 3, b, r] + a31 * b13 + a32 * b23 + a33 * b33

    return

end
