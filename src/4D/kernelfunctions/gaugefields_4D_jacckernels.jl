

function jacckernel_identityGaugefields!(i, U, NC)
    @inbounds for ic = 1:NC
        U[ic, ic, i] = 1
    end
end

function jacckernel_randomGaugefields!(i, U, NC)
    @inbounds for ic = 1:NC
        for jc = 1:NC
            U[jc, ic, i] = rand() - 0.5 + im * (rand() - 0.5)
        end
    end
end

function jacckernel_normalize_U_NC2!(i, u)
    α = u[1, 1, i]
    β = u[2, 1, i]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, i] = α / detU
    u[2, 1, i] = β / detU
    u[1, 2, i] = -conj(β) / detU
    u[2, 2, i] = conj(α) / detU
    return
end

function jacckernel_normalize_U_NC3!(i, u)
    #b = Int64(CUDA.threadIdx().x)
    #r = Int64(CUDA.blockIdx().x)
    w1 = 0
    w2 = 0
    @inbounds for ic = 1:3
        w1 += u[2, ic, i] * conj(u[1, ic, i])
        w2 += u[1, ic, i] * conj(u[1, ic, i])
    end
    zerock2 = w2
    w1 = -w1 / w2

    x4 = (u[2, 1, i]) + w1 * u[1, 1, i]
    x5 = (u[2, 2, i]) + w1 * u[1, 2, i]
    x6 = (u[2, 3, i]) + w1 * u[1, 3, i]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3

    u[2, 1, i] = x4
    u[2, 2, i] = x5
    u[2, 3, i] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1, i] = u[1, 1, i] * w2
    u[1, 2, i] = u[1, 2, i] * w2
    u[1, 3, i] = u[1, 3, i] * w2
    u[2, 1, i] = u[2, 1, i] * w3
    u[2, 2, i] = u[2, 2, i] * w3
    u[2, 3, i] = u[2, 3, i] * w3

    aa1 = real(u[1, 1, i])
    aa2 = imag(u[1, 1, i])
    aa3 = real(u[1, 2, i])
    aa4 = imag(u[1, 2, i])
    aa5 = real(u[1, 3, i])
    aa6 = imag(u[1, 3, i])
    aa7 = real(u[2, 1, i])
    aa8 = imag(u[2, 1, i])
    aa9 = real(u[2, 2, i])
    aa10 = imag(u[2, 2, i])
    aa11 = real(u[2, 3, i])
    aa12 = imag(u[2, 3, i])

    aa13 =
        aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
    aa14 =
        aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
    aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
    aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
    aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
    aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

    u[3, 1, i] = aa13 + im * aa14
    u[3, 2, i] = aa15 + im * aa16
    u[3, 3, i] = aa17 + im * aa18

    return

end

function jacckernel_normalize_U_NC!(i, u, A, NC)
    aa = view(A, :, :, i)
    gramschmidt!(aa)
    u[:, :, i] .= aa#A[:, :]
    return
end


function jacckernel_mul_NC!(i, C, A, B, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = 0
            for k3 = 1:NC
                C[k1, k2, i] +=
                    A[k1, k3, i] * B[k3, k2, i]
            end
        end
    end
end

function jacckernel_mul_NC!(i, C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, i] * B[k3, k2, i]
            end
        end
    end
end

@inbounds function jacckernel_mul_NC3!(i, C, A, B)
    a11 = A[1, 1, i]
    a21 = A[2, 1, i]
    a31 = A[3, 1, i]
    a12 = A[1, 2, i]
    a22 = A[2, 2, i]
    a32 = A[3, 2, i]
    a13 = A[1, 3, i]
    a23 = A[2, 3, i]
    a33 = A[3, 3, i]
    b11 = B[1, 1, i]
    b21 = B[2, 1, i]
    b31 = B[3, 1, i]
    b12 = B[1, 2, i]
    b22 = B[2, 2, i]
    b32 = B[3, 2, i]
    b13 = B[1, 3, i]
    b23 = B[2, 3, i]
    b33 = B[3, 3, i]
    C[1, 1, i] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = a31 * b13 + a32 * b23 + a33 * b33

    return

end

@inbounds function jacckernel_mul_NC3!(i, C, A, B, α, β)
    a11 = α * A[1, 1, i]
    a21 = α * A[2, 1, i]
    a31 = α * A[3, 1, i]
    a12 = α * A[1, 2, i]
    a22 = α * A[2, 2, i]
    a32 = α * A[3, 2, i]
    a13 = α * A[1, 3, i]
    a23 = α * A[2, 3, i]
    a33 = α * A[3, 3, i]
    b11 = B[1, 1, i]
    b21 = B[2, 1, i]
    b31 = B[3, 1, i]
    b12 = B[1, 2, i]
    b22 = B[2, 2, i]
    b32 = B[3, 2, i]
    b13 = B[1, 3, i]
    b23 = B[2, 3, i]
    b33 = B[3, 3, i]
    C[1, 1, i] = β * C[1, 1, i] + a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = β * C[2, 1, i] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = β * C[3, 1, i] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = β * C[1, 2, i] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = β * C[2, 2, i] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = β * C[3, 2, i] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = β * C[1, 3, i] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = β * C[2, 3, i] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = β * C[3, 3, i] + a31 * b13 + a32 * b23 + a33 * b33

    return

end


@inbounds function jacckernel_mul_NC3_abdag!(i, C, A, B, α, β)
    a11 = α * A[1, 1, i]
    a21 = α * A[2, 1, i]
    a31 = α * A[3, 1, i]
    a12 = α * A[1, 2, i]
    a22 = α * A[2, 2, i]
    a32 = α * A[3, 2, i]
    a13 = α * A[1, 3, i]
    a23 = α * A[2, 3, i]
    a33 = α * A[3, 3, i]
    b11 = conj(B[1, 1, i])
    b21 = conj(B[1, 2, i])
    b31 = conj(B[1, 3, i])
    b12 = conj(B[2, 1, i])
    b22 = conj(B[2, 2, i])
    b32 = conj(B[2, 3, i])
    b13 = conj(B[3, 1, i])
    b23 = conj(B[3, 2, i])
    b33 = conj(B[3, 3, i])
    C[1, 1, i] = β * C[1, 1, i] + a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = β * C[2, 1, i] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = β * C[3, 1, i] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = β * C[1, 2, i] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = β * C[2, 2, i] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = β * C[3, 2, i] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = β * C[1, 3, i] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = β * C[2, 3, i] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = β * C[3, 3, i] + a31 * b13 + a32 * b23 + a33 * b33

    return

end

function jacckernel_mul_NC3_abdag!(i, C, A, B)
    a11 = A[1, 1, i]
    a21 = A[2, 1, i]
    a31 = A[3, 1, i]
    a12 = A[1, 2, i]
    a22 = A[2, 2, i]
    a32 = A[3, 2, i]
    a13 = A[1, 3, i]
    a23 = A[2, 3, i]
    a33 = A[3, 3, i]
    b11 = conj(B[1, 1, i])
    b21 = conj(B[1, 2, i])
    b31 = conj(B[1, 3, i])
    b12 = conj(B[2, 1, i])
    b22 = conj(B[2, 2, i])
    b32 = conj(B[2, 3, i])
    b13 = conj(B[3, 1, i])
    b23 = conj(B[3, 2, i])
    b33 = conj(B[3, 3, i])
    C[1, 1, i] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = a31 * b13 + a32 * b23 + a33 * b33

    return

end


function jacckernel_mul_NC_abdag!(i, C, A, B, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = 0

            for k3 = 1:NC
                C[k1, k2, i] +=
                    A[k1, k3, i] * conj(B[k2, k3, i])
            end
        end
    end
end

function jacckernel_mul_NC_abdag!(i, C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, i] * conj(B[k2, k3, i])
            end
        end
    end
end

@inbounds function jacckernel_mul_NC3_adagbdag!(i, C, A, B, α, β)
    a11 = α * conj(A[1, 1, i])
    a21 = α * conj(A[1, 2, i])
    a31 = α * conj(A[1, 3, i])
    a12 = α * conj(A[2, 1, i])
    a22 = α * conj(A[2, 2, i])
    a32 = α * conj(A[2, 3, i])
    a13 = α * conj(A[3, 1, i])
    a23 = α * conj(A[3, 2, i])
    a33 = α * conj(A[3, 3, i])
    b11 = conj(B[1, 1, i])
    b21 = conj(B[1, 2, i])
    b31 = conj(B[1, 3, i])
    b12 = conj(B[2, 1, i])
    b22 = conj(B[2, 2, i])
    b32 = conj(B[2, 3, i])
    b13 = conj(B[3, 1, i])
    b23 = conj(B[3, 2, i])
    b33 = conj(B[3, 3, i])
    C[1, 1, i] = β * C[1, 1, i] + a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = β * C[2, 1, i] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = β * C[3, 1, i] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = β * C[1, 2, i] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = β * C[2, 2, i] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = β * C[3, 2, i] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = β * C[1, 3, i] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = β * C[2, 3, i] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = β * C[3, 3, i] + a31 * b13 + a32 * b23 + a33 * b33

    return

end

@inbounds function jacckernel_mul_NC3_adagbdag!(i, C, A, B)
    a11 = conj(A[1, 1, i])
    a21 = conj(A[1, 2, i])
    a31 = conj(A[1, 3, i])
    a12 = conj(A[2, 1, i])
    a22 = conj(A[2, 2, i])
    a32 = conj(A[2, 3, i])
    a13 = conj(A[3, 1, i])
    a23 = conj(A[3, 2, i])
    a33 = conj(A[3, 3, i])
    b11 = conj(B[1, 1, i])
    b21 = conj(B[1, 2, i])
    b31 = conj(B[1, 3, i])
    b12 = conj(B[2, 1, i])
    b22 = conj(B[2, 2, i])
    b32 = conj(B[2, 3, i])
    b13 = conj(B[3, 1, i])
    b23 = conj(B[3, 2, i])
    b33 = conj(B[3, 3, i])
    C[1, 1, i] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = a31 * b13 + a32 * b23 + a33 * b33

    return

end

function jacckernel_mul_NC_adagbdag!(i, C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, i]) * conj(B[k2, k3, i])
            end
        end
    end
end

function jacckernel_mul_NC_adagbdag!(i, C, A, B, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = 0

            for k3 = 1:NC
                C[k1, k2, i] +=
                    conj(A[k3, k1, i]) * conj(B[k2, k3, i])
            end
        end
    end
end

@inbounds function jacckernel_mul_NC3_adagb!(i, C, A, B, α, β)
    a11 = α * conj(A[1, 1, i])
    a21 = α * conj(A[1, 2, i])
    a31 = α * conj(A[1, 3, i])
    a12 = α * conj(A[2, 1, i])
    a22 = α * conj(A[2, 2, i])
    a32 = α * conj(A[2, 3, i])
    a13 = α * conj(A[3, 1, i])
    a23 = α * conj(A[3, 2, i])
    a33 = α * conj(A[3, 3, i])
    b11 = B[1, 1, i]
    b21 = B[2, 1, i]
    b31 = B[3, 1, i]
    b12 = B[1, 2, i]
    b22 = B[2, 2, i]
    b32 = B[3, 2, i]
    b13 = B[1, 3, i]
    b23 = B[2, 3, i]
    b33 = B[3, 3, i]
    C[1, 1, i] = β * C[1, 1, i] + a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = β * C[2, 1, i] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = β * C[3, 1, i] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = β * C[1, 2, i] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = β * C[2, 2, i] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = β * C[3, 2, i] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = β * C[1, 3, i] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = β * C[2, 3, i] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = β * C[3, 3, i] + a31 * b13 + a32 * b23 + a33 * b33

    return

end

@inbounds function jacckernel_mul_NC3_adagb!(i, C, A, B)
    a11 = conj(A[1, 1, i])
    a21 = conj(A[1, 2, i])
    a31 = conj(A[1, 3, i])
    a12 = conj(A[2, 1, i])
    a22 = conj(A[2, 2, i])
    a32 = conj(A[2, 3, i])
    a13 = conj(A[3, 1, i])
    a23 = conj(A[3, 2, i])
    a33 = conj(A[3, 3, i])
    b11 = B[1, 1, i]
    b21 = B[2, 1, i]
    b31 = B[3, 1, i]
    b12 = B[1, 2, i]
    b22 = B[2, 2, i]
    b32 = B[3, 2, i]
    b13 = B[1, 3, i]
    b23 = B[2, 3, i]
    b33 = B[3, 3, i]
    C[1, 1, i] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, i] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, i] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, i] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, i] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, i] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, i] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, i] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, i] = a31 * b13 + a32 * b23 + a33 * b33

    return

end


function jacckernel_mul_NC_adagb!(i, C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, i]) * B[k3, k2, i]
            end
        end
    end
end

function jacckernel_mul_NC_adagb!(i, C, A, B, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = zero(eltype(C))

            for k3 = 1:NC
                C[k1, k2, i] +=
                    conj(A[k3, k1, i]) * B[k3, k2, i]
            end
        end
    end
end


function jacckernel_mul_NC_abshift!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, i] * B[k3, k2, bshifted, rshifted]
            end
        end
    end
end


function jacckernel_mul_NC_ashiftb!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, bshifted, rshifted] * B[k3, k2, i]
            end
        end
    end
end


function jacckernel_mul_NC_ashiftbshift!(i, C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(i, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(i, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, bshifted_a, rshifted_a] * B[k3, k2, bshifted_b, rshifted_b]
            end
        end
    end
end

function jacckernel_mul_NC_ashiftbshiftdag!(i, C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(i, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(i, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, bshifted_a, rshifted_a] * conj(B[k2, k3, bshifted_b, rshifted_b])
            end
        end
    end
end

function jacckernel_mul_NC_adagbshift!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, i]) * B[k3, k2, bshifted, rshifted]
            end
        end
    end
end

function jacckernel_mul_NC_adagbshiftdag!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, i]) * conj(B[k2, k3, bshifted, rshifted])
            end
        end
    end
end

function jacckernel_mul_NC_ashiftbdag!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, bshifted, rshifted] * conj(B[k2, k3, i])
            end
        end
    end
end

function jacckernel_mul_NC_ashiftdagbdag!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)

    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, bshifted, rshifted]) * conj(B[k2, k3, i])
            end
        end
    end
end

function jacckernel_mul_NC_abshiftdag!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * A[k1, k3, i] * conj(B[k2, k3, bshifted, rshifted])
            end
        end
    end
end

function jacckernel_mul_NC_ashiftdagb!(i, C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, bshifted, rshifted]) * B[k3, k2, i]
            end
        end
    end
end

function jacckernel_mul_NC_ashiftdagbshiftdag!(i, C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(i, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(i, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, bshifted_a, rshifted_a]) * conj(B[k2, k3, bshifted_b, rshifted_b])
            end
        end
    end
end

function jacckernel_mul_NC_ashiftdagbshift!(i, C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(i, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(i, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, i] = β * C[k1, k2, i]

            for k3 = 1:NC
                C[k1, k2, i] +=
                    α * conj(A[k3, k1, bshifted_a, rshifted_a]) * B[k3, k2, bshifted_b, rshifted_b]
            end
        end
    end
end

function jacckernel_tr!(i, U, NC)
    res = zero(eltype(U))
    @inbounds for k = 1:NC
        res += U[k, k, i]
    end
    return res

    #temp_volume[i] = 0
    #@inbounds for k = 1:NC
    #    temp_volume[i] += U[k, k, i]
    #end
    #return
end

function jacckernel_add_U!(i, c, a, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, i] += a[k2, k1, i]
        end
    end
    return
end


function jacckernel_add_U_αa!(i, c, a, α, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, i] += α * a[k2, k1, i]
        end
    end
    return
end

function jacckernel_add_U_αadag!(i, c, a, α, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, i] += α * conj(a[k1, k2, i])
        end
    end
    return
end

function jacckernel_clear_U!(i, c, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, i] = zero(eltype(c))
        end
    end
    return
end




function jacckernel_exptU_TAwuww_NC!(i, uout, u, t, NC, NG, generators, temp1, temp2, temp3)
    a = view(u, :, i)
    u0 = view(temp2, :, :, i)
    #for k = 1:length(a)
    #    a[k] = u[k, ix, iy, iz, it]
    #end

    lie2matrix_tuple!(u0, a, NG, generators, NC, t)
    #uout[:, :, b,r] = exp(t * (im / 2) * u0)
    nmax = 10

    matrixexponential!(view(uout, :, :, i), u0, NC, nmax, view(temp1, :, :, i), view(temp3, :, :, i))
end

function matrixexponential!(aout, a, Nc, nmax, temp1, temp2)
    atemp = temp1
    atemp2 = temp2

    for j = 1:Nc
        for i = 1:Nc
            aout[i, j] = 0
            atemp2[i, j] = 0
            #b[i,j] = 0
            #btemp[i,j] = 0
        end
    end
    for i = 1:Nc
        aout[i, i] = 1
        atemp2[i, i] = 1
    end

    for k = 1:nmax
        for j = 1:Nc
            for i = 1:Nc
                atemp[i, j] = atemp2[i, j]
                atemp2[i, j] = 0
            end
        end

        for jc = 1:Nc
            for ic = 1:Nc
                for kc = 1:Nc
                    atemp2[ic, jc] += a[ic, kc] * atemp[kc, jc] / k
                end
            end
        end

        for j = 1:Nc
            for i = 1:Nc
                aout[i, j] += atemp2[i, j]
            end
        end

    end


end




function lie2matrix_tuple!(matrix, a, NG, generators, NC, t)
    matrix .= 0
    for i = 1:NG
        #for (i, genmatrix) in enumerate(g.generator)
        for jc = 1:NC
            for ic = 1:NC
                matrix[ic, jc] += a[i] * generators[i][ic, jc] * t * (im / 2)
            end
        end
    end
    return
end

function jacckernel_exptU_TAwuww_NC2!(i, uout, u, t)
    u1 = t * u[1, i] / 2
    u2 = t * u[2, i] / 2
    u3 = t * u[3, i] / 2
    R = sqrt(u1^2 + u2^2 + u3^2) + tinyvalue
    sR = sin(R) / R
    #sR = ifelse(R == 0,1,sR)
    a0 = cos(R)
    a1 = u1 * sR
    a2 = u2 * sR
    a3 = u3 * sR

    uout[1, 1, i] = cos(R) + im * a3
    uout[1, 2, i] = im * a1 + a2
    uout[2, 1, i] = im * a1 - a2
    uout[2, 2, i] = cos(R) - im * a3
end

function jacckernel_exptU_TAwuww_NC3!(i, w, u, ww, t)
    c1 = t * u[1, i] * 0.5
    c2 = t * u[2, i] * 0.5
    c3 = t * u[3, i] * 0.5
    c4 = t * u[4, i] * 0.5
    c5 = t * u[5, i] * 0.5
    c6 = t * u[6, i] * 0.5
    c7 = t * u[7, i] * 0.5
    c8 = t * u[8, i] * 0.5
    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    if csum == 0
        w[1, 1, i] = 1
        w[1, 2, i] = 0
        w[1, 3, i] = 0
        w[2, 1, i] = 0
        w[2, 2, i] = 1
        w[2, 3, i] = 0
        w[3, 1, i] = 0
        w[3, 2, i] = 0
        w[3, 3, i] = 1

        ww[1, 1, i] = 1
        ww[1, 2, i] = 0
        ww[1, 3, i] = 0
        ww[2, 1, i] = 0
        ww[2, 2, i] = 1
        ww[2, 3, i] = 0
        ww[3, 1, i] = 0
        ww[3, 2, i] = 0
        ww[3, 3, i] = 1
        return
    end


    #x[1,1,icum] =  c3+sr3i*c8 +im*(  0.0 )
    v1 = c3 + sr3i * c8
    v2 = 0.0
    #x[1,2,icum] =  c1         +im*( -c2   )
    v3 = c1
    v4 = -c2
    #x[1,3,icum] =  c4         +im*(-c5   )
    v5 = c4
    v6 = -c5

    #x[2,1,icum] =  c1         +im*(  c2   )
    v7 = c1
    v8 = c2

    #x[2,2,icum] =  -c3+sr3i*c8+im*(  0.0 )
    v9 = -c3 + sr3i * c8
    v10 = 0.0

    #x[2,3,icum] =  c6         +im*( -c7   )
    v11 = c6
    v12 = -c7

    #x[3,1,icum] =  c4         +im*(  c5   )
    v13 = c4
    v14 = c5

    #x[3,2,icum] =  c6         +im*(  c7   )
    v15 = c6
    v16 = c7
    #x[3,3,icum] =  -sr3i2*c8  +im*(  0.0 )
    v17 = -sr3i2 * c8
    v18 = 0.0


    #c find eigenvalues of v
    trv3 = (v1 + v9 + v17) / 3.0
    cofac =
        v1 * v9 - v3^2 - v4^2 + v1 * v17 - v5^2 - v6^2 + v9 * v17 - v11^2 -
        v12^2
    det =
        v1 * v9 * v17 - v1 * (v11^2 + v12^2) - v9 * (v5^2 + v6^2) -
        v17 * (v3^2 + v4^2) +
        (v5 * (v3 * v11 - v4 * v12) + v6 * (v3 * v12 + v4 * v11)) * 2.0

    p3 = cofac / 3.0 - trv3^2
    q = trv3 * cofac - det - 2.0 * trv3^3
    x = sqrt(-4.0 * p3) + tinyvalue

    arg = q / (x * p3)

    arg = min(1, max(-1, arg))
    theta = acos(arg) / 3.0
    e1 = x * cos(theta) + trv3
    theta = theta + pi23
    e2 = x * cos(theta) + trv3
    #       theta = theta + pi23
    #       e3 = x * cos(theta) + trv3
    e3 = 3.0 * trv3 - e1 - e2

    # solve for eigenvectors

    w1 = v5 * (v9 - e1) - v3 * v11 + v4 * v12
    w2 = -v6 * (v9 - e1) + v4 * v11 + v3 * v12
    w3 = (v1 - e1) * v11 - v3 * v5 - v4 * v6
    w4 = -(v1 - e1) * v12 - v4 * v5 + v3 * v6
    w5 = -(v1 - e1) * (v9 - e1) + v3^2 + v4^2
    w6 = 0.0
    #println("1c $w1 $w2 $w3 $w4 $w5 $w6 ",)
    #coeffv = sqrt(w1^2 + w2^2 + w3^2 + w4^2 + w5^2)

    #coeff = ifelse(coeffv == zero(coeffv),0,coeffv)
    coeff = 1.0 / sqrt(w1^2 + w2^2 + w3^2 + w4^2 + w5^2)
    #println("1 ",coeff)

    w1 = w1 * coeff
    w2 = w2 * coeff
    w3 = w3 * coeff
    w4 = w4 * coeff
    w5 = w5 * coeff

    w7 = v5 * (v9 - e2) - v3 * v11 + v4 * v12
    w8 = -v6 * (v9 - e2) + v4 * v11 + v3 * v12
    w9 = (v1 - e2) * v11 - v3 * v5 - v4 * v6
    w10 = -(v1 - e2) * v12 - v4 * v5 + v3 * v6
    w11 = -(v1 - e2) * (v9 - e2) + v3^2 + v4^2
    w12 = 0.0

    coeff = 1.0 / sqrt(w7^2 + w8^2 + w9^2 + w10^2 + w11^2)

    w7 = w7 * coeff
    w8 = w8 * coeff
    w9 = w9 * coeff
    w10 = w10 * coeff
    w11 = w11 * coeff

    w13 = v5 * (v9 - e3) - v3 * v11 + v4 * v12
    w14 = -v6 * (v9 - e3) + v4 * v11 + v3 * v12
    w15 = (v1 - e3) * v11 - v3 * v5 - v4 * v6
    w16 = -(v1 - e3) * v12 - v4 * v5 + v3 * v6
    w17 = -(v1 - e3) * (v9 - e3) + v3^2 + v4^2
    w18 = 0.0

    coeff = 1.0 / sqrt(w13^2 + w14^2 + w15^2 + w16^2 + w17^2)
    w13 = w13 * coeff
    w14 = w14 * coeff
    w15 = w15 * coeff
    w16 = w16 * coeff
    w17 = w17 * coeff

    # construct the projection v
    c1 = cos(e1)
    s1 = sin(e1)
    ww1 = w1 * c1 - w2 * s1
    ww2 = w2 * c1 + w1 * s1
    ww3 = w3 * c1 - w4 * s1
    ww4 = w4 * c1 + w3 * s1
    ww5 = w5 * c1 - w6 * s1
    ww6 = w6 * c1 + w5 * s1

    c2 = cos(e2)
    s2 = sin(e2)
    ww7 = w7 * c2 - w8 * s2
    ww8 = w8 * c2 + w7 * s2
    ww9 = w9 * c2 - w10 * s2
    ww10 = w10 * c2 + w9 * s2
    ww11 = w11 * c2 - w12 * s2
    ww12 = w12 * c2 + w11 * s2

    c3 = cos(e3)
    s3 = sin(e3)
    ww13 = w13 * c3 - w14 * s3
    ww14 = w14 * c3 + w13 * s3
    ww15 = w15 * c3 - w16 * s3
    ww16 = w16 * c3 + w15 * s3
    ww17 = w17 * c3 - w18 * s3
    ww18 = w18 * c3 + w17 * s3

    w[1, 1, i] = w1 + im * w2
    w[1, 2, i] = w3 + im * w4
    w[1, 3, i] = w5 + im * w6
    w[2, 1, i] = w7 + im * w8
    w[2, 2, i] = w9 + im * w10
    w[2, 3, i] = w11 + im * w12
    w[3, 1, i] = w13 + im * w14
    w[3, 2, i] = w15 + im * w16
    w[3, 3, i] = w17 + im * w18

    ww[1, 1, i] = ww1 + im * ww2
    ww[1, 2, i] = ww3 + im * ww4
    ww[1, 3, i] = ww5 + im * ww6
    ww[2, 1, i] = ww7 + im * ww8
    ww[2, 2, i] = ww9 + im * ww10
    ww[2, 3, i] = ww11 + im * ww12
    ww[3, 1, i] = ww13 + im * ww14
    ww[3, 2, i] = ww15 + im * ww16
    ww[3, 3, i] = ww17 + im * ww18

    return
end

function jacckernel_exptU_wvww_NC3!(i, w, v, ww, t)
    v11 = v[1, 1, i]
    v22 = v[2, 2, i]
    v33 = v[3, 3, i]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = v[1, 2, i]
    v13 = v[1, 3, i]
    v21 = v[2, 1, i]
    v23 = v[2, 3, i]
    v31 = v[3, 1, i]
    v32 = v[3, 2, i]

    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32

    c1_0 = (imag(y12) + imag(y21))
    c2_0 = (real(y12) - real(y21))
    c3_0 = (imag(y11) - imag(y22))
    c4_0 = (imag(y13) + imag(y31))
    c5_0 = (real(y13) - real(y31))

    c6_0 = (imag(y23) + imag(y32))
    c7_0 = (real(y23) - real(y32))
    c8_0 = sr3i * (imag(y11) + imag(y22) - 2 * imag(y33))

    c1 = t * c1_0 * 0.5
    c2 = t * c2_0 * 0.5
    c3 = t * c3_0 * 0.5
    c4 = t * c4_0 * 0.5
    c5 = t * c5_0 * 0.5
    c6 = t * c6_0 * 0.5
    c7 = t * c7_0 * 0.5
    c8 = t * c8_0 * 0.5
    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    if csum == 0
        w[1, 1, i] = 1
        w[1, 2, i] = 0
        w[1, 3, i] = 0
        w[2, 1, i] = 0
        w[2, 2, i] = 1
        w[2, 3, i] = 0
        w[3, 1, i] = 0
        w[3, 2, i] = 0
        w[3, 3, i] = 1

        ww[1, 1, i] = 1
        ww[1, 2, i] = 0
        ww[1, 3, i] = 0
        ww[2, 1, i] = 0
        ww[2, 2, i] = 1
        ww[2, 3, i] = 0
        ww[3, 1, i] = 0
        ww[3, 2, i] = 0
        ww[3, 3, i] = 1
        return
    end


    #x[1,1,icum] =  c3+sr3i*c8 +im*(  0.0 )
    v1 = c3 + sr3i * c8
    v2 = 0.0
    #x[1,2,icum] =  c1         +im*( -c2   )
    v3 = c1
    v4 = -c2
    #x[1,3,icum] =  c4         +im*(-c5   )
    v5 = c4
    v6 = -c5

    #x[2,1,icum] =  c1         +im*(  c2   )
    v7 = c1
    v8 = c2

    #x[2,2,icum] =  -c3+sr3i*c8+im*(  0.0 )
    v9 = -c3 + sr3i * c8
    v10 = 0.0

    #x[2,3,icum] =  c6         +im*( -c7   )
    v11 = c6
    v12 = -c7

    #x[3,1,icum] =  c4         +im*(  c5   )
    v13 = c4
    v14 = c5

    #x[3,2,icum] =  c6         +im*(  c7   )
    v15 = c6
    v16 = c7
    #x[3,3,icum] =  -sr3i2*c8  +im*(  0.0 )
    v17 = -sr3i2 * c8
    v18 = 0.0


    #c find eigenvalues of v
    trv3 = (v1 + v9 + v17) / 3.0
    cofac =
        v1 * v9 - v3^2 - v4^2 + v1 * v17 - v5^2 - v6^2 + v9 * v17 - v11^2 -
        v12^2
    det =
        v1 * v9 * v17 - v1 * (v11^2 + v12^2) - v9 * (v5^2 + v6^2) -
        v17 * (v3^2 + v4^2) +
        (v5 * (v3 * v11 - v4 * v12) + v6 * (v3 * v12 + v4 * v11)) * 2.0
    p3 = cofac / 3.0 - trv3^2
    q = trv3 * cofac - det - 2.0 * trv3^3
    x = sqrt(-4.0 * p3) + tinyvalue
    arg = q / (x * p3)

    arg = min(1, max(-1, arg))
    theta = acos(arg) / 3.0
    e1 = x * cos(theta) + trv3
    theta = theta + pi23
    e2 = x * cos(theta) + trv3
    #       theta = theta + pi23
    #       e3 = x * cos(theta) + trv3
    e3 = 3.0 * trv3 - e1 - e2

    # solve for eigenvectors

    w1 = v5 * (v9 - e1) - v3 * v11 + v4 * v12
    w2 = -v6 * (v9 - e1) + v4 * v11 + v3 * v12
    w3 = (v1 - e1) * v11 - v3 * v5 - v4 * v6
    w4 = -(v1 - e1) * v12 - v4 * v5 + v3 * v6
    w5 = -(v1 - e1) * (v9 - e1) + v3^2 + v4^2
    w6 = 0.0

    coeff = 1.0 / sqrt(w1^2 + w2^2 + w3^2 + w4^2 + w5^2)


    w1 = w1 * coeff
    w2 = w2 * coeff
    w3 = w3 * coeff
    w4 = w4 * coeff
    w5 = w5 * coeff

    w7 = v5 * (v9 - e2) - v3 * v11 + v4 * v12
    w8 = -v6 * (v9 - e2) + v4 * v11 + v3 * v12
    w9 = (v1 - e2) * v11 - v3 * v5 - v4 * v6
    w10 = -(v1 - e2) * v12 - v4 * v5 + v3 * v6
    w11 = -(v1 - e2) * (v9 - e2) + v3^2 + v4^2
    w12 = 0.0

    coeff = 1.0 / sqrt(w7^2 + w8^2 + w9^2 + w10^2 + w11^2)

    w7 = w7 * coeff
    w8 = w8 * coeff
    w9 = w9 * coeff
    w10 = w10 * coeff
    w11 = w11 * coeff

    w13 = v5 * (v9 - e3) - v3 * v11 + v4 * v12
    w14 = -v6 * (v9 - e3) + v4 * v11 + v3 * v12
    w15 = (v1 - e3) * v11 - v3 * v5 - v4 * v6
    w16 = -(v1 - e3) * v12 - v4 * v5 + v3 * v6
    w17 = -(v1 - e3) * (v9 - e3) + v3^2 + v4^2
    w18 = 0.0

    coeff = 1.0 / sqrt(w13^2 + w14^2 + w15^2 + w16^2 + w17^2)
    w13 = w13 * coeff
    w14 = w14 * coeff
    w15 = w15 * coeff
    w16 = w16 * coeff
    w17 = w17 * coeff

    # construct the projection v
    c1 = cos(e1)
    s1 = sin(e1)
    ww1 = w1 * c1 - w2 * s1
    ww2 = w2 * c1 + w1 * s1
    ww3 = w3 * c1 - w4 * s1
    ww4 = w4 * c1 + w3 * s1
    ww5 = w5 * c1 - w6 * s1
    ww6 = w6 * c1 + w5 * s1

    c2 = cos(e2)
    s2 = sin(e2)
    ww7 = w7 * c2 - w8 * s2
    ww8 = w8 * c2 + w7 * s2
    ww9 = w9 * c2 - w10 * s2
    ww10 = w10 * c2 + w9 * s2
    ww11 = w11 * c2 - w12 * s2
    ww12 = w12 * c2 + w11 * s2

    c3 = cos(e3)
    s3 = sin(e3)
    ww13 = w13 * c3 - w14 * s3
    ww14 = w14 * c3 + w13 * s3
    ww15 = w15 * c3 - w16 * s3
    ww16 = w16 * c3 + w15 * s3
    ww17 = w17 * c3 - w18 * s3
    ww18 = w18 * c3 + w17 * s3

    w[1, 1, i] = w1 + im * w2
    w[1, 2, i] = w3 + im * w4
    w[1, 3, i] = w5 + im * w6
    w[2, 1, i] = w7 + im * w8
    w[2, 2, i] = w9 + im * w10
    w[2, 3, i] = w11 + im * w12
    w[3, 1, i] = w13 + im * w14
    w[3, 2, i] = w15 + im * w16
    w[3, 3, i] = w17 + im * w18

    ww[1, 1, i] = ww1 + im * ww2
    ww[1, 2, i] = ww3 + im * ww4
    ww[1, 3, i] = ww5 + im * ww6
    ww[2, 1, i] = ww7 + im * ww8
    ww[2, 2, i] = ww9 + im * ww10
    ww[2, 3, i] = ww11 + im * ww12
    ww[3, 1, i] = ww13 + im * ww14
    ww[3, 2, i] = ww15 + im * ww16
    ww[3, 3, i] = ww17 + im * ww18

    #a = ww[:,:,ix,iy,iz,it]
    #b = w[:,:,ix,iy,iz,it]
    #println(b'*a)
    #println(exp(im*t*v[:,:,ix,iy,iz,it]))
    #error("d")



end




function jacckernel_Traceless_antihermitian_NC3!(i, vout, vin)

    fac13 = 1 / 3

    v11 = vin[1, 1, i]
    v21 = vin[2, 1, i]
    v31 = vin[3, 1, i]

    v12 = vin[1, 2, i]
    v22 = vin[2, 2, i]
    v32 = vin[3, 2, i]

    v13 = vin[1, 3, i]
    v23 = vin[2, 3, i]
    v33 = vin[3, 3, i]


    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im



    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    #=
    vout[1,2,ix,iy,iz,it] = 0.5  * x12
    vout[1,3,ix,iy,iz,it] = 0.5  * x13
    vout[2,1,ix,iy,iz,it] = 0.5  * x21
    vout[2,3,ix,iy,iz,it] = 0.5  * x23
    vout[3,1,ix,iy,iz,it] = 0.5  * x31
    vout[3,2,ix,iy,iz,it] = 0.5  * x32
    =#
    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32


    vout[1, 1, i] = y11
    vout[2, 1, i] = y21
    vout[3, 1, i] = y31

    vout[1, 2, i] = y12
    vout[2, 2, i] = y22
    vout[3, 2, i] = y32

    vout[1, 3, i] = y13
    vout[2, 3, i] = y23
    vout[3, 3, i] = y33

    return

end

function jacckernel_tr!(i, temp_volume, A, B, NC)
    temp_volume[i] = 0
    @inbounds for k = 1:NC
        for k2 = 1:NC
            temp_volume[i] += A[k, k2, i] * B[k2, k, i]
        end
    end
    return
end

function jacckernel_NC_shiftedU!(i, Ushifted, U,
    shift, blockinfo, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            Ushifted[k2, k1, i] = U[k2, k1, bshifted, rshifted]
        end
    end
end

function jacckernel_add_U_αshifta!(i, c, a, α, shift, blockinfo, NC)
    bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, i] += α * a[k2, k1, bshifted, rshifted]
        end
    end
    return
end



function jacckernel_substitute_TAU!(i, Uμ, pwork, NumofBasis, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    #    ix, iy, iz, it = fourdim_cordinate(i, blockinfo)
    @inbounds for k = 1:NumofBasis
        icount = ((((it - 1) * NZ + iz - 1) * NY + iy - 1) * NX + ix - 1) * NumofBasis + k
        Uμ[k, i] = pwork[icount]
    end
end


function jacckernel_mult_xTAyTA!(i, x, y, NumofBasis)
    res = zero(eltype(x))
    @inbounds for k = 1:NumofBasis
        res += x[k, i] * y[k, i]
    end
    return res
end


function jacckernel_Traceless_antihermitian_add_TAU_NC2!(i,
    c, vin, factor)
    v11 = vin[1, 1, i]
    v22 = vin[2, 2, i]

    tri = fac12 * (imag(v11) + imag(v22))

    v12 = vin[1, 2, i]
    #v13 = vin[1,3,ix,iy,iz,it]
    v21 = vin[2, 1, i]

    x12 = v12 - conj(v21)

    x21 = -conj(x12)

    y11 = (imag(v11) - tri) * im
    y12 = 0.5 * x12
    y21 = 0.5 * x21
    y22 = (imag(v22) - tri) * im

    c[1, i] =
        (imag(y12) + imag(y21)) * factor + c[1, i]
    c[2, i] =
        (real(y12) - real(y21)) * factor + c[2, i]
    c[3, i] =
        (imag(y11) - imag(y22)) * factor + c[3, i]

    return

end

function matrix2lie_tuple!(a, NG, generators, NC, A)
    for i = 1:NG
        #println("i = $i")
        #display(g.generator[i]*g.generator[i])
        #println("\t")
        #println(tr(g.generator[i]*A)/2)
        a[i] = 0.0im
        for ic = 1:NC
            for kc = 1:NC
                a[i] += generators[i][ic, kc] * A[kc, ic] / 2
            end
        end#  tr(g.generator[i] * A) / 2
    end
    return
end

function jacckernel_Traceless_antihermitian_add_TAU_NC!(i,
    c, vin, factor, NC, NG, generators, temp, tempa)
    matrix = view(temp, :, :, i)
    a = tempa


    tri = 0.0
    for k = 1:NC
        tri += imag(vin[k, k, i])
    end
    tri *= 1 / NC
    for k = 1:NC
        #vout[k,k,ix,iy,iz,it] = (imag(vin[k,k,ix,iy,iz,it])-tri)*im
        matrix[k, k] = (imag(vin[k, k, i]) - tri) * im
    end

    for k1 = 1:NC
        for k2 = (k1+1):NC
            vv =
                0.5 * (
                    vin[k1, k2, i] -
                    conj(vin[k2, k1, i])
                )
            #vout[k1,k2,ix,iy,iz,it] = vv
            #vout[k2,k1,ix,iy,iz,it] = -conj(vv)
            matrix[k1, k2] = vv
            matrix[k2, k1] = -conj(vv)
        end
    end
    #display(matrix)
    matrix2lie_tuple!(a, NG, generators, NC, matrix)
    #display(a)
    #error("l")
    #matrix2lie!(a, g, matrix)
    for k = 1:NG
        c[k, i] = 2 * imag(a[k]) * factor + c[k, i]
    end
end

function jacckernel_Traceless_antihermitian_add_TAU_NC3!(i,
    c, vin, factor)

    fac13 = 1 / 3


    v11 = vin[1, 1, i]
    v22 = vin[2, 2, i]
    v33 = vin[3, 3, i]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = vin[1, 2, i]
    v13 = vin[1, 3, i]
    v21 = vin[2, 1, i]
    v23 = vin[2, 3, i]
    v31 = vin[3, 1, i]
    v32 = vin[3, 2, i]

    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    #=
    vout[1,2,ix,iy,iz,it] = 0.5  * x12
    vout[1,3,ix,iy,iz,it] = 0.5  * x13
    vout[2,1,ix,iy,iz,it] = 0.5  * x21
    vout[2,3,ix,iy,iz,it] = 0.5  * x23
    vout[3,1,ix,iy,iz,it] = 0.5  * x31
    vout[3,2,ix,iy,iz,it] = 0.5  * x32
    =#
    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32


    c[1, i] =
        (imag(y12) + imag(y21)) * factor + c[1, i]
    c[2, i] =
        (real(y12) - real(y21)) * factor + c[2, i]
    c[3, i] =
        (imag(y11) - imag(y22)) * factor + c[3, i]
    c[4, i] =
        (imag(y13) + imag(y31)) * factor + c[4, i]
    c[5, i] =
        (real(y13) - real(y31)) * factor + c[5, i]

    c[6, i] =
        (imag(y23) + imag(y32)) * factor + c[6, i]
    c[7, i] =
        (real(y23) - real(y32)) * factor + c[7, i]
    c[8, i] =
        sr3i * (imag(y11) + imag(y22) - 2 * imag(y33)) * factor +
        c[8, i]

    return

end

function jacckernel_clear_TAU!(i, c, NumofBasis)
    @inbounds for k1 = 1:NumofBasis
        c[k1, i] = 0
    end
    return
end

function jacckernel_add_TAU!(i, c, a, NumofBasis)
    @inbounds for k1 = 1:NumofBasis
        c[k1, i] += a[k1, i]
    end
    return
end


function jacckernel_add_TAU!(i, c, t::Number, a, NumofBasis)
    @inbounds for k1 = 1:NumofBasis
        c[k1, i] += t * a[k1, i]
    end
    return
end



