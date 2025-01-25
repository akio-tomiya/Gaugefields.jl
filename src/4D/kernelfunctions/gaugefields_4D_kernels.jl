struct Blockindices
    blocks::NTuple{4,Int64}
    blocks_s::NTuple{4,Int64}
    blocknumbers::NTuple{4,Int64}
    blocknumbers_s::NTuple{4,Int64}
    blocksize::Int64 #num. of Threads 
    rsize::Int64 #num. of blocks

    function Blockindices(L, blocks)
        blocknumbers = div.(L, blocks)

        dim = length(L)
        blocks_s = ones(dim)
        blocknumbers_s = ones(dim)
        for i in 2:dim
            for j in 1:i-1
                blocknumbers_s[i] = blocknumbers_s[i] * blocknumbers[j]
                blocks_s[i] = blocks_s[i] * blocks[j]
            end
        end

        blocksize = prod(blocks)
        rsize = prod(blocknumbers)

        return new(Tuple(blocks), Tuple(blocks_s), Tuple(blocknumbers), Tuple(blocknumbers_s), blocksize, rsize)

    end
end

@inline convert_x(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)
@inline convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s) = 1 + convert_x(b, blocks, blocks_s) +
                                                                           convert_x(r, blocknumbers, blocknumbers_s) * blocks

function fourdim_cordinate(b, r, blockinfo)
    blocks = blockinfo.blocks[1]
    blocks_s = blockinfo.blocks_s[1]
    blocknumbers = blockinfo.blocknumbers[1]
    blocknumbers_s = blockinfo.blocknumbers_s[1]
    ix = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[2]
    blocks_s = blockinfo.blocks_s[2]
    blocknumbers = blockinfo.blocknumbers[2]
    blocknumbers_s = blockinfo.blocknumbers_s[2]
    iy = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[3]
    blocks_s = blockinfo.blocks_s[3]
    blocknumbers = blockinfo.blocknumbers[3]
    blocknumbers_s = blockinfo.blocknumbers_s[3]
    iz = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[4]
    blocks_s = blockinfo.blocks_s[4]
    blocknumbers = blockinfo.blocknumbers[4]
    blocknumbers_s = blockinfo.blocknumbers_s[4]
    it = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    return ix, iy, iz, it
end


@inline function minusshift(x, xb, xb_s)
    x - (xb - 1) * xb_s
end
@inline function plusshift(x, xb, xb_s)
    x + (xb - 1) * xb_s
end

@inline function shiftedindex_each_plus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bshifted = b
    rshifted = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if (ic1 == blocks - 1)
        bshifted = minusshift(bshifted, blocks, blocks_s)
        #bshifted += - (block-1)*block_s

        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if (ic2 == blocknumbers - 1)
            rshifted = minusshift(rshifted, blocknumbers, blocknumbers_s)
            #rshifted = rshiftfunc(rshifted,blocknumbers ,blocknumbers_s) 
            #rshifted += - (blocknumbers -1)*blocknumbers_s
        else
            rshifted += +blocknumbers_s
        end
    else
        bshifted += blocks_s #shift in idim direction 
        #rshifted = r

    end

    return bshifted, rshifted

end

@inline function shiftedindex_each_minus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bshifted = b
    rshifted = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == 0
        #bshifted += + (blocks-1)*blocks_s
        bshifted = plusshift(bshifted, blocks, blocks_s)

        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == 0
            rshifted = plusshift(rshifted, blocknumbers, blocknumbers_s)
            #rshifted += + (blocknumbers -1)*blocknumbers_s
        else
            rshifted += -blocknumbers_s
        end

    else
        bshifted += -blocks_s #shift in idim direction 
        #rshifted = r
    end

    return bshifted, rshifted

end

@inline function shiftedindex(b, r, shift, blockinfo)
    bshifted = b
    rshifted = r
    for idim = 1:4
        blocks = blockinfo.blocks[idim]
        blocks_s = blockinfo.blocks_s[idim]
        blocknumbers = blockinfo.blocknumbers[idim]
        blocknumbers_s = blockinfo.blocknumbers_s[idim]

        isplus = ifelse(shift[idim] > 0, true, false)
        numshift = abs(shift[idim])
        if isplus
            for ishift = 1:numshift
                bshifted, rshifted = shiftedindex_each_plus(bshifted, rshifted, blocks, blocks_s, blocknumbers, blocknumbers_s)
            end
        else
            for ishift = 1:numshift
                bshifted, rshifted = shiftedindex_each_minus(bshifted, rshifted, blocks, blocks_s, blocknumbers, blocknumbers_s)
            end
        end
    end

    return bshifted, rshifted
end



function kernel_identityGaugefields!(b,r,U, NC)
    @inbounds for ic = 1:NC
        U[ic, ic, b, r] = 1
    end
end

function kernel_randomGaugefields!(b,r,U, NC)
    @inbounds for ic = 1:NC
        for jc = 1:NC
            U[jc, ic, b, r] = rand() - 0.5 + im * (rand() - 0.5)
        end
    end
end

function kernel_normalize_U_NC2!(b,r,u)
    α = u[1, 1, b, r]
    β = u[2, 1, b, r]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, b, r] = α / detU
    u[2, 1, b, r] = β / detU
    u[1, 2, b, r] = -conj(β) / detU
    u[2, 2, b, r] = conj(α) / detU
    return
end

function kernel_normalize_U_NC3!(b,r,u)
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


function kernel_mul_NC!(b,r,C, A, B, NC)
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

function kernel_mul_NC!(b,r,C, A, B, α, β, NC)
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

function kernel_mul_NC3!(b,r,C, A, B)
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

function kernel_mul_NC3!(b,r,C, A, B, α, β)
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


function kernel_mul_NC_abdag!(b,r,C, A, B, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = 0

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    A[k1, k3, b, r] * conj(B[k2, k3, b, r])
            end
        end
    end
end

function kernel_mul_NC_abdag!(b,r,C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * conj(B[k2, k3, b, r])
            end
        end
    end
end


function kernel_mul_NC_adagbdag!(b,r,C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * conj(B[k2, k3, b, r])
            end
        end
    end
end

function kernel_mul_NC_adagbdag!(b,r,C, A, B, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = 0

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    conj(A[k3, k1, b, r]) * conj(B[k2, k3, b, r])
            end
        end
    end
end

function kernel_mul_NC_adagb!(b,r,C, A, B, α, β, NC)
    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * B[k3, k2, b, r]
            end
        end
    end
end


function kernel_mul_NC_abshift!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * B[k3, k2, bshifted, rshifted]
            end
        end
    end
end


function kernel_mul_NC_ashiftb!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted, rshifted] * B[k3, k2, b, r]
            end
        end
    end
end


function kernel_mul_NC_ashiftbshift!(b,r,C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted_a, rshifted_a] * B[k3, k2, bshifted_b, rshifted_b]
            end
        end
    end
end

function kernel_mul_NC_ashiftbshiftdag!(b,r,C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted_a, rshifted_a] * conj(B[k2, k3, bshifted_b, rshifted_b])
            end
        end
    end
end

function kernel_mul_NC_adagbshift!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * B[k3, k2, bshifted, rshifted]
            end
        end
    end
end

function kernel_mul_NC_adagbshiftdag!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * conj(B[k2, k3, bshifted, rshifted])
            end
        end
    end
end

function kernel_mul_NC_ashiftbdag!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted, rshifted] * conj(B[k2, k3, b, r])
            end
        end
    end
end

function kernel_mul_NC_ashiftdagbdag!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted, rshifted]) * conj(B[k2, k3, b, r])
            end
        end
    end
end

function kernel_mul_NC_abshiftdag!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * conj(B[k2, k3, bshifted, rshifted])
            end
        end
    end
end

function kernel_mul_NC_ashiftdagb!(b,r,C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted, rshifted]) * B[k3, k2, b, r]
            end
        end
    end
end

function kernel_mul_NC_ashiftdagbshiftdag!(b,r,C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted_a, rshifted_a]) * conj(B[k2, k3, bshifted_b, rshifted_b])
            end
        end
    end
end

function kernel_mul_NC_ashiftdagbshift!(b,r,C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted_a, rshifted_a]) * B[k3, k2, bshifted_b, rshifted_b]
            end
        end
    end
end

function kernel_tr!(b,r,temp_volume, U, NC)
    temp_volume[b, r] = 0
    @inbounds for k = 1:NC
        temp_volume[b, r] += U[k, k, b, r]
    end
    return
end

function kernel_add_U!(b,r,c, a, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += a[k2, k1, b, r]
        end
    end
    return
end

function kernel_add_U_αa!(b,r,c, a, α, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * a[k2, k1, b, r]
        end
    end
    return
end

function kernel_add_U_αadag!(b,r,c, a, α, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * conj(a[k1, k2, b, r])
        end
    end
    return
end

function kernel_clear_U!(b,r,c, NC)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] = 0
        end
    end
    return
end

function kernel_exptU_wvww!(b,r,w, v, ww, t, NC)
    v11 = v[1, 1, b, r]
    v22 = v[2, 2, b, r]
    v33 = v[3, 3, b, r]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = v[1, 2, b, r]
    v13 = v[1, 3, b, r]
    v21 = v[2, 1, b, r]
    v23 = v[2, 3, b, r]
    v31 = v[3, 1, b, r]
    v32 = v[3, 2, b, r]

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
        w[1, 1, b, r] = 1
        w[1, 2, b, r] = 0
        w[1, 3, b, r] = 0
        w[2, 1, b, r] = 0
        w[2, 2, b, r] = 1
        w[2, 3, b, r] = 0
        w[3, 1, b, r] = 0
        w[3, 2, b, r] = 0
        w[3, 3, b, r] = 1

        ww[1, 1, b, r] = 1
        ww[1, 2, b, r] = 0
        ww[1, 3, b, r] = 0
        ww[2, 1, b, r] = 0
        ww[2, 2, b, r] = 1
        ww[2, 3, b, r] = 0
        ww[3, 1, b, r] = 0
        ww[3, 2, b, r] = 0
        ww[3, 3, b, r] = 1
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

    w[1, 1, b, r] = w1 + im * w2
    w[1, 2, b, r] = w3 + im * w4
    w[1, 3, b, r] = w5 + im * w6
    w[2, 1, b, r] = w7 + im * w8
    w[2, 2, b, r] = w9 + im * w10
    w[2, 3, b, r] = w11 + im * w12
    w[3, 1, b, r] = w13 + im * w14
    w[3, 2, b, r] = w15 + im * w16
    w[3, 3, b, r] = w17 + im * w18

    ww[1, 1, b, r] = ww1 + im * ww2
    ww[1, 2, b, r] = ww3 + im * ww4
    ww[1, 3, b, r] = ww5 + im * ww6
    ww[2, 1, b, r] = ww7 + im * ww8
    ww[2, 2, b, r] = ww9 + im * ww10
    ww[2, 3, b, r] = ww11 + im * ww12
    ww[3, 1, b, r] = ww13 + im * ww14
    ww[3, 2, b, r] = ww15 + im * ww16
    ww[3, 3, b, r] = ww17 + im * ww18

    #a = ww[:,:,ix,iy,iz,it]
    #b = w[:,:,ix,iy,iz,it]
    #println(b'*a)
    #println(exp(im*t*v[:,:,ix,iy,iz,it]))
    #error("d")



end

function kernel_Traceless_antihermitian_NC3!(vout, vin,b,r)

    fac13 = 1 / 3

    v11 = vin[1, 1, b, r]
    v21 = vin[2, 1, b, r]
    v31 = vin[3, 1, b, r]

    v12 = vin[1, 2, b, r]
    v22 = vin[2, 2, b, r]
    v32 = vin[3, 2, b, r]

    v13 = vin[1, 3, b, r]
    v23 = vin[2, 3, b, r]
    v33 = vin[3, 3, b, r]


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


    vout[1, 1, b, r] = y11
    vout[2, 1, b, r] = y21
    vout[3, 1, b, r] = y31

    vout[1, 2, b, r] = y12
    vout[2, 2, b, r] = y22
    vout[3, 2, b, r] = y32

    vout[1, 3, b, r] = y13
    vout[2, 3, b, r] = y23
    vout[3, 3, b, r] = y33

    return

end

function kernel_tr!(b,r,temp_volume, A, B, NC)
    temp_volume[b, r] = 0
    @inbounds for k = 1:NC
        for k2 = 1:NC
            temp_volume[b, r] += A[k, k2, b, r] * B[k2, k, b, r]
        end
    end
    return
end

function kernel_add_U_αshifta!(b,r,c, a, α, shift, blockinfo, NC)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * a[k2, k1, bshifted, rshifted]
        end
    end
    return
end



function kernel_substitute_TAU!(b,r,Uμ,pwork,blockinfo,NumofBasis,NX,NY,NZ,NT)
    ix,iy,iz,it = fourdim_cordinate(b,r,blockinfo)
    @inbounds for k=1:NumofBasis
        icount = ((((it-1)*NZ+iz-1)*NY+iy-1)*NX+ix-1)*NumofBasis+k
        Uμ[k, b,r] = pwork[icount]
    end
end


function kernel_mult_xTAyTA!(b,r,temp,x,y,NumofBasis)
    temp[b,r] = 0
    @inbounds for k = 1:NumofBasis
        temp[b,r] += x[k, b,r] * y[k, b,r]
    end
    return
end


function kernel_exptU_TAwuww_NC3!(b,r,w,u,ww,t)
    c1 = t * u[1,b,r] * 0.5
    c2 = t * u[2,b,r] * 0.5
    c3 = t * u[3,b,r] * 0.5
    c4 = t * u[4,b,r] * 0.5
    c5 = t * u[5,b,r] * 0.5
    c6 = t * u[6,b,r] * 0.5
    c7 = t * u[7,b,r] * 0.5
    c8 = t * u[8,b,r] * 0.5
    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    if csum == 0
        w[1, 1,b,r] = 1
        w[1, 2,b,r] = 0
        w[1, 3,b,r] = 0
        w[2, 1,b,r] = 0
        w[2, 2,b,r] = 1
        w[2, 3,b,r] = 0
        w[3, 1,b,r] = 0
        w[3, 2,b,r] = 0
        w[3, 3,b,r] = 1

        ww[1, 1,b,r] = 1
        ww[1, 2,b,r] = 0
        ww[1, 3,b,r] = 0
        ww[2, 1,b,r] = 0
        ww[2, 2,b,r] = 1
        ww[2, 3,b,r] = 0
        ww[3, 1,b,r] = 0
        ww[3, 2,b,r] = 0
        ww[3, 3,b,r] = 1
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

    w[1, 1,b,r] = w1 + im * w2
    w[1, 2,b,r] = w3 + im * w4
    w[1, 3,b,r] = w5 + im * w6
    w[2, 1,b,r] = w7 + im * w8
    w[2, 2,b,r] = w9 + im * w10
    w[2, 3,b,r] = w11 + im * w12
    w[3, 1,b,r] = w13 + im * w14
    w[3, 2,b,r] = w15 + im * w16
    w[3, 3,b,r] = w17 + im * w18

    ww[1, 1,b,r] = ww1 + im * ww2
    ww[1, 2,b,r] = ww3 + im * ww4
    ww[1, 3,b,r] = ww5 + im * ww6
    ww[2, 1,b,r] = ww7 + im * ww8
    ww[2, 2,b,r] = ww9 + im * ww10
    ww[2, 3,b,r] = ww11 + im * ww12
    ww[3, 1,b,r] = ww13 + im * ww14
    ww[3, 2,b,r] = ww15 + im * ww16
    ww[3, 3,b,r] = ww17 + im * ww18

    return
end


function kernel_Traceless_antihermitian_add_TAU_NC3!(b,r,
    c,vin,factor)

    fac13 = 1 / 3


    v11 = vin[1, 1, b,r]
    v22 = vin[2, 2, b,r]
    v33 = vin[3, 3, b,r]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = vin[1, 2, b,r]
    v13 = vin[1, 3, b,r]
    v21 = vin[2, 1, b,r]
    v23 = vin[2, 3, b,r]
    v31 = vin[3, 1, b,r]
    v32 = vin[3, 2, b,r]

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


    c[1, b,r] =
        (imag(y12) + imag(y21)) * factor + c[1, b,r]
    c[2, b,r] =
        (real(y12) - real(y21)) * factor + c[2, b,r]
    c[3, b,r] =
        (imag(y11) - imag(y22)) * factor + c[3, b,r]
    c[4, b,r] =
        (imag(y13) + imag(y31)) * factor + c[4, b,r]
    c[5, b,r] =
        (real(y13) - real(y31)) * factor + c[5, b,r]

    c[6, b,r] =
        (imag(y23) + imag(y32)) * factor + c[6, b,r]
    c[7, b,r] =
        (real(y23) - real(y32)) * factor + c[7, b,r]
    c[8, b,r] =
        sr3i * (imag(y11) + imag(y22) - 2 * imag(y33)) * factor +
        c[8, b,r]

    return

end