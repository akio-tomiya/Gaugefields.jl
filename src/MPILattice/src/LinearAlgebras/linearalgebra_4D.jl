@inline launch3d(PN::NTuple{4,Int}) = (PN[1], PN[2], PN[3] * PN[4])


function LinearAlgebra.mul!(C::LatticeVector{4,T,AT}, A::LatticeVector{4,T,AT}, B::LatticeVector{4,T,AT}) where {T,AT}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dvector_mul!, C.A, A.A, B.A, C.NC, nw, C.PN
    )
    #set_halo!(C)
end

#=
@inline function get_4Dindex(i, dims)
    #i = (((it-1)*dims[3]+iz-1)*dims[2]+iy-1)*dims[1]+ix
    Nx, Ny, Nz, Nt = dims
    o = i - 1                      # zero-based offset
    ix = (o % Nx) + 1
    o ÷= Nx
    iy = (o % Ny) + 1
    o ÷= Ny
    iz = (o % Nz) + 1
    o ÷= Nz
    it = o + 1
    return ix, iy, iz, it
end
=#

# 3D( i1,i2,i3 ) → 4D(ix,iy,iz,it) 展開
@inline function get_4Dindex(i1::I, i2::I, i3::I, dims::NTuple{4,I}) where {I<:Integer}
    Nx, Ny, Nz, Nt = dims
    ix = i1
    iy = i2
    o = i3 - one(I)
    o, rz = divrem(o, Nz)      # rz: 0..Nz-1,  o: 0..Nt-1
    iz = rz + one(I)
    it = o + one(I)
    return ix, iy, iz, it
end


# Generic, fast path using divrem (works for any sizes)
@inline function get_4Dindex(i::I, dims::NTuple{4,I}) where {I<:Integer}
    # Decode linear index i (1-based) into four 1-based coordinates.
    # Use divrem to compute quotient and remainder in one shot, reducing idiv count.
    @inbounds begin
        Nx, Ny, Nz, Nt = dims
        o = i - one(I)                  # zero-based offset
        o, rx = divrem(o, Nx)
        ix = rx + one(I)
        o, ry = divrem(o, Ny)
        iy = ry + one(I)
        o, rz = divrem(o, Nz)
        iz = rz + one(I)
        it = o + one(I)                 # remaining quotient
        return ix, iy, iz, it
    end
end


struct Mulkernel{NC1,NC2,NC3}
end

@inline function kernel_4Dvector_mul!(i, C, A, B, NC, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for ic = 1:NC
        C[ic, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, ix+nw, iy+nw, iz+nw, it+nw] * B[ic, ix+nw, iy+nw, iz+nw, it+nw]
    end
end


#C = A B 
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end




@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = zero(eltype(C))
        end

        for kc = 1:NC3
            b = B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for ic = 1:NC1
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * b# B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end



#C = A B 
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC1,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC1,nw}, B::LatticeMatrix{4,T3,AT3,NC1,NC1,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(nw), C.PN
    )
    #set_halo!(C)
end

@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{nw}, PN) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC1
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = zero(eltype(C))
        end

        for kc = 1:NC1
            b = B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for ic = 1:NC1
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * b# B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{3}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]
        a21 = A[2, 1, ix, iy, iz, it]
        a31 = A[3, 1, ix, iy, iz, it]
        a12 = A[1, 2, ix, iy, iz, it]
        a22 = A[2, 2, ix, iy, iz, it]
        a32 = A[3, 2, ix, iy, iz, it]
        a13 = A[1, 3, ix, iy, iz, it]
        a23 = A[2, 3, ix, iy, iz, it]
        a33 = A[3, 3, ix, iy, iz, it]

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{2}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]
        a21 = A[2, 1, ix, iy, iz, it]
        a12 = A[1, 2, ix, iy, iz, it]
        a22 = A[2, 2, ix, iy, iz, it]


        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]

        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22

    end
end





@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]
        a21 = A[2, 1, ix, iy, iz, it]
        a31 = A[3, 1, ix, iy, iz, it]
        a12 = A[1, 2, ix, iy, iz, it]
        a22 = A[2, 2, ix, iy, iz, it]
        a32 = A[3, 2, ix, iy, iz, it]
        a13 = A[1, 3, ix, iy, iz, it]
        a23 = A[2, 3, ix, iy, iz, it]
        a33 = A[3, 3, ix, iy, iz, it]
        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]
        a21 = A[2, 1, ix, iy, iz, it]
        a12 = A[1, 2, ix, iy, iz, it]
        a22 = A[2, 2, ix, iy, iz, it]

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]

        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22

    end
end




#C = A B α + C β
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}, α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α, β
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α, β) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, α, β) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = α * A[1, 1, ix, iy, iz, it]
        a21 = α * A[2, 1, ix, iy, iz, it]
        a31 = α * A[3, 1, ix, iy, iz, it]
        a12 = α * A[1, 2, ix, iy, iz, it]
        a22 = α * A[2, 2, ix, iy, iz, it]
        a32 = α * A[3, 2, ix, iy, iz, it]
        a13 = α * A[1, 3, ix, iy, iz, it]
        a23 = α * A[2, 3, ix, iy, iz, it]
        a33 = α * A[3, 3, ix, iy, iz, it]

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end

end

@inline function kernel_4Dmatrix_mul!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, PN, α, β) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = α * A[1, 1, ix, iy, iz, it]
        a21 = α * A[2, 1, ix, iy, iz, it]
        a12 = α * A[1, 2, ix, iy, iz, it]
        a22 = α * A[2, 2, ix, iy, iz, it]


        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]

        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]


        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22
    end

end


function expt!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, t::S=one(S)) where {T,AT,NC1,NC2,S<:Number,T1,AT1,nw}
    @assert NC1 == NC2 "Matrix exponentiation requires square matrices, but got $(NC1) x $(NC2)."

    JACC.parallel_for(
        prod(C.PN), kernel_4Dexpt!, C.A, A.A, C.PN, Val(nw), t, Val(NC1)
    )
    return
    #set_halo!(C)
end

@inline function kernel_4Dexpt!(i, C, A, PN, ::Val{nw}, t, ::Val{3}) where nw
    ix, iy, iz, it = get_4Dindex(i, PN)
    a11 = A[1, 1, ix+nw, iy+nw, iz+nw, it+nw]
    a12 = A[1, 2, ix+nw, iy+nw, iz+nw, it+nw]
    a13 = A[1, 3, ix+nw, iy+nw, iz+nw, it+nw]
    a21 = A[2, 1, ix+nw, iy+nw, iz+nw, it+nw]
    a22 = A[2, 2, ix+nw, iy+nw, iz+nw, it+nw]
    a23 = A[2, 3, ix+nw, iy+nw, iz+nw, it+nw]
    a31 = A[3, 1, ix+nw, iy+nw, iz+nw, it+nw]
    a32 = A[3, 2, ix+nw, iy+nw, iz+nw, it+nw]
    a33 = A[3, 3, ix+nw, iy+nw, iz+nw, it+nw]

    c11, c12, c13, c21, c22, c23, c31, c32, c33 = exp3x3_pade(a11, a12, a13, a21, a22, a23, a31, a32, a33, t)
    C[1, 1, ix+nw, iy+nw, iz+nw, it+nw] = c11
    C[1, 2, ix+nw, iy+nw, iz+nw, it+nw] = c12
    C[1, 3, ix+nw, iy+nw, iz+nw, it+nw] = c13
    C[2, 1, ix+nw, iy+nw, iz+nw, it+nw] = c21
    C[2, 2, ix+nw, iy+nw, iz+nw, it+nw] = c22
    C[2, 3, ix+nw, iy+nw, iz+nw, it+nw] = c23
    C[3, 1, ix+nw, iy+nw, iz+nw, it+nw] = c31
    C[3, 2, ix+nw, iy+nw, iz+nw, it+nw] = c32
    C[3, 3, ix+nw, iy+nw, iz+nw, it+nw] = c33

end

@inline function kernel_4Dexpt!(i, C, A, PN, ::Val{nw}, t, ::Val{2}) where nw
    ix, iy, iz, it = get_4Dindex(i, PN)
    a11 = A[1, 1, ix+nw, iy+nw, iz+nw, it+nw]
    a21 = A[2, 1, ix+nw, iy+nw, iz+nw, it+nw]
    a12 = A[1, 2, ix+nw, iy+nw, iz+nw, it+nw]
    a22 = A[2, 2, ix+nw, iy+nw, iz+nw, it+nw]
    c11, c12, c21, c22 = exp2x2_elem(a11, a12, a21, a22, t)

    C[1, 1, ix+nw, iy+nw, iz+nw, it+nw] = c11
    C[1, 2, ix+nw, iy+nw, iz+nw, it+nw] = c12
    C[2, 1, ix+nw, iy+nw, iz+nw, it+nw] = c21
    C[2, 2, ix+nw, iy+nw, iz+nw, it+nw] = c22
end



@inline function kernel_4Dexpt!(i, C, A, PN, ::Val{nw}, t, ::Val{N}) where {N,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    expm_pade13_writeback!(C, A, ix + nw, iy + nw, iz + nw, it + nw, t, Val(N))
    #C[:, :, ix, iy, iz, it] = expm_pade13(A[:, :, ix, iy, iz, it], t)
end

function expt!(C::LatticeMatrix{4,T,AT,NC1,NC1,nw}, TA::LatticeMatrix{4,T1,AT1,Num,1,nw2}, t::S=one(S)) where {T,AT,NC1,Num,S<:Number,T1<:Real,AT1,nw,nw2}

    if NC1 > 3
        error("In NC > 3 case, this function should not be used")
    else
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt_TA!, C.A, TA.A, C.PN, Val(nw), t, Val(NC1), Val(nw2)
        )
    end
    return
    #set_halo!(C)
end

function kernel_4Dexpt_TA!(i, uout, A, PN, ::Val{nw}, t, ::Val{2}, ::Val{nw2}) where {nw,nw2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixt = ix + nw2
    iyt = iy + nw2
    izt = iz + nw2
    itt = it + nw2
    ix += nw
    iy += nw
    iz += nw
    it += nw

    c1_0 = A[1, 1, ixt, iyt, izt, itt]
    c2_0 = A[2, 1, ixt, iyt, izt, itt]
    c3_0 = A[3, 1, ixt, iyt, izt, itt]

    #icum = (((it-1)*NX+iz-1)*NY+iy-1)*NX+ix  
    u1 = t * c1_0 / 2
    u2 = t * c2_0 / 2
    u3 = t * c3_0 / 2
    R = sqrt(u1^2 + u2^2 + u3^2) + tinyvalue
    sR = sin(R) / R
    #sR = ifelse(R == 0,1,sR)
    a0 = cos(R)
    a1 = u1 * sR
    a2 = u2 * sR
    a3 = u3 * sR

    uout[1, 1, ix, iy, iz, it] = cos(R) + im * a3
    uout[1, 2, ix, iy, iz, it] = im * a1 + a2
    uout[2, 1, ix, iy, iz, it] = im * a1 - a2
    uout[2, 2, ix, iy, iz, it] = cos(R) - im * a3
end


function kernel_4Dexpt_TA!(i, C, A, PN, ::Val{nw}, t, ::Val{3}, ::Val{nw2}) where {nw,nw2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    T = eltype(C)
    ixt = ix + nw2
    iyt = iy + nw2
    izt = iz + nw2
    itt = it + nw2
    ix += nw
    iy += nw
    iz += nw
    it += nw

    c1_0 = A[1, 1, ixt, iyt, izt, itt]
    c2_0 = A[2, 1, ixt, iyt, izt, itt]
    c3_0 = A[3, 1, ixt, iyt, izt, itt]
    c4_0 = A[4, 1, ixt, iyt, izt, itt]
    c5_0 = A[5, 1, ixt, iyt, izt, itt]

    c6_0 = A[6, 1, ixt, iyt, izt, itt]
    c7_0 = A[7, 1, ixt, iyt, izt, itt]
    c8_0 = A[8, 1, ixt, iyt, izt, itt]

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
        c = Mat3{eltype(C)}(one(eltype(C)))
        C[1, 1, ix, iy, iz, it] = c.a11
        C[1, 2, ix, iy, iz, it] = c.a12
        C[1, 3, ix, iy, iz, it] = c.a13
        C[2, 1, ix, iy, iz, it] = c.a21
        C[2, 2, ix, iy, iz, it] = c.a22
        C[2, 3, ix, iy, iz, it] = c.a23
        C[3, 1, ix, iy, iz, it] = c.a31
        C[3, 2, ix, iy, iz, it] = c.a32
        C[3, 3, ix, iy, iz, it] = c.a33

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


    w = Mat3{T}(w1 + im * w2,
        w3 + im * w4,
        w5 + im * w6,
        w7 + im * w8,
        w9 + im * w10,
        w11 + im * w12,
        w13 + im * w14,
        w15 + im * w16,
        w17 + im * w18)
    ww = Mat3{T}(ww1 + im * ww2,
        ww3 + im * ww4,
        ww5 + im * ww6,
        ww7 + im * ww8,
        ww9 + im * ww10,
        ww11 + im * ww12,
        ww13 + im * ww14,
        ww15 + im * ww16,
        ww17 + im * ww18)
    c = mul3(conjugate3(w), ww)

    C[1, 1, ix, iy, iz, it] = c.a11
    C[1, 2, ix, iy, iz, it] = c.a12
    C[1, 3, ix, iy, iz, it] = c.a13
    C[2, 1, ix, iy, iz, it] = c.a21
    C[2, 2, ix, iy, iz, it] = c.a22
    C[2, 3, ix, iy, iz, it] = c.a23
    C[3, 1, ix, iy, iz, it] = c.a31
    C[3, 2, ix, iy, iz, it] = c.a32
    C[3, 3, ix, iy, iz, it] = c.a33



end


#C = A'*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]'
        a12 = A[2, 1, ix, iy, iz, it]'
        a13 = A[3, 1, ix, iy, iz, it]'

        a21 = A[1, 2, ix, iy, iz, it]'
        a22 = A[2, 2, ix, iy, iz, it]'
        a23 = A[3, 2, ix, iy, iz, it]'

        a31 = A[1, 3, ix, iy, iz, it]'
        a32 = A[2, 3, ix, iy, iz, it]'
        a33 = A[3, 3, ix, iy, iz, it]'

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

@inline function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]'
        a12 = A[2, 1, ix, iy, iz, it]'

        a21 = A[1, 2, ix, iy, iz, it]'
        a22 = A[2, 2, ix, iy, iz, it]'


        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]

        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]

        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22
    end
end

#C = α*A'*B+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end


@inline function kernel_4Dmatrix_mul_AdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = α * A[1, 1, ix, iy, iz, it]'
        a12 = α * A[2, 1, ix, iy, iz, it]'
        a13 = α * A[3, 1, ix, iy, iz, it]'

        a21 = α * A[1, 2, ix, iy, iz, it]'
        a22 = α * A[2, 2, ix, iy, iz, it]'
        a23 = α * A[3, 2, ix, iy, iz, it]'

        a31 = α * A[1, 3, ix, iy, iz, it]'
        a32 = α * A[2, 3, ix, iy, iz, it]'
        a33 = α * A[3, 3, ix, iy, iz, it]'

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end
end



#C = A*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_ABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = α* A*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_ABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end

        for kc = 1:NC3
            b = conj(B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw])
            @simd for ic = 1:NC1
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * b#B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_ABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = α * A[1, 1, ix, iy, iz, it]
        a21 = α * A[2, 1, ix, iy, iz, it]
        a31 = α * A[3, 1, ix, iy, iz, it]
        a12 = α * A[1, 2, ix, iy, iz, it]
        a22 = α * A[2, 2, ix, iy, iz, it]
        a32 = α * A[3, 2, ix, iy, iz, it]
        a13 = α * A[1, 3, ix, iy, iz, it]
        a23 = α * A[2, 3, ix, iy, iz, it]
        a33 = α * A[3, 3, ix, iy, iz, it]


        b11 = B[1, 1, ix, iy, iz, it]'
        b12 = B[2, 1, ix, iy, iz, it]'
        b13 = B[3, 1, ix, iy, iz, it]'
        b21 = B[1, 2, ix, iy, iz, it]'
        b22 = B[2, 2, ix, iy, iz, it]'
        b23 = B[3, 2, ix, iy, iz, it]'
        b31 = B[1, 3, ix, iy, iz, it]'
        b32 = B[2, 3, ix, iy, iz, it]'
        b33 = B[3, 3, ix, iy, iz, it]'

        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end
end

#C = A'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = A[1, 1, ix, iy, iz, it]'
        a12 = A[2, 1, ix, iy, iz, it]'
        a13 = A[3, 1, ix, iy, iz, it]'
        a21 = A[1, 2, ix, iy, iz, it]'
        a22 = A[2, 2, ix, iy, iz, it]'
        a23 = A[3, 2, ix, iy, iz, it]'
        a31 = A[1, 3, ix, iy, iz, it]'
        a32 = A[2, 3, ix, iy, iz, it]'
        a33 = A[3, 3, ix, iy, iz, it]'


        b11 = B[1, 1, ix, iy, iz, it]'
        b12 = B[2, 1, ix, iy, iz, it]'
        b13 = B[3, 1, ix, iy, iz, it]'
        b21 = B[1, 2, ix, iy, iz, it]'
        b22 = B[2, 2, ix, iy, iz, it]'
        b23 = B[3, 2, ix, iy, iz, it]'
        b31 = B[1, 3, ix, iy, iz, it]'
        b32 = B[2, 3, ix, iy, iz, it]'
        b33 = B[3, 3, ix, iy, iz, it]'

        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end

#C =  α* A'*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_AdagBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        a11 = α * A[1, 1, ix, iy, iz, it]'
        a12 = α * A[2, 1, ix, iy, iz, it]'
        a13 = α * A[3, 1, ix, iy, iz, it]'
        a21 = α * A[1, 2, ix, iy, iz, it]'
        a22 = α * A[2, 2, ix, iy, iz, it]'
        a23 = α * A[3, 2, ix, iy, iz, it]'
        a31 = α * A[1, 3, ix, iy, iz, it]'
        a32 = α * A[2, 3, ix, iy, iz, it]'
        a33 = α * A[3, 3, ix, iy, iz, it]'


        b11 = B[1, 1, ix, iy, iz, it]'
        b12 = B[2, 1, ix, iy, iz, it]'
        b13 = B[3, 1, ix, iy, iz, it]'
        b21 = B[1, 2, ix, iy, iz, it]'
        b22 = B[2, 2, ix, iy, iz, it]'
        b23 = B[3, 2, ix, iy, iz, it]'
        b31 = B[1, 3, ix, iy, iz, it]'
        b32 = B[2, 3, ix, iy, iz, it]'
        b33 = B[3, 3, ix, iy, iz, it]'

        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::LatticeMatrix{4,T2,AT2,NC1,NC2,nw}) where {T1,T2,AT1,AT2,NC1,NC2,nw}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute!, C.A, A.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end

@inline function kernel_4Dsubstitute!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2,nw}}) where {T1,T2,AT1,AT2,NC1,NC2,nw}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_dag!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end

@inline function kernel_4Dsubstitute_dag!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[jc, ic, ix+nw, iy+nw, iz+nw, it+nw]'
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2,nw},shift}) where {T1,T2,AT1,AT2,NC1,NC2,shift,nw}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_shift!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end
export substitute!

@inline function kernel_4Dsubstitute_shift!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN, shift) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    #println("ix, iy, iz, it = ", (ix, iy, iz, it))
    #println("ix, iy, iz, it = ", (ixp, iyp, izp, itp))
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2,nw},shift}}) where {T1,T2,AT1,AT2,NC1,NC2,shift,nw}
    JACC.parallel_for(
        prod(C.PN), kernel_4Dsubstitute_shiftdag!, C.A, A.data.data.A, Val(NC1), Val(NC2), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end
export substitute!

@inline function kernel_4Dsubstitute_shiftdag!(i, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN, shift) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[jc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]'
        end
    end
end

#C = shiftedA*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = A[1, 1, ixp, iyp, izp, itp]
        a21 = A[2, 1, ixp, iyp, izp, itp]
        a31 = A[3, 1, ixp, iyp, izp, itp]
        a12 = A[1, 2, ixp, iyp, izp, itp]
        a22 = A[2, 2, ixp, iyp, izp, itp]
        a32 = A[3, 2, ixp, iyp, izp, itp]
        a13 = A[1, 3, ixp, iyp, izp, itp]
        a23 = A[2, 3, ixp, iyp, izp, itp]
        a33 = A[3, 3, ixp, iyp, izp, itp]

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α shiftedA*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_shiftAB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = α * A[1, 1, ixp, iyp, izp, itp]
        a21 = α * A[2, 1, ixp, iyp, izp, itp]
        a31 = α * A[3, 1, ixp, iyp, izp, itp]
        a12 = α * A[1, 2, ixp, iyp, izp, itp]
        a22 = α * A[2, 2, ixp, iyp, izp, itp]
        a32 = α * A[3, 2, ixp, iyp, izp, itp]
        a13 = α * A[1, 3, ixp, iyp, izp, itp]
        a23 = α * A[2, 3, ixp, iyp, izp, itp]
        a33 = α * A[3, 3, ixp, iyp, izp, itp]
        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end


end



#C = A*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end



@inline function kernel_4Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = A[1, 1, ix, iy, iz, it]
        a21 = A[2, 1, ix, iy, iz, it]
        a31 = A[3, 1, ix, iy, iz, it]
        a12 = A[1, 2, ix, iy, iz, it]
        a22 = A[2, 2, ix, iy, iz, it]
        a32 = A[3, 2, ix, iy, iz, it]
        a13 = A[1, 3, ix, iy, iz, it]
        a23 = A[2, 3, ix, iy, iz, it]
        a33 = A[3, 3, ix, iy, iz, it]
        b11 = B[1, 1, ixp, iyp, izp, itp]
        b21 = B[2, 1, ixp, iyp, izp, itp]
        b31 = B[3, 1, ixp, iyp, izp, itp]
        b12 = B[1, 2, ixp, iyp, izp, itp]
        b22 = B[2, 2, ixp, iyp, izp, itp]
        b32 = B[3, 2, ixp, iyp, izp, itp]
        b13 = B[1, 3, ixp, iyp, izp, itp]
        b23 = B[2, 3, ixp, iyp, izp, itp]
        b33 = B[3, 3, ixp, iyp, izp, itp]
        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end




#C = α A*shiftedB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}
    βin = T1(β)
    αin = T1(α)
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, αin, βin
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]


    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end



@inline function kernel_4Dmatrix_mul_AshiftB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]

        
        a11 = A[1,1,ix,iy,iz,it]; a21 = A[2,1,ix,iy,iz,it]; a31 = A[3,1,ix,iy,iz,it]
        a12 = A[1,2,ix,iy,iz,it]; a22 = A[2,2,ix,iy,iz,it]; a32 = A[3,2,ix,iy,iz,it]
        a13 = A[1,3,ix,iy,iz,it]; a23 = A[2,3,ix,iy,iz,it]; a33 = A[3,3,ix,iy,iz,it]

        b11 = B[1,1,ixp, iyp, izp, itp]; b21 = B[2,1,ixp, iyp, izp, itp]; b31 = B[3,1,ixp, iyp, izp, itp]
        c11 = a11*b11 + a12*b21 + a13*b31
        c21 = a21*b11 + a22*b21 + a23*b31
        c31 = a31*b11 + a32*b21 + a33*b31

        # ----  j=2 ----
        b12 = B[1,2,ixp, iyp, izp, itp]; b22 = B[2,2,ixp, iyp, izp, itp]; b32 = B[3,2,ixp, iyp, izp, itp]
        c12 = a11*b12 + a12*b22 + a13*b32
        c22 = a21*b12 + a22*b22 + a23*b32
        c32 = a31*b12 + a32*b22 + a33*b32

        # ----  j=3 ----
        b13 = B[1,3,ixp, iyp, izp, itp]; b23 = B[2,3,ixp, iyp, izp, itp]; b33 = B[3,3,ixp, iyp, izp, itp]
        c13 = a11*b13 + a12*b23 + a13*b33
        c23 = a21*b13 + a22*b23 + a23*b33
        c33 = a31*b13 + a32*b23 + a33*b33

        if iszero(β)
            C[1,1,ix,iy,iz,it] = α*c11; C[2,1,ix,iy,iz,it] = α*c21; C[3,1,ix,iy,iz,it] = α*c31
            C[1,2,ix,iy,iz,it] = α*c12; C[2,2,ix,iy,iz,it] = α*c22; C[3,2,ix,iy,iz,it] = α*c32
            C[1,3,ix,iy,iz,it] = α*c13; C[2,3,ix,iy,iz,it] = α*c23; C[3,3,ix,iy,iz,it] = α*c33
        else
            C[1,1,ix,iy,iz,it] = α*c11 + β*C[1,1,ix,iy,iz,it]
            C[2,1,ix,iy,iz,it] = α*c21 + β*C[2,1,ix,iy,iz,it]
            C[3,1,ix,iy,iz,it] = α*c31 + β*C[3,1,ix,iy,iz,it]
            C[1,2,ix,iy,iz,it] = α*c12 + β*C[1,2,ix,iy,iz,it]
            C[2,2,ix,iy,iz,it] = α*c22 + β*C[2,2,ix,iy,iz,it]
            C[3,2,ix,iy,iz,it] = α*c32 + β*C[3,2,ix,iy,iz,it]
            C[1,3,ix,iy,iz,it] = α*c13 + β*C[1,3,ix,iy,iz,it]
            C[2,3,ix,iy,iz,it] = α*c23 + β*C[2,3,ix,iy,iz,it]
            C[3,3,ix,iy,iz,it] = α*c33 + β*C[3,3,ix,iy,iz,it]
        end
        

        #=
        a11 = α * A[1, 1, ix, iy, iz, it]
        a21 = α * A[2, 1, ix, iy, iz, it]
        a31 = α * A[3, 1, ix, iy, iz, it]
        a12 = α * A[1, 2, ix, iy, iz, it]
        a22 = α * A[2, 2, ix, iy, iz, it]
        a32 = α * A[3, 2, ix, iy, iz, it]
        a13 = α * A[1, 3, ix, iy, iz, it]
        a23 = α * A[2, 3, ix, iy, iz, it]
        a33 = α * A[3, 3, ix, iy, iz, it]
        b11 = B[1, 1, ixp, iyp, izp, itp]
        b21 = B[2, 1, ixp, iyp, izp, itp]
        b31 = B[3, 1, ixp, iyp, izp, itp]
        b12 = B[1, 2, ixp, iyp, izp, itp]
        b22 = B[2, 2, ixp, iyp, izp, itp]
        b32 = B[3, 2, ixp, iyp, izp, itp]
        b13 = B[1, 3, ixp, iyp, izp, itp]
        b23 = B[2, 3, ixp, iyp, izp, itp]
        b33 = B[3, 3, ixp, iyp, izp, itp]
        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
        =#
    end


end







#C = shiftedA'*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = A[1, 1, ixp, iyp, izp, itp]'
        a12 = A[2, 1, ixp, iyp, izp, itp]'
        a13 = A[3, 1, ixp, iyp, izp, itp]'
        a21 = A[1, 2, ixp, iyp, izp, itp]'
        a22 = A[2, 2, ixp, iyp, izp, itp]'
        a23 = A[3, 2, ixp, iyp, izp, itp]'
        a31 = A[1, 3, ixp, iyp, izp, itp]'
        a32 = A[2, 3, ixp, iyp, izp, itp]'
        a33 = A[3, 3, ixp, iyp, izp, itp]'

        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α*shiftedA'*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_shiftAdagB!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = α * A[1, 1, ixp, iyp, izp, itp]'
        a12 = α * A[2, 1, ixp, iyp, izp, itp]'
        a13 = α * A[3, 1, ixp, iyp, izp, itp]'
        a21 = α * A[1, 2, ixp, iyp, izp, itp]'
        a22 = α * A[2, 2, ixp, iyp, izp, itp]'
        a23 = α * A[3, 2, ixp, iyp, izp, itp]'
        a31 = α * A[1, 3, ixp, iyp, izp, itp]'
        a32 = α * A[2, 3, ixp, iyp, izp, itp]'
        a33 = α * A[3, 3, ixp, iyp, izp, itp]'
        b11 = B[1, 1, ix, iy, iz, it]
        b21 = B[2, 1, ix, iy, iz, it]
        b31 = B[3, 1, ix, iy, iz, it]
        b12 = B[1, 2, ix, iy, iz, it]
        b22 = B[2, 2, ix, iy, iz, it]
        b32 = B[3, 2, ix, iy, iz, it]
        b13 = B[1, 3, ix, iy, iz, it]
        b23 = B[2, 3, ix, iy, iz, it]
        b33 = B[3, 3, ix, iy, iz, it]
        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end


end


#C = shiftedA*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

@inline function kernel_4Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = A[1, 1, ixp, iyp, izp, itp]
        a21 = A[2, 1, ixp, iyp, izp, itp]
        a31 = A[3, 1, ixp, iyp, izp, itp]
        a12 = A[1, 2, ixp, iyp, izp, itp]
        a22 = A[2, 2, ixp, iyp, izp, itp]
        a32 = A[3, 2, ixp, iyp, izp, itp]
        a13 = A[1, 3, ixp, iyp, izp, itp]
        a23 = A[2, 3, ixp, iyp, izp, itp]
        a33 = A[3, 3, ixp, iyp, izp, itp]

        b11 = B[1, 1, ix, iy, iz, it]'
        b12 = B[2, 1, ix, iy, iz, it]'
        b13 = B[3, 1, ix, iy, iz, it]'
        b21 = B[1, 2, ix, iy, iz, it]'
        b22 = B[2, 2, ix, iy, iz, it]'
        b23 = B[3, 2, ix, iy, iz, it]'
        b31 = B[1, 3, ix, iy, iz, it]'
        b32 = B[2, 3, ix, iy, iz, it]'
        b33 = B[3, 3, ix, iy, iz, it]'

        C[1, 1, ix, iy, iz, it] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = a31 * b13 + a32 * b23 + a33 * b33
    end
end


#C = α*shiftedA*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixp+nw, iyp+nw, izp+nw, itp+nw] * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end




@inline function kernel_4Dmatrix_mul_shiftABdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]


        a11 = α * A[1, 1, ixp, iyp, izp, itp]
        a21 = α * A[2, 1, ixp, iyp, izp, itp]
        a31 = α * A[3, 1, ixp, iyp, izp, itp]
        a12 = α * A[1, 2, ixp, iyp, izp, itp]
        a22 = α * A[2, 2, ixp, iyp, izp, itp]
        a32 = α * A[3, 2, ixp, iyp, izp, itp]
        a13 = α * A[1, 3, ixp, iyp, izp, itp]
        a23 = α * A[2, 3, ixp, iyp, izp, itp]
        a33 = α * A[3, 3, ixp, iyp, izp, itp]
        b11 = B[1, 1, ix, iy, iz, it]'
        b12 = B[2, 1, ix, iy, iz, it]'
        b13 = B[3, 1, ix, iy, iz, it]'

        b21 = B[1, 2, ix, iy, iz, it]'
        b22 = B[2, 2, ix, iy, iz, it]'
        b23 = B[3, 2, ix, iy, iz, it]'

        b31 = B[1, 3, ix, iy, iz, it]'
        b32 = B[2, 3, ix, iy, iz, it]'
        b33 = B[3, 3, ix, iy, iz, it]'

        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
    end

end


#C = shiftedA'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C = α*shiftedA'*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end


#C = A'*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end

#C = α*A'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
            end
        end
    end
end


#C = A*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            #C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            C[ic, jc, ix, iy, iz, it] = zero(eltype(C))
        end
        for kc = 1:NC3
            b = conj(B[jc, kc, ixp, iyp, izp, itp])
            for ic = 1:NC1
                C[ic, jc, ix, iy, iz, it] += A[ic, kc, ix, iy, iz, it] * b
            end
        end
    end
end

#C = α*A*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end


@inline function kernel_4Dmatrix_mul_AshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN, shift, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        ixp = ix + shift[1]
        iyp = iy + shift[2]
        izp = iz + shift[3]
        itp = it + shift[4]

        a11 = A[1,1,ix,iy,iz,it]; a21 = A[2,1,ix,iy,iz,it]; a31 = A[3,1,ix,iy,iz,it]
        a12 = A[1,2,ix,iy,iz,it]; a22 = A[2,2,ix,iy,iz,it]; a32 = A[3,2,ix,iy,iz,it]
        a13 = A[1,3,ix,iy,iz,it]; a23 = A[2,3,ix,iy,iz,it]; a33 = A[3,3,ix,iy,iz,it]

        b11 = B[1,1,ixp, iyp, izp, itp]'; b21 = B[1,2,ixp, iyp, izp, itp]'; b31 = B[1,3,ixp, iyp, izp, itp]'
        c11 = a11*b11 + a12*b21 + a13*b31
        c21 = a21*b11 + a22*b21 + a23*b31
        c31 = a31*b11 + a32*b21 + a33*b31

        # ----  j=2 ----
        b12 = B[2,1,ixp, iyp, izp, itp]'; b22 = B[2,2,ixp, iyp, izp, itp]'; b32 = B[2,3,ixp, iyp, izp, itp]'
        c12 = a11*b12 + a12*b22 + a13*b32
        c22 = a21*b12 + a22*b22 + a23*b32
        c32 = a31*b12 + a32*b22 + a33*b32

        # ----  j=3 ----
        b13 = B[3,1,ixp, iyp, izp, itp]'; b23 = B[3,2,ixp, iyp, izp, itp]'; b33 = B[3,3,ixp, iyp, izp, itp]'
        c13 = a11*b13 + a12*b23 + a13*b33
        c23 = a21*b13 + a22*b23 + a23*b33
        c33 = a31*b13 + a32*b23 + a33*b33

        if iszero(β)
            C[1,1,ix,iy,iz,it] = α*c11; C[2,1,ix,iy,iz,it] = α*c21; C[3,1,ix,iy,iz,it] = α*c31
            C[1,2,ix,iy,iz,it] = α*c12; C[2,2,ix,iy,iz,it] = α*c22; C[3,2,ix,iy,iz,it] = α*c32
            C[1,3,ix,iy,iz,it] = α*c13; C[2,3,ix,iy,iz,it] = α*c23; C[3,3,ix,iy,iz,it] = α*c33
        else
            C[1,1,ix,iy,iz,it] = α*c11 + β*C[1,1,ix,iy,iz,it]
            C[2,1,ix,iy,iz,it] = α*c21 + β*C[2,1,ix,iy,iz,it]
            C[3,1,ix,iy,iz,it] = α*c31 + β*C[3,1,ix,iy,iz,it]
            C[1,2,ix,iy,iz,it] = α*c12 + β*C[1,2,ix,iy,iz,it]
            C[2,2,ix,iy,iz,it] = α*c22 + β*C[2,2,ix,iy,iz,it]
            C[3,2,ix,iy,iz,it] = α*c32 + β*C[3,2,ix,iy,iz,it]
            C[1,3,ix,iy,iz,it] = α*c13 + β*C[1,3,ix,iy,iz,it]
            C[2,3,ix,iy,iz,it] = α*c23 + β*C[2,3,ix,iy,iz,it]
            C[3,3,ix,iy,iz,it] = α*c33 + β*C[3,3,ix,iy,iz,it]
        end

        #=
        a11 = α * A[1, 1, ix, iy, iz, it]
        a21 = α * A[2, 1, ix, iy, iz, it]
        a31 = α * A[3, 1, ix, iy, iz, it]
        a12 = α * A[1, 2, ix, iy, iz, it]
        a22 = α * A[2, 2, ix, iy, iz, it]
        a32 = α * A[3, 2, ix, iy, iz, it]
        a13 = α * A[1, 3, ix, iy, iz, it]
        a23 = α * A[2, 3, ix, iy, iz, it]
        a33 = α * A[3, 3, ix, iy, iz, it]
        b11 = conj(B[1, 1, ixp, iyp, izp, itp])
        b12 = conj(B[2, 1, ixp, iyp, izp, itp])
        b13 = conj(B[3, 1, ixp, iyp, izp, itp])

        b21 = conj(B[1, 2, ixp, iyp, izp, itp])
        b22 = conj(B[2, 2, ixp, iyp, izp, itp])
        b23 = conj(B[3, 2, ixp, iyp, izp, itp])

        b31 = conj(B[1, 3, ixp, iyp, izp, itp])
        b32 = conj(B[2, 3, ixp, iyp, izp, itp])
        b33 = conj(B[3, 3, ixp, iyp, izp, itp])

        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33
        =#
    end

end





#C = A'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end

#C = α*A'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ixp+nw, iyp+nw, izp+nw, itp+nw]'
            end
        end
    end
end



#C = shiftA*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shiftA,shiftB,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = α*shiftA*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = shiftA'*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = α*shiftA'*shiftedB+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftB!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[kc, jc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]
            end
        end
    end
end

#C = shiftA*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#C = α* shiftA*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number}

    #println((shiftA, shiftB))
    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#=
@inline function kernel_4Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{2}, ::Val{2}, ::Val{2}, ::Val{nw},
     PN, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    begin
        ixpA = ix + shiftA[1]
        iypA = iy + shiftA[2]
        izpA = iz + shiftA[3]
        itpA = it + shiftA[4]

        ixpB = ix + shiftB[1]
        iypB = iy + shiftB[2]
        izpB = iz + shiftB[3]
        itpB = it + shiftB[4]

        a11 = α * A[1, 1, ixpA, iypA, izpA, itpA]
        a21 = α * A[2, 1, ixpA, iypA, izpA, itpA]
        a12 = α * A[1, 2, ixpA, iypA, izpA, itpA]
        a22 = α * A[2, 2, ixpA, iypA, izpA, itpA]

        b11 = B[1, 1, ixpB, iypB, izpB, itpB]'
        b12 = B[2, 1, ixpB, iypB, izpB, itpB]'
        b21 = B[1, 2, ixpB, iypB, izpB, itpB]'
        b22 = B[2, 2, ixpB, iypB, izpB, itpB]'


        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 
    end



end
=#

@inline function kernel_4Dmatrix_mul_shiftAshiftBdag!(i, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw},
     PN, shiftA, shiftB, α::S, β::S) where {nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    @inbounds begin
        ixpA = ix + shiftA[1]
        iypA = iy + shiftA[2]
        izpA = iz + shiftA[3]
        itpA = it + shiftA[4]

        ixpB = ix + shiftB[1]
        iypB = iy + shiftB[2]
        izpB = iz + shiftB[3]
        itpB = it + shiftB[4]

        a11 = α * A[1, 1, ixpA, iypA, izpA, itpA]
        a21 = α * A[2, 1, ixpA, iypA, izpA, itpA]
        a31 = α * A[3, 1, ixpA, iypA, izpA, itpA]
        a12 = α * A[1, 2, ixpA, iypA, izpA, itpA]
        a22 = α * A[2, 2, ixpA, iypA, izpA, itpA]
        a32 = α * A[3, 2, ixpA, iypA, izpA, itpA]
        a13 = α * A[1, 3, ixpA, iypA, izpA, itpA]
        a23 = α * A[2, 3, ixpA, iypA, izpA, itpA]
        a33 = α * A[3, 3, ixpA, iypA, izpA, itpA]

        b11 = B[1, 1, ixpB, iypB, izpB, itpB]'
        b12 = B[2, 1, ixpB, iypB, izpB, itpB]'
        b13 = B[3, 1, ixpB, iypB, izpB, itpB]'

        b21 = B[1, 2, ixpB, iypB, izpB, itpB]'
        b22 = B[2, 2, ixpB, iypB, izpB, itpB]'
        b23 = B[3, 2, ixpB, iypB, izpB, itpB]'


        b31 = B[1, 3, ixpB, iypB, izpB, itpB]'
        b32 = B[2, 3, ixpB, iypB, izpB, itpB]'
        b33 = B[3, 3, ixpB, iypB, izpB, itpB]'


        C[1, 1, ix, iy, iz, it] = β * C[1, 1, ix, iy, iz, it] + a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1, ix, iy, iz, it] = β * C[2, 1, ix, iy, iz, it] + a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1, ix, iy, iz, it] = β * C[3, 1, ix, iy, iz, it] + a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2, ix, iy, iz, it] = β * C[1, 2, ix, iy, iz, it] + a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2, ix, iy, iz, it] = β * C[2, 2, ix, iy, iz, it] + a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2, ix, iy, iz, it] = β * C[3, 2, ix, iy, iz, it] + a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3, ix, iy, iz, it] = β * C[1, 3, ix, iy, iz, it] + a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3, ix, iy, iz, it] = β * C[2, 3, ix, iy, iz, it] + a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3, ix, iy, iz, it] = β * C[3, 3, ix, iy, iz, it] + a31 * b13 + a32 * b23 + a33 * b33

    end



end


#C = shiftA'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#C = α*shiftA'*shiftedB'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw,S<:Number}

    JACC.parallel_for(
        prod(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftBdag!(i, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixpA = ix + shiftA[1]
    iypA = iy + shiftA[2]
    izpA = iz + shiftA[3]
    itpA = it + shiftA[4]

    ixpB = ix + shiftB[1]
    iypB = iy + shiftB[2]
    izpB = iz + shiftB[3]
    itpB = it + shiftB[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ixpA+nw, iypA+nw, izpA+nw, itpA+nw]' * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

function LinearAlgebra.tr(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}) where {T1,AT1,NC1,NC2,nw}
    @assert NC1 == NC2 "Trace is only defined for square matrices"
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_tr_4D, C.A, Val(NC1), C.PN, Val(nw); init=zero(eltype(C.A)))::T1
    return s
end

@inline _preduce(n, op, kern, A, vNC1, PN, vnw, init::T) where {T} =
    JACC.parallel_reduce(n, op, kern, A, vNC1, PN, vnw; init=init)::T

function LinearAlgebra.tr(C::LatticeMatrix{4,T1,AT1,NC1,NC1,nw}) where {T1,AT1,NC1,nw}
    return _preduce(prod(C.PN), +, kernel_tr_4D, C.A, Val(NC1), C.PN, Val(nw), zero(T1))::T1
end


@inline function kernel_tr_4D(i, A, ::Val{NC1}, PN, ::Val{nw}) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    s = zero(eltype(A))
    @inbounds for ic = 1:NC1
        s += A[ic, ic, ix+nw, iy+nw, iz+nw, it+nw]
    end
    return s
end

function LinearAlgebra.dot(A::LatticeMatrix{4,T1,AT1,NC1,1,nw}, B::LatticeMatrix{4,T2,AT2,NC1,1,nw}) where {T1<:Real,T2<:Real,AT1,AT2,NC1,nw}
    s = JACC.parallel_reduce(prod(A.PN), +, kernel_dot_real_1,
        A.A, B.A, A.PN, Val(NC1), Val(nw); init=zero(eltype(A.A)))
end

@inline function kernel_dot_real_1(i, A, B, PN, ::Val{NC1}, ::Val{nw}) where {NC1,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    s = zero(eltype(A))

    @inbounds for ic = 1:NC1
        s += A[ic, 1, ix, iy, iz, it] * B[ic, 1, ix, iy, iz, it]
    end
    return s
end



#=
function LinearAlgebra.tr(C::LatticeMatrix{4,T1,AT1,3,3}) where {T1,AT1}
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_tr_4D_NC3, C.A, C.PN, Val(nw); init=zero(eltype(C.A)))
end

function kernel_tr_4D_NC3(i1,i2,i3, A, PN, nw)
    ix, iy, iz, it = get_4Dindex(i, PN)
    s = zero(eltype(A))
    for ic = 1:3
        s += A[ic, ic, ix+nw, iy+nw, iz+nw, it+nw]
    end
    return s
end
=#

function partial_trace(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, μ::Int, position::Int=1) where {T1,AT1,NC1,NC2,nw}
    s = JACC.parallel_reduce(prod(C.PN), +, kernel_partial_trace_4D, C.A, NC1, C.PN, μ, position, Val(nw); init=zero(eltype(C.A)))
    return s
end
export partial_trace

@inline function kernel_partial_trace_4D(i, A, NC, PN, μ, position, ::Val{nw}) where nw
    NN = get_4Dindex(i, PN)

    ix, iy, iz, it = NN

    s = zero(eltype(A))
    if NN[μ] == position
        for ic = 1:NC
            s += A[ic, ic, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
    return s
end

# ========== host side ==========
function normalize_matrix!(C::LatticeMatrix{4,T,AT,NC,NC,nw}) where {T,AT,NC,nw}
    if NC == 2
        JACC.parallel_for(prod(C.PN), kernel_normalize_NC2!, C.A, C.PN, Val(nw))
    elseif NC == 3
        JACC.parallel_for(prod(C.PN), kernel_normalize_NC3!, C.A, C.PN, Val(nw))
    else
        # Generic: modified Gram–Schmidt per site (unitarize columns)
        JACC.parallel_for(prod(C.PN), kernel_normalize_generic!, C.A, C.PN, NC, Val(nw))
    end
    #set_halo!(C)
end
export normalize_matrix!


@inline function kernel_normalize_NC2!(i, u, PN, ::Val{nw}) where nw
    ix, iy, iz, it = get_4Dindex(i, PN)
    α = u[1, 1, ix+nw, iy+nw, iz+nw, it+nw]
    β = u[2, 1, ix+nw, iy+nw, iz+nw, it+nw]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, ix+nw, iy+nw, iz+nw, it+nw] = α / detU
    u[2, 1, ix+nw, iy+nw, iz+nw, it+nw] = β / detU
    u[1, 2, ix+nw, iy+nw, iz+nw, it+nw] = -conj(β) / detU
    u[2, 2, ix+nw, iy+nw, iz+nw, it+nw] = conj(α) / detU
end

@inline function kernel_normalize_NC3!(i, u, PN, ::Val{nw}) where nw
    ix, iy, iz, it = get_4Dindex(i, PN)
    w1 = 0
    w2 = 0
    @inbounds for ic = 1:3
        w1 += u[2, ic, ix+nw, iy+nw, iz+nw, it+nw] * conj(u[1, ic, ix+nw, iy+nw, iz+nw, it+nw])
        w2 += u[1, ic, ix+nw, iy+nw, iz+nw, it+nw] * conj(u[1, ic, ix+nw, iy+nw, iz+nw, it+nw])
    end
    zerock2 = w2
    w1 = -w1 / w2

    x4 = (u[2, 1, ix+nw, iy+nw, iz+nw, it+nw]) + w1 * u[1, 1, ix+nw, iy+nw, iz+nw, it+nw]
    x5 = (u[2, 2, ix+nw, iy+nw, iz+nw, it+nw]) + w1 * u[1, 2, ix+nw, iy+nw, iz+nw, it+nw]
    x6 = (u[2, 3, ix+nw, iy+nw, iz+nw, it+nw]) + w1 * u[1, 3, ix+nw, iy+nw, iz+nw, it+nw]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3

    u[2, 1, ix+nw, iy+nw, iz+nw, it+nw] = x4
    u[2, 2, ix+nw, iy+nw, iz+nw, it+nw] = x5
    u[2, 3, ix+nw, iy+nw, iz+nw, it+nw] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1, ix+nw, iy+nw, iz+nw, it+nw] = u[1, 1, ix+nw, iy+nw, iz+nw, it+nw] * w2
    u[1, 2, ix+nw, iy+nw, iz+nw, it+nw] = u[1, 2, ix+nw, iy+nw, iz+nw, it+nw] * w2
    u[1, 3, ix+nw, iy+nw, iz+nw, it+nw] = u[1, 3, ix+nw, iy+nw, iz+nw, it+nw] * w2
    u[2, 1, ix+nw, iy+nw, iz+nw, it+nw] = u[2, 1, ix+nw, iy+nw, iz+nw, it+nw] * w3
    u[2, 2, ix+nw, iy+nw, iz+nw, it+nw] = u[2, 2, ix+nw, iy+nw, iz+nw, it+nw] * w3
    u[2, 3, ix+nw, iy+nw, iz+nw, it+nw] = u[2, 3, ix+nw, iy+nw, iz+nw, it+nw] * w3

    aa1 = real(u[1, 1, ix+nw, iy+nw, iz+nw, it+nw])
    aa2 = imag(u[1, 1, ix+nw, iy+nw, iz+nw, it+nw])
    aa3 = real(u[1, 2, ix+nw, iy+nw, iz+nw, it+nw])
    aa4 = imag(u[1, 2, ix+nw, iy+nw, iz+nw, it+nw])
    aa5 = real(u[1, 3, ix+nw, iy+nw, iz+nw, it+nw])
    aa6 = imag(u[1, 3, ix+nw, iy+nw, iz+nw, it+nw])
    aa7 = real(u[2, 1, ix+nw, iy+nw, iz+nw, it+nw])
    aa8 = imag(u[2, 1, ix+nw, iy+nw, iz+nw, it+nw])
    aa9 = real(u[2, 2, ix+nw, iy+nw, iz+nw, it+nw])
    aa10 = imag(u[2, 2, ix+nw, iy+nw, iz+nw, it+nw])
    aa11 = real(u[2, 3, ix+nw, iy+nw, iz+nw, it+nw])
    aa12 = imag(u[2, 3, ix+nw, iy+nw, iz+nw, it+nw])

    aa13 =
        aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
    aa14 =
        aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
    aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
    aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
    aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
    aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

    u[3, 1, ix+nw, iy+nw, iz+nw, it+nw] = aa13 + im * aa14
    u[3, 2, ix+nw, iy+nw, iz+nw, it+nw] = aa15 + im * aa16
    u[3, 3, ix+nw, iy+nw, iz+nw, it+nw] = aa17 + im * aa18

end



# ========== device side (generic N) ==========
# Normalize columns in-place to form a unitary (QR with Q-only), per lattice site
@inline function kernel_normalize_generic!(i, u, PN, NC, ::Val{nw}) where nw
    # Index decode
    ix, iy, iz, it = get_4Dindex(i, PN)

    # Type helpers
    T = eltype(u)
    rT = real(one(T))
    epsT = sqrt(eps(rT))  # tolerance for near-zero norms

    # Modified Gram–Schmidt over columns j = 1..NC
    @inbounds for j = 1:NC
        # Orthogonalize column j against columns 1..j-1
        for k = 1:j-1
            # inner = ⟨u[:,k], u[:,j]⟩ = sum(conj(u[k]) * u[j])
            inner = zero(T)
            for r = 1:NC
                inner += conj(u[r, k, ix+nw, iy+nw, iz+nw, it+nw]) * u[r, j, ix+nw, iy+nw, iz+nw, it+nw]
            end
            # u[:,j] -= inner * u[:,k]
            for r = 1:NC
                u[r, j, ix+nw, iy+nw, iz+nw, it+nw] -= inner * u[r, k, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end

        # Compute 2-norm of column j
        nrm2 = zero(rT)
        for r = 1:NC
            nrm2 += abs2(u[r, j, ix+nw, iy+nw, iz+nw, it+nw])
        end
        nrm = sqrt(nrm2)

        # Handle near-zero; fall back to a canonical basis vector
        if nrm < epsT
            # Zero column then set j-th row to 1 (produces consistent unitary completion)
            for r = 1:NC
                u[r, j, ix+nw, iy+nw, iz+nw, it+nw] = zero(T)
            end
            u[j, j, ix+nw, iy+nw, iz+nw, it+nw] = one(T)
        else
            # Normalize column j
            invn = one(rT) / nrm
            invnT = convert(T, invn)  # keep type stability for Complex/Real T
            for r = 1:NC
                u[r, j, ix+nw, iy+nw, iz+nw, it+nw] *= invnT
            end
        end
    end

    # Optional: single re-orthogonalization sweep for improved numerical stability
    # (uncomment if needed)
    # @inbounds for j = 1:NC
    #     for k = 1:j-1
    #         inner = zero(T)
    #         for r = 1:NC
    #             inner += conj(u[r,k,ix,iy,iz,it]) * u[r,j,ix,iy,iz,it]
    #         end
    #         for r = 1:NC
    #             u[r,j,ix,iy,iz,it] -= inner * u[r,k,ix,iy,iz,it]
    #         end
    #     end
    #     nrm2 = zero(rT)
    #     for r = 1:NC
    #         nrm2 += abs2(u[r,j,ix,iy,iz,it])
    #     end
    #     nrm = sqrt(nrm2)
    #     invnT = convert(T, one(rT)/max(nrm, epsT))
    #     for r = 1:NC
    #         u[r,j,ix,iy,iz,it] *= invnT
    #     end
    # end

    return nothing
end

#=
function randomize_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}) where {T,AT,NC1,NC2,nw}
    JACC.parallel_for(prod(C.PN), kernel_randomize_4D!, C.A, C.PN, NC1, NC2)
    #set_halo!(C)
end
export randomize_matrix!

@inline function kernel_randomize_4D!(i1,i2,i3, u, PN, NC1, NC2)
    ix, iy, iz, it = get_4Dindex(i, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix, iy, iz, it] = pcgrand(rng,eltype(u)) - 0.5 + im * (pcgrand(rng,eltype(u)) - 0.5)
        end
    end

end
=#

# Host wrapper: choose a fixed or time-based seed and launch
function randomize_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}) where {T,AT,NC1,NC2,nw}
    seed0 = UInt64(0x12345678ABCDEF01)  # or UInt64(time_ns())
    JACC.parallel_for(prod(C.PN), kernel_randomize_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw), seed0)
    set_halo!(C)
end
export randomize_matrix!

# We split on element type at compile time via Val to avoid dynamic branches.
@inline function kernel_randomize_4D!(i, u, PN, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    T = eltype(u)

    if T === ComplexF32
        _rand_fill!(Val(:c32), ix, iy, iz, it, u, Val(NC1), Val(NC2), Val(nw), seed0)
    elseif T === ComplexF64
        _rand_fill!(Val(:c64), ix, iy, iz, it, u, Val(NC1), Val(NC2), Val(nw), seed0)
    elseif T === Float32
        _rand_fill!(Val(:r32), ix, iy, iz, it, u, Val(NC1), Val(NC2), Val(nw), seed0)
    elseif T === Float64
        _rand_fill!(Val(:r64), ix, iy, iz, it, u, Val(NC1), Val(NC2), Val(nw), seed0)
    else
        # If you ever support other types, you can add more specializations.
        # For now, throw a clear error on host side before launching widely.
        @assert false "Unsupported eltype in randomize: $(T)"
    end
    return nothing
end

# --- Specializations (no convert(T, ...) inside) ---

# ComplexF32
@inline function _rand_fill!(::Val{:c32}, ix, iy, iz, it, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(ix + nw, iy + nw, iz + nw, it + nw, ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        state, r2 = pcg32_step(state, inc)
        realv = u01_f32(r1) - 0.5f0
        imagv = u01_f32(r2) - 0.5f0
        u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = ComplexF32(realv, imagv)
    end
    return nothing
end

# ComplexF64
@inline function _rand_fill!(::Val{:c64}, ix, iy, iz, it, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(ix + nw, iy + nw, iz + nw, it + nw, ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        state, r2 = pcg32_step(state, inc)
        realv = u01_f64(r1, r2) - 0.5
        state, i1 = pcg32_step(state, inc)
        state, i2 = pcg32_step(state, inc)
        imagv = u01_f64(i1, i2) - 0.5
        u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = ComplexF64(realv, imagv)
    end
    return nothing
end

# Float32
@inline function _rand_fill!(::Val{:r32}, ix, iy, iz, it, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(ix + nw, iy + nw, iz + nw, it + nw, ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        realv = u01_f32(r1) - 0.5f0
        u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = realv  # already Float32
    end
    return nothing
end

# Float64
@inline function _rand_fill!(::Val{:r64}, ix, iy, iz, it, u, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    @inbounds for jc = 1:NC2, ic = 1:NC1
        state, inc = mix_seed(ix + nw, iy + nw, iz + nw, it + nw, ic, jc, seed0)
        state, r1 = pcg32_step(state, inc)
        state, r2 = pcg32_step(state, inc)
        realv = u01_f64(r1, r2) - 0.5
        u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = realv  # already Float64
    end
    return nothing
end

function clear_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}) where {T,AT,NC1,NC2,nw}
    JACC.parallel_for(prod(C.PN), kernel_clear_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
end
export clear_matrix!

@inline function kernel_clear_4D!(i, u, PN, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = zero(eltype(u))
        end
    end

end

function makeidentity_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}) where {T,AT,NC1,NC2,nw}
    JACC.parallel_for(prod(C.PN), kernel_makeidentity_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
end
export makeidentity_matrix!


export makeidentity_matrix!

@inline function kernel_makeidentity_4D!(i, u, PN, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = ifelse(ic == jc, one(eltype(u)), zero(eltype(u)))
        end
    end

end


@inline function kernel_makeidentity_4D!(i, u, PN, ::Val{3}, ::Val{3}, ::Val{nw}) where {nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    v1 = one(eltype(u))
    v0 = zero(eltype(u))
    u[1, 1, ix, iy, iz, it] = v1
    u[2, 1, ix, iy, iz, it] = v0
    u[3, 1, ix, iy, iz, it] = v0
    u[1, 2, ix, iy, iz, it] = v0
    u[2, 2, ix, iy, iz, it] = v1
    u[3, 2, ix, iy, iz, it] = v0
    u[1, 3, ix, iy, iz, it] = v0
    u[2, 3, ix, iy, iz, it] = v0
    u[3, 3, ix, iy, iz, it] = v1

end


#C = C+ α*A
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,nw,S<:Number}
    JACC.parallel_for(prod(C.PN), kernel_add_4D!, C.A, A.A, C.PN, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end
export add_matrix!

@inline function kernel_add_4D!(i, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * v[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
end

#C = C+ α*shiftA
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2,nw},shift}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,shift,nw,S<:Number}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_shift!, C.A, A.data.A, C.PN, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end


@inline function kernel_add_4D_shift!(i, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * v[ic, jc, ixp+nw, iyp+nw, izp+nw, itp+nw]
        end
    end
end

#C = C+ α*Adag
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::Adjoint_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2,nw}}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,nw,S<:Number}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_dag!, C.A, A.data.A, C.PN, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end

@inline function kernel_add_4D_dag!(i, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * v[jc, ic, ix+nw, iy+nw, iz+nw, it+nw]'
        end
    end
end

#C = C+ α*shiftAdag
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2,nw},shift}}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,shift,nw,S<:Number}
    JACC.parallel_for(prod(C.PN), kernel_add_4D_shiftdag!, C.A, A.data.data.A, C.PN, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end


@inline function kernel_add_4D_shiftdag!(i, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * v[jc, ic, ixp+nw, iyp+nw, izp+nw, itp+nw]'
        end
    end
end

function applyfunction!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, f::Function, variables...) where {T,AT,NC1,NC2,nw}
    JACC.parallel_for(prod(C.PN), kernel_apply_function_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw), f, variables...)
    #set_halo!(C)
end
export applyfunction!

@inline function kernel_apply_function_4D!(i, u, PN, ::Val{N1}, ::Val{N2}, ::Val{nw}, f, variables...) where {N1,N2,nw}
    ix, iy, iz, it = get_4Dindex(i, PN)
    At = MMatrix{N1,N2,eltype(u)}(undef)

    @inbounds for jc = 1:N2
        for ic = 1:N1
            At[ic, jc] = u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
    Aout = f(At, variables...)

    for jc = 1:N2
        for ic = 1:N1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = Aout[ic, jc]
        end
    end
end

function traceless_antihermitian_add!(C::LatticeMatrix{4,T,AT,NG,1,nw}, factor,
    A::LatticeMatrix{4,T2,AT2,NC,NC,nw2}) where {T<:Real,AT,NG,nw,T2,AT2,NC,nw2}
    JACC.parallel_for(prod(C.PN), kernel_4d_Traceless_antihermitian_add!, C.A, A.A, factor, C.PN, Val(NG), Val(NC), Val(nw), Val(nw2))
end

function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, PN, ::Val{NG}, ::Val{NC}, ::Val{nw}, ::Val{nw2}) where {NC,NG,nw,nw2}
    error("NC > 3 is not supported")
end

const fac12 = 1/2

function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, PN, ::Val{NG}, ::Val{2}, ::Val{nw}, ::Val{nw2}) where {NG,nw,nw2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix2 = ix + nw2
    iy2 = iy + nw2
    iz2 = iz + nw2
    it2 = it + nw2
    ix += nw
    iy += nw
    iz += nw
    it += nw

    v11 = vin[1, 1, ix2, iy2, iz2, it2]
    v22 = vin[2, 2, ix2, iy2, iz2, it2]

    tri = fac12 * (imag(v11) + imag(v22))

    v12 = vin[1, 2, ix2, iy2, iz2, it2]
    #v13 = vin[1,3,ix,iy,iz,it]
    v21 = vin[2, 1, ix2, iy2, iz2, it2]

    x12 = v12 - conj(v21)

    x21 = -conj(x12)

    y11 = (imag(v11) - tri) * im
    y12 = 0.5 * x12
    y21 = 0.5 * x21
    y22 = (imag(v22) - tri) * im

    c[1, 1, ix, iy, iz, it] =
        (imag(y12) + imag(y21)) * factor + c[1, 1, ix, iy, iz, it]
    c[2, 1, ix, iy, iz, it] =
        (real(y12) - real(y21)) * factor + c[2, 1, ix, iy, iz, it]
    c[3, 1, ix, iy, iz, it] =
        (imag(y11) - imag(y22)) * factor + c[3, 1, ix, iy, iz, it]

end


function kernel_4d_Traceless_antihermitian_add!(i, c, vin, factor, PN, ::Val{NG}, ::Val{3}, ::Val{nw}, ::Val{nw2}) where {NG,nw,nw2}
    ix, iy, iz, it = get_4Dindex(i, PN)
    ix2 = ix + nw2
    iy2 = iy + nw2
    iz2 = iz + nw2
    it2 = it + nw2
    ix += nw
    iy += nw
    iz += nw
    it += nw

    fac13 = 1 / 3


    v11 = vin[1, 1, ix2, iy2, iz2, it2]
    v22 = vin[2, 2, ix2, iy2, iz2, it2]
    v33 = vin[3, 3, ix2, iy2, iz2, it2]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = vin[1, 2, ix2, iy2, iz2, it2]
    v13 = vin[1, 3, ix2, iy2, iz2, it2]
    v21 = vin[2, 1, ix2, iy2, iz2, it2]
    v23 = vin[2, 3, ix2, iy2, iz2, it2]
    v31 = vin[3, 1, ix2, iy2, iz2, it2]
    v32 = vin[3, 2, ix2, iy2, iz2, it2]

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


    c[1, 1, ix, iy, iz, it] =
        (imag(y12) + imag(y21)) * factor + c[1, 1, ix, iy, iz, it]
    c[2, 1, ix, iy, iz, it] =
        (real(y12) - real(y21)) * factor + c[2, 1, ix, iy, iz, it]
    c[3, 1, ix, iy, iz, it] =
        (imag(y11) - imag(y22)) * factor + c[3, 1, ix, iy, iz, it]
    c[4, 1, ix, iy, iz, it] =
        (imag(y13) + imag(y31)) * factor + c[4, 1, ix, iy, iz, it]
    c[5, 1, ix, iy, iz, it] =
        (real(y13) - real(y31)) * factor + c[5, 1, ix, iy, iz, it]

    c[6, 1, ix, iy, iz, it] =
        (imag(y23) + imag(y32)) * factor + c[6, 1, ix, iy, iz, it]
    c[7, 1, ix, iy, iz, it] =
        (real(y23) - real(y32)) * factor + c[7, 1, ix, iy, iz, it]
    c[8, 1, ix, iy, iz, it] =
        sr3i * (imag(y11) + imag(y22) - 2 * imag(y33)) * factor +
        c[8, 1, ix, iy, iz, it]
end