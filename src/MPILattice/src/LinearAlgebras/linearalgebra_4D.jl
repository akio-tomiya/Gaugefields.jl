@inline launch3d(PN::NTuple{4,Int}) = (PN[1], PN[2], PN[3] * PN[4])


function LinearAlgebra.mul!(C::LatticeVector{4,T,AT}, A::LatticeVector{4,T,AT}, B::LatticeVector{4,T,AT}) where {T,AT}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dvector_mul!, C.A, A.A, B.A, C.NC, nw, C.PN
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

@inline function kernel_4Dvector_mul!(i1, i2, i3, C, A, B, NC, nw, PN)
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for ic = 1:NC
        C[ic, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, ix+nw, iy+nw, iz+nw, it+nw] * B[ic, ix+nw, iy+nw, iz+nw, it+nw]
    end
end


#C = A B 
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#=
@inline function kernel_4Dmatrix_mul!(i1, i2, i3, C, A, B, ::Val{3}, ::Val{3}, ::Val{3}, ::Val{nw}, PN) where {nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw

    a11 = A[1, 1, ix,iy,iz,it]
    a21 = A[2, 1, ix,iy,iz,it]
    a31 = A[3, 1, ix,iy,iz,it]
    a12 = A[1, 2, ix,iy,iz,it]
    a22 = A[2, 2, ix,iy,iz,it]
    a32 = A[3, 2, ix,iy,iz,it]
    a13 = A[1, 3, ix,iy,iz,it]
    a23 = A[2, 3, ix,iy,iz,it]
    a33 = A[3, 3, ix,iy,iz,it]
    b11 = B[1, 1, ix,iy,iz,it]
    b21 = B[2, 1, ix,iy,iz,it]
    b31 = B[3, 1, ix,iy,iz,it]
    b12 = B[1, 2, ix,iy,iz,it]
    b22 = B[2, 2, ix,iy,iz,it]
    b32 = B[3, 2, ix,iy,iz,it]
    b13 = B[1, 3, ix,iy,iz,it]
    b23 = B[2, 3, ix,iy,iz,it]
    b33 = B[3, 3, ix,iy,iz,it]
    C[1, 1, ix,iy,iz,it] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, ix,iy,iz,it] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, ix,iy,iz,it] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, ix,iy,iz,it] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, ix,iy,iz,it] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, ix,iy,iz,it] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, ix,iy,iz,it] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, ix,iy,iz,it] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, ix,iy,iz,it] = a31 * b13 + a32 * b23 + a33 * b33

end
=#




#C = A B α + C β
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}, α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul!, C.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α, β
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α, β) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ix+nw, iy+nw, iz+nw, it+nw] * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end




function expt!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, t::S=one(S)) where {T,AT,NC1,NC2,S<:Number,T1,AT1,nw}
    @assert NC1 == NC2 "Matrix exponentiation requires square matrices, but got $(NC1) x $(NC2)."
    if NC1 == 3
        JACC.parallel_for(
            launch3d(C.PN), kernel_4Dexpt_NC3!, C.A, A.A, C.PN, Val(nw), t
        )
    elseif NC1 == 2
        JACC.parallel_for(
            launch3d(C.PN), kernel_4Dexpt_NC2!, C.A, A.A, C.PN, Val(nw), t
        )
    else
        JACC.parallel_for(
            launch3d(C.PN), kernel_4Dexpt!, C.A, A.A, C.PN, Val(nw), t, Val(NC1)
        )
    end
    #set_halo!(C)
end

@inline function kernel_4Dexpt_NC3!(i1, i2, i3, C, A, PN, ::Val{nw}, t) where nw
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

@inline function kernel_4Dexpt_NC2!(i1, i2, i3, C, A, PN, ::Val{nw}, t) where nw
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

@inline function kernel_4Dexpt!(i1, i2, i3, C, A, PN, ::Val{nw}, t, ::Val{N}) where {N,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    expm_pade13_writeback!(C, A, ix + nw, iy + nw, iz + nw, it + nw, t, Val(N))
    #C[:, :, ix, iy, iz, it] = expm_pade13(A[:, :, ix, iy, iz, it], t)
end



#C = A'*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end

#C = α*A'*B+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[kc, jc, ix+nw, iy+nw, iz+nw, it+nw]
            end
        end
    end
end


#C = A*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_ABdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_ABdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = A'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw}
    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = 0
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

#C =  α* A'*B' + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,nw,S<:Number}
    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagBdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = β * C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
            for kc = 1:NC3
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[kc, ic, ix+nw, iy+nw, iz+nw, it+nw]' * B[jc, kc, ix+nw, iy+nw, iz+nw, it+nw]'
            end
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::LatticeMatrix{4,T2,AT2,NC1,NC2,nw}) where {T1,T2,AT1,AT2,NC1,NC2,nw}
    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dsubstitute!, C.A, A.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end

@inline function kernel_4Dsubstitute!(i1, i2, i3, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2,nw}}) where {T1,T2,AT1,AT2,NC1,NC2,nw}
    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dsubstitute_dag!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.PN
    )
    #set_halo!(C)
end

@inline function kernel_4Dsubstitute_dag!(i1, i2, i3, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = A[jc, ic, ix+nw, iy+nw, iz+nw, it+nw]'
        end
    end
end

function substitute!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC2,nw},shift}) where {T1,T2,AT1,AT2,NC1,NC2,shift,nw}
    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dsubstitute_shift!, C.A, A.data.A, Val(NC1), Val(NC2), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end
export substitute!

@inline function kernel_4Dsubstitute_shift!(i1, i2, i3, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN, shift) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dsubstitute_shiftdag!, C.A, A.data.data.A, Val(NC1), Val(NC2), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end
export substitute!

@inline function kernel_4Dsubstitute_shiftdag!(i1, i2, i3, C, A, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, PN, shift) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = α shiftedA*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAB!, C.A, A.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = A*shiftedB
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = α A*shiftedB + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::LatticeMatrix{4,T2,AT2,NC1,NC3,nw}, B::Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AshiftB!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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





#C = shiftedA'*B
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = α*shiftedA'*B + β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::LatticeMatrix{4,T3,AT3,NC3,NC2,nw},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagB!, C.A, A.data.data.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = shiftedA*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftABdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = α*shiftedA*B'+β*C
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}},
    α::S, β::S) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw,S<:Number}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftABdag!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftABdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = shiftedA'*B'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shift}}, B::Adjoint_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagBdag!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_AshiftBdag!, C.A, A.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#C = A'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw}}, B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shift}}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3,shift,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_AdagshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shift, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_AdagshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shift, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAshiftB!, C.A, A.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftB!, C.A, A.data.data.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftB!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAshiftBdag!, C.A, A.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
                C[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * A[ic, kc, ixpA+nw, iypA+nw, izpA+nw, itpA+nw] * B[jc, kc, ixpB+nw, iypB+nw, izpB+nw, itpB+nw]'
            end
        end
    end
end

#C = shiftA'*shiftedB'
function LinearAlgebra.mul!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw},
    A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T2,AT2,NC1,NC3,nw},shiftA}},
    B::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T3,AT3,NC3,NC2,nw},shiftB}}) where {T1,T2,T3,AT1,AT2,
    AT3,NC1,NC2,NC3,shiftA,shiftB,nw}

    JACC.parallel_for(
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        launch3d(C.PN), kernel_4Dmatrix_mul_shiftAdagshiftBdag!, C.A, A.data.data.A, B.data.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), C.PN, shiftA, shiftB, α::S, β::S
    )
    #set_halo!(C)
end


@inline function kernel_4Dmatrix_mul_shiftAdagshiftBdag!(i1, i2, i3, C, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, PN, shiftA, shiftB, α::S, β::S) where {NC1,NC2,NC3,nw,S<:Number}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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

#=
function LinearAlgebra.tr(C::LatticeMatrix{4,T1,AT1,3,3}) where {T1,AT1}
    s = JACC.parallel_reduce(launch3d(C.PN), +, kernel_tr_4D_NC3, C.A, C.PN, Val(nw); init=zero(eltype(C.A)))
end

function kernel_tr_4D_NC3(i1,i2,i3, A, PN, nw)
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
        JACC.parallel_for(launch3d(C.PN), kernel_normalize_NC2!, C.A, C.PN, Val(nw))
    elseif NC == 3
        JACC.parallel_for(launch3d(C.PN), kernel_normalize_NC3!, C.A, C.PN, Val(nw))
    else
        # Generic: modified Gram–Schmidt per site (unitarize columns)
        JACC.parallel_for(launch3d(C.PN), kernel_normalize_generic!, C.A, C.PN, NC, Val(nw))
    end
    #set_halo!(C)
end
export normalize_matrix!


@inline function kernel_normalize_NC2!(i1, i2, i3, u, PN, ::Val{nw}) where nw
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
    α = u[1, 1, ix+nw, iy+nw, iz+nw, it+nw]
    β = u[2, 1, ix+nw, iy+nw, iz+nw, it+nw]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, ix+nw, iy+nw, iz+nw, it+nw] = α / detU
    u[2, 1, ix+nw, iy+nw, iz+nw, it+nw] = β / detU
    u[1, 2, ix+nw, iy+nw, iz+nw, it+nw] = -conj(β) / detU
    u[2, 2, ix+nw, iy+nw, iz+nw, it+nw] = conj(α) / detU
end

@inline function kernel_normalize_NC3!(i1, i2, i3, u, PN, ::Val{nw}) where nw
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
@inline function kernel_normalize_generic!(i1, i2, i3, u, PN, NC, ::Val{nw}) where nw
    # Index decode
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)

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
    JACC.parallel_for(launch3d(C.PN), kernel_randomize_4D!, C.A, C.PN, NC1, NC2)
    #set_halo!(C)
end
export randomize_matrix!

@inline function kernel_randomize_4D!(i1,i2,i3, u, PN, NC1, NC2)
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)

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
    JACC.parallel_for(launch3d(C.PN), kernel_randomize_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw), seed0)
    set_halo!(C)
end
export randomize_matrix!

# We split on element type at compile time via Val to avoid dynamic branches.
@inline function kernel_randomize_4D!(i1, i2, i3, u, PN, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, seed0::UInt64) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
    JACC.parallel_for(launch3d(C.PN), kernel_clear_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
end
export clear_matrix!

@inline function kernel_clear_4D!(i1, i2, i3, u, PN, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = zero(eltype(u))
        end
    end

end

function makeidentity_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}) where {T,AT,NC1,NC2,nw}
    JACC.parallel_for(launch3d(C.PN), kernel_makeidentity_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw))
    set_halo!(C)
end
export makeidentity_matrix!

@inline function kernel_makeidentity_4D!(i1, i2, i3, u, PN, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] = ifelse(ic == jc, one(eltype(u)), zero(eltype(u)))
        end
    end

end

#C = C+ α*A
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::LatticeMatrix{4,T1,AT1,NC1,NC2,nw}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,nw,S<:Number}
    JACC.parallel_for(launch3d(C.PN), kernel_add_4D!, C.A, A.A, C.PN, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end
export add_matrix!

@inline function kernel_add_4D!(i1, i2, i3, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * v[ic, jc, ix+nw, iy+nw, iz+nw, it+nw]
        end
    end
end

#C = C+ α*shiftA
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2,nw},shift}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,shift,nw,S<:Number}
    JACC.parallel_for(launch3d(C.PN), kernel_add_4D_shift!, C.A, A.data.A, C.PN, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end


@inline function kernel_add_4D_shift!(i1, i2, i3, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
    JACC.parallel_for(launch3d(C.PN), kernel_add_4D_dag!, C.A, A.data.A, C.PN, Val(NC1), Val(NC2), α, Val(nw))
    #set_halo!(C)
end

@inline function kernel_add_4D_dag!(i1, i2, i3, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            u[ic, jc, ix+nw, iy+nw, iz+nw, it+nw] += α * v[jc, ic, ix+nw, iy+nw, iz+nw, it+nw]'
        end
    end
end

#C = C+ α*shiftAdag
function add_matrix!(C::LatticeMatrix{4,T,AT,NC1,NC2,nw}, A::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T1,AT1,NC1,NC2,nw},shift}}, α::S=1) where {T,T1,AT,AT1,NC1,NC2,shift,nw,S<:Number}
    JACC.parallel_for(launch3d(C.PN), kernel_add_4D_shiftdag!, C.A, A.data.data.A, C.PN, Val(NC1), Val(NC2), α, shift, Val(nw))
    #set_halo!(C)
end


@inline function kernel_add_4D_shiftdag!(i1, i2, i3, u, v, PN, ::Val{NC1}, ::Val{NC2}, α, shift, ::Val{nw}) where {NC1,NC2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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
    JACC.parallel_for(launch3d(C.PN), kernel_apply_function_4D!, C.A, C.PN, Val(NC1), Val(NC2), Val(nw), f, variables...)
    #set_halo!(C)
end
export applyfunction!

@inline function kernel_apply_function_4D!(i1, i2, i3, u, PN, ::Val{N1}, ::Val{N2}, ::Val{nw}, f, variables...) where {N1,N2,nw}
    ix, iy, iz, it = get_4Dindex(i1, i2, i3, PN)
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