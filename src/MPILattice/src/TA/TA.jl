struct TALattice{D,T,AT,N} <: Lattice{D,T,AT}
    lt::LatticeMatrix{D,T,AT,N,N}

    function TALattice(A::LatticeMatrix{D,T,AT,N,N2}) where {D,T,AT,N,N2}
        @assert N == N2 "The number of columns must match the number of rows."
        su = new{D,T,AT,N}(A)
        traceless_antihermitian!(su)
        return su
    end
end


export TALattice

const tinyvalue = 1e-100
const pi23 = 2pi / 3

const fac13 = 1 / 3
const sr3 = sqrt(3)
const sr3i = 1 / sr3
const sr3ih = 0.5 * sr3i
const sqr3inv = sr3i
const sr3i2 = 2 * sr3i

function traceless_antihermitian!(A::LatticeMatrix{4,T,AT,N,N}) where {T,AT,N}
    if N == 3
        JACC.parallel_for(
            prod(A.PN), kernel_traceless_antihermitian_4DNC3!, A.A, A.nw, A.PN)
    elseif N == 2
        JACC.parallel_for(
            prod(A.PN), kernel_traceless_antihermitian_4DNC2!, A.A, A.nw, A.PN)
    elseif N == 1
        @warn("No traceless antihermitian condition applied for SU(1). This is a scalar lattice, so no special unitary condition is needed.")
        # For N=1, no SU(N) condition is needed, as it is just a scalar.
    else
        JACC.parallel_for(
            prod(A.PN), kernel_traceless_antihermitian_4D!, A.A, N, A.nw, A.PN)
        #error("Unsupported number of colors for special unitary lattice: $N")
    end
    set_halo!(A)
end

function traceless_antihermitian!(A::TALattice{4,T,AT,N}) where {T,AT,N}
    traceless_antihermitian!(A.lt)
    #=
    if N == 3
        JACC.parallel_for(
            prod(A.lt.PN), kernel_traceless_antihermitian_4DNC3!, A.lt.A, A.lt.nw, A.lt.PN)
    elseif N == 2
        JACC.parallel_for(
            prod(A.lt.PN), kernel_traceless_antihermitian_4DNC2!, A.lt.A, A.lt.nw, A.lt.PN)
    elseif N == 1
        @warn("No traceless antihermitian condition applied for SU(1). This is a scalar lattice, so no special unitary condition is needed.")
        # For N=1, no SU(N) condition is needed, as it is just a scalar.
    else
        JACC.parallel_for(
            prod(A.lt.PN), kernel_traceless_antihermitian_4D!, A.lt.A, N, A.lt.nw, A.lt.PN)
        #error("Unsupported number of colors for special unitary lattice: $N")
    end
    set_halo!(A.lt)
    =#
end


function kernel_traceless_antihermitian_4DNC2!(i, v, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    v11 = v[1, 1, ix, iy, iz, it]
    v22 = v[2, 2, ix, iy, iz, it]

    tri = (imag(v11) + imag(v22)) * 0.5

    v12 = v[1, 2, ix, iy, iz, it]
    v21 = v[2, 1, ix, iy, iz, it]

    x12 = v12 - conj(v21)

    x21 = -conj(x12)

    v[1, 1, ix, iy, iz, it] = (imag(v11) - tri) * im
    v[1, 2, ix, iy, iz, it] = 0.5 * x12
    v[2, 1, ix, iy, iz, it] = 0.5 * x21
    v[2, 2, ix, iy, iz, it] = (imag(v22) - tri) * im

end

function kernel_traceless_antihermitian_4DNC3!(i, v, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    v11 = v[1, 1, ix, iy, iz, it]
    v21 = v[2, 1, ix, iy, iz, it]
    v31 = v[3, 1, ix, iy, iz, it]

    v12 = v[1, 2, ix, iy, iz, it]
    v22 = v[2, 2, ix, iy, iz, it]
    v32 = v[3, 2, ix, iy, iz, it]

    v13 = v[1, 3, ix, iy, iz, it]
    v23 = v[2, 3, ix, iy, iz, it]
    v33 = v[3, 3, ix, iy, iz, it]


    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

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

    v[1, 1, ix, iy, iz, it] = y11
    v[2, 1, ix, iy, iz, it] = y21
    v[3, 1, ix, iy, iz, it] = y31

    v[1, 2, ix, iy, iz, it] = y12
    v[2, 2, ix, iy, iz, it] = y22
    v[3, 2, ix, iy, iz, it] = y32

    v[1, 3, ix, iy, iz, it] = y13
    v[2, 3, ix, iy, iz, it] = y23
    v[3, 3, ix, iy, iz, it] = y33

end

function kernel_traceless_antihermitian_4D!(i, v, N, nw, PN)
    ix, iy, iz, it = get_4Dindex(i, PN)
    fac1N = 1 / N
    tri = 0.0
    for k = 1:N
        tri += imag(v[k, k, ix, iy, iz, it])
    end
    tri *= fac1N
    for k = 1:N
        v[k, k, ix, iy, iz, it] =
            (imag(v[k, k, ix, iy, iz, it]) - tri) * im
    end


    for k1 = 1:N
        for k2 = k1+1:N
            vv =
                0.5 * (
                    v[k1, k2, ix, iy, iz, it] -
                    conj(v[k2, k1, ix, iy, iz, it])
                )
            v[k1, k2, ix, iy, iz, it] = vv
            v[k2, k1, ix, iy, iz, it] = -conj(vv)
        end
    end


end


function expt!(C::LatticeMatrix{4,T,AT,NC1,NC2}, A::TALattice{4,T1,AT1,NC1}, t::S=one(S)) where {T,AT,NC1,NC2,S<:Number,T1,AT1}
    @assert NC1 == NC2 "Matrix exponentiation requires square matrices, but got $(NC1) x $(NC2)."
    if NC1 == 3
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt_SU3!, C.A, A.lt.A, C.PN, t
        )
    elseif NC1 == 2
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt_SU2!, C.A, A.lt.A, C.PN, t
        )
    else
        JACC.parallel_for(
            prod(C.PN), kernel_4Dexpt!, C.A, A.lt.A, C.PN, t, Val(NC1)
        )
    end
    set_halo!(C)
end

function kernel_4Dexpt_SU3!(i, C, A, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)
    T = eltype(C)

    y11 = A[1, 1, ix, iy, iz, it]
    y22 = A[2, 2, ix, iy, iz, it]
    y33 = A[3, 3, ix, iy, iz, it]
    y12 = A[1, 2, ix, iy, iz, it]
    y13 = A[1, 3, ix, iy, iz, it]
    y21 = A[2, 1, ix, iy, iz, it]
    y23 = A[2, 3, ix, iy, iz, it]
    y31 = A[3, 1, ix, iy, iz, it]
    y32 = A[3, 2, ix, iy, iz, it]

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
    #C[:, :, ix, iy, iz, it] = T[c.a11 c.a12 c.a13;
    #    c.a21 c.a22 c.a23;
    #    c.a31 c.a32 c.a33]

    C[1, 1, ix, iy, iz, it] = c.a11
    C[1, 2, ix, iy, iz, it] = c.a12
    C[1, 3, ix, iy, iz, it] = c.a13
    C[2, 1, ix, iy, iz, it] = c.a21
    C[2, 2, ix, iy, iz, it] = c.a22
    C[2, 3, ix, iy, iz, it] = c.a23
    C[3, 1, ix, iy, iz, it] = c.a31
    C[3, 2, ix, iy, iz, it] = c.a32
    C[3, 3, ix, iy, iz, it] = c.a33

    #=
    w[1, 1, ix, iy, iz, it] = w1 + im * w2
    w[1, 2, ix, iy, iz, it] = w3 + im * w4
    w[1, 3, ix, iy, iz, it] = w5 + im * w6
    w[2, 1, ix, iy, iz, it] = w7 + im * w8
    w[2, 2, ix, iy, iz, it] = w9 + im * w10
    w[2, 3, ix, iy, iz, it] = w11 + im * w12
    w[3, 1, ix, iy, iz, it] = w13 + im * w14
    w[3, 2, ix, iy, iz, it] = w15 + im * w16
    w[3, 3, ix, iy, iz, it] = w17 + im * w18

    ww[1, 1, ix, iy, iz, it] = ww1 + im * ww2
    ww[1, 2, ix, iy, iz, it] = ww3 + im * ww4
    ww[1, 3, ix, iy, iz, it] = ww5 + im * ww6
    ww[2, 1, ix, iy, iz, it] = ww7 + im * ww8
    ww[2, 2, ix, iy, iz, it] = ww9 + im * ww10
    ww[2, 3, ix, iy, iz, it] = ww11 + im * ww12
    ww[3, 1, ix, iy, iz, it] = ww13 + im * ww14
    ww[3, 2, ix, iy, iz, it] = ww15 + im * ww16
    ww[3, 3, ix, iy, iz, it] = ww17 + im * ww18
    =#

end

function kernel_4Dexpt_SU2!(i, uout, v, PN, t)
    ix, iy, iz, it = get_4Dindex(i, PN)

    y11 = v[1, 1, ix, iy, iz, it]
    y12 = v[1, 2, ix, iy, iz, it]
    y21 = v[2, 1, ix, iy, iz, it]
    y22 = v[2, 2, ix, iy, iz, it]

    c1_0 = (imag(y12) + imag(y21))
    c2_0 = (real(y12) - real(y21))
    c3_0 = (imag(y11) - imag(y22))

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