struct TA_Gaugefields_4D_mpi{NC,NumofBasis} <: TA_Gaugefields_4D{NC}
    a::Array{Float64,5}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NC::Int64
    NumofBasis::Int64
    generators::Union{Nothing,Generator}

    PEs::NTuple{4,Int64}
    PN::NTuple{4,Int64}
    mpiinit::Bool
    myrank::Int64
    nprocs::Int64
    myrank_xyzt::NTuple{4,Int64}

    function TA_Gaugefields_4D_mpi(u::Gaugefields_4D{NC}) where {NC}
        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        return new{NC,NumofBasis}(
            zeros(Float64, NumofBasis, u.PN[1], u.PN[2], u.PN[3], u.PN[4]),
            u.NX,
            u.NY,
            u.NZ,
            u.NT,
            u.NC,
            NumofBasis,
            generators,
            u.PEs,
            u.PN,
            true,
            u.myrank,
            u.nprocs,
            u.myrank_xyzt,
        )
    end
end

function Base.setindex!(x::T, v, i...) where {T<:TA_Gaugefields_4D_mpi}
    @inbounds x.a[i...] = v
end

function Base.getindex(x::T, i...) where {T<:TA_Gaugefields_4D_mpi}
    @inbounds return x.a[i...]
end


function barrier(x::TA_Gaugefields_4D_mpi)
    MPI.Barrier(comm)
end

function add_U!(
    c::TA_Gaugefields_4D_mpi{NC,NumofBasis},
    α::N,
    a::TA_Gaugefields_4D_mpi{NC,NumofBasis},
) where {NC,N<:Number,NumofBasis}
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    #NumofBasis = c.NumofBasis
    for it = 1:c.PN[4]
        for iz = 1:c.PN[3]
            for iy = 1:c.PN[2]
                for ix = 1:c.PN[1]
                    for k = 1:NumofBasis
                        c[k, ix, iy, iz, it] =
                            c[k, ix, iy, iz, it] + α * a[k, ix, iy, iz, it]
                    end
                end
            end
        end
    end

    barrier(c)
    #error("add_U! is not implemented in type $(typeof(c)) ")
end

function clear_U!(U::TA_Gaugefields_4D_mpi{NC,NumofBasis}) where {NC,NumofBasis}
    #NumofBasis = Uμ.NumofBasis
    for it = 1:U.PN[4]
        for iz = 1:U.PN[3]
            for iy = 1:U.PN[2]
                for ix = 1:U.PN[1]
                    for k = 1:NumofBasis
                        @inbounds U[k, ix, iy, iz, it] = 0
                    end
                end
            end
        end
    end
    barrier(U)
end

function gauss_distribution!(
    p::TA_Gaugefields_4D_mpi{NC,NumofBasis};
    σ = 1.0,
) where {NC,NumofBasis}
    d = Normal(0.0, σ)
    #NumofBasis = Uμ.NumofBasis
    pwork = rand(d, prod(p.PN) * NumofBasis)
    icount = 0
    @inbounds for it = 1:p.PN[4]
        for iz = 1:p.PN[3]
            for iy = 1:p.PN[2]
                @simd for ix = 1:p.PN[1]
                    for k = 1:NumofBasis
                        icount += 1
                        p[k, ix, iy, iz, it] = pwork[icount]
                    end
                end
            end
        end
    end
    barrier(p)
end

function Base.:*(
    x::TA_Gaugefields_4D_mpi{NC,NumofBasis},
    y::TA_Gaugefields_4D_mpi{NC,NumofBasis},
) where {NC,NumofBasis}
    #NumofBasis = Uμ.NumofBasis
    s = 0.0
    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                @simd for ix = 1:x.PN[1]
                    for k = 1:NumofBasis
                        s += x[k, ix, iy, iz, it] * y[k, ix, iy, iz, it]
                    end
                end
            end
        end
    end
    s = MPI.Allreduce(s, MPI.SUM, comm)
    barrier(x)

    return s
end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_mpi{3,NumofBasis},
    factor,
    vin::Union{Gaugefields_4D_wing_mpi{3},Gaugefields_4D_nowing_mpi{3}},
) where {NumofBasis}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
    fac13 = 1 / 3
    NX = vin.NX
    NY = vin.NY
    NZ = vin.NZ
    NT = vin.NT

    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    v11 = getvalue(vin, 1, 1, ix, iy, iz, it)
                    v22 = getvalue(vin, 2, 2, ix, iy, iz, it)
                    v33 = getvalue(vin, 3, 3, ix, iy, iz, it)

                    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

                    #=
                    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
                    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
                    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
                    =#
                    y11 = (imag(v11) - tri) * im
                    y22 = (imag(v22) - tri) * im
                    y33 = (imag(v33) - tri) * im

                    v12 = getvalue(vin, 1, 2, ix, iy, iz, it)
                    v13 = getvalue(vin, 1, 3, ix, iy, iz, it)
                    v21 = getvalue(vin, 2, 1, ix, iy, iz, it)
                    v23 = getvalue(vin, 2, 3, ix, iy, iz, it)
                    v31 = getvalue(vin, 3, 1, ix, iy, iz, it)
                    v32 = getvalue(vin, 3, 2, ix, iy, iz, it)

                    x12 = v12 - conj(v21)
                    x13 = v13 - conj(v31)
                    x23 = v23 - conj(v32)

                    x21 = -conj(x12)
                    x31 = -conj(x13)
                    x32 = -conj(x23)

                    #=
                    vout[1,2,ix,iy,iz,it) = 0.5  * x12
                    vout[1,3,ix,iy,iz,it) = 0.5  * x13
                    vout[2,1,ix,iy,iz,it) = 0.5  * x21
                    vout[2,3,ix,iy,iz,it) = 0.5  * x23
                    vout[3,1,ix,iy,iz,it) = 0.5  * x31
                    vout[3,2,ix,iy,iz,it) = 0.5  * x32
                    =#
                    y12 = 0.5 * x12
                    y13 = 0.5 * x13
                    y21 = 0.5 * x21
                    y23 = 0.5 * x23
                    y31 = 0.5 * x31
                    y32 = 0.5 * x32


                    c[1, ix, iy, iz, it] =
                        (imag(y12) + imag(y21)) * factor + c[1, ix, iy, iz, it]
                    c[2, ix, iy, iz, it] =
                        (real(y12) - real(y21)) * factor + c[2, ix, iy, iz, it]
                    c[3, ix, iy, iz, it] =
                        (imag(y11) - imag(y22)) * factor + c[3, ix, iy, iz, it]
                    c[4, ix, iy, iz, it] =
                        (imag(y13) + imag(y31)) * factor + c[4, ix, iy, iz, it]
                    c[5, ix, iy, iz, it] =
                        (real(y13) - real(y31)) * factor + c[5, ix, iy, iz, it]

                    c[6, ix, iy, iz, it] =
                        (imag(y23) + imag(y32)) * factor + c[6, ix, iy, iz, it]
                    c[7, ix, iy, iz, it] =
                        (real(y23) - real(y32)) * factor + c[7, ix, iy, iz, it]
                    c[8, ix, iy, iz, it] =
                        sr3i * (imag(y11) + imag(y22) - 2 * imag(y33)) * factor +
                        c[8, ix, iy, iz, it]


                end
            end
        end
    end
    barrier(c)


end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_mpi{2,NumofBasis},
    factor,
    vin::Union{Gaugefields_4D_wing_mpi{2},Gaugefields_4D_nowing_mpi{2}},
) where {NumofBasis}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
    fac12 = 1 / 2
    NX = vin.NX
    NY = vin.NY
    NZ = vin.NZ
    NT = vin.NT

    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    v11 = getvalue(vin, 1, 1, ix, iy, iz, it)
                    v22 = getvalue(vin, 2, 2, ix, iy, iz, it)

                    tri = fac12 * (imag(v11) + imag(v22))



                    v12 = getvalue(vin, 1, 2, ix, iy, iz, it)
                    #v13 = getvalue(vin,1,3,ix,iy,iz,it)
                    v21 = getvalue(vin, 2, 1, ix, iy, iz, it)

                    x12 = v12 - conj(v21)

                    x21 = -conj(x12)

                    y11 = (imag(v11) - tri) * im
                    y12 = 0.5 * x12
                    y21 = 0.5 * x21
                    y22 = (imag(v22) - tri) * im

                    c[1, ix, iy, iz, it] =
                        (imag(y12) + imag(y21)) * factor + c[1, ix, iy, iz, it]
                    c[2, ix, iy, iz, it] =
                        (real(y12) - real(y21)) * factor + c[2, ix, iy, iz, it]
                    c[3, ix, iy, iz, it] =
                        (imag(y11) - imag(y22)) * factor + c[3, ix, iy, iz, it]

                end
            end
        end
    end
    barrier(c)


end

"""
-----------------------------------------------------c
     !!!!!   vin and vout should be different vectors

     Projectin of the etraceless antiermite part 
     vout = x/2 - Tr(x)/6
     wher   x = vin - Conjg(vin)      
-----------------------------------------------------c
    """
function Traceless_antihermitian!(
    c::TA_Gaugefields_4D_mpi{3,NumofBasis},
    vin::Union{Gaugefields_4D_wing_mpi{3},Gaugefields_4D_nowing_mpi{3}},
) where {NumofBasis}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
    fac13 = 1 / 3
    NX = vin.NX
    NY = vin.NY
    NZ = vin.NZ
    NT = vin.NT

    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    v11 = getvalue(vin, 1, 1, ix, iy, iz, it)
                    v22 = getvalue(vin, 2, 2, ix, iy, iz, it)
                    v33 = getvalue(vin, 3, 3, ix, iy, iz, it)

                    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

                    #=
                    vout[1,1,ix,iy,iz,it) = (imag(v11)-tri)*im
                    vout[2,2,ix,iy,iz,it) = (imag(v22)-tri)*im
                    vout[3,3,ix,iy,iz,it) = (imag(v33)-tri)*im
                    =#
                    y11 = (imag(v11) - tri) * im
                    y22 = (imag(v22) - tri) * im
                    y33 = (imag(v33) - tri) * im

                    v12 = getvalue(vin, 1, 2, ix, iy, iz, it)
                    v13 = getvalue(vin, 1, 3, ix, iy, iz, it)
                    v21 = getvalue(vin, 2, 1, ix, iy, iz, it)
                    v23 = getvalue(vin, 2, 3, ix, iy, iz, it)
                    v31 = getvalue(vin, 3, 1, ix, iy, iz, it)
                    v32 = getvalue(vin, 3, 2, ix, iy, iz, it)

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

                    c[1, ix, iy, iz, it] = (imag(y12) + imag(y21))
                    c[2, ix, iy, iz, it] = (real(y12) - real(y21))
                    c[3, ix, iy, iz, it] = (imag(y11) - imag(y22))
                    c[4, ix, iy, iz, it] = (imag(y13) + imag(y31))
                    c[5, ix, iy, iz, it] = (real(y13) - real(y31))

                    c[6, ix, iy, iz, it] = (imag(y23) + imag(y32))
                    c[7, ix, iy, iz, it] = (real(y23) - real(y32))
                    c[8, ix, iy, iz, it] = sr3i * (imag(y11) + imag(y22) - 2 * imag(y33))
                end
            end
        end
    end
    barrier(c)


end

function Traceless_antihermitian!(
    c::TA_Gaugefields_4D_mpi{2,NumofBasis},
    vin::Union{Gaugefields_4D_wing_mpi{2},Gaugefields_4D_nowing_mpi{2}},
) where {NumofBasis}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
    fac12 = 1 / 2
    NX = vin.NX
    NY = vin.NY
    NZ = vin.NZ
    NT = vin.NT

    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    v11 = vin[1, 1, ix, iy, iz, it]
                    v22 = vin[2, 2, ix, iy, iz, it]

                    tri = fac12 * (imag(v11) + imag(v22))



                    v12 = vin[1, 2, ix, iy, iz, it]
                    #v13 = vin[1,3,ix,iy,iz,it]
                    v21 = vin[2, 1, ix, iy, iz, it]

                    x12 = v12 - conj(v21)

                    x21 = -conj(x12)

                    y11 = (imag(v11) - tri) * im
                    y12 = 0.5 * x12
                    y21 = 0.5 * x21
                    y22 = (imag(v22) - tri) * im

                    c[1, ix, iy, iz, it] = (imag(y12) + imag(y21))
                    c[2, ix, iy, iz, it] = (real(y12) - real(y21))
                    c[3, ix, iy, iz, it] = (imag(y11) - imag(y22))
                end
            end
        end
    end


end

function Traceless_antihermitian!(
    c::TA_Gaugefields_4D_mpi{NC,NumofBasis},
    vin::Union{Gaugefields_4D_wing_mpi{NC},Gaugefields_4D_nowing_mpi{NC}},
) where {NC,NumofBasis}
    @assert NC != 3 && NC != 2
    #NC = vout.NC
    fac1N = 1 / NC
    nv = vin.NV

    NX = vin.NX
    NY = vin.NY
    NZ = vin.NZ
    NT = vin.NT
    g = c.generators
    matrix = zeros(ComplexF64, NC, NC)
    a = zeros(ComplexF64, length(g))

    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    tri = 0.0
                    @simd for k = 1:NC
                        v = getvalue(vin, k, k, ix, iy, iz, it)
                        tri += imag(v)
                    end
                    tri *= fac1N
                    @simd for k = 1:NC
                        #vout[k,k,ix,iy,iz,it] = (imag(getvalue(vin,k,k,ix,iy,iz,it))-tri)*im
                        v = getvalue(vin, k, k, ix, iy, iz, it)
                        matrix[k, k] = (imag(v) - tri) * im
                    end

                    for k1 = 1:NC
                        @simd for k2 = k1+1:NC
                            v12 = getvalue(vin, k1, k2, ix, iy, iz, it)
                            v21 = getvalue(vin, k2, k1, ix, iy, iz, it)
                            vv = 0.5 * ( v12 - conj(v21) )
                            #vout[k1,k2,ix,iy,iz,it] = vv
                            #vout[k2,k1,ix,iy,iz,it] = -conj(vv)
                            matrix[k1, k2] = vv
                            matrix[k2, k1] = -conj(vv)
                        end
                    end

                    matrix2lie!(a, g, matrix)
                    for k = 1:length(g)
                        c[k, ix, iy, iz, it] = 2 * imag(a[k])
                    end

                end
            end
        end
    end
    barrier(c)



end


function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_mpi{NC,NumofBasis},
    factor,
    vin::Union{Gaugefields_4D_wing_mpi{NC},Gaugefields_4D_nowing_mpi{NC}},
) where {NC,NumofBasis}
    @assert NC != 3 && NC != 2 "NC should be NC >4! in this function"
    #NC = vout.NC
    fac1N = 1 / NC
    nv = vin.NV

    NX = vin.NX
    NY = vin.NY
    NZ = vin.NZ
    NT = vin.NT
    g = c.generators
    matrix = zeros(ComplexF64, NC, NC)
    a = zeros(ComplexF64, length(g))

    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    tri = 0.0
                    @simd for k = 1:NC
                        v = getvalue(vin, k, k, ix, iy, iz, it)
                        tri += imag(v)
                    end
                    tri *= fac1N
                    @simd for k = 1:NC
                        #vout[k,k,ix,iy,iz,it] = (imag(getvalue(vin,k,k,ix,iy,iz,it))-tri)*im
                        v = getvalue(vin, k, k, ix, iy, iz, it)
                        matrix[k, k] = (imag(v) - tri) * im
                    end

                    for k1 = 1:NC
                        @simd for k2 = k1+1:NC
                            v12 = getvalue(vin, k1, k2, ix, iy, iz, it)
                            v21 = getvalue(vin, k2, k1, ix, iy, iz, it)
                            vv = 0.5 * ( v12 - conj(v21) )
                            #vout[k1,k2,ix,iy,iz,it] = vv
                            #vout[k2,k1,ix,iy,iz,it] = -conj(vv)
                            matrix[k1, k2] = vv
                            matrix[k2, k1] = -conj(vv)
                        end
                    end

                    matrix2lie!(a, g, matrix)
                    for k = 1:length(g)
                        c[k, ix, iy, iz, it] =
                            2 * imag(a[k]) * factor + c[k, ix, iy, iz, it]
                    end

                end
            end
        end
    end
    barrier(c)



end

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_mpi{NC,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D,NC,NumofBasis} #uout = exp(t*u)
    @assert NC != 3 && NC != 2 "This function is for NC != 2,3, now, NC = $NC, and NumofBasis = $NumofBasis"
    g = u.generators
    NT = u.NT
    NZ = u.NZ
    NY = u.NY
    NX = u.NX
    V = zeros(ComplexF64, NC, NC)

    u0 = zeros(ComplexF64, NC, NC)
    a = zeros(Float64, length(g))
    for it = 1:vin.PN[4]
        for iz = 1:vin.PN[3]
            for iy = 1:vin.PN[2]
                @simd for ix = 1:vin.PN[1]
                    for k = 1:length(a)
                        a[k] = u[k, ix, iy, iz, it]
                    end

                    lie2matrix!(u0, g, a)
                    V[:, :] = exp(t * (im / 2) * u0)
                    for k2 = 1:NC
                        for k1 = 1:NC
                            v = V[k1, k2]
                            setvalue!(uout, v, k1, k2, ix, iy, iz, it)
                        end
                    end

                    #uout[:,:,ix,iy,iz,it] = exp(t*(im/2)*u0)

                end
            end
        end
    end
    barrier(uout)
    #error("exptU! is not implemented in type $(typeof(u)) ")
end

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_mpi{3,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D,NumofBasis} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    NT = u.NT
    NZ = u.NZ
    NY = u.NY
    NX = u.NX



    for it = 1:u.PN[4]
        for iz = 1:u.PN[3]
            for iy = 1:u.PN[2]
                for ix = 1:u.PN[1]
                    c1 = t * u[1, ix, iy, iz, it] * 0.5
                    c2 = t * u[2, ix, iy, iz, it] * 0.5
                    c3 = t * u[3, ix, iy, iz, it] * 0.5
                    c4 = t * u[4, ix, iy, iz, it] * 0.5
                    c5 = t * u[5, ix, iy, iz, it] * 0.5
                    c6 = t * u[6, ix, iy, iz, it] * 0.5
                    c7 = t * u[7, ix, iy, iz, it] * 0.5
                    c8 = t * u[8, ix, iy, iz, it] * 0.5
                    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
                    if csum == 0
                        v = 1
                        setvalue!(w, v, 1, 1, ix, iy, iz, it)# =   1 
                        v = 0
                        setvalue!(w, v, 1, 2, ix, iy, iz, it)
                        setvalue!(w, v, 1, 3, ix, iy, iz, it)# =    0 
                        setvalue!(w, v, 2, 1, ix, iy, iz, it)# =  0
                        v = 1
                        setvalue!(w, v, 2, 2, ix, iy, iz, it)# =   1 
                        v = 0
                        setvalue!(w, v, 2, 3, ix, iy, iz, it)# =   0
                        setvalue!(w, v, 3, 1, ix, iy, iz, it)# =   0 
                        setvalue!(w, v, 3, 2, ix, iy, iz, it)# =   0  
                        v = 1
                        setvalue!(w, v, 3, 3, ix, iy, iz, it)# =   1  

                        v = 1
                        setvalue!(ww, v, 1, 1, ix, iy, iz, it)# =   1
                        v = 0
                        setvalue!(ww, v, 1, 2, ix, iy, iz, it)# =   0
                        setvalue!(ww, v, 1, 3, ix, iy, iz, it)# =   0
                        setvalue!(ww, v, 2, 1, ix, iy, iz, it)# =   0
                        v = 1
                        setvalue!(ww, v, 2, 2, ix, iy, iz, it)# =   1 
                        v = 0
                        setvalue!(ww, v, 2, 3, ix, iy, iz, it)# =   0 
                        setvalue!(ww, v, 3, 1, ix, iy, iz, it)# =   0
                        setvalue!(ww, v, 3, 2, ix, iy, iz, it)# =   0  
                        v = 1
                        setvalue!(ww, v, 3, 3, ix, iy, iz, it)# =   1
                        continue
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

                    v = w1 + im * w2
                    setvalue!(w, v, 1, 1, ix, iy, iz, it)# =   1 
                    v = w3 + im * w4
                    setvalue!(w, v, 1, 2, ix, iy, iz, it)
                    v = w5 + im * w6
                    setvalue!(w, v, 1, 3, ix, iy, iz, it)# =    0 
                    v = w7 + im * w8
                    setvalue!(w, v, 2, 1, ix, iy, iz, it)# =  0
                    v = w9 + im * w10

                    setvalue!(w, v, 2, 2, ix, iy, iz, it)# =   1 
                    v = w11 + im * w12
                    setvalue!(w, v, 2, 3, ix, iy, iz, it)# =   0
                    v = w13 + im * w14
                    setvalue!(w, v, 3, 1, ix, iy, iz, it)# =   0 
                    v = w15 + im * w16
                    setvalue!(w, v, 3, 2, ix, iy, iz, it)# =   0  
                    v = 1 \ w17 + im * w18
                    setvalue!(w, v, 3, 3, ix, iy, iz, it)# =   1  

                    v = ww1 + im * ww2
                    setvalue!(ww, v, 1, 1, ix, iy, iz, it)# =   1
                    v = ww3 + im * ww4
                    setvalue!(ww, v, 1, 2, ix, iy, iz, it)# =   0
                    v = ww5 + im * ww6
                    setvalue!(ww, v, 1, 3, ix, iy, iz, it)# =   0
                    v = ww7 + im * ww8
                    setvalue!(ww, v, 2, 1, ix, iy, iz, it)# =   0
                    v = ww9 + im * ww10
                    setvalue!(ww, v, 2, 2, ix, iy, iz, it)# =   1 
                    v = ww11 + im * ww12
                    setvalue!(ww, v, 2, 3, ix, iy, iz, it)# =   0 
                    v = ww13 + im * ww14
                    setvalue!(ww, v, 3, 1, ix, iy, iz, it)# =   0
                    v = ww15 + im * ww16
                    setvalue!(ww, v, 3, 2, ix, iy, iz, it)# =   0  
                    v = ww17 + im * ww18
                    setvalue!(ww, v, 3, 3, ix, iy, iz, it)# =   1


                end
            end
        end
    end
    barrier(uout)

    mul!(uout, w', ww)
    barrier(uout)


end
const tinyvalue = 1e-100


function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_mpi{2,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D,NumofBasis} #uout = exp(t*u)     
    NT = u.NT
    NZ = u.NZ
    NY = u.NY
    NX = u.NX


    for it = 1:u.PN[4]
        for iz = 1:u.PN[3]
            for iy = 1:u.PN[2]
                @simd for ix = 1:u.PN[1]
                    #icum = (((it-1)*NX+iz-1)*NY+iy-1)*NX+ix  
                    u1 = t * u[1, ix, iy, iz, it] / 2
                    u2 = t * u[2, ix, iy, iz, it] / 2
                    u3 = t * u[3, ix, iy, iz, it] / 2
                    R = sqrt(u1^2 + u2^2 + u3^2) + tinyvalue
                    sR = sin(R) / R
                    #sR = ifelse(R == 0,1,sR)
                    a0 = cos(R)
                    a1 = u1 * sR
                    a2 = u2 * sR
                    a3 = u3 * sR

                    v = cos(R) + im * a3
                    setvalue!(uout, v, 1, 1, ix, iy, iz, it)
                    v = im * a1 + a2
                    setvalue!(uout, v, 1, 2, ix, iy, iz, it)
                    #uout[2,1,ix,iy,iz,it) = 
                    v = im * a1 - a2
                    setvalue!(uout, v, 2, 1, ix, iy, iz, it)
                    #uout[2,2,ix,iy,iz,it)=
                    v = cos(R) - im * a3
                    setvalue!(uout, v, 2, 2, ix, iy, iz, it)

                end
            end
        end
    end

    barrier(uout)




end
