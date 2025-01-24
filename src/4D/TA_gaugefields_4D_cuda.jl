import CUDA

struct TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv} <: TA_Gaugefields_4D{NC}
    a::Ta
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NC::Int64
    NumofBasis::Int64
    generators::Union{Nothing,Generator}
    blockinfo::Blockindices
    temp_volume::TUv

    function TA_Gaugefields_4D_cuda(NC, NX, NY, NZ, NT,
        blocks)
        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        L = [NX,NY,NZ,NT]
        blockinfo = Blockindices(L,blocks)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize


        a = zeros(Float64, NumofBasis,blocksize,rsize ) |> CUDA.CuArray
        Ta = typeof(a)

        temp_volume = zeros(Float64, blocksize,rsize) |> CUDA.CuArray
        TUv = typeof(temp_volume)

        return new{NC,NumofBasis,Ta,TUv}(
            a,
            NX,
            NY,
            NZ,
            NT,
            NC,
            NumofBasis,
            generators,
            blockinfo,
            temp_volume 
        )
    end
end

function initialize_TA_Gaugefields(u::Gaugefields_4D_cuda) 
    return TA_Gaugefields_4D_cuda(u.NC, u.NX, u.NY, u.NZ, u.NT,
                u.blockinfo.blocks)
end

function cudakernel_substitute_TAU!(Uμ,pwork,blockinfo,NumofBasis,NX,NY,NZ,NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    ix,iy,iz,it = fourdim_cordinate(b,r,blockinfo)
    @inbounds for k=1:NumofBasis
        icount = ((((it-1)*NZ+iz-1)*NY+iy-1)*NX+ix-1)*NumofBasis+k
        Uμ[k, b,r] = pwork[icount]
    end

end

function substitute_U!(
    Uμ::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta},
    pwork::CUDA.CuArray,
) where {NC,NumofBasis,Ta}

    NT = Uμ.NT
    NZ = Uμ.NZ 
    NY = Uμ.NY
    NX = Uμ.NX

    CUDA.@sync begin
        CUDA.@cuda threads=Uμ.blockinfo.blocksize blocks=Uμ.blockinfo.rsize cudakernel_substitute_TAU!(
                Uμ.a,pwork,Uμ.blockinfo,NumofBasis,NX,NY,NZ,NT)
    end

end

function substitute_U!(
    Uμ::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta},
    pwork,
) where {NC,NumofBasis,Ta}

    pwork_gpu = pwork |> CUDA.CuArray
    substitute_U!(
        Uμ,
        pwork_gpu)
end

function Base.similar(u::TA_Gaugefields_4D_cuda{NC,NumofBasis}) where {NC,NumofBasis}
    return TA_Gaugefields_4D_cuda(NC, u.NX, u.NY, u.NZ, u.NT,u.blockinfo.blocks)
    #error("similar! is not implemented in type $(typeof(U)) ")
end


function cudakernel_mult_xTAyTA!(temp,x,y,NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    temp[b,r] = 0
    @inbounds for k = 1:NumofBasis
        temp[b,r] += x[k, b,r] * y[k, b,r]
    end
    return
end

function Base.:*(
    x::TA_Gaugefields_4D_cuda{NC,NumofBasis},
    y::TA_Gaugefields_4D_cuda{NC,NumofBasis},
) where {NC,NumofBasis}

    CUDA.@sync begin
        CUDA.@cuda threads=x.blockinfo.blocksize blocks=x.blockinfo.rsize cudakernel_mult_xTAyTA!(x.temp_volume,
                x.a,y.a,NumofBasis)
    end
    s = CUDA.reduce(+, x.temp_volume)
    return s
end



function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_cuda{2,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis} #uout = exp(t*u)
    error("exptU with 2 for type $(typeof(v)) is not implemented")
end

function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_cuda{NC,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis,NC} #uout = exp(t*u)
    error("exptU with general NC for type $(typeof(v)) is n ot implemented")
end

function cudakernel_exptU_TAwuww_NC3!(w,u,ww,t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)


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


function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_cuda{3,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    #clear_U!(uout)

    CUDA.@sync begin
        CUDA.@cuda threads=uout.blockinfo.blocksize blocks=uout.blockinfo.rsize cudakernel_exptU_TAwuww_NC3!(
            w.U,u.a,ww.U,t) #w,u,ww,t
    end


    mul!(uout, w', ww)

end

function cudakernel_Traceless_antihermitian_add_TAU_NC3!(
    c,vin,factor)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

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


function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_cuda{3,NumofBasis},
    factor,
    vin::Gaugefields_4D_cuda{3},
) where {NumofBasis}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")



    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize cudakernel_Traceless_antihermitian_add_TAU_NC3!(
            c.a,vin.U,factor)
    end
end