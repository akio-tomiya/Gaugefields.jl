
function substitute_U!(
    Uμ::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc},
    pwork,
) where {NC,NumofBasis,Ta,TUv}

    NT = Uμ.NT
    NZ = Uμ.NZ
    NY = Uμ.NY
    NX = Uμ.NX

    N = NX * NY * NZ * NT
    JACC.parallel_for(N, jacckernel_substitute_TAU!, Uμ.a, JACC.array(pwork), NumofBasis, NX, NY, NZ, NT)

    #CUDA.@sync begin
    #    CUDA.@cuda threads = Uμ.blockinfo.blocksize blocks = Uμ.blockinfo.rsize jacckernel_substitute_TAU!(
    #        Uμ.a, pwork, Uμ.blockinfo, NumofBasis, NX, NY, NZ, NT)
    #end

end




function Base.:*(
    x::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc},
    y::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc},
) where {NC,NumofBasis,Ta,TUv}
    N = x.NX * x.NY * x.NZ * x.NT
    s = JACC.parallel_reduce(N, +, jacckernel_mult_xTAyTA!,
        x.a, y.a, NumofBasis; init=zero(eltype(x.a)))
    return s
end


function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{2,NumofBasis,Ta,TUv,:jacc},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta,TUv} #uout = exp(t*u)     

    NN = u.NX * u.NY * u.NZ * u.NT
    JACC.parallel_for(NN, jacckernel_exptU_TAwuww_NC2!,
        uout.U, u.a, t) #w,u,ww,t

    #error("not implemented")
    #CUDA.@sync begin
    #    CUDA.@cuda threads = uout.blockinfo.blocksize blocks = uout.blockinfo.rsize jacckernel_exptU_TAwuww_NC2!(
    #        uout.U, u.a, t) #w,u,ww,t
    #end


end

using InteractiveUtils

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc},
    temps::Array{T,1},
) where {NC,N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta,TUv} #uout = exp(t*u)     

    generators = Tuple(JACC.Array.(u.generators.generator))
    NG = length(generators)

    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]

    NN = u.NX * u.NY * u.NZ * u.NT
    JACC.parallel_for(NN, jacckernel_exptU_TAwuww_NC!,
        uout.U, u.a, t, NC, NG, generators, temp1.U, temp2.U, temp3.U)

    #=
    uoutcpu = Array(uout.U)
    ucpu = Array(u.a)
    temp1cpu = Array(temp1.U)
    temp2cpu = Array(temp2.U)
    temp3cpu = Array(temp3.U)
    generatorscpu = Array.(generators)
    b =1
    r = 1

    kernel_exptU_TAwuww_NC!(b,r,
            uoutcpu, ucpu, t,NC,NG,generatorscpu,temp1cpu,temp2cpu,temp3cpu)
    display(uoutcpu[:,:,b,r])  
    error("dd")
    =#

    #error("not implemented")
    #CUDA.@sync begin
    #    CUDA.@cuda threads = uout.blockinfo.blocksize blocks = uout.blockinfo.rsize jacckernel_exptU_TAwuww_NC!(
    #        uout.U, u.a, t, NC, NG, generators, temp1.U, temp2.U, temp3.U) #w,u,ww,t
    #end


end

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{3,NumofBasis,Ta,TUv,:jacc},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta,TUv} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    #clear_U!(uout)

    NN = u.NX * u.NY * u.NZ * u.NT
    JACC.parallel_for(NN, jacckernel_exptU_TAwuww_NC3!,
        w.U, u.a, ww.U, t) #w,u,ww,t)
    mul!(uout, w', ww)



end



function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{2,NumofBasis,Ta,TUv,:jacc},
    factor,
    vin::Gaugefields_4D_accelerator{2},
) where {NumofBasis,Ta,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_Traceless_antihermitian_add_TAU_NC2!,
        c.a, vin.U, factor)

end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{3,NumofBasis,Ta,TUv,:jacc},
    factor,
    vin::Gaugefields_4D_accelerator{3},
) where {NumofBasis,Ta,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_Traceless_antihermitian_add_TAU_NC3!,
        c.a, vin.U, factor)


end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc},
    factor,
    vin::Gaugefields_4D_accelerator{NC},
) where {NC,NumofBasis,Ta,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
    generators = Tuple(JACC.array.(c.generators.generator))
    NG = length(generators)
    temp1 = deepcopy(vin.U)
    N = c.NX * c.NY * c.NZ * c.NT
    tempa = JACC.array(zeros(eltype(vin.U), NumofBasis, N))

    JACC.parallel_for(N, jacckernel_Traceless_antihermitian_add_TAU_NC!,
        c.a, vin.U, factor, NC, NG, generators, temp1, tempa)

    return


    error("not implemented for NC > 3 case")

    generators = Tuple(CUDA.CuArray.(c.generators.generator))
    NG = length(generators)

    #temp1 = vin.Ushifted #reuse
    temp1 = deepcopy(vin.U)
    tempa = CUDA.CuArray{ComplexF64}(undef, NumofBasis, c.blockinfo.blocksize, c.blockinfo.rsize)


    #=
        CUDA.@sync begin
            CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize jacckernel_Traceless_antihermitian_add_TAU_NC!(
                c.a, vin.U, factor, NC, NG, generators, temp1, tempa)
        end
        =#
    #display(Array(c.a)[:,1,1])
    #error("ta")
end




function clear_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc}) where {NC,NumofBasis,Ta,TUv}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_clear_TAU!, c.a, NumofBasis)

end



function add_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc}, a::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc}) where {NC,NumofBasis,Ta,TUv}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_add_TAU!, c.a, a.a, NumofBasis)

end



function add_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis}, t::Number, a::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc}) where {NC,NumofBasis,Ta,TUv}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_add_TAU!, c.a, t, a.a, NumofBasis)

end

function gauss_distribution!(
    p::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc};
    σ=1.0,
) where {NC,NumofBasis,Ta,TUv}
    d = Normal(0.0, σ)
    NN = p.NX * p.NY * p.NZ * p.NT
    pwork = rand(d, NumofBasis, NN)
    p.a .= JACC.array(pwork)
end

function substitute_U!(A::TA_Gaugefields_4D_serial{NC}, B::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc}) where {NC,NumofBasis,Ta,TUv}
    bcpu = Array(B.a)

    NV = A.NX * A.NY * A.NZ * A.NT

    for i = 1:NV
        ix, iy, iz, it = index_to_coords(i, A.NX, A.NY, A.NZ, A.NT)
        for ic = 1:NumofBasis
            A[ic, ix, iy, iz, it] = bcpu[ic, i]
        end
    end
end



function substitute_U!(B::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:jacc}, A::TA_Gaugefields_4D_serial{NC}) where {NC,NumofBasis,Ta,TUv}
    bcpu = Array(B.a)

    NV = A.NX * A.NY * A.NZ * A.NT

    for i = 1:NV
        ix, iy, iz, it = index_to_coords(i, A.NX, A.NY, A.NZ, A.NT)
        for ic = 1:NumofBasis
            bcpu[ic, i] = A[ic, ix, iy, iz, it]
        end
    end

    B.a .= JACC.Array(bcpu)

end
