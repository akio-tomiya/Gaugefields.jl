function LinearAlgebra.tr(a::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc}) where {NC,TU,TUv}
    N = a.NX * a.NY * a.NZ * a.NT
    s = JACC.parallel_reduce(N, jacckernel_tr!, a.U, NC)#, c.U, A.U, B.U, NC)
    return s
    #CUDA.@sync begin
    #    CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, NC)
    #end
    #s = CUDA.reduce(+, a.temp_volume)
    #return s
end

#A*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::T1,
    B::T2) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.U, B.U, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.U, B.U, NC)
    #end
end

#alpha*A*B + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb,
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, A.U, B.U, α, β, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.U, B.U, α, β, NC)
    #end
end

#A*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::T1,
    B::Adjoint_Gaugefields{T2}) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!, c.U, A.U, B.parent.U, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(c.U, A.U, B.parent.U, NC)
    #end
end

#alpha*A*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!, c.U, A.U, B.parent.U, α, β, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(c.U, A.U, B.parent.U, α, β, NC)
    #end
end

#A'*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!, c.U, A.parent.U, B.parent.U, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  
    #     jacckernel_mul_NC_adagbdag!(c.U, A.parent.U, B.parent.U, NC)
    #end
end

#c = alpha * A'* B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!, c.U, A.parent.U, B.parent.U, α, β, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(c.U, A.parent.U, B.parent.U, α, β, NC)
    #end
end

#A'*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!, c.U, A.parent.U, B.U, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.U, B.U, NC)
    #end


end


#alpha*A'*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!, c.U, A.parent.U, B.U, α, β, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.U, B.U, α, β, NC)
    #end
end


#alpha*A*shiftB+C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abshift!, c.U, A.U, B.parent.U, α, β,
        B.shift, B.parent.blockinfo, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abshift!(c.U, A.U, B.parent.U, α, β,
    #        B.shift, B.parent.blockinfo, NC)
    #end

end

#alpha*A*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TshiftedU},
    A::T1,
    B::T2
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TshiftedU}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.U, B.parent.Ushifted, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.U, B.parent.Ushifted, NC)
    #end
end


#alpha*A*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TshiftedU},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TshiftedU}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.U, B.parent.Ushifted, α, β, NC)

end


#alpha*shiftA*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftb!, c.U, A.parent.U, B.U, α, β,
        A.shift, B.parent.blockinfo, NC)

    #        jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftb!(c.U, A.parent.U, B.U, α, β,
    #        A.shift, B.parent.blockinfo, NC)
    #end

end

#shiftA*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.parent.Ushifted, B.U, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.parent.Ushifted, B.U, NC)
    #end


end

#alpha*shiftA*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.parent.Ushifted, B.U, α, β, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.parent.Ushifted, B.U, α, β, NC)
    #end


end

#alpha*shiftA*shiftB +beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftbshift!, c.U, A.parent.U, B.parent.U, α, β,
        A.shift, B.shift, B.parent.blockinfo, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftbshift!(c.U, A.parent.U, B.parent.U, α, β,
    #        A.shift, B.shift, B.parent.blockinfo, NC)
    #end

end



#shiftA*shiftB 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.parent.Ushifted, B.parent.Ushifted, NC)



    #     jacc.@sync begin
    #         jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.parent.Ushifted, B.parent.Ushifted, NC)
    #    end


end

#alpha*shiftA*shiftB +beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC!, c.U, A.parent.Ushifted, B.parent.Ushifted, α, β, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC!(c.U, A.parent.Ushifted, B.parent.Ushifted, α, β, NC)
    #end

end



#alpha*shiftA*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftbshiftdag!,
        c.U, A.parent.U, B.parent.parent.U, α, β,
        A.shift, B.parent.shift, A.parent.blockinfo, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftbshiftdag!(
    #        c.U, A.parent.U, B.parent.parent.U, α, β,
    #        A.shift, B.parent.shift, A.parent.blockinfo, NC)
    #end


end

#shiftA*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!,
        c.U, A.parent.Ushifted, B.parent.parent.Ushifted, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(
    #        c.U, A.parent.Ushifted, B.parent.parent.Ushifted, NC)
    #end


end


#alpha*shiftA*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!,
        c.U, A.parent.Ushifted, B.parent.parent.Ushifted, α, β, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(
    #       c.U, A.parent.Ushifted, B.parent.parent.Ushifted, α, β, NC)
    #end

end

#alpha*A'*shiftB+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbshift!, c.U, A.parent.U, B.parent.U, α, β,
        B.shift, B.parent.blockinfo, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbshift!(c.U, A.parent.U, B.parent.U, α, β,
    #        B.shift, B.parent.blockinfo, NC)
    #end

end

#A'*shiftB
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!, c.U, A.parent.U, B.parent.Ushifted, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.U, B.parent.Ushifted, NC)
    #end

end

#alpha*A'*shiftB+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!,
        c.U, A.parent.U, B.parent.Ushifted, α, β, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.U, B.parent.Ushifted, α, β, NC)
    #end

end

#alpha*A'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbshiftdag!,
        c.U, A.parent.U, B.parent.parent.U, α, β,
        B.parent.shift, B.parent.parent.blockinfo, NC)



    #jacc.@sync begin
    #    jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbshiftdag!(
    #       c.U, A.parent.U, B.parent.parent.U, α, β,
    #       B.parent.shift, B.parent.parent.blockinfo, NC)
    #end


end

#A'*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!,
        c.U, A.parent.U, B.parent.parent.Ushifted, NC)



    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(
    #        c.U, A.parent.U, B.parent.parent.Ushifted, NC)
    #end

end

#alpha*A'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!,
        c.U, A.parent.U, B.parent.parent.Ushifted, α, β, NC)



    #jacc.@sync begin
    #         jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(
    #            c.U, A.parent.U, B.parent.parent.Ushifted, α, β, NC)
    #    end


end

#alpha*shiftA*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftbdag!,
        c.U, A.parent.U, B.parent.U, α, β,
        A.shift, A.parent.blockinfo, NC)



    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftbdag!(c.U, A.parent.U, B.parent.U, α, β,
    #        A.shift, A.parent.blockinfo, NC)
    #end

end


#shiftA*B' 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!,
        c.U, A.parent.Ushifted, B.parent.U, NC)



    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(c.U, A.parent.Ushifted, B.parent.U, NC)
    #end


end


#alpha*shiftA*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!,
        c.U, A.parent.Ushifted, B.parent.U, α, β, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(c.U, A.parent.Ushifted, B.parent.U, α, β, NC)
    #end
end

#alpha*shiftA'*B'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftdagbdag!,
        c.U, A.parent.parent.U, B.parent.U, α, β,
        A.parent.shift, A.parent.parent.blockinfo, NC)



    #     jacc.@sync begin
    #         jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftdagbdag!(
    #            c.U, A.parent.parent.U, B.parent.U, α, β,
    #            A.parent.shift, A.parent.parent.blockinfo, NC)
    #    end

end


#shiftA'*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!,
        c.U, A.parent.parent.Ushifted, B.parent.U, NC)


    #     jacc.@sync begin
    #         jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(
    #            c.U, A.parent.parent.Ushifted, B.parent.U, NC)
    #    end


end


#alpha*shiftA'*B'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!,
        c.U, A.parent.parent.Ushifted, B.parent.U, α, β, NC)


    #     jacc.@sync begin
    #         jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(
    #            c.U, A.parent.parent.Ushifted, B.parent.U, α, β, NC)
    #    end



end

#alpha*A*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abshiftdag!,
        c.U, A.U, B.parent.parent.U, α, β,
        B.parent.shift, B.parent.parent.blockinfo, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abshiftdag!(c.U, A.U, B.parent.parent.U, α, β,
    #        B.parent.shift, B.parent.parent.blockinfo, NC)
    #end

end

#A*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!,
        c.U, A.U, B.parent.parent.Ushifted, NC)



    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(c.U, A.U, B.parent.parent.Ushifted, NC)
    #end

end

#alpha*A*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_abdag!,
        c.U, A.U, B.parent.parent.Ushifted, α, β, NC)



    # jacc.@sync begin
    #
    #        jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_abdag!(c.U, A.U, B.parent.parent.Ushifted, α, β, NC)
    #  end

end


#alpha*shiftA'*B+ beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftdagb!,
        c.U, A.parent.parent.U, B.U, α, β,
        A.parent.shift, A.parent.parent.blockinfo, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftdagb!(c.U, A.parent.parent.U, B.U, α, β,
    #        A.parent.shift, A.parent.parent.blockinfo, NC)
    #end


end

#shiftA'*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!,
        c.U, A.parent.Ushifted, B.U, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.Ushifted, B.U, NC)
    #end

end



#alpha*shiftA'*B+ beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!,
        c.U, A.parent.Ushifted, B.U, α, β, NC)



    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.Ushifted, B.U, α, β, NC)
    #end

end


#alpha*shiftA'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftdagbshiftdag!,
        c.U, A.parent.parent.U, B.parent.parent.U, α, β,
        A.parent.shift, B.parent.shift, A.parent.parent.blockinfo, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftdagbshiftdag!(c.U, A.parent.parent.U, B.parent.parent.U, α, β,
    #        A.parent.shift, B.parent.shift, A.parent.parent.blockinfo, NC)
    #end


end

#shiftA'*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!,
        c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted, NC)


    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted, NC)
    #end

end


#alpha*shiftA'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagbdag!,
        c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted, α, β, NC)



    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagbdag!(c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted, α, β, NC)
    #end


end

#alpha*shiftA'*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_ashiftdagbshift!,
        c.U, A.parent.parent.U, B.parent.U, α, β,
        A.parent.shift, B.shift, A.parent.parent.blockinfo, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_ashiftdagbshift!(c.U, A.parent.parent.U, B.parent.U, α, β,
    #        A.parent.shift, B.shift, A.parent.parent.blockinfo, NC)
    #end

end

#shiftA'*shiftB 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!,
        c.U, A.parent.parent.Ushifted, B.parent.Ushifted, NC)

    # jacc.@sync begin
    #     jacc.@ jacc threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize  jacckernel_mul_NC_adagb!(c.U, A.parent.parent.Ushifted, B.parent.Ushifted, NC)
    #end

end

#alpha*shiftA'*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_mul_NC_adagb!,
        c.U, A.parent.parent.Ushifted, B.parent.Ushifted, α, β, NC)

end













