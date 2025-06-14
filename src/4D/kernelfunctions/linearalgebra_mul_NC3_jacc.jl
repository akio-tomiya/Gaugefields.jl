#A*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::T1,
    B::T2) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.U, B.U)
    end
end

#alpha*A*B + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb,
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.U, B.U, α, β)
    end
end

#A*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::T1,
    B::Adjoint_Gaugefields{T2}) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(c.U, A.U, B.parent.U)
    end
end

#alpha*A*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(c.U, A.U, B.parent.U, α, β)
    end
end

#A'*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(c.U, A.parent.U, B.parent.U)
    end
end

#c = alpha * A'* B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(c.U, A.parent.U, B.parent.U, α, β)
    end
end

#A'*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.U, B.U)
    end


end


#alpha*A'*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.U, B.U, α, β)
    end
end


#alpha*A*shiftB+C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abshift!(c.U, A.U, B.parent.U, α, β,
            B.shift, B.parent.blockinfo)
    end

end

#alpha*A*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TshiftedU},
    A::T1,
    B::T2
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TshiftedU}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.U, B.parent.Ushifted)
    end
end


#alpha*A*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TshiftedU},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TshiftedU}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.U, B.parent.Ushifted, α, β)
    end

end


#alpha*shiftA*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftb!(c.U, A.parent.U, B.U, α, β,
            A.shift, B.parent.blockinfo)
    end

end

#shiftA*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.parent.Ushifted, B.U)
    end


end

#alpha*shiftA*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.parent.Ushifted, B.U, α, β)
    end


end

#alpha*shiftA*shiftB +beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftbshift!(c.U, A.parent.U, B.parent.U, α, β,
            A.shift, B.shift, B.parent.blockinfo)
    end

end



#shiftA*shiftB 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.parent.Ushifted, B.parent.Ushifted)
    end


end

#alpha*shiftA*shiftB +beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, A.parent.Ushifted, B.parent.Ushifted, α, β)
    end

end



#alpha*shiftA*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftbshiftdag!(
            c.U, A.parent.U, B.parent.parent.U, α, β,
            A.shift, B.parent.shift, A.parent.blockinfo)
    end


end

#shiftA*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(
            c.U, A.parent.Ushifted, B.parent.parent.Ushifted)
    end


end


#alpha*shiftA*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(
            c.U, A.parent.Ushifted, B.parent.parent.Ushifted, α, β)
    end

end

#alpha*A'*shiftB+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbshift!(c.U, A.parent.U, B.parent.U, α, β,
            B.shift, B.parent.blockinfo)
    end

end

#A'*shiftB
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.U, B.parent.Ushifted)
    end

end

#alpha*A'*shiftB+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.U, B.parent.Ushifted, α, β)
    end

end

#alpha*A'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbshiftdag!(
            c.U, A.parent.U, B.parent.parent.U, α, β,
            B.parent.shift, B.parent.parent.blockinfo)
    end


end

#A'*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(
            c.U, A.parent.U, B.parent.parent.Ushifted)
    end

end

#alpha*A'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(
            c.U, A.parent.U, B.parent.parent.Ushifted, α, β)
    end


end

#alpha*shiftA*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftbdag!(c.U, A.parent.U, B.parent.U, α, β,
            A.shift, A.parent.blockinfo)
    end

end


#shiftA*B' 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(c.U, A.parent.Ushifted, B.parent.U)
    end


end


#alpha*shiftA*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(c.U, A.parent.Ushifted, B.parent.U, α, β)
    end
end

#alpha*shiftA'*B'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftdagbdag!(
            c.U, A.parent.parent.U, B.parent.U, α, β,
            A.parent.shift, A.parent.parent.blockinfo)
    end

end


#shiftA'*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(
            c.U, A.parent.parent.Ushifted, B.parent.U)
    end



end


#alpha*shiftA'*B'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(
            c.U, A.parent.parent.Ushifted, B.parent.U, α, β)
    end



end

#alpha*A*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abshiftdag!(c.U, A.U, B.parent.parent.U, α, β,
            B.parent.shift, B.parent.parent.blockinfo)
    end

end

#A*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(c.U, A.U, B.parent.parent.Ushifted)
    end

end

#alpha*A*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_abdag!(c.U, A.U, B.parent.parent.Ushifted, α, β)
    end

end


#alpha*shiftA'*B+ beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftdagb!(c.U, A.parent.parent.U, B.U, α, β,
            A.parent.shift, A.parent.parent.blockinfo)
    end


end

#shiftA'*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.Ushifted, B.U)
    end

end



#alpha*shiftA'*B+ beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.Ushifted, B.U, α, β)
    end

end


#alpha*shiftA'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftdagbshiftdag!(c.U, A.parent.parent.U, B.parent.parent.U, α, β,
            A.parent.shift, B.parent.shift, A.parent.parent.blockinfo)
    end


end

#shiftA'*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted)
    end

end


#alpha*shiftA'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagbdag!(c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted, α, β)
    end


end

#alpha*shiftA'*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_ashiftdagbshift!(c.U, A.parent.parent.U, B.parent.U, α, β,
            A.parent.shift, B.shift, A.parent.parent.blockinfo)
    end

end

#shiftA'*shiftB 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.parent.Ushifted, B.parent.Ushifted)
    end

end

#alpha*shiftA'*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3_adagb!(c.U, A.parent.parent.Ushifted, B.parent.Ushifted, α, β)
    end

end













