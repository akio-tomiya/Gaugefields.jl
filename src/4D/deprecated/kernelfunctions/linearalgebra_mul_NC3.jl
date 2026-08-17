#A*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::T1,
    B::T2) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.U, B.U)
        end
    end
end

#alpha*A*B + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb,
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.U, B.U, α, β)
        end
    end
end

#A*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::T1,
    B::Adjoint_Gaugefields{T2}) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.U, B.parent.U)
        end
    end
end

#alpha*A*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.U, B.parent.U, α, β)
        end
    end
end

#A'*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r, c.U, A.parent.U, B.parent.U)
        end
    end

end

#c = alpha * A'* B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r, c.U, A.parent.U, B.parent.U, α, β)
        end
    end
end

#A'*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.U, B.U)
        end
    end

end


#alpha*A'*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.U, B.U, α, β)
        end
    end

end


#alpha*A*shiftB+C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abshift!(b, r, c.U, A.U, B.parent.U, α, β,
                B.shift, B.parent.blockinfo)
        end
    end
end

#alpha*A*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TshiftedU},
    A::T1,
    B::T2
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TshiftedU}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.U, B.parent.Ushifted)
        end
    end
end


#alpha*A*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TshiftedU},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TshiftedU}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.U, B.parent.Ushifted, α, β)
        end
    end
end


#alpha*shiftA*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftb!(b, r, c.U, A.parent.U, B.U, α, β,
                A.shift, B.parent.blockinfo)
        end
    end
end

#shiftA*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.parent.Ushifted, B.U)
        end
    end

end

#alpha*shiftA*B+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.parent.Ushifted, B.U, α, β)
        end
    end

end

#alpha*shiftA*shiftB +beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftbshift!(b, r, c.U, A.parent.U, B.parent.U, α, β,
                A.shift, B.shift, B.parent.blockinfo)
        end
    end
end



#shiftA*shiftB 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.parent.Ushifted, B.parent.Ushifted)
        end
    end

end

#alpha*shiftA*shiftB +beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.parent.Ushifted, B.parent.Ushifted, α, β)
        end
    end

end



#alpha*shiftA*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftbshiftdag!(b, r,
                c.U, A.parent.U, B.parent.parent.U, α, β,
                A.shift, B.parent.shift, A.parent.blockinfo)
        end
    end

end

#shiftA*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r,
                c.U, A.parent.Ushifted, B.parent.parent.Ushifted)
        end
    end

end


#alpha*shiftA*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r,
                c.U, A.parent.Ushifted, B.parent.parent.Ushifted, α, β)
        end
    end

end

#alpha*A'*shiftB+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbshift!(b, r, c.U, A.parent.U, B.parent.U, α, β,
                B.shift, B.parent.blockinfo)
        end
    end
end

#A'*shiftB
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.U, B.parent.Ushifted)
        end
    end
end

#alpha*A'*shiftB+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.U, B.parent.Ushifted, α, β)
        end
    end
end

#alpha*A'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbshiftdag!(b, r,
                c.U, A.parent.U, B.parent.parent.U, α, β,
                B.parent.shift, B.parent.parent.blockinfo)
        end
    end

end

#A'*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r,
                c.U, A.parent.U, B.parent.parent.Ushifted)
        end
    end

end

#alpha*A'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r,
                c.U, A.parent.U, B.parent.parent.Ushifted, α, β)
        end
    end

end

#alpha*shiftA*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftbdag!(b, r, c.U, A.parent.U, B.parent.U, α, β,
                A.shift, A.parent.blockinfo)
        end
    end
end


#shiftA*B' 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.parent.Ushifted, B.parent.U)
        end
    end
end


#alpha*shiftA*B' + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.parent.Ushifted, B.parent.U, α, β)
        end
    end
end

#alpha*shiftA'*B'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftdagbdag!(b, r,
                c.U, A.parent.parent.U, B.parent.U, α, β,
                A.parent.shift, A.parent.parent.blockinfo)
        end
    end

end


#shiftA'*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r,
                c.U, A.parent.parent.Ushifted, B.parent.U)
        end
    end

end


#alpha*shiftA'*B'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r,
                c.U, A.parent.parent.Ushifted, B.parent.U, α, β)
        end
    end

end

#alpha*A*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abshiftdag!(b, r, c.U, A.U, B.parent.parent.U, α, β,
                B.parent.shift, B.parent.parent.blockinfo)
        end
    end
end

#A*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2}
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.U, B.parent.parent.Ushifted)
        end
    end
end

#alpha*A*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.U, B.parent.parent.Ushifted, α, β)
        end
    end
end


#alpha*shiftA'*B+ beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftdagb!(b, r, c.U, A.parent.parent.U, B.U, α, β,
                A.parent.shift, A.parent.parent.blockinfo)
        end
    end

end

#shiftA'*B
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.Ushifted, B.U)
        end
    end

end



#alpha*shiftA'*B+ beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.Ushifted, B.U, α, β)
        end
    end

end


#alpha*shiftA'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftdagbshiftdag!(b, r, c.U, A.parent.parent.U, B.parent.parent.U, α, β,
                A.parent.shift, B.parent.shift, A.parent.parent.blockinfo)
        end
    end
end

#shiftA'*shiftB'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r, c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted)
        end
    end
end


#alpha*shiftA'*shiftB'+beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagbdag!(b, r, c.U, A.parent.parent.Ushifted, B.parent.parent.Ushifted, α, β)
        end
    end
end

#alpha*shiftA'*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS<:Nothing}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_ashiftdagbshift!(b, r, c.U, A.parent.parent.U, B.parent.U, α, β,
                A.parent.shift, B.shift, A.parent.parent.blockinfo)
        end
    end
end

#shiftA'*shiftB 
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,
    TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.parent.Ushifted, B.parent.Ushifted)
        end
    end
end

#alpha*shiftA'*shiftB + beta*C
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv,TS}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_adagb!(b, r, c.U, A.parent.parent.Ushifted, B.parent.Ushifted, α, β)
        end
    end
end








#A*B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:threads},
    A::T1,
    B::Adjoint_Gaugefields{T2}) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3_abdag!(b, r, c.U, A.U, B.parent.U)
        end
    end
end



function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{3,TU,TUv,:threads},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb,
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC3!(b, r, c.U, A.U, B.U, α, β)
        end
    end
end







