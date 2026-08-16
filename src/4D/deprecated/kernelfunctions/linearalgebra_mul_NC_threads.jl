function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftdagbshift!(b, r, c.U, A.parent.parent.U, B.parent.U, α, β,
                A.parent.shift, B.shift, A.parent.parent.blockinfo, NC)
        end
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftdagbshiftdag!(b, r, c.U, A.parent.parent.U, B.parent.parent.U, α, β,
                A.parent.shift, B.parent.shift, A.parent.parent.blockinfo, NC)
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftdagb!(b, r, c.U, A.parent.parent.U, B.U, α, β,
                A.parent.shift, A.parent.parent.blockinfo, NC)
        end
    end

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_abshiftdag!(b, r, c.U, A.U, B.parent.parent.U, α, β,
                B.parent.shift, B.parent.parent.blockinfo, NC)
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftdagbdag!(b, r,
                c.U, A.parent.parent.U, B.parent.U, α, β,
                A.parent.shift, A.parent.parent.blockinfo, NC)
        end
    end

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftbdag!(b, r, c.U, A.parent.U, B.parent.U, α, β,
                A.shift, A.parent.blockinfo, NC)
        end
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_adagbshiftdag!(b, r,
                c.U, A.parent.U, B.parent.parent.U, α, β,
                B.parent.shift, B.parent.parent.blockinfo, NC)
        end
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_adagbshift!(b, r, c.U, A.parent.U, B.parent.U, α, β,
                B.shift, B.parent.blockinfo, NC)
        end
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftbshiftdag!(b, r,
                c.U, A.parent.U, B.parent.parent.U, α, β,
                A.shift, B.parent.shift, A.parent.blockinfo, NC)
        end
    end

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftbshift!(b, r, c.U, A.parent.U, B.parent.U, α, β,
                A.shift, B.shift, B.parent.blockinfo, NC)
        end
    end

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_ashiftb!(b, r, c.U, A.parent.U, B.U, α, β,
                A.shift, B.parent.blockinfo, NC)
        end
    end

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Shifted_Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}
    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_abshift!(b, r, c.U, A.U, B.parent.U, α, β,
                B.shift, B.parent.blockinfo, NC)
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_adagb!(b, r, c.U, A.parent.U, B.U, α, β, NC)
        end
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2}
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_adagbdag!(b, r, c.U, A.parent.U, B.parent.U, NC)
        end
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}

    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_adagbdag!(b, r, c.U, A.parent.U, B.parent.U, α, β, NC)
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,Ta<:Number,Tb<:Number,TU,TUv}
    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC_abdag!(b, r, c.U, A.U, B.parent.U, α, β, NC)
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    A::T1,
    B::T2) where {NC,T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator,TU,TUv}
    Threads.@threads for r = 1:c.blockinfo.rsize
        Threads.@threads for b = 1:c.blockinfo.blocksize
            kernel_mul_NC!(b, r, c.U, A.U, B.U, NC)
        end
    end
end



