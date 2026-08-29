import ..AbstractGaugefields_module: Gaugefields_4D_accelerator, get_tempU

include("kernelfunctions/stout_kernels.jl")

function CdexpQdQ!(CdeQdQ::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    C::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    Q::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS}; eps_Q=1e-18) where {TU,TUv,TS}  # C star dexpQ/dQ

    temp1 = get_tempU(CdeQdQ)
    temp2 = get_tempU(C)
    temp3 = get_tempU(Q)

    for r = 1:CdeQdQ.blockinfo.rsize
        for b = 1:CdeQdQ.blockinfo.blocksize
            kernel_CdexpQdQ_NC3!(b, r, CdeQdQ.U, C.U, Q.U, temp1, temp2, temp3; eps_Q)
        end
    end

end

function construct_Λmatrix_forSTOUT!(
    Λ::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    δ_current::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    Q::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
    u::Gaugefields_4D_accelerator{3,TU,TUv,:none,TS},
) where {TU,TUv,TS}
    return _construct_Λmatrix_forSTOUT_with_pullback!(Λ, δ_current, Q, u)
end
