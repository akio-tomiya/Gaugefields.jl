function CdexpQdQ!(CdeQdQ::Gaugefields_4D_accelerator{3,TU,TUv,:cuda,TS},
    C::Gaugefields_4D_accelerator{3,TU,TUv,:cuda,TS},
    Q::Gaugefields_4D_accelerator{3,TU,TUv,:cuda,TS}; eps_Q=1e-18) where {TU,TUv,TS}  # C star dexpQ/dQ

    temp1 = get_tempU(CdeQdQ)
    temp2 = get_tempU(C)
    temp3 = get_tempU(Q)

   
    CUDA.@sync begin
        CUDA.@cuda threads = CdeQdQ.blockinfo.blocksize blocks = CdeQdQ.blockinfo.rsize cudakernel_CdexpQdQ_NC3!(CdeQdQ.U, C.U, Q.U, temp1, temp2, temp3,eps_Q)
    end


end



function cudakernel_CdexpQdQ_NC3!(CdeQdQin, Cin, Qin, temp1, temp2, temp3,eps_Q)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_CdexpQdQ_NC3!(b, r, CdeQdQin, Cin, Qin, temp1, temp2, temp3; eps_Q)
end