function LinearAlgebra.mul!(C::LatticeVector{1,T,AT}, A::LatticeVector{1,T,AT}, B::LatticeVector{1,T,AT}) where {T,AT}

    JACC.parallel_for(
        C.PN[1], kernel_1Dvector_mul!, C.A, A.A, B.A, C.NC, C.nw
    )
    set_halo!(C)
end

function LinearAlgebra.mul!(C::LatticeMatrix{1,T1,AT1,NC1,NC2},
    A::LatticeMatrix{1,T2,AT2,NC1,NC3}, B::LatticeMatrix{1,T3,AT3,NC3,NC2}) where {T1,T2,T3,AT1,AT2,AT3,NC1,NC2,NC3}
    JACC.parallel_for(
        C.PN[1], kernel_1Dmatrix_mul!, C.A, A.A, B.A, NC1, NC2, NC3, C.nw
    )
    set_halo!(C)
end

function kernel_1Dvector_mul!(i, C, A, B, NC, nw)
    @inbounds for ic = 1:NC
        C[ic, i+nw] = A[ic, i+nw] * B[ic, i+nw]
    end
end

function kernel_1Dmatrix_mul!(i, C, A, B, NC1, NC2, NC3, nw)
    @inbounds for ic = 1:NC1
        for jc = 1:NC2
            C[ic, jc, i+nw] = 0
            for kc = 1:NC3
                C[ic, jc, i+nw] += A[ic, kc, i+nw] * B[kc, jc, i+nw]
            end
        end
    end
end