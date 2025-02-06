


function kernel_CdexpQdQ_NC3!(b, r, CdeQdQin, Cin, Qin, temp1, temp2, temp3; eps_Q=1e-18)
    NC = 3
    CdeQdQ = view(CdeQdQin, :, :, b, r)
    Cn = view(Cin, :, :, b, r)
    Q = view(Qin, :, :, b, r)
    Qnim = view(temp1, :, :, b, r)
    B1 = view(temp2, :, :, b, r)
    B2 = view(temp3, :, :, b, r)

    trQ2 = 0.0im
    for i = 1:3
        for j = 1:3
            trQ2 += Q[i, j] * Q[j, i]
        end
    end

    #println("abs(trQ2) $(abs(trQ2))")
    if abs(trQ2) > eps_Q
        for jc = 1:NC
            for ic = 1:NC
                Qnim[ic, jc] = Q[ic, jc] / im
            end
        end
        f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qnim)
        #if ix == iy == iz == it == 1
        #    println((f0, f1, f2, b10, b11, b12, b20, b21, b22))
        #end

        construct_B1B2!(B1, B2, Qnim, b10, b11, b12, b20, b21, b22)
        trCB1, trCB2 = construct_trCB1B2(B1, B2, Cn)
        #CdeQdQn = B1
        #=
        if b * r == 1
            println("before")
            display(CdeQdQ)
            println("Qnim")
            display(Qnim)
            println("Cn")
            display(Cn)
        end
        =#
        construct_CdeQdQ_3!(CdeQdQ, trCB1, trCB2, f1, f2, Qnim, Cn)
        #=
        if b * r == 1
            println("after")
            display(CdeQdQ)
        end
        =#
        #for jc = 1:NC
        #    for ic = 1:NC
        #        CdeQdQ[ic, jc] = CdeQdQn[ic, jc]
        #    end
        #end
    else
        #println("abs(trQ2) < eps_Q")
        #for jc = 1:NC
        #    for ic = 1:NC
        #        #CdeQdQ[ic, jc, b,r] = C[ic, jc, b,r]
        #    end
        #end
    end
end