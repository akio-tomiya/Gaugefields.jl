function CdexpQdQ!(CdeQdQ::Gaugefields_4D_nowing_mpi{3}, C::Gaugefields_4D_nowing_mpi,
    Q::Gaugefields_4D_nowing_mpi; eps_Q=1e-18) # C star dexpQ/dQ
    NT = Q.NT
    NY = Q.NY
    NZ = Q.NZ
    NX = Q.NX
    NC = 3
    Qnim = zeros(ComplexF64, NC, NC) #Qn/im
    B1 = zero(Qnim)
    B2 = zero(Qnim)
    Cn = zero(Qnim)
    CdeQdQn = zero(Qnim)

    for it = 1:Q.PN[4]
        for iz = 1:Q.PN[3]
            for iy = 1:Q.PN[2]
                for ix = 1:Q.PN[1]

                    trQ2 = 0.0im
                    for i = 1:3
                        for j = 1:3
                            q1 = getvalue(Q, i, j, ix, iy, iz, it)
                            q2 = getvalue(Q, j, i, ix, iy, iz, it)
                            #trQ2 += Q[i, j, ix, iy, iz, it] * Q[j, i, ix, iy, iz, it]
                            trQ2 += q1 * q2

                        end
                    end

                    if abs(trQ2) > eps_Q
                        for jc = 1:NC
                            for ic = 1:NC
                                q = getvalue(Q, ic, jc, ix, iy, iz, it)
                                Qnim[ic, jc] = q / im#Q[ic, jc, ix, iy, iz, it] / im
                                c = getvalue(C, ic, jc, ix, iy, iz, it)
                                Cn[ic, jc] = c#C[ic, jc, ix, iy, iz, it]
                            end
                        end
                        f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qnim)
                        #if ix == iy == iz == it == 1
                        #    println((f0, f1, f2, b10, b11, b12, b20, b21, b22))
                        #end

                        construct_B1B2!(B1, B2, Qnim, b10, b11, b12, b20, b21, b22)
                        trCB1, trCB2 = construct_trCB1B2(B1, B2, Cn)
                        construct_CdeQdQ_3!(CdeQdQn, trCB1, trCB2, f1, f2, Qnim, Cn)

                        for jc = 1:NC
                            for ic = 1:NC
                                #CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
                                v = CdeQdQn[ic, jc]
                                setvalue!(CdeQdQ, v, ic, jc, ix, iy, iz, it)
                            end
                        end
                    else
                        for jc = 1:NC
                            for ic = 1:NC
                                #CdeQdQ[ic, jc, ix, iy, iz, it] = C[ic, jc, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(CdeQdQ)
end
function CdexpQdQ!(CdeQdQ::Gaugefields_4D_nowing_mpi{2}, C::Gaugefields_4D_nowing_mpi{2},
    Q::Gaugefields_4D_nowing_mpi{2}; eps_Q=1e-18) # C star dexpQ/dQ
    NT = Q.NT
    NY = Q.NY
    NZ = Q.NZ
    NX = Q.NX
    NC = 2
    Qn = zeros(ComplexF64, NC, NC) #Qn
    B = zero(Qn)
    B2 = zero(Qn)
    Cn = zero(Qn)
    CdeQdQn = zero(Qn)

    for it = 1:Q.PN[4]
        for iz = 1:Q.PN[3]
            for iy = 1:Q.PN[2]
                for ix = 1:Q.PN[1]

                    trQ2 = 0.0im
                    for i = 1:2
                        for j = 1:2
                            q1 = getvalue(Q, i, j, ix, iy, iz, it)
                            q2 = getvalue(Q, j, i, ix, iy, iz, it)
                            trQ2 += q1 * q2
                            #trQ2 += Q[i, j, ix, iy, iz, it] * Q[j, i, ix, iy, iz, it]
                        end
                    end


                    if abs(trQ2) > eps_Q
                        q = sqrt((-1 / 2) * trQ2)
                        for jc = 1:NC
                            for ic = 1:NC
                                Qn[ic, jc] = getvalue(Q, ic, jc, ix, iy, iz, it)#Q[ic, jc, ix, iy, iz, it]
                                Cn[ic, jc] = getvalue(C, ic, jc, ix, iy, iz, it)#C[ic, jc, ix, iy, iz, it]
                            end
                        end
                        calc_Bmatrix!(B, q, Qn, NC)
                        trsum = 0.0im
                        for i = 1:2
                            for j = 1:2
                                trsum += Cn[i, j] * B[j, i]
                            end
                        end
                        for i = 1:2
                            for j = 1:2
                                CdeQdQn[j, i] = (sin(q) / q) * Cn[j, i] + trsum * Qn[j, i]
                            end
                        end

                        for jc = 1:NC
                            for ic = 1:NC
                                v = CdeQdQn[ic, jc]
                                setvalue!(CdeQdQ, v, ic, jc, ix, iy, iz, it)

                                #CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
                            end
                        end
                    else
                        for jc = 1:NC
                            for ic = 1:NC
                                #CdeQdQ[ic, jc, ix, iy, iz, it] = C[ic, jc, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(CdeQdQ)
end
