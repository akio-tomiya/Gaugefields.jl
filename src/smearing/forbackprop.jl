
import ..AbstractGaugefields_module: clear_U!, add_U!, Gaugefields_4D_nowing
import ..Abstractsmearing_module: Abstractsmearing, STOUT_dataset
import Wilsonloop:
    Wilsonline,
    DwDU,
    make_loopforactions,
    make_Cμ,
    derive_U,
    derive_Udag,
    get_leftlinks,
    get_rightlinks
import ..AbstractGaugefields_module: calc_coefficients_Q

function CdexpQdQ!(CdeQdQ, C, Q)
    error("CdexpQdQ! is not implemented in types of CdeQdQ, C, Q: $(typeof(CdeQdQ)) $(typeof(C)) $(typeof(Q)) ")
end
function construct_B1B2!(B1, B2, Qn, b10, b11, b12, b20, b21, b22)
    B1 .= 0
    B2 .= 0
    for i = 1:3
        B1[i, i] = b10
        B2[i, i] = b20
    end
    for j = 1:3
        for i = 1:3
            B1[i, j] += b11 * Qn[i, j]
            B2[i, j] += b21 * Qn[i, j]
            for k = 1:3
                B1[i, j] += b12 * Qn[i, k] * Qn[k, j]
                B2[i, j] += b22 * Qn[i, k] * Qn[k, j]
            end
        end
    end
end
function construct_trCB1B2(B1, B2, C)
    trB1 = 0.0im
    trB2 = 0.0im
    for i = 1:3
        for j = 1:3
            trB1 += C[i, j] * B1[j, i]
            trB2 += C[i, j] * B2[j, i]
        end
    end
    return trB1, trB2
end

function construct_CdeQdQ_3!(CdeQdQn, trCB1, trCB2, f1, f2, Qn, Cn)
    for j = 1:3
        for i = 1:3
            CdeQdQn[i, j] = trCB1 * Qn[i, j] + f1 * Cn[i, j]
            for k = 1:3
                CdeQdQn[i, j] +=
                    trCB2 * Qn[i, k] * Qn[k, j] +
                    f2 * (Qn[i, k] * Cn[k, j] + Cn[i, k] * Qn[k, j])
            end
        end
    end
    CdeQdQn ./= im
    return
end

function calc_dSdQ!(dSdQ, dSdUbar, Qμ, Uμ, temps_g)
    dSdUU = temps_g[1]
    mul!(dSdUU, Uμ, dSdUbar)
    CdexpQdQ!(dSdQ, dSdUU, Qμ)
end
export calc_dSdQ!

function calc_dSdΩ!(dSdΩ, dSdQ)
    LdQdΩ!(dSdΩ, dSdQ)
end
export calc_dSdΩ!

function calc_dSdC!(dSdC, dSdΩ, Uμ)
    mul!(dSdC, Uμ', dSdΩ)
end
export calc_dSdC!

function calc_dSdUdag!(dSdUdag, dSdΩ, C)
    mul!(dSdUdag, dSdΩ, C)
end
export calc_dSdUdag!

function calc_dSdU_fromUdagfromC!(dSdU, dSdUdag)
    substitute_U!(dSdU, dSdUdag')
end
export calc_dSdU_fromUdagfromC!
function calc_dSdU_fromu1!(dSdU, dSdu1)
    substitute_U!(dSdU, dSdu1)
end
export calc_dSdU_fromu1!

function calc_dSdu1!(dSdu1, dSdUbar, expQ) # Ubar = exp(Q)*U
    mul!(dSdu1, dSdUbar, expQ)
end
export calc_dSdu1!

#=
function calc_STOUT_c_i(closedloops; Dim=4)
    #one of loopset, like plaq. There are several loops. 
    numloop = length(closedloops) #number of loops 
    Cμs = Vector{Vector{Wilsonline{Dim}}}(undef, Dim)
    for μ = 1:Dim
        Cμs[μ] = Vector{Wilsonline{Dim}}[] #set of staples. In the case of plaq, there are six staples. 
    end
    for i = 1:numloop
        glinks = closedloops[i]
        for μ = 1:Dim
            Cμ = make_Cμ(glinks, μ)
            for j = 1:length(Cμ)
                push!(Cμs[μ], Cμ[j])
            end
        end
    end

    dCμdUν = Matrix{Vector{DwDU{Dim}}}(undef, Dim, Dim)
    dCμdagdUν = Matrix{Vector{DwDU{Dim}}}(undef, Dim, Dim)

    for ν = 1:Dim
        for μ = 1:Dim
            dCμdUν[μ, ν] = Vector{DwDU{Dim}}[]
            dCμdagdUν[μ, ν] = Vector{DwDU{Dim}}[]
            Cμ = Cμs[μ]
            numCμ = length(Cμ)
            for j = 1:numCμ
                Cμj = Cμ[j]
                dCμjν = derive_U(Cμj, ν)
                numdCμjν = length(dCμjν)
                for k = 1:numdCμjν
                    push!(dCμdUν[μ, ν], dCμjν[k])
                end

                dCμjνdag = derive_U(Cμj', ν)
                numdCμjνdag = length(dCμjνdag)
                for k = 1:numdCμjνdag
                    push!(dCμdagdUν[μ, ν], dCμjνdag[k])
                end

            end
            #println("dC$(μ)/dU$(ν): ")
            #show(CmudUnu[μ,ν])
        end
    end
    return Cμs, dCμdUν, dCμdagdUν
end
=#

function calc_STOUT_C(loops_smearing, L; Dim=4)
    num = length(loops_smearing)
    loopset = make_loopforactions(loops_smearing, L)
    dataset = Array{STOUT_dataset{Dim},1}(undef, num)
    for i = 1:num
        closedloops = loopset[i]
        dataset[i] = STOUT_dataset(closedloops; Dim)
        #Cμs, dCμdUν, dCμdagdUν = calc_STOUT_c_i(closedloops; Dim)
    end
    return dataset
end
export calc_STOUT_C

function calc_C!(C, μ, ρs, dataset::Vector{STOUT_dataset{Dim}}, Uin, temps_g) where {Dim}
    temp1 = temps_g[1]
    temp2 = temps_g[2]
    temp3 = temps_g[3]
    num = length(ρs)
    clear_U!(C)
    for i = 1:num
        #println("ρi  = ",ρs[i] )
        loops = dataset[i].Cμ[μ]
        evaluate_gaugelinks!(temp3, loops, Uin, [temp1, temp2])
        #println("i = $i")
        #println(temp3[1,1,1,1,1,1])
        add_U!(C, ρs[i], temp3)
    end
    #println("U ", Uin[1][1,1,1,1,1,1])
end
export calc_C!

function calc_dSdUν_fromdSCμ_add!(dSdU, dataset::Vector{STOUT_dataset{Dim}}, dSdCμ, ρs, Us, μ, ν, temps_g) where {Dim}  #use pullback for C(U): dS/dCμ star dCμ/dUν
    temp1 = temps_g[1]
    temp2 = temps_g[2]
    temp3 = temps_g[3]
    temp4 = temps_g[4]

    numterms = length(ρs)
    for iterm = 1:numterms
        dCμdUν = dataset[iterm].dCμdUν
        dCμdagdUν = dataset[iterm].dCμdagdUν
        ρi = ρs[iterm]

        #for μ = 1:Dim
        numdCμdUν = length(dCμdUν[μ, ν])
        for j = 1:numdCμdUν
            dCμdUν_j = dCμdUν[μ, ν][j]
            position = dCμdUν_j.position
            m = Tuple(-collect(position))
            dSdCμm = shift_U(dSdCμ, m)

            leftlinks = get_leftlinks(dCμdUν_j)
            rightlinks = get_rightlinks(dCμdUν_j)

            A = temp3
            evaluate_gaugelinks!(A, leftlinks, Us, [temp1, temp2])

            B = temp4
            evaluate_gaugelinks!(B, rightlinks, Us, [temp1, temp2])
            LdCdU_i_add!(dSdU, dSdCμm, A, B, ρi, temps_g)
        end

        numdCμdagdUν = length(dCμdagdUν[μ, ν])
        for j = 1:numdCμdagdUν
            dCμdagdUν_j = dCμdagdUν[μ, ν][j]

            position = dCμdagdUν_j.position
            m = Tuple(-collect(position))
            dSdCμm = shift_U(dSdCμ, m)
            leftlinks = get_leftlinks(dCμdagdUν_j)
            rightlinks = get_rightlinks(dCμdagdUν_j)

            barA = temp3
            evaluate_gaugelinks!(barA, leftlinks, Us, [temp1, temp2])
            barB = temp4
            evaluate_gaugelinks!(barB, rightlinks, Us, [temp1, temp2])
            LdCdU_i_add!(dSdU, dSdCμm', barA, barB, ρi, temps_g)
        end
        #end

    end
end
export calc_dSdUν_fromdSCμ_add!


function LdCdU_i_add!(LdCdU, L, A, B, ρ, temps_g) #dCdU = ρ sum_i A_i otimes B_i , dCdagdU = ρ sum_i Abar_i otimes Bbar_i
    BL = temps_g[1]
    BLA = temps_g[2]
    mul!(BL, B, L)
    mul!(BLA, BL, A)
    add_U!(LdCdU, ρ, BLA)
end





function CdexpQdQ!(CdeQdQ::Gaugefields_4D_nowing{3}, C::Gaugefields_4D_nowing{3},
    Q::Gaugefields_4D_nowing{3}) # C star dexpQ/dQ
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


    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX

                    for jc = 1:NC
                        for ic = 1:NC
                            Qnim[ic, jc] = Q[ic, jc, ix, iy, iz, it] / im
                            Cn[ic, jc] = C[ic, jc, ix, iy, iz, it]
                        end
                    end
                    f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qnim)
                    construct_B1B2!(B1, B2, Qnim, b10, b11, b12, b20, b21, b22)
                    trCB1, trCB2 = construct_trCB1B2(B1, B2, Cn)
                    construct_CdeQdQ_3!(CdeQdQn, trCB1, trCB2, f1, f2, Qnim, Cn)

                    for jc = 1:NC
                        for ic = 1:NC
                            CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
                        end
                    end
                end
            end
        end
    end
end
export CdexpQdQ!

function LdQdΩ!(LdQdΩ, L) # L star dQ/dΩ
    Traceless_antihermitian!(LdQdΩ, L)
end

#=
function LdQdΩ!(LdQdΩ::Gaugefields_4D_nowing{3}, L::Gaugefields_4D_nowing{3}) # L star dQ/dΩ
    Gaugefields.Traceless_antihermitian!(LdQdΩ, L)
    return
end
=#
export LdQdΩ!

function expU(U::Gaugefields_4D_nowing{NC}) where {NC}
    NT = U.NT
    NY = U.NY
    NZ = U.NZ
    NX = U.NX
    eU = similar(U)
    Umat = zeros(ComplexF64, NC, NC)
    eUmat = zeros(ComplexF64, NC, NC)
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX

                    for jc = 1:NC
                        for ic = 1:NC
                            Umat[ic, jc] = U[ic, jc, ix, iy, iz, it]
                        end
                    end
                    eUmat[:, :] = exp(Umat)

                    for jc = 1:NC
                        for ic = 1:NC
                            eU[ic, jc, ix, iy, iz, it] = eUmat[ic, jc]
                        end
                    end
                end
            end
        end
    end
    return eU
end
export expU