
import ..AbstractGaugefields_module: clear_U!, add_U!, Gaugefields_4D_nowing, substitute_U!
import ..AbstractGaugefields_module: calc_coefficients_Q,calc_Bmatrix!

mutable struct STOUTsmearing_layer{T,Dim,Tρ} <: CovLayer{Dim}
    ρs::Tρ#Vector{Tρ}
    const dataset::Vector{STOUT_dataset{Dim}}
    const Uinα::Vector{T}
    const Uinβ::Vector{T}
    const eQs::Vector{T}
    const Cs::Vector{T}
    const Qs::Vector{T}
    const temps::Vector{T}
    const dSdCs::Vector{T}
    islocalρ::Bool
    isαβsame::Bool
    hasdSdCs::Bool
    dSdρ::Union{Nothing,Tρ}
end
export STOUTsmearing_layer

function STOUTsmearing_layer(loops_smearing, L, U::Vector{<:AbstractGaugefields{NC,Dim}}, ρs=zeros(Float64, length(loops_smearing))) where {NC,Dim}
    loopset = make_loopforactions(loops_smearing, L)
    return STOUTsmearing_layer(loopset, U, ρs)
end

function STOUTsmearing_layer(loopset::Vector{Vector{Wilsonline{Dim}}}, U::Vector{<:AbstractGaugefields{NC,Dim}}, ρs=zeros(Float64, length(loopset))) where {NC,Dim}
    T = eltype(U)
    numg = 5 + Dim - 1
    temps = Vector{T}(undef, numg)
    for i = 1:numg
        temps[i] = similar(U[1])
    end
    #num = length(loops_smearing)
    #loopset = make_loopforactions(loops_smearing, L)
    num = length(loopset)
    #display(loopset)
    #error("d")
    dataset = Array{STOUT_dataset{Dim},1}(undef, num)
    for i = 1:num
        closedloops = loopset[i] #one of loopset, like plaq. There are several loops. 
        dataset[i] = STOUT_dataset(closedloops, Dim=Dim)
    end
    #ρs = zeros(Float64, num)
    #
    Tρ = typeof(ρs)#eltype(ρs)
    if eltype(ρs) <: Number
        islocalρ = false
    else
        islocalρ = true
    end

    Uinα = Vector{T}(undef, Dim)
    Uinβ = Vector{T}(undef, Dim)
    eQs = Vector{T}(undef, Dim)
    Cs = Vector{T}(undef, Dim)
    Qs = Vector{T}(undef, Dim)
    dSdCs = Vector{T}(undef, Dim)

    for μ = 1:Dim
        Uinα[μ] = similar(U[1])
        Uinβ[μ] = similar(U[1])
        eQs[μ] = similar(U[1])
        Cs[μ] = similar(U[1])
        Qs[μ] = similar(U[1])
        dSdCs[μ] = similar(U[1])
    end
    dSdρ = zero(ρs)

    return STOUTsmearing_layer{T,Dim,Tρ}(ρs, dataset, Uinα, Uinβ, eQs, Cs, Qs, temps, dSdCs, islocalρ, false, false, dSdρ)
end

function set_parameters!(s::STOUTsmearing_layer, ρs)
    s.ρs[:] .= ρs
    #println(ρs)
end

function get_parameters(s::STOUTsmearing_layer)
    return deepcopy(s.ρs)
end

function get_parameter_derivatives(s::STOUTsmearing_layer)
    return deepcopy(s.dSdρ)
end

function get_numparameters(s::STOUTsmearing_layer)
    return length(s.ρs)
end

function zero_grad!(s::STOUTsmearing_layer)
    if s.dSdρ != nothing
        s.dSdρ .= 0
    end
end

function Base.length(layer::STOUTsmearing_layer)
    return length(layer.ρs)
end

function get_Cμ(layer::STOUTsmearing_layer, i)
    return layer.dataset[i].Cμ
end

function get_dCμdUν(layer::STOUTsmearing_layer, i)
    return layer.dataset[i].dCμdUν
end

function get_dCμdagdUν(layer::STOUTsmearing_layer, i)
    return layer.dataset[i].dCμdagdUν
end

function get_ρ(layer::STOUTsmearing_layer, i)
    return layer.ρs[i]
end


function make_longstaple_pair(μ, ν, s)
    loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
    #loop = [(ν, +s), (μ, +1), (ν, -s)]
    w1 = Wilsonline(loop)
    #w = Vector{typeof(w1)}(undef, 2)
    w = Vector{typeof(w1)}(undef, 1)
    w[1] = w1
    #loop = [(ν, -s), (μ, +1), (ν, +s)]
    #w2 = Wilsonline(loop)
    #w[2] = w2
    return w
end
export make_longstaple_pair

function apply_layer!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    layer::STOUTsmearing_layer{T,Dim},
    Uin,
    temps,
    tempf,
) where {NC,Dim,T}


    ρs = layer.ρs
    forward!(layer, Uout, ρs, Uin, Uin)
    set_wing_U!(Uout)
    return
end

function layer_pullback!(
    δ_prev::Array{<:AbstractGaugefields{NC,Dim},1},
    δ_current,
    layer::STOUTsmearing_layer{T,Dim},
    Uprev,
    temps,
    tempf,
) where {NC,Dim,T}
    clear_U!(δ_prev)

    dSdρ = layer.dSdρ

    dSdU2 = temps[1:Dim]
    clear_U!(dSdU2)
    backward_dSdUαUβρ_add!(layer, δ_prev, dSdU2, dSdρ, δ_current)
    add_U!(δ_prev, 1, dSdU2)


    #backward_dSdUα_add!(layer, δ_prev, δ_current)
    #backward_dSdUβ_add!(layer, δ_prev, δ_current)
    #backward_dSdρ_add!(layer, dSdρ, δ_current)
    set_wing_U!(δ_prev)
    return
end


function forward!(s::STOUTsmearing_layer{T,Dim}, Uout, ρs::Vector{TN}, Uinα, Uinβ) where {T,Dim,TN<:Number} #Uout = exp(Q(Uin,ρs))*Uinα
    s.isαβsame = (Uinα == Uinβ)
    s.islocalρ = false
    #println("is $isαβsame")
    substitute_U!(s.Uinα, Uinα)
    if s.isαβsame == false
        substitute_U!(s.Uinβ, Uinβ)
    end
    for i = 1:length(s.ρs)
        s.ρs[i] = deepcopy(ρs[i])
    end
    temps = s.temps
    Ω = temps[end]
    for μ = 1:Dim
        calc_C!(s.Cs[μ], μ, ρs, s.dataset, Uinβ, s.temps)
        #display(s.Cs[μ][:, :, 1, 1, 1, 1])
        mul!(Ω, s.Cs[μ], Uinβ[μ]') #Ω = C*Udag
        Traceless_antihermitian!(s.Qs[μ], Ω)
        exptU!(s.eQs[μ], 1, s.Qs[μ], temps[1:2])
        mul!(Uout[μ], s.eQs[μ], Uinα[μ])
    end
    set_wing_U!(Uout)
    s.hasdSdCs = false
end
export forward!



function backward_dSdU_add!(s::STOUTsmearing_layer{T,Dim}, dSdUin, dSdUout) where {T,Dim}
    backward_dSdUα_add!(s, dSdUin, dSdUout)
    backward_dSdUβ_add!(s, dSdUin, dSdUout)
end
export backward_dSdU_add!

function backward_dSdUαUβρ_add!(s::STOUTsmearing_layer{T,Dim}, dSdUα, dSdUβ, dSdρ, dSdUout) where {T,Dim}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    temps = s.temps
    temp1 = temps[1]
    dSdQ = temps[2]
    dSdΩ = temps[3]
    dSdUdag = temps[4]
    dSdCs = temps[5:5+Dim-1]

    if s.isαβsame
        Uin = s.Uinα
    else
        Uin = s.Uinβ
    end


    for μ = 1:Dim

        #dS/dUα
        calc_dSdu1!(temp1, dSdUout[μ], s.eQs[μ])
        add_U!(dSdUα[μ], temp1)

        #dS/dUβ
        Cμ = s.Cs[μ]
        Qμ = s.Qs[μ]

        calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, s.Uinα[μ], temp1)
        calc_dSdΩ!(dSdΩ, dSdQ)
        calc_dSdC!(dSdCs[μ], dSdΩ, Uin[μ])

        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        add_U!(dSdUβ[μ], dSdUdag')

        if s.islocalρ == false
            Cμi = temps[4] #dSdUdag
            #dS/dρ
            num = length(s.ρs)
            for i = 1:num
                loops = s.dataset[i].Cμ[μ]
                evaluate_gaugelinks!(Cμi, loops, Uin, temps[1:2])
                mul!(temp1, dSdCs[μ], Cμi)
                dSdρ[i] += real(tr(temp1)) * 2
            end
        else
            error("not supported yet")
        end

    end

    for ν = 1:Dim
        for μ = 1:Dim
            calc_dSdUν_fromdSCμ_add!(dSdUβ[ν], s.dataset, dSdCs[μ], s.ρs, Uin, μ, ν, temps)
        end
    end
end
export backward_dSdUαUβρ_add!



function backward_dSdρ_add!(s::STOUTsmearing_layer{T,Dim,Tρ}, dSdρ, dSdUout) where {T,Dim,Tρ}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    temps = s.temps
    temp1 = temps[1]
    dSdQ = temps[2]
    dSdΩ = temps[3]
    #dSdUdag = temps[4]
    #dSdCs = temps[5:5+Dim-1]
    if s.isαβsame
        Uin = s.Uinα
    else
        Uin = s.Uinβ
    end


    for μ = 1:Dim

        if s.hasdSdCs == false
            Qμ = s.Qs[μ]
            calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, Uin[μ], temp1)
            calc_dSdΩ!(dSdΩ, dSdQ)
            calc_dSdC!(s.dSdCs[μ], dSdΩ, Uin[μ])
        end
        if s.islocalρ == false
            Cμi = temps[4] #dSdUdag
            #dS/dρ
            num = length(s.ρs)
            for i = 1:num
                loops = s.dataset[i].Cμ[μ]
                #println(loops)
                evaluate_gaugelinks!(Cμi, loops, Uin, temps[1:2])
                #display(Cμi[:, :, 1, 1, 1, 1])
                #error("d")
                mul!(temp1, s.dSdCs[μ], Cμi)
                #display(s.dSdCs[μ][:, :, 1, 1, 1, 1])
                dSdρ[i] += real(tr(temp1)) * 2
            end
        else
            error("not supported yet")
        end

    end
    s.hasdSdCs = true
end
export backward_dSdρ_add!



function backward_dSdUβ_add!(s::STOUTsmearing_layer{T,Dim}, dSdUβ, dSdUout) where {T,Dim} # Uout =  exp(Q(Uin,ρs))*Uinα
    temps = s.temps
    temps = similar(s.temps)
    temp1 = temps[1]
    dSdQ = temps[2]
    dSdΩ = temps[3]
    dSdUdag = temps[4]
    #dSdCs = temps[5:5+Dim-1]
    if s.isαβsame
        Uin = s.Uinα
    else
        Uin = s.Uinβ
    end
    #println("is $isαβsame")
    #Uprev = deepcopy(Uin)
    #display(Uin[1][:, :, 1, 1, 1, 1])


    #Cμ = similar(temp1)
    #Qμ = similar(temp1)
    #eQ = similar(temp1)
    #Ω = similar(temp1)
    ρs = s.ρs
    #println(ρs)
    dataset = s.dataset



    for μ = 1:Dim

        #calc_C!(Cμ, μ, ρs, dataset, Uin, temps)
        #mul!(Ω, Cμ, Uin[μ]') #Ω = C*Udag
        #Traceless_antihermitian!(Qμ, Ω)
        #exptU!(eQ, 1, Qμ, temps[1:2])


        Cμ = s.Cs[μ]
        Qμ = s.Qs[μ]
        #calc_dSdu1!(dSdUin[μ], dSdUout[μ], s.eQ)
        #println("Cμ")
        #display(Cμ[:, :, 1, 1, 1, 1])
        #println("Qμ")
        #display(Qμ[:, :, 1, 1, 1, 1])
        calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, Uin[μ], temp1)
        #println("dSdQ")
        #display(dSdQ[:, :, 1, 1, 1, 1])

        #dSdQ_n = calc_dSdQ(Qμ, Uin[μ], 1, 1, 1, 1)
        #println("dSdQ_n")
        #display(dSdQ_n)
        #println("dSdQ")
        #display(dSdQ[:, :, 1, 1, 1, 1])

        calc_dSdΩ!(dSdΩ, dSdQ)
        #println("dSdΩ")
        #display(dSdΩ[:, :, 1, 1, 1, 1])

        #dSdΩ_n = calc_dSdΩ(Ω, Uin[μ], 1, 1, 1, 1)
        #display(dSdΩ_n)
        #display(dSdΩ[:, :, 1, 1, 1, 1])
        #error("ee")

        calc_dSdC!(s.dSdCs[μ], dSdΩ, Uin[μ])
        #println("dSdCs")
        #display(s.dSdCs[μ][:, :, 1, 1, 1, 1])
        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        add_U!(dSdUβ[μ], dSdUdag')
    end

    for ν = 1:Dim
        for μ = 1:Dim
            calc_dSdUν_fromdSCμ_add!(dSdUβ[ν], s.dataset, s.dSdCs[μ], s.ρs, Uin, μ, ν, temps)
        end
    end
    s.hasdSdCs = true
end
export backward_dSdUβ_add!


function backward_dSdUα_add!(s::STOUTsmearing_layer{T,Dim}, dSdUα, dSdUout) where {T,Dim}
    temps = s.temps
    temps = similar(s.temps)
    temp1 = temps[1]

    for μ = 1:Dim
        calc_dSdu1!(temp1, dSdUout[μ], s.eQs[μ])
        add_U!(dSdUα[μ], temp1)
    end
end
export backward_dSdUα_add!

function calc_C!(C, μ, ρs::Vector{TN}, dataset::Vector{STOUT_dataset{Dim}}, Uin, temps_g) where {Dim,TN<:Number}
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



function calc_dSdu1!(dSdu1, dSdUbar, expQ) # Ubar = exp(Q)*U
    mul!(dSdu1, dSdUbar, expQ)
end
export calc_dSdu1!

function calc_dSdQ!(dSdQ, dSdUbar, Qμ, Uμ, temp)
    dSdUU = temp
    mul!(dSdUU, Uμ, dSdUbar)
    CdexpQdQ!(dSdQ, dSdUU, Qμ)
end
export calc_dSdQ!

function LdQdΩ!(LdQdΩ, L) # L star dQ/dΩ
    Traceless_antihermitian!(LdQdΩ, L)
end

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



function LdCdU_i_add!(LdCdU, L, A, B, ρ, temps_g) #dCdU = ρ sum_i A_i otimes B_i , dCdagdU = ρ sum_i Abar_i otimes Bbar_i
    BL = temps_g[1]
    BLA = temps_g[2]
    mul!(BL, B, L)
    mul!(BLA, BL, A)
    add_U!(LdCdU, ρ, BLA)
end

export calc_dSdUν_fromdSCμ_add!
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

function calc_dSdΩ(Ω, Uμ, ix, iy, iz, it, eta=1e-12; NC=3)
    numg = 5
    temps_g = Vector{typeof(Ω)}(undef, numg)
    for i = 1:numg
        temps_g[i] = similar(Ω)
    end
    Qμ = similar(Ω)
    Traceless_antihermitian!(Qμ, Ω)
    eQ = similar(Qμ)
    barU = similar(Qμ)
    eQ = expU(Qμ)
    mul!(barU, eQ, Uμ)
    S = real(tr(barU * barU))

    dSdΩ_n = zeros(ComplexF64, NC, NC)
    Qd = similar(Qμ)
    for ic = 1:NC
        for jc = 1:NC
            Ωd = deepcopy(Ω)
            Ωd[jc, ic, ix, iy, iz, it] += eta

            Traceless_antihermitian!(Qd, Ωd)
            eQ = expU(Qd)
            mul!(barU, eQ, Uμ)
            Sd = real(tr(barU * barU))


            Ωd = deepcopy(Ω)
            Ωd[jc, ic, ix, iy, iz, it] += im * eta
            Traceless_antihermitian!(Qd, Ωd)
            eQ = expU(Qd)
            mul!(barU, eQ, Uμ)
            Sdm = real(tr(barU * barU))

            fg_n = (Sd - S) / eta
            fg_n_im = (Sdm - S) / eta
            dSdΩ_n[ic, jc] = (fg_n - im * fg_n_im) / 2

        end
    end
    return dSdΩ_n
end




function calc_dSdQ(Q, Uμ, ix, iy, iz, it, eta=1e-12; NC=3)

    numg = 5
    temps_g = Vector{typeof(Uμ)}(undef, numg)
    for i = 1:numg
        temps_g[i] = similar(Q)
    end
    #eta = 1e-14
    eQ = similar(Q)
    barU = similar(Q)
    eQ = expU(Q)
    #Gaugefields.exptU!(eQ, 1, Q, temps_g[1:2]) # exp(Q)
    mul!(barU, eQ, Uμ)
    #display(eQ[:, :, ix, iy, iz, it])
    #S = real(tr(eQ * eQ))
    #S = real(tr(eQ))
    #S = real(tr(barU))
    S = real(tr(barU * barU))


    dSdQ_n = zeros(ComplexF64, NC, NC)
    for ic = 1:NC
        for jc = 1:NC


            #S = real(tr(barU * barU))

            Qd = deepcopy(Q)
            Qd[jc, ic, ix, iy, iz, it] += eta
            eQ = expU(Qd)
            #Gaugefields.exptU!(eQ, 1, Qd, temps_g[1:2]) # exp(Q)
            #display(eQ[:, :, ix, iy, iz, it])
            mul!(barU, eQ, Uμ)
            Sd = real(tr(barU * barU))
            #Sd = real(tr(barU))

            #Sd = real(tr(eQ))
            #Sd = real(tr(eQ * eQ))


            Qd = deepcopy(Q)
            Qd[jc, ic, ix, iy, iz, it] += im * eta
            eQ = expU(Qd)
            #Gaugefields.exptU!(eQ, 1, Qd, temps_g[1:2]) # exp(Q)
            mul!(barU, eQ, Uμ)
            Sdm = real(tr(barU * barU))
            #Sdm = real(tr(eQ * eQ))
            #Sdm = real(tr(barU))
            #Sdm = real(tr(eQ))

            fg_n = (Sd - S) / eta
            fg_n_im = (Sdm - S) / eta
            dSdQ_n[ic, jc] = (fg_n - im * fg_n_im) / 2
        end
    end
    return dSdQ_n

end

function CdexpQdQ!(CdeQdQ::Gaugefields_4D_nowing{3}, C::Gaugefields_4D_nowing{3},
    Q::Gaugefields_4D_nowing{3}; eps_Q=1e-18) # C star dexpQ/dQ
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

                    trQ2 = 0.0im
                    for i = 1:3
                        for j = 1:3
                            trQ2 += Q[i, j, ix, iy, iz, it] * Q[j, i, ix, iy, iz, it]
                        end
                    end

                    if abs(trQ2) > eps_Q
                        for jc = 1:NC
                            for ic = 1:NC
                                Qnim[ic, jc] = Q[ic, jc, ix, iy, iz, it] / im
                                Cn[ic, jc] = C[ic, jc, ix, iy, iz, it]
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
                                CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
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
end
function CdexpQdQ!(CdeQdQ::Gaugefields_4D_nowing{2}, C::Gaugefields_4D_nowing{2},
    Q::Gaugefields_4D_nowing{2}; eps_Q=1e-18) # C star dexpQ/dQ
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


    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX

                    trQ2 = 0.0im
                    for i = 1:3
                        for j = 1:3
                            trQ2 += Q[i, j, ix, iy, iz, it] * Q[j, i, ix, iy, iz, it]
                        end
                    end

                    if abs(trQ2) > eps_Q
                        q = sqrt((-1 / 2) * trQ2)
                        for jc = 1:NC
                            for ic = 1:NC
                                Qn[ic, jc] = Q[ic, jc, ix, iy, iz, it] 
                                Cn[ic, jc] = C[ic, jc, ix, iy, iz, it]
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
                                CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
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
end
export CdexpQdQ!


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
