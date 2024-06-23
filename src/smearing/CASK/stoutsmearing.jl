function STOUTsmearing_layer(U::Vector{<:AbstractGaugefields{NC,Dim}}, attentionmatrix::WeightMatrix_layer) where {NC,Dim}
    T = eltype(U)
    numg = 5 + Dim - 1
    temps = Vector{T}(undef, numg)
    for i = 1:numg
        temps[i] = similar(U[1])
    end

    maxS = attentionmatrix.maxS
    loopset = []
    count = 0
    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:maxS
                count += 1
                loop = make_longstaple_pair(μ, ν, s)
                push!(loopset, loop)
            end
        end
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
    Tρ = typeof(attentionmatrix)#eltype(ρs)
    islocal = true


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
    dSdρ = nothing

    return STOUTsmearing_layer{T,Dim,Tρ}(attentionmatrix, dataset, Uinα, Uinβ, eQs, Cs, Qs, temps, dSdCs, islocal, false, false, dSdρ)
end

function forward!(s::STOUTsmearing_layer{T,Dim}, Uout, attentionmatrix::WeightMatrix_layer, Uinα, Uinβ) where {T,Dim} #Uout = exp(Q(Uin,ρs))*Uinα
    s.isαβsame = (Uinα == Uinβ)
    s.islocalρ = true
    #println("is $isαβsame")
    substitute_U!(s.Uinα, Uinα)
    if s.isαβsame == false
        substitute_U!(s.Uinβ, Uinβ)
    else
        substitute_U!(s.Uinβ, Uinα)
    end
    s.ρs = attentionmatrix

    #display(s.ρs.data[:, :, 1, 1, 1, 1, 1])
    #for i = 1:length(s.ρs)
    #    s.ρs[i] = deepcopy(ρs[i])
    #end
    temps = s.temps
    #temps = [similar(s.Uinα[1]), similar(s.Uinα[2])]
    Ω = temps[end]
    for μ = 1:Dim
        calc_C!(s.Cs[μ], μ, attentionmatrix, s.dataset, Uinβ, s.temps)
        #println("Cs")
        #display(s.Cs[μ][:, :, 1, 1, 1, 1])
        mul!(Ω, s.Cs[μ], Uinβ[μ]') #Ω = C*Udag
        #println("Ω")
        #display(Ω[:, :, 1, 1, 1, 1])

        Traceless_antihermitian!(s.Qs[μ], Ω)
        #println("f Q$μ")
        #display(s.Qs[μ][:, :, 1, 1, 1, 1])
        #println("μ = $μ Qs:")
        #display(s.Qs[μ][:, :, 1, 1, 1, 1])
        exptU!(s.eQs[μ], 1, s.Qs[μ], temps[1:2])
        mul!(Uout[μ], s.eQs[μ], Uinα[μ])
    end
    set_wing_U!(Uout)
    s.hasdSdCs = false
end
export forward!


function backward_dSdUαUβρ_add!(s::STOUTsmearing_layer{T,Dim,Tρ}, dSdUα, dSdUβ::Vector{<:AbstractGaugefields{NC,Dim}}, dSdρ::Array{N,7}, dSdUout) where {NC,T,Dim,Tρ<:WeightMatrix_layer,N<:Number}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    temps = s.temps
    temp1 = temps[1]
    dSdQ = temps[2]
    dSdΩ = temps[3]
    dSdUdag = temps[4]

    #filterfunc(x) = ifelse(x > 0, 1, zero(x))
    dNC = 0.1
    filterfunc(x) = ifelse(x > 0, (1 + x^2) * π / (2(NC + dNC)), zero(x))
    #tempnew = similar(dSdUout)
    #dSdCs = temps[5:5+Dim-1]
    #error(s.isαβsame)

    #dSdρtemp = deepcopy(dSdρ)

    #=
    if s.isαβsame
        Uin = s.Uinα
    else
        Uin = s.Uinβ
    end
    =#
    maxS = get_maxS(s)

    count = 0
    for μ = 1:Dim

        #dS/dUα
        calc_dSdu1!(temp1, dSdUout[μ], s.eQs[μ])
        add_U!(dSdUα[μ], temp1)

        #dS/dUβ
        Cμ = s.Cs[μ]
        Qμ = s.Qs[μ]
        #println("Q$μ")
        #display(Qμ[:, :, 1, 1, 1, 1])
        #println("μ = $μ d")
        calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, s.Uinα[μ], temp1) #(U^{alpha} dS/dQ ) star dexp(Q)/dQ
        #println("Uin[$μ]")
        #display(Uin[μ][:, :, 1, 1, 1, 1])
        calc_dSdΩ!(dSdΩ, dSdQ)
        calc_dSdC!(s.dSdCs[μ], dSdΩ, s.Uinβ[μ]) # U^{beta dagger} dSd/Omega
        #calc_dSdC!(s.dSdCs[μ], dSdΩ, Uin[μ])
        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        add_U!(dSdUβ[μ], dSdUdag')
        #display(dSdUβ[μ][:, :, 1, 1, 1, 1])
        #println("μ = $μ d")

        Cμi = temps[4]
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for si = 1:maxS
                count += 1
                #println(count)
                loops = s.dataset[count].Cμ[μ]
                evaluate_gaugelinks!(Cμi, loops, s.Uinβ, temps[1:2])
                #evaluate_gaugelinks!(Cμi, loops, Uin, temps[1:2])

                mul!(temp1, s.dSdCs[μ], Cμi)
                #site_realtrace_add!(view(dSdρ, μ, ν, si, :, :, :, :), temp1, 2)
                #site_realtrace_add!(view(dSdρtemp, μ, ν, si, :, :, :, :), temp1, 2)
                #map!(x -> ifelse(x > 0, x, zero(x)), dSdρtemp, s.ρs.data)
                site_realtrace_filter_add!(view(dSdρ, μ, ν, si, :, :, :, :), temp1, view(s.ρs.data, μ, ν, si, :, :, :, :), filterfunc, 2)
            end
        end

    end
    #map!(x -> ifelse(x > 0, x, zero(x)), dSdρ, s.ρs.data)


    #for ν = 1:Dim
    for μ = 1:Dim
        #for μ = 1:Dim
        calc_dSdUν_fromdSCμ_add!(dSdUβ[μ], s.dataset, s.dSdCs, s.ρs, s.Uinβ, temps, μ)
        #calc_dSdUν_fromdSCμ_add!(dSdUβ[μ], s.dataset, s.dSdCs, s.ρs, Uin, temps, μ)
        #calc_dSdUν_fromdSCμ_add!(dSdUβ[ν], s.dataset, s.dSdCs[μ], s.ρs, Uin, temps, μ, ν)
    end
    #end
end

function backward_dSdρ_add!(s::STOUTsmearing_layer{T,Dim,Tρ}, dSdρ::Array{N,7}, dSdUout) where {T,Dim,Tρ<:WeightMatrix_layer,N<:Real}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    temps = s.temps
    temps = similar(s.temps)
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
    maxS = get_maxS(s)

    count = 0
    for μ = 1:Dim

        if s.hasdSdCs == false
            Qμ = s.Qs[μ]
            calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, Uin[μ], temp1)
            calc_dSdΩ!(dSdΩ, dSdQ)
            calc_dSdC!(s.dSdCs[μ], dSdΩ, Uin[μ])
        end
        if s.islocalρ == true
            Cμi = temps[4] #dSdUdag
            for ν = 1:Dim
                if μ == ν
                    continue
                end
                for si = 1:maxS
                    count += 1
                    #println(count)
                    loops = s.dataset[count].Cμ[μ]
                    evaluate_gaugelinks!(Cμi, loops, Uin, temps[1:2])
                    mul!(temp1, s.dSdCs[μ], Cμi)
                    site_realtrace_add!(view(dSdρ, μ, ν, si, :, :, :, :), temp1, 2)
                end
            end
        else
            error("not supported yet s.islocalρ = $(s.islocalρ)")
        end


    end
    s.hasdSdCs = true
end

function calc_C!(C, μ, attentionmatrix::WeightMatrix_layer, dataset::Vector{STOUT_dataset{Dim}}, Uin, temps_g) where {Dim}
    temp1 = temps_g[1]
    temp2 = temps_g[2]
    temp3 = temps_g[3]
    #num = length(ρs)
    clear_U!(C)
    maxS = attentionmatrix.maxS

    i = 0
    for μ2 = 1:μ-1
        for ν = 1:Dim
            if μ2 == ν
                continue
            end
            for s = 1:maxS
                i += 1
            end
        end
    end

    #i = 0
    #for μ = 1:Dim
    for ν = 1:Dim
        if μ == ν
            continue
        end
        for s = 1:maxS
            i += 1
            #println("i = $i")
            loops = dataset[i].Cμ[μ]
            evaluate_gaugelinks!(temp3, loops, Uin, [temp1, temp2])
            #println("temp3")
            #display(temp3[:, :, 1, 1, 1, 1])
            #println(attentionmatrix.data[μ, ν, s, 1, 1, 1, 1])
            mul!(temp1, view(attentionmatrix.data, μ, ν, s, :, :, :, :), temp3)
            #println(attentionmatrix.data[μ, ν, s, 1, 1, 1, 1])
            #println("temp1")
            #display(temp1[:, :, 1, 1, 1, 1])
            add_U!(C, temp1)
            #count += 1
            #loop = make_longstaple_pair(μ, ν, s)
            #push!(loopset, loop)
        end
    end
    #end


    #println("U ", Uin[1][1,1,1,1,1,1])
end
export calc_C!

function LdCdU_i_add!(LdCdU, L, A, B, attentionmatrix::WeightMatrix_layer, temps_g, μ, ν, s) #dCdU = ρ sum_i A_i otimes B_i , dCdagdU = ρ sum_i Abar_i otimes Bbar_i
    BL = temps_g[1]
    BLA = temps_g[2]
    mul!(BL, B, L)
    mul!(BLA, BL, A)
    aBLA = temps_g[1]
    mul!(aBLA, view(attentionmatrix.data, μ, ν, s, :, :, :, :), BLA)
    add_U!(LdCdU, aBLA)
    #add_U!(LdCdU, ρ, BLA)
end


function calc_dSdUν_fromdSCμ_add!(dSdUμ, dataset::Vector{STOUT_dataset{Dim}}, dSdCs, attentionmatrix::WeightMatrix_layer, Us, temps_g, μ) where {Dim}  #use pullback for C(U): dS/dCμ' star dCμ'/dUμ
    temp1 = temps_g[1]
    temp2 = temps_g[2]
    temp3 = temps_g[3]
    temp4 = temps_g[4]

    maxS = attentionmatrix.maxS
    count = 0

    for μd = 1:Dim
        dSdCμd = dSdCs[μd]
        for ν = 1:Dim
            if μd == ν
                continue
            end
            for s = 1:maxS
                count += 1
                dCμddUμ = dataset[count].dCμdUν
                numdCμddUμ = length(dCμddUμ[μd, μ])
                for j = 1:numdCμddUμ
                    dCμdUμ_j = dCμddUμ[μd, μ][j]
                    position = dCμdUμ_j.position
                    m = Tuple(-collect(position))
                    dSdCμm = shift_U(dSdCμd, m)

                    leftlinks = get_leftlinks(dCμdUμ_j)
                    numleft = length(leftlinks)
                    A = temp3
                    #show(leftlinks)
                    if numleft != 0
                        evaluate_gaugelinks!(A, leftlinks, Us, [temp1, temp2])
                        #evaluate_gaugelinks!(A, leftlinks, UK, [temp_U1, temp_U2])
                        mul!(temp1, dSdCμm, A)
                    else
                        substitute_U!(temp1, dSdCμm)
                    end

                    rightlinks = get_rightlinks(dCμdUμ_j)
                    numright = length(rightlinks)
                    B = temp4
                    if numright != 0
                        evaluate_gaugelinks!(B, rightlinks, Us, [temp2, temp3])
                        mul!(temp2, B, temp1)
                    else
                        substitute_U!(temp2, temp1)
                    end
                    mul_withshift!(temp1, view(attentionmatrix.data, μd, ν, s, :, :, :, :), temp2, m)
                    #mul!(temp1, view(attentionmatrix.data, μd, ν, s, :, :, :, :), temp2)
                    add_U!(dSdUμ, temp1)
                    #add_U!(temp_U, temp1)
                end

                dCμddagdUμ = dataset[count].dCμdagdUν
                numdCμddagdUμ = length(dCμddagdUμ[μd, μ])
                for j = 1:numdCμddagdUμ
                    dCμdagdUμ_j = dCμddagdUμ[μd, μ][j]
                    position = dCμdagdUμ_j.position
                    m = Tuple(-collect(position))
                    dSdCμm = shift_U(dSdCμd, m)

                    leftlinks = get_leftlinks(dCμdagdUμ_j)
                    numleft = length(leftlinks)
                    A = temp3
                    #show(leftlinks)
                    if numleft != 0
                        evaluate_gaugelinks!(A, leftlinks, Us, [temp1, temp2])
                        #evaluate_gaugelinks!(A, leftlinks, UK, [temp_U1, temp_U2])
                        mul!(temp1, dSdCμm', A)
                    else
                        substitute_U!(temp1, dSdCμm')
                    end

                    rightlinks = get_rightlinks(dCμdagdUμ_j)
                    numright = length(rightlinks)
                    B = temp4
                    if numright != 0
                        evaluate_gaugelinks!(B, rightlinks, Us, [temp2, temp3])
                        mul!(temp2, B, temp1)
                    else
                        substitute_U!(temp2, temp1)
                    end
                    #mul!(temp1, view(attentionmatrix.data, μd, ν, s, :, :, :, :), temp2)
                    mul_withshift!(temp1, view(attentionmatrix.data, μd, ν, s, :, :, :, :), temp2, m)
                    add_U!(dSdUμ, temp1)
                    #add_U!(temp_U, temp1)
                end
            end
        end
    end
end

function calc_dSdUν_fromdSCμ_add!(dSdUν, dataset::Vector{STOUT_dataset{Dim}}, dSdCμ, attentionmatrix::WeightMatrix_layer, Us, temps_g, μ, ν) where {Dim}  #use pullback for C(U): dS/dCμ star dCμ/dUν
    temp1 = temps_g[1]
    temp2 = temps_g[2]
    temp3 = temps_g[3]
    temp4 = temps_g[4]

    maxS = attentionmatrix.maxS

    #temp_U = similar(Us)

    count = 0
    for μd = 1:μ-1
        for νd = 1:Dim
            if μd == νd
                continue
            end

            for s = 1:maxS
                count += 1
            end
        end
    end

    for νd = 1:Dim
        if μ == νd
            continue
        end

        for s = 1:maxS
            count += 1
            #println("μ =  $μ, count = $count")


            #count = 0
            #for μd = 1:Dim
            #    for νd = 1:Dim
            #        if μd == νd
            #            continue
            #        end

            #        for s = 1:maxS
            #            count += 1
            #            if μd == μ




            Cμ = dataset[count].Cμ[μ]
            println("1")
            display(attentionmatrix.data[:, :, 1, 1, 1, 1, 1])
            println("2")
            display(attentionmatrix.data[:, :, 1, 2, 2, 2, 2])
            #attentionmatrix.data[:, :, 1, 2, 2, 2, 2] = attentionmatrix.data[:, :, 1, 1, 1, 1, 1]
            for j = 1:length(Cμ)
                V1 = Cμ[j]
                backward_dSdUK_add_fromVK!(dSdUν, V1, dSdCμ, Us, ν, temps_g, view(attentionmatrix.data, μ, νd, s, :, :, :, :))
                #backward_dSdUK_add_fromVK!(dSdUν, V1, dSdCμ, Us, ν, temp_U, view(attentionmatrix.data, μ, νd, s, :, :, :, :))
            end

            #=

            clear_U!(temp_U)
            dCμdUν = dataset[count].dCμdUν
            V1 = dataset[count].Cμ[μ]
            dV1 = Wilsonloop.derive_U(V1, ν)
            #dV1 = dCμdUν[μ, ν]
            backward_dSdUKα_add_fromVK!(dSdUν, dV1, dSdCμ, Us, temps_g)
            backward_dSdUKα_add_fromVK!(temp_U, dV1, dSdCμ, Us, temps_g)

            dCμdagdUν = dataset[count].dCμdagdUν
            V1 = dataset[count].Cμ[μ]
            dV1 = Wilsonloop.derive_U(V1', ν)
            #dV1 = dCμdagdUν[μ, ν]
            backward_dSdUKα_add_fromVK!(dSdUν, dV1, dSdCμ', Us, temps_g)
            backward_dSdUKα_add_fromVK!(temp_U, dV1, dSdCμ', Us, temps_g)
            println("temp_U1")
            display(temp_U[:, :, 1, 1, 1, 1])
            println("temp_U2")

            clear_U!(temp_U)

            numdCμdUν = length(dCμdUν[μ, ν])
            for j = 1:numdCμdUν
                dCμdUν_j = dCμdUν[μ, ν][j]
                position = dCμdUν_j.position
                m = Tuple(-collect(position))
                dSdCμm = shift_U(dSdCμ, m)

                leftlinks = get_leftlinks(dCμdUν_j)

                numleft = length(leftlinks)
                A = temp3
                #show(leftlinks)
                if numleft != 0
                    evaluate_gaugelinks!(A, leftlinks, Us, [temp1, temp2])
                    #evaluate_gaugelinks!(A, leftlinks, UK, [temp_U1, temp_U2])
                    mul!(temp1, dSdCμm, A)
                else
                    substitute_U!(temp1, dSdCμm)
                end


                rightlinks = get_rightlinks(dCμdUν_j)
                numright = length(rightlinks)
                B = temp4
                if numright != 0
                    evaluate_gaugelinks!(B, rightlinks, Us, [temp2, temp3])
                    mul!(temp2, B, temp1)
                else
                    substitute_U!(temp2, temp1)
                end
                mul!(temp1, view(attentionmatrix.data, μ, νd, s, :, :, :, :), temp2)
                #add_U!(dSdUν, temp1)
                add_U!(temp_U, temp1)

                #evaluate_gaugelinks!(B, rightlinks, Us, [temp1, temp2])
                #LdCdU_i_add!(dSdUν, dSdCμm, A, B, attentionmatrix, temps_g, μ, ν, s)
                #LdCdU_i_add!(dSdUν, dSdCμm, A, B, attentionmatrix, temps_g, μ, νd, s)
                #LdCdU_i_add!(dSdU, dSdCμm, A, B, ρi, temps_g)
            end

            dCμdagdUν = dataset[count].dCμdagdUν
            numdCμdagdUν = length(dCμdagdUν[μ, ν])
            for j = 1:numdCμdagdUν
                dCμdagdUν_j = dCμdagdUν[μ, ν][j]

                position = dCμdagdUν_j.position
                m = Tuple(-collect(position))
                dSdCμm = shift_U(dSdCμ, m)


                leftlinks = get_leftlinks(dCμdagdUν_j)

                numleft = length(leftlinks)
                A = temp3
                #show(leftlinks)
                if numleft != 0
                    evaluate_gaugelinks!(A, leftlinks, Us, [temp1, temp2])
                    #evaluate_gaugelinks!(A, leftlinks, UK, [temp_U1, temp_U2])
                    mul!(temp1, dSdCμm', A)
                else
                    substitute_U!(temp1, dSdCμm')
                end


                rightlinks = get_rightlinks(dCμdagdUν_j)
                numright = length(rightlinks)
                B = temp4
                if numright != 0
                    evaluate_gaugelinks!(B, rightlinks, Us, [temp2, temp3])
                    mul!(temp2, B, temp1)
                else
                    substitute_U!(temp2, temp1)
                end
                mul!(temp1, view(attentionmatrix.data, μ, νd, s, :, :, :, :), temp2)
                #add_U!(dSdUν, temp1)
                add_U!(temp_U, temp1)





                #barA = temp3
                #evaluate_gaugelinks!(barA, leftlinks, Us, [temp1, temp2])
                #barB = temp4
                #evaluate_gaugelinks!(barB, rightlinks, Us, [temp1, temp2])
                #LdCdU_i_add!(dSdUν, dSdCμm', barA, barB, attentionmatrix, temps_g, μ, ν, s)
                #LdCdU_i_add!(dSdUν, dSdCμm', barA, barB, attentionmatrix, temps_g, μ, νd, s)
                #LdCdU_i_add!(dSdU, dSdCμm', barA, barB, ρi, temps_g)
            end
            display(temp_U[:, :, 1, 1, 1, 1])

            =#

            #            end
            #end
        end
    end


    return

    count = 0
    for μd = 1:Dim
        for ν = 1:Dim
            if μd == ν
                continue
            end
            for s = 1:maxS
                count += 1
                #println("count = $count")
                dCμddUμ = dataset[count].dCμdUν

                numdCμddUμ = length(dCμddUμ[μd, μ])
                for j = 1:numdCμddUμ
                    dCμdUμ_j = dCμddUμ[μd, μ][j]
                    position = dCμdUμ_j.position
                    m = Tuple(-collect(position))
                    dSdCμm = shift_U(dSdCs[μd], m)

                    leftlinks = get_leftlinks(dCμdUμ_j)
                    rightlinks = get_rightlinks(dCμdUμ_j)

                    A = temp3
                    evaluate_gaugelinks!(A, leftlinks, Us, [temp1, temp2])

                    B = temp4
                    evaluate_gaugelinks!(B, rightlinks, Us, [temp1, temp2])
                    LdCdU_i_add!(dSdU, dSdCμm, A, B, attentionmatrix, temps_g, μd, ν, s)
                end

                dCμddagdUμ = dataset[count].dCμdagdUν
                numdCμdagdUμ = length(dCμddagdUμ[μd, μ])
                for j = 1:numdCμdagdUμ
                    dCμddagdUμ_j = dCμddagdUμ[μd, μ][j]

                    position = dCμddagdUμ_j.position
                    m = Tuple(-collect(position))
                    dSdCμm = shift_U(dSdCs[μd], m)
                    leftlinks = get_leftlinks(dCμddagdUμ_j)
                    rightlinks = get_rightlinks(dCμddagdUμ_j)

                    barA = temp3
                    evaluate_gaugelinks!(barA, leftlinks, Us, [temp1, temp2])
                    barB = temp4
                    evaluate_gaugelinks!(barB, rightlinks, Us, [temp1, temp2])
                    LdCdU_i_add!(dSdU, dSdCμm', barA, barB, attentionmatrix, temps_g, μd, ν, s)
                end

            end
        end
    end




end