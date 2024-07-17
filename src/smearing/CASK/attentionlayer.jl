import Wilsonloop: make_staple

get_weightmatrix(at::WeightMatrix_layer) = at.data
export get_weightmatrix

function WeightMatrix_layer(loopset::Vector{Vector{Wilsonline{Dim}}},
    U::Vector{<:AbstractGaugefields{NC,Dim}}, maxS=1,
    ρs=zeros(Float64, length(loopset))
) where {NC,Dim}
    return WeightMatrix_layer(loopset, loopset, U, maxS, ρs, ρs)
end

function WeightMatrix_layer(loopset_Q::Vector{Vector{Wilsonline{Dim}}}, loopset_K::Vector{Vector{Wilsonline{Dim}}},
    U::Vector{<:AbstractGaugefields{NC,Dim}}, maxS=1,
    ρs_Q=zeros(Float64, length(loopset_Q)),
    ρs_K=zeros(Float64, length(loopset_K)),
) where {NC,Dim}
    T = eltype(U)
    @assert Dim == 4 "Dim = 4 is only supported"
    Qstout = STOUTsmearing_layer(loopset_Q, U, ρs_Q)
    Kstout = STOUTsmearing_layer(loopset_K, U, ρs_K)
    _, _, nx, ny, nz, nt = size(U[1])

    UQ = similar(U)
    UK = similar(U)



    Tρ = eltype(ρs_Q)
    data = zeros(Float64, Dim, Dim, maxS, nx, ny, nz, nt)
    Dim3 = Dim + 3

    numg = Dim #+ 2
    temps = Vector{T}(undef, numg)
    for i = 1:numg
        temps[i] = similar(U[1])
    end

    dSdatilde = zero(data)
    return WeightMatrix_layer{T,Dim,Dim3,Tρ}(data, maxS, Qstout, Kstout, UQ, UK, dSdatilde, temps)
end

#function STOUTsmearing_layer(loopset::Vector{Vector{Wilsonline{Dim}}}, U::Vector{<:AbstractGaugefields{NC,Dim}}, ρs=zeros(Float64, length(loopset))) where {NC,Dim}

function forward!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, UQ::Vector{<:AbstractGaugefields{NC,Dim}}, UK::Vector{T}) where {T,Dim,Dim3,Tρ,NC}
    #for i = 1:length(a.Qstout.ρs)
    #    a.Qstout.ρs[i] = deepcopy(ρs_Q[i])
    #end
    #for i = 1:length(a.Kstout.ρs)
    #    a.Kstout.ρs[i] = deepcopy(ρs_K[i])
    #end

    #forward!(a.Qstout, a.UQ, ρs_Q, Uin, Uin)
    #forward!(a.Kstout, a.UK, ρs_K, Uin, Uin)


    temp_U1 = a.Kstout.temps[1]
    VK = a.Kstout.temps[2]
    UQVK = a.Kstout.temps[3]

    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:a.maxS
                loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
                w = Wilsonline(loop)
                Vloop = make_staple(w, μ)[1]
                #display(Vloop)
                evaluate_gaugelinks!(VK, Vloop, UK, [temp_U1])
                mul!(UQVK, UQ[μ], VK)
                site_realtrace!(view(a.data, μ, ν, s, :, :, :, :), UQVK)


            end

        end
    end
    #map!(x -> ifelse(x > 0, x, zero(x)), a.data, a.data)
    #z = ReLU(x)*π/2NC, -NC <=  z <= NC
    dNC = 0.01
    map!(x -> tan(x * (π / 4(NC + dNC))), a.data, a.data)
    #map!(x -> ifelse(x > 0, tan(x * (π / 2(NC + dNC))), zero(x)), a.data, a.data)
    #(s::STOUTsmearing_layer{T,Dim}, Uout, ρs::Vector{TN}, Uinα, Uinβ) where {T,Dim,TN<:Number}
end

```
re(tr(UQ VK)) - re(tr(U V))
```
function forward!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, UQ::Vector{<:AbstractGaugefields{NC,Dim}}, UK::Vector{T}, UIN::Vector{T}) where {T,Dim,Dim3,Tρ,NC}
    #for i = 1:length(a.Qstout.ρs)
    #    a.Qstout.ρs[i] = deepcopy(ρs_Q[i])
    #end
    #for i = 1:length(a.Kstout.ρs)
    #    a.Kstout.ρs[i] = deepcopy(ρs_K[i])
    #end

    #forward!(a.Qstout, a.UQ, ρs_Q, Uin, Uin)
    #forward!(a.Kstout, a.UK, ρs_K, Uin, Uin)


    temp_U1 = a.Kstout.temps[1]
    VK = a.Kstout.temps[2]
    UQVK = a.Kstout.temps[3]

    #a = (reTr[UQVK] - reTr[UV])

    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:a.maxS
                loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
                w = Wilsonline(loop)
                Vloop = make_staple(w, μ)[1]
                #display(Vloop)
                evaluate_gaugelinks!(VK, Vloop, UK, [temp_U1])
                mul!(UQVK, UQ[μ], VK)
                site_realtrace_add!(view(a.data, μ, ν, s, :, :, :, :), UQVK)

                V = VK
                UV = UQVK
                evaluate_gaugelinks!(V, Vloop, UIN, [temp_U1])
                mul!(UV, UIN[μ], V)
                site_realtrace_add!(view(a.data, μ, ν, s, :, :, :, :), UV, -1)
                #reTr[UV]

            end

        end
    end
    #map!(x -> ifelse(x > 0, x, zero(x)), a.data, a.data)
    #z = ReLU(x)*π/2NC, -NC <=  z <= NC
    dNC = 0.01
    map!(x -> tan(x * (π / 4(NC + dNC))), a.data, a.data)
    #map!(x -> ifelse(x > 0, tan(x * (π / 2(NC + dNC))), zero(x)), a.data, a.data)
    #(s::STOUTsmearing_layer{T,Dim}, Uout, ρs::Vector{TN}, Uinα, Uinβ) where {T,Dim,TN<:Number}
end


function forward!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, Uin, ρs_Q::Vector{TN}, ρs_K::Vector{TN}) where {T,Dim,Dim3,Tρ,TN<:Number}
    for i = 1:length(a.Qstout.ρs)
        a.Qstout.ρs[i] = deepcopy(ρs_Q[i])
    end
    for i = 1:length(a.Kstout.ρs)
        a.Kstout.ρs[i] = deepcopy(ρs_K[i])
    end

    forward!(a.Qstout, a.UQ, ρs_Q, Uin, Uin)
    #add_U!(a.UQ, -1, Uin)
    forward!(a.Kstout, a.UK, ρs_K, Uin, Uin)
    #add_U!(a.UK, -1, Uin)

    #display(a.UQ[1][:, :, 1, 1, 1, 1])
    # display(a.UK[1][:, :, 1, 1, 1, 1])

    #forward!(a, a.UQ, a.UK)
    forward!(a, a.UQ, a.UK, Uin)
    return

end

```
-----------------------------------
a = re(tr(UQ VK)) - re(tr(Uin_1 V(Uin_2)))
Input: dS/da
Outputs: dS/dUin,dS/dρQ, dS/dρK
-----------------------------------
Output: dS/dUin = dS/dUQa dUQa/dUin + dS/dUQb  dUQb/dUin +  dS/dUKa  dUKa/dUin + dS/dUKb  dUKb/dUin + dS/dUin_0 dUin_0/dUin
dS/dUin_0 = dS/dUin_1 dUin_1/dUin_0 +  dS/dVin_2 dVin_2/dUin_0 

dS/dUQa = dS/dUQ dUQ/dUQa
dS/dUQb = dS/dUQ dUQ/dUQb
dS/dUQ = dS/da da/dUQ

dS/dUKa = dS/dUK dUK/dUa
dS/dUKb = dS/dUK dUK/dUb
dS/dUK = dS/da da/dUK



Output: dS/dρQ = dS/dUQ dUQ/dρQ
Output: dS/dρK = dS/dUK dUK/dρK
```
function backward_dSdU_add_fromdSda!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdUin::Vector{T}, dSdρQ, dSdρK, dSda) where {T,Dim,Dim3,Tρ}
    dSdUQ = a.temps[1:Dim]
    dSdUQ = similar(a.temps[1:Dim]) #debug
    clear_U!(dSdUQ)
    backward_dSdUQ_add!(a, dSdUQ, dSda) #dS/dUQ = dSda da/dUQ
    #dSdUαQ = a.temps[Dim+1]
    #dSdUβQ = a.temps[Dim+2]

    #UQ = exp(f(UQb)) UQa
    #dS/dUin = dS/dUQa + dS/dUQb
    backward_dSdUα_add!(a.Qstout, dSdUin, dSdUQ) #dS/dUQa = dS/dUQ dUQ/dUQa
    backward_dSdUβ_add!(a.Qstout, dSdUin, dSdUQ) #dS/dUQb = dS/dUQ dUQ/dUQb

    backward_dSdρ_add!(a.Qstout, dSdρQ, dSdUQ) #dS/dρQ = dS/dUQ dUQ/dρQ

    #backward_dSdUαUβρ_add!(a.Kstout, dSdUαQ, dSdUβQ, dSdρQ, dSdUQ)
    #add_U!(dSdUout, dSdUαQ)
    #add_U!(dSdUout, dSdUβQ)

    #UK = exp(f(UKb)) UKa
    dSdUK = a.temps[1:Dim]
    dSdUK = similar(a.temps[1:Dim])
    clear_U!(dSdUK)
    backward_dSdUK_add!(a, dSdUK, dSda) #dS/dUK = dS/da da/dUK
    #dS/dUin += dS/dUKa + dS/dUKb
    backward_dSdUα_add!(a.Kstout, dSdUin, dSdUK) #dS/dUKa = dS/dUK dUK/dUa
    backward_dSdUβ_add!(a.Kstout, dSdUin, dSdUK) #dS/dUKb = dS/dUK dUK/dUb
    #println("UK") 
    #display(dSdUK)
    backward_dSdρ_add!(a.Kstout, dSdρK, dSdUK) #dS/dρK = dS/dUK dUK/dρK
    #display(dSdρK)

    #dSdUαK = a.temps[Dim+1]
    #dSdUβK = a.temps[Dim+2]
    #backward_dSdUαUβρ_add!(a.Qstout, dSdUαK, dSdUβK, dSdρK, dSdUK)
    #add_U!(dSdUout, dSdUαK)
    #add_U!(dSdUout, dSdUβK)

    backward_dSdUQ_add!(a, dSdUin, dSda, Uin, -1)
    backward_dSdUK_add!(a, dSdUin, dSda, Uin, Uin, -1)


end
export backward_dSdU_add_fromdSda!

function backward_dSdU_add_fromdSda!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdUin::Vector{T}, dSda) where {T,Dim,Dim3,Tρ}
    dSdUQ = a.temps[1:Dim]
    dSdUQ = similar(a.temps[1:Dim]) #debug
    clear_U!(dSdUQ)
    backward_dSdUQ_add!(a, dSdUQ, dSda)
    add_U!(dSdUin, 1, dSdUQ)

    dSdUK = a.temps[1:Dim]
    dSdUK = similar(a.temps[1:Dim])
    clear_U!(dSdUK)

    backward_dSdUK_add!(a, dSdUK, dSda)
    add_U!(dSdUin, 1, dSdUK)
end


function backward_dSdUQ_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdUQ::Vector{T}, dSda) where {T,Dim,Dim3,Tρ}
    backward_dSdUQ_add!(a, dSdUQ, dSda, a.UK, 1)
    return
    #map!(x -> ifelse(x > 0, one(x), zero(x)), a.dSdatilde, dSda)
    temp_U1 = a.Kstout.temps[1]
    VK = a.Kstout.temps[2]
    aVK = a.Kstout.temps[3]

    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:a.maxS
                loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
                w = Wilsonline(loop)
                Vloop = make_staple(w, μ)[1]
                evaluate_gaugelinks!(VK, Vloop, a.UK, [temp_U1])

                mul!(aVK, view(dSda, μ, ν, s, :, :, :, :), VK)
                add_U!(dSdUQ[μ], 0.5, aVK) #real(tr(AB)) = (tr(AB + B^dag A^dag))/2
            end
        end
    end
end
export backward_dSdUQ_add!

function backward_dSdUQ_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdUQ::Vector{T}, dSda, UK, factor) where {T,Dim,Dim3,Tρ}
    #map!(x -> ifelse(x > 0, one(x), zero(x)), a.dSdatilde, dSda)
    temp_U1 = a.Kstout.temps[1]
    VK = a.Kstout.temps[2]
    aVK = a.Kstout.temps[3]

    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:a.maxS
                loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
                w = Wilsonline(loop)
                Vloop = make_staple(w, μ)[1]
                evaluate_gaugelinks!(VK, Vloop, UK, [temp_U1])

                mul!(aVK, view(dSda, μ, ν, s, :, :, :, :), VK)
                add_U!(dSdUQ[μ], factor * 0.5, aVK) #real(tr(AB)) = (tr(AB + B^dag A^dag))/2
            end
        end
    end
end

#function backward_dSdUatilde_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdatilde, dSda) where {T,Dim,Dim3,Tρ}
#    map!(x -> ifelse(x > 0, x, zero(x)), dSdatilde, dSda)
#end

function backward_dSdUK_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdUK::Vector{T}, dSda) where {T,Dim,Dim3,Tρ}
    backward_dSdUK_add!(a, dSdUK, dSda, a.UQ, a.UK, 1)
    return

    dSdVKμ = a.Kstout.temps[4]
    temp = a.Kstout.temps[1]
    temps = a.Qstout.temps



    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:a.maxS
                loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
                w = Wilsonline(loop)
                Vloop = make_staple(w, μ)[1]
                clear_U!(dSdVKμ)
                #if α == 1
                #    println("dSdV")
                #    display(dSdVKμ[:, :, 1, 1, 1, 1])
                #end
                backward_dSdVKνs_add!(a, dSdVKμ, dSda, μ, ν, s, temp)

                #backward_dSdUK_add_fromVK!(dSdUK, Vloop, dSdVKμ, a.UK, temps)
                #if α == 1
                #    println("μ = $μ ν = $ν s = $s")
                #    display(dSdUK[1][:, :, 1, 1, 1, 1])
                #end
                for α = 1:Dim
                    backward_dSdUK_add_fromVK!(dSdUK[α], Vloop, dSdVKμ, a.UK, α, temps)
                end
                #=
                if α == 1
                    println("μ = $μ ν = $ν s = $s")
                    println("dSda")
                    display(dSda[μ, ν, s, 1, 1, 1, 1])
                    println("dSdV")
                    display(dSdVKμ[:, :, 1, 1, 1, 1])
                    println("dSdU")
                    display(dSdUK[1][:, :, 1, 1, 1, 1])
                end
                =#
                #println("μ = $μ ν = $ν s = $s")
                #display(dSdUK[1][:, :, 1, 1, 1, 1])
            end
        end
    end


end
export backward_dSdUK_add!

function backward_dSdUK_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdUK::Vector{T}, dSda, UQ, UK, factor) where {T,Dim,Dim3,Tρ}
    dSdVKμ = a.Kstout.temps[4]
    temp = a.Kstout.temps[1]
    temps = a.Qstout.temps



    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            for s = 1:a.maxS
                loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
                w = Wilsonline(loop)
                Vloop = make_staple(w, μ)[1]
                clear_U!(dSdVKμ)
                #if α == 1
                #    println("dSdV")
                #    display(dSdVKμ[:, :, 1, 1, 1, 1])
                #end
                backward_dSdVKνs_add!(a, dSdVKμ, dSda, μ, ν, s, temp, UQ, factor)

                for α = 1:Dim
                    backward_dSdUK_add_fromVK!(dSdUK[α], Vloop, dSdVKμ, UK, α, temps)
                end
            end
        end
    end


end
export backward_dSdUin_2_add!

function backward_dSdVKνs_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdVK::T, dSda, μ, ν, s, temp, UQ, factor) where {T,Dim,Dim3,Tρ}
    #map!(x -> ifelse(x > 0, one(x), zero(x)), a.dSdatilde, dSda)
    aUQ = temp#a.Kstout.temps[1]
    #for μ = 1:Dim
    mul!(aUQ, view(dSda, μ, ν, s, :, :, :, :), UQ[μ])
    #println("UQ")
    #display(a.UQ[μ][:, :, 1, 1, 1, 1])
    #println("aUQ")
    #display(aUQ[:, :, 1, 1, 1, 1])
    #println("predSdVK")
    #display(dSdVK[:, :, 1, 1, 1, 1])
    add_U!(dSdVK, 0.5 * factor, aUQ)
    #println("dSdVK")
    #display(dSdVK[:, :, 1, 1, 1, 1])
    #end
end

function backward_dSdVKνs_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdVK::T, dSda, μ, ν, s, temp) where {T,Dim,Dim3,Tρ}
    backward_dSdVKνs_add!(a::WeightMatrix_layer{T,Dim,Dim3,Tρ}, dSdVK::T, dSda, μ, ν, s, temp, a.UQ, 1)
    return

    #map!(x -> ifelse(x > 0, one(x), zero(x)), a.dSdatilde, dSda)
    aUQ = temp#a.Kstout.temps[1]
    #for μ = 1:Dim
    mul!(aUQ, view(dSda, μ, ν, s, :, :, :, :), a.UQ[μ])
    #println("UQ")
    #display(a.UQ[μ][:, :, 1, 1, 1, 1])
    #println("aUQ")
    #display(aUQ[:, :, 1, 1, 1, 1])
    #println("predSdVK")
    #display(dSdVK[:, :, 1, 1, 1, 1])
    add_U!(dSdVK, 0.5, aUQ)
    #println("dSdVK")
    #display(dSdVK[:, :, 1, 1, 1, 1])
    #end
end
export backward_dSdVKνs_add!

function backward_dSdUKα_add_fromVK!(dSdUKα, dV1, dSdVK, UK, temps)
    temp_U1 = temps[1]
    temp_U2 = temps[2]
    temp_U3 = temps[3]
    temp_U4 = temps[4]
    L = dSdVK

    numdV1 = length(dV1)
    for i = 1:numdV1
        dV1_i = dV1[i]
        leftlinks = get_leftlinks(dV1_i)
        rightlinks = get_rightlinks(dV1_i)

        position = dV1_i.position
        m = Tuple(-collect(position))
        Lμm = shift_U(L, m)

        # left
        A = temp_U3
        #println("left")
        numleft = length(leftlinks)
        #println("numleft $numleft")
        #show(leftlinks)
        if numleft != 0
            evaluate_gaugelinks!(A, leftlinks, UK, [temp_U1, temp_U2])
            mul!(temp_U1, Lμm, A)
        else
            substitute_U!(temp_U1, Lμm)
        end

        #right
        B = temp_U4
        numright = length(rightlinks)
        #println("right")
        # display(temp_U1[:, :, 1, 1, 1, 1])
        #show(rightlinks)
        #l1 = L[:, :, 1, 1, 1, 1]
        #u1 = U[2][:, :, 1+1, 1, 1, 1]
        #u2 = U[1][:, :, 1, 1+1, 1, 1]'
        #u1u2l1 = u1 * u2 * l1
        if numright != 0
            evaluate_gaugelinks!(B, rightlinks, UK, [temp_U4, temp_U2])
            mul!(temp_U2, B, temp_U1)
        else
            substitute_U!(temp_U2, temp_U1)
        end
        #B*L*A

        add_U!(dSdUKα, temp_U2)
        #display(L[:, :, 1, 1, 1, 1])
        #display(LdVdagdU[:, :, 1, 1, 1, 1])
        #println("analytical")
        #display(u1u2l1)
    end
end


function backward_dSdUKα_add_fromVK!(dSdUKα, dV1, dSdVK, UK, temps, attentionmatrix)
    temp_U1 = temps[1]
    temp_U2 = temps[2]
    temp_U3 = temps[3]
    temp_U4 = temps[4]
    L = dSdVK

    numdV1 = length(dV1)
    for i = 1:numdV1
        dV1_i = dV1[i]
        leftlinks = get_leftlinks(dV1_i)
        rightlinks = get_rightlinks(dV1_i)

        position = dV1_i.position
        m = Tuple(-collect(position))
        Lμm = shift_U(L, m)

        # left
        A = temp_U3
        #println("left")
        numleft = length(leftlinks)
        #println("numleft $numleft")
        #show(leftlinks)
        if numleft != 0
            evaluate_gaugelinks!(A, leftlinks, UK, [temp_U1, temp_U2])
            mul!(temp_U1, Lμm, A)
        else
            substitute_U!(temp_U1, Lμm)
        end

        #right
        B = temp_U4
        numright = length(rightlinks)
        #println("right")
        # display(temp_U1[:, :, 1, 1, 1, 1])
        #show(rightlinks)
        #l1 = L[:, :, 1, 1, 1, 1]
        #u1 = U[2][:, :, 1+1, 1, 1, 1]
        #u2 = U[1][:, :, 1, 1+1, 1, 1]'
        #u1u2l1 = u1 * u2 * l1
        if numright != 0
            evaluate_gaugelinks!(B, rightlinks, UK, [temp_U4, temp_U2])
            mul!(temp_U2, B, temp_U1)
        else
            substitute_U!(temp_U2, temp_U1)
        end
        #B*L*A

        mul!(temp_U1, attentionmatrix, temp_U2)
        #println("temp_U1")
        #display(temp_U1[:, :, 1, 1, 1, 1])
        add_U!(dSdUKα, temp_U1)

        #Gaugefields.add_U!(dSdUKα, temp_U2)
        #display(L[:, :, 1, 1, 1, 1])
        #display(LdVdagdU[:, :, 1, 1, 1, 1])
        #println("analytical")
        #display(u1u2l1)
    end
end


function backward_dSdUK_add_fromVK!(dSdUKα, V1::Wilsonline{Dim}, dSdVKμ, UK, α, temps) where {Dim} #dVmudUKalpha
    #=
    dS/dUK = sum dS/dVK dVK/dU
    =#
    #temp_U1 = temps[1]
    #temp_U2 = temps[2]
    #temp_U3 = temps[3]
    #temp_U4 = temps[4]
    #L = dSdVK
    #LdVdagdU = similar(U[1])

    #V1 = make_Vdag(μ,ν)
    #println("α = $α")
    #show(V1)
    #for α = 1:Dim
    dV1 = derive_U(V1, α)
    #show(dV1)
    #println("---- make_LdV1dU_add-----------")
    #show(dV1)
    #println("---------------------------")
    #for μ = 1:Dim
    backward_dSdUKα_add_fromVK!(dSdUKα, dV1, dSdVKμ, UK, temps)
    #end

    dV1dag = derive_U(V1', α)
    #for μ = 1:Dim
    backward_dSdUKα_add_fromVK!(dSdUKα, dV1dag, dSdVKμ', UK, temps)
    #end
    #end

end


function backward_dSdUK_add_fromVK!(dSdUKα, V1::Wilsonline{Dim}, dSdVKμ, UK, α, temps, attentionmatrix) where {Dim} #dVmudUKalpha
    #=
    dS/dUK = sum dS/dVK dVK/dU
    =#
    #temp_U1 = temps[1]
    #temp_U2 = temps[2]
    #temp_U3 = temps[3]
    #temp_U4 = temps[4]
    #L = dSdVK
    #LdVdagdU = similar(U[1])

    #V1 = make_Vdag(μ,ν)
    #println("α = $α")
    #show(V1)
    #for α = 1:Dim
    dV1 = derive_U(V1, α)
    #show(dV1)
    #println("---- make_LdV1dU_add-----------")
    #show(dV1)
    #println("---------------------------")
    #for μ = 1:Dim
    backward_dSdUKα_add_fromVK!(dSdUKα, dV1, dSdVKμ, UK, temps, attentionmatrix)
    #end

    dV1dag = derive_U(V1', α)
    #for μ = 1:Dim
    backward_dSdUKα_add_fromVK!(dSdUKα, dV1dag, dSdVKμ', UK, temps, attentionmatrix)
    #end
    #end

end
