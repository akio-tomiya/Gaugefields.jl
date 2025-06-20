module Gradientflow_module
import ..AbstractGaugefields_module:
    AbstractGaugefields,
    Abstractfields,
    initialize_TA_Gaugefields,
    add_force!,
    exp_aF_U!,
    clear_U!,
    add_U!,
    evaluate_wilson_loops!,
    exptU!,
    Traceless_antihermitian_add!,
    set_wing_U!,
    Traceless_antihermitian,
    evaluate_gaugelinks!,
    construct_Λmatrix_forSTOUT!,
    Traceless_antihermitian!,
    shift_U
import ..GaugeAction_module: GaugeAction, get_temporary_gaugefields, calc_dSdUμ!
import ..Abstractsmearing_module: Abstractsmearing
import Wilsonloop: make_loops_fromname, Wilsonline
using LinearAlgebra
import Wilsonloop: LinearAlgebra.adjoint
import ..Temporalfields_module: Temporalfields, unused!, get_temp


struct Gradientflow_general{Dim,TA,T} <: Abstractsmearing
    Nflow::Int64
    eps::Float64
    gaugeaction::GaugeAction{Dim,T}
    _temporal_TA_field::Array{TA,1}
    _temporal_G_field::Temporalfields{T}
    #_temporal_G_field::Array{T,1}
    _temporal_U_field::Temporalfields{Vector{T}}#Array{Array{T,1},1}

    function Gradientflow_general(
        U::Array{<:AbstractGaugefields{NC,Dim},1},
        linknames,
        linkvalues;
        Nflow=1,
        eps=0.01,
    ) where {NC,Dim}
        @assert length(linknames) == length(linkvalues)
        numlinks = length(linknames)
        links = Vector{Vector{Wilsonline{Dim}}}(undef, numlinks)
        for i = 1:numlinks
            links[i] = make_loops_fromname(linknames[i], Dim=Dim)
        end

        return Gradientflow_general(U, links, linkvalues, Nflow=Nflow, eps=eps)
    end

    function Gradientflow_general(
        U::Array{<:AbstractGaugefields{NC,Dim},1},
        links::Vector{Vector{Wilsonline{Dim}}},
        linkvalues;
        Nflow=1,
        eps=0.01,
    ) where {NC,Dim}
        F0 = initialize_TA_Gaugefields(U)
        #println("F0 $(typeof(F0))")
        Ftemps = Array{typeof(F0),1}(undef, 4)
        Ftemps[1] = F0
        for i = 2:4
            Ftemps[i] = initialize_TA_Gaugefields(U)
        end
        T = eltype(U)
        Utemps = Temporalfields(U, num=2)
        #Utemps = Array{Temporalfields{T},1}(undef, 2)
        #Utemps = Array{Array{T,1},1}(undef, 2)
        #for i = 1:2
        #    Utemps[i] = similar(U)
        #end

        tempG = Temporalfields(U[1], num=4)#Array{T,1}(undef, 3)
        #for i = 1:3
        #    tempG[i] = similar(U[1])
        #end

        gaugeaction = GaugeAction(U)
        @assert length(links) == length(linkvalues)
        numlinks = length(links)
        for i = 1:numlinks
            loop = links[i]#make_loops_fromname(linknames[i],Dim=Dim)
            factor = linkvalues[i]
            if typeof(factor) <: Real
                append!(loop, loop')
                push!(gaugeaction, factor, loop)
            elseif typeof(factor) <: Number
                push!(gaugeaction, factor, loop)
                push!(gaugeaction, factor', loop')
            else
                error("type of factor $(typeof(factor)) is not supported")
            end
            #append!(loop,loop')
            #factor = linkvalues[i]
            #push!(gaugeaction,factor,loop)
        end

        return new{Dim,typeof(F0),T}(Nflow, eps, gaugeaction, Ftemps, tempG, Utemps)
    end
end

mutable struct Gradientflow{TA,T} <: Abstractsmearing
    Nflow::Int64
    eps::Float64
    _temporal_TA_field::Array{TA,1}
    _temporal_G_field::Temporalfields{T}
    #_temporal_G_field::Array{T,1}
    #_temporal_U_field::Array{Array{T,1},1}
    _temporal_U_field::Temporalfields{Vector{T}}

    function Gradientflow(
        U::Array{T,1};
        Nflow=1,
        eps=0.01,
        mpi=false,
    ) where {T<:AbstractGaugefields}
        F0 = initialize_TA_Gaugefields(U)
        Ftemps = Array{typeof(F0),1}(undef, 4)
        Ftemps[1] = F0
        for i = 2:4
            Ftemps[i] = initialize_TA_Gaugefields(U)
        end

        Utemps = Temporalfields(U, num=2)
        #Utemps = Array{Array{T,1},1}(undef, 2)
        #for i = 1:2
        #    Utemps[i] = similar(U)
        #end
        tempG = Temporalfields(U[1], num=10)
        #tempG = Array{T,1}(undef, 3)
        ##for i = 1:3
        #    tempG[i] = similar(U[1])
        #end

        return new{typeof(F0),T}(Nflow, eps, Ftemps, tempG, Utemps)
    end

end

function get_tempG(x::T) where {T<:Gradientflow}
    return x._temporal_G_field
end

function get_eps(x::T) where {T<:Gradientflow}
    return x.eps
end

function flow!(U, g::T) where {T<:Gradientflow}
    #@assert Dim == 4 "Dimension should be 4. But Dim = $Dim "
    Ftemps = g._temporal_TA_field
    Utemps = g._temporal_U_field
    temps = g._temporal_G_field

    F0 = Ftemps[1]
    F1 = Ftemps[2]
    F2 = Ftemps[3]
    Ftmp = Ftemps[4]

    #Ftmp = similar(U)
    #W1 = Utemps[1]
    #W2 = Utemps[2]
    W1, it_W1 = get_temp(Utemps)
    W2, it_W2 = get_temp(Utemps)
    #temp1, it_temp1 = get_temp(temps)
    #temp2, it_temp2 = get_temp(temps)
    #temp3, it_temp3 = get_temp(temps)
    eps = g.eps

    for istep = 1:g.Nflow #RK4 integrator -> RK3?
        clear_U!(F0)
        add_force!(F0, U, temps, plaqonly=true)

        #add_force!(F0,U,[temp1,temp2,temp3],gparam)

        exp_aF_U!(W1, -eps * (1 / 4), F0, U, temps) #exp(a*F)*U

        #println("W1 ",W1[1][1,1,1,1,1,1])
        #
        clear_U!(F1)
        add_force!(F1, W1, temps, plaqonly=true)
        #add_force!(F1,W1,[temp1,temp2,temp3],gparam) #F
        #println("F1 ",F1[1][1,1,1,1,1,1])
        clear_U!(Ftmp)
        add_U!(Ftmp, -(8 / 9 * eps), F1)
        #println("Ftmp ",Ftmp[1][1,1,1,1,1,1])
        add_U!(Ftmp, (17 / 36 * eps), F0)
        #println("Ftmp1 ",Ftmp[1][1,1,1,1,1,1])
        exp_aF_U!(W2, 1, Ftmp, W1, temps) #exp(a*F)*U
        #exp_aF_U!(W2,1,Ftmp,U,[temp1,temp2,temp3]) #exp(a*F)*U
        #println("W2 ",W2[1][1,1,1,1,1,1])

        #
        clear_U!(F2)
        add_force!(F2, W2, temps, plaqonly=true)
        #add_force!(F2,W2,[temp1,temp2,temp3],gparam) #F
        #calc_gaugeforce!(F2,W2,univ) #F
        clear_U!(Ftmp)
        add_U!(Ftmp, -(3 / 4 * eps), F2)
        add_U!(Ftmp, (8 / 9 * eps), F1)
        add_U!(Ftmp, -(17 / 36 * eps), F0)
        #exp_aF_U!(W1,1,Ftmp,U,[temp1,temp2,temp3]) #exp(a*F)*U  
        exp_aF_U!(U, 1, Ftmp, W2, temps) #exp(a*F)*U  

        #println(typeof(U[1]))
        #println(U[1][1,1,1,1,1,1])

        #error("U")
    end
    unused!(Utemps, it_W1)
    unused!(Utemps, it_W2)
    #unused!(temps, it_temp1)
    #unused!(temps, it_temp2)
    #unused!(temps, it_temp3)

end

function flow!(U, g::Gradientflow_general{Dim,TA,T}) where {Dim,TA,T}
    #@assert Dim == 4 "Dimension should be 4. But Dim = $Dim "
    Ftemps = g._temporal_TA_field
    Utemps = g._temporal_U_field
    temps = g._temporal_G_field

    F0 = Ftemps[1]
    F1 = Ftemps[2]
    F2 = Ftemps[3]
    Ftmp = Ftemps[4]

    #Ftmp = similar(U)
    W1, it_W1 = get_temp(Utemps)
    W2, it_W2 = get_temp(Utemps)
    #temp1, it_temp1 = get_temp(temps)
    #temp2, it_temp2 = get_temp(temps)
    #temp3, it_temp3 = get_temp(temps)
    #W1 = Utemps[1]
    #W2 = Utemps[2]
    #temp1 = temps[1]
    #temp2 = temps[2]
    #temp3 = temps[3]
    eps = g.eps

    for istep = 1:g.Nflow #RK4 integrator -> RK3?
        clear_U!(F0)
        F_update!(F0, U, 1, Dim, g.gaugeaction)

        #add_force!(F0,U,temps,plaqonly = true)

        #add_force!(F0,U,[temp1,temp2,temp3],gparam)
        exp_aF_U!(W1, -eps * (1 / 4), F0, U, temps) #exp(a*F)*U

        #println("W1 ",W1[1][1,1,1,1,1,1])
        #
        clear_U!(F1)
        F_update!(F1, W1, 1, Dim, g.gaugeaction)
        #add_force!(F1,W1,[temp1,temp2,temp3],plaqonly = true)
        #add_force!(F1,W1,[temp1,temp2,temp3],gparam) #F
        #println("F1 ",F1[1][1,1,1,1,1,1])
        clear_U!(Ftmp)
        add_U!(Ftmp, -(8 / 9 * eps), F1)
        #println("Ftmp ",Ftmp[1][1,1,1,1,1,1])
        add_U!(Ftmp, (17 / 36 * eps), F0)
        #println("Ftmp1 ",Ftmp[1][1,1,1,1,1,1])
        exp_aF_U!(W2, 1, Ftmp, W1, temps) #exp(a*F)*U
        #exp_aF_U!(W2,1,Ftmp,U,[temp1,temp2,temp3]) #exp(a*F)*U
        #println("W2 ",W2[1][1,1,1,1,1,1])

        #
        clear_U!(F2)
        F_update!(F2, W2, 1, Dim, g.gaugeaction)
        #add_force!(F2,W2,[temp1,temp2,temp3],plaqonly = true)
        #add_force!(F2,W2,[temp1,temp2,temp3],gparam) #F
        #calc_gaugeforce!(F2,W2,univ) #F
        clear_U!(Ftmp)
        #println(istep)
        add_U!(Ftmp, -(3 / 4 * eps), F2)
        add_U!(Ftmp, (8 / 9 * eps), F1)
        add_U!(Ftmp, -(17 / 36 * eps), F0)
        #println(istep)
        #exp_aF_U!(W1,1,Ftmp,U,[temp1,temp2,temp3]) #exp(a*F)*U  
        exp_aF_U!(U, 1, Ftmp, W2, temps) #exp(a*F)*U  

        #println(typeof(U[1]))
        #println(U[1][1,1,1,1,1,1])

        #error("U")
    end

    unused!(Utemps, it_W1)
    unused!(Utemps, it_W2)
    ##unused!(temps, it_temp1)
    #unused!(temps, it_temp2)
    #unused!(temps, it_temp3)



end

function F_update!(F, U, factor, Dim, gauge_action) # F -> F +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    #dSdUμ = similar(U[1])

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(F[μ], factor, temp1)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_dSdUμ)
    #error("d")

end

include("./gradientflow_Bfields.jl")

end
