function Gradientflow_general(
    U::Array{T1,1},
    B::Bfield{T,Dim},
    links::Vector{Vector{Wilsonline{Dim}}},
    linkvalues;
    Nflow=1,
    eps=0.01,
) where {NC,Dim,T1<:AbstractGaugefields{NC,Dim},T}
    F0 = initialize_TA_Gaugefields(U)
    Ftemps = Array{typeof(F0),1}(undef, 4)
    Ftemps[1] = F0
    for i = 2:4
        Ftemps[i] = initialize_TA_Gaugefields(U)
    end
    #T = eltype(U)
    Utemps = Temporalfields(U, num=2)
    #Utemps = Array{Array{T,1},1}(undef, 2)
    #for i = 1:2
    #    Utemps[i] = similar(U)
    #end
    tempG = Temporalfields(U[1], num=3)
    #tempG = Array{T,1}(undef, 3)
    #for i = 1:3
    #    tempG[i] = similar(U[1])
    #end

    gaugeaction = GaugeAction(U, B)
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
    end

    return Gradientflow_general(Nflow, eps, gaugeaction, Ftemps, tempG, Utemps)
end


function flow!(U, B, g::T) where {T<:Gradientflow}
    Ftemps = g._temporal_TA_field
    Utemps = g._temporal_U_field
    temps = g._temporal_G_field

    F0 = Ftemps[1]
    F1 = Ftemps[2]
    F2 = Ftemps[3]
    Ftmp = Ftemps[4]

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
        add_force!(F0, U, B, temps, plaqonly=true)

        exp_aF_U!(W1, -eps * (1 / 4), F0, U, temps) #exp(a*F)*U

        #
        clear_U!(F1)
        add_force!(F1, W1, B, temps, plaqonly=true)
        clear_U!(Ftmp)
        add_U!(Ftmp, -(8 / 9 * eps), F1)
        add_U!(Ftmp, (17 / 36 * eps), F0)
        exp_aF_U!(W2, 1, Ftmp, W1, temps) #exp(a*F)*U
        #
        clear_U!(F2)
        add_force!(F2, W2, B, temps, plaqonly=true)
        clear_U!(Ftmp)
        add_U!(Ftmp, -(3 / 4 * eps), F2)
        add_U!(Ftmp, (8 / 9 * eps), F1)
        add_U!(Ftmp, -(17 / 36 * eps), F0)
        exp_aF_U!(U, 1, Ftmp, W2, temps) #exp(a*F)*U  
    end
    unused!(Utemps, it_W1)
    unused!(Utemps, it_W2)
    #unused!(temps, it_temp1)
    #unused!(temps, it_temp2)
    #unused!(temps, it_temp3)

end

function flow!(U, B, g::T) where {T<:Gradientflow_general}
    Ftemps = g._temporal_TA_field
    Utemps = g._temporal_U_field
    temps = g._temporal_G_field

    F0 = Ftemps[1]
    F1 = Ftemps[2]
    F2 = Ftemps[3]
    Ftmp = Ftemps[4]

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

    for istep = 1:g.Nflow
        clear_U!(F0)

        F_update!(F0, U, B, 1, Dim, g.gaugeaction)

        exp_aF_U!(W1, -eps * (1 / 4), F0, U, temps) #exp(a*F)*U

        #
        clear_U!(F1)
        F_update!(F1, W1, B, 1, Dim, g.gaugeaction)
        clear_U!(Ftmp)
        add_U!(Ftmp, -(8 / 9 * eps), F1)
        add_U!(Ftmp, (17 / 36 * eps), F0)
        exp_aF_U!(W2, 1, Ftmp, W1, temps) #exp(a*F)*U
        #
        clear_U!(F2)
        F_update!(F2, W2, B, 1, Dim, g.gaugeaction)
        clear_U!(Ftmp)
        add_U!(Ftmp, -(3 / 4 * eps), F2)
        add_U!(Ftmp, (8 / 9 * eps), F1)
        add_U!(Ftmp, -(17 / 36 * eps), F0)
        exp_aF_U!(U, 1, Ftmp, W2, temps) #exp(a*F)*U  
    end

    unused!(Utemps, it_W1)
    unused!(Utemps, it_W2)
    #unused!(temps, it_temp1)
    #unused!(temps, it_temp2)
    #unused!(temps, it_temp3)

end

function F_update!(F, U, B, factor, Dim, gauge_action) # F -> F +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    #dSdUμ = similar(U[1])

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U, B)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(F[μ], factor, temp1)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_dSdUμ)
end
