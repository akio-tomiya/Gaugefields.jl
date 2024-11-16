module Obs_measure_module

using Random
using LinearAlgebra
using Wilsonloop

import ..AbstractGaugefields_module:
    AbstractGaugefields,
    evaluate_gaugelinks!,
    Traceless_antihermitian,
    Traceless_antihermitian!,
    substitute_U!
import ..GaugeAction_module: GaugeAction, evaluate_staple_eachindex!
import ..Gradientflow_module: Gradientflow_general, flow!
import ..Temporalfields_module: Temporalfields, unused!, get_temp

function calculate_topological_charge_plaq(
    U::Array{T,1},
    temp_UμνTA,
    temps::Temporalfields,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end
function calculate_topological_charge_plaq(
    U::Array{T,1},
    B::Array{T,2},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, B, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end

function calculate_topological_charge_clover(
    U::Array{T,1},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end
function calculate_topological_charge_clover(
    U::Array{T,1},
    B::Array{T,2},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, B, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end

function calculate_topological_charge_improved(
    U::Array{T,1},
    temp_UμνTA,
    Qclover,
    temps::Temporalfields,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA

    numofloops = calc_UμνTA!(UμνTA, "rect", U, temps)
    Qrect = 2 * calc_Q(UμνTA, numofloops, U)
    c1 = -1 / 12
    c0 = 5 / 3
    Q = c0 * Qclover + c1 * Qrect
    return Q
end
function calculate_topological_charge_improved(
    U::Array{T,1},
    B::Array{T,2},
    temp_UμνTA,
    Qclover,
    temps::Temporalfields,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA

    numofloops = calc_UμνTA!(UμνTA, "rect", U, B, temps)
    Qrect = 2 * calc_Q(UμνTA, numofloops, U)
    c1 = -1 / 12
    c0 = 5 / 3
    Q = c0 * Qclover + c1 * Qrect
    return Q
end

function calc_UμνTA!(
    temp_UμνTA,
    name::String,
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps::Temporalfields,
) where {NC,Dim}
    loops_μν, numofloops = calc_loopset_μν_name(name, Dim)
    calc_UμνTA!(temp_UμνTA, loops_μν, U, temps)
    return numofloops
end
function calc_UμνTA!(
    temp_UμνTA,
    name::String,
    U::Array{T,1},
    B::Array{T,2},
    temps::Temporalfields,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    loops_μν, numofloops = calc_loopset_μν_name(name, Dim)
    calc_UμνTA!(temp_UμνTA, loops_μν, U, B, temps)
    return numofloops
end


function calc_UμνTA!(
    temp_UμνTA,
    loops_μν,
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps_g::Temporalfields,
) where {NC,Dim}
    UμνTA = temp_UμνTA

    temp, it_temp   = get_temp(temps_g)
    temps, it_temps = get_temp(temps_g,5)
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end
            evaluate_gaugelinks!(temp, loops_μν[μ, ν], U, temps)
            Traceless_antihermitian!(UμνTA[μ, ν], temp)
        end
    end
    unused!(temps_g,it_temp)
    unused!(temps_g,it_temps)
    return
end
function calc_UμνTA!(
    temp_UμνTA,
    loops_μν,
    U::Array{T,1},
    B::Array{T,2},
    temps_g::Temporalfields,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA

    temp, it_temp   = get_temp(temps_g)
    temps, it_temps = get_temp(temps_g,8)
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end
            evaluate_gaugelinks!(temp, loops_μν[μ, ν], U, B, temps)
            Traceless_antihermitian!(UμνTA[μ, ν], temp)
        end
    end
    unused!(temps_g,it_temp)
    unused!(temps_g,it_temps)
    return
end


#=
implementation of topological charge is based on
https://arxiv.org/abs/1509.04259
=#
function calc_Q(UμνTA, numofloops, U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    Q = 0.0
    if Dim == 4
        ε(μ, ν, ρ, σ) = epsilon_tensor(μ, ν, ρ, σ)
    else
        error("Dimension $Dim is not supported")
    end
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end
            Uμν = UμνTA[μ, ν]
            for ρ = 1:Dim
                for σ = 1:Dim
                    if ρ == σ
                        continue
                    end
                    Uρσ = UμνTA[ρ, σ]
                    s = tr(Uμν, Uρσ)
                    Q += ε(μ, ν, ρ, σ) * s / numofloops^2
                end
            end
        end
    end

    return -Q / (32 * (π^2))
end

#topological charge
function epsilon_tensor(mu::Int, nu::Int, rho::Int, sigma::Int)
    sign = 1 # (3) 1710.09474 extended epsilon tensor
    if mu < 0
        sign *= -1
        mu = -mu
    end
    if nu < 0
        sign *= -1
        nu = -nu
    end
    if rho < 0
        sign *= -1
        rh = -rho
    end
    if sigma < 0
        sign *= -1
        sigma = -sigma
    end
    epsilon = zeros(Int, 4, 4, 4, 4)
    epsilon[1, 2, 3, 4] = 1
    epsilon[1, 2, 4, 3] = -1
    epsilon[1, 3, 2, 4] = -1
    epsilon[1, 3, 4, 2] = 1
    epsilon[1, 4, 2, 3] = 1
    epsilon[1, 4, 3, 2] = -1
    epsilon[2, 1, 3, 4] = -1
    epsilon[2, 1, 4, 3] = 1
    epsilon[2, 3, 1, 4] = 1
    epsilon[2, 3, 4, 1] = -1
    epsilon[2, 4, 1, 3] = -1
    epsilon[2, 4, 3, 1] = 1
    epsilon[3, 1, 2, 4] = 1
    epsilon[3, 1, 4, 2] = -1
    epsilon[3, 2, 1, 4] = -1
    epsilon[3, 2, 4, 1] = 1
    epsilon[3, 4, 1, 2] = 1
    epsilon[3, 4, 2, 1] = -1
    epsilon[4, 1, 2, 3] = -1
    epsilon[4, 1, 3, 2] = 1
    epsilon[4, 2, 1, 3] = 1
    epsilon[4, 2, 3, 1] = -1
    epsilon[4, 3, 1, 2] = -1
    epsilon[4, 3, 2, 1] = 1
    return epsilon[mu, nu, rho, sigma] * sign
end

function calc_loopset_μν_name(name, Dim)
    loops_μν = Array{Vector{Wilsonline{Dim}},2}(undef, Dim, Dim)
    if name == "plaq"
        numofloops = 1
        for μ = 1:Dim
            for ν = 1:Dim
                loops_μν[μ, ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                plaq = make_plaq(μ, ν, Dim = Dim)
                push!(loops_μν[μ, ν], plaq)
            end
        end
    elseif name == "clover"
        numofloops = 4
        for μ = 1:Dim
            for ν = 1:Dim
                loops_μν[μ, ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                loops_μν[μ, ν] = make_cloverloops_topo(μ, ν, Dim = Dim)
            end
        end
    elseif name == "rect"
        numofloops = 8
        for μ = 1:4
            for ν = 1:4
                if ν == μ
                    continue
                end
                loops = Wilsonline{Dim}[]
                loop_righttop = Wilsonline([(μ, 2), (ν, 1), (μ, -2), (ν, -1)])
                loop_lefttop = Wilsonline([(ν, 1), (μ, -2), (ν, -1), (μ, 2)])
                loop_rightbottom = Wilsonline([(ν, -1), (μ, 2), (ν, 1), (μ, -2)])
                loop_leftbottom = Wilsonline([(μ, -2), (ν, -1), (μ, 2), (ν, 1)])
                push!(loops, loop_righttop)
                push!(loops, loop_lefttop)
                push!(loops, loop_rightbottom)
                push!(loops, loop_leftbottom)

                loop_righttop = Wilsonline([(μ, 1), (ν, 2), (μ, -1), (ν, -2)])
                loop_lefttop = Wilsonline([(ν, 2), (μ, -1), (ν, -2), (μ, 1)])
                loop_rightbottom = Wilsonline([(ν, -2), (μ, 1), (ν, 2), (μ, -1)])
                loop_leftbottom = Wilsonline([(μ, -1), (ν, -2), (μ, 1), (ν, 2)])
                push!(loops, loop_righttop)
                push!(loops, loop_lefttop)
                push!(loops, loop_rightbottom)
                push!(loops, loop_leftbottom)

                loops_μν[μ, ν] = loops
            end
        end
    else
        error("$name is not supported")
    end
    return loops_μν, numofloops
end


function make_cloverloops_topo(μ, ν; Dim = 4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)])
    loop_lefttop = Wilsonline([(ν, 1), (μ, -1), (ν, -1), (μ, 1)])
    loop_rightbottom = Wilsonline([(ν, -1), (μ, 1), (ν, 1), (μ, -1)])
    loop_leftbottom = Wilsonline([(μ, -1), (ν, -1), (μ, 1), (ν, 1)])
    push!(loops, loop_righttop)
    push!(loops, loop_lefttop)
    push!(loops, loop_rightbottom)
    push!(loops, loop_leftbottom)
    return loops
end

# Gauge coupling
function calculate_gauge_coupling_plaq(
    U::Array{T,1},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end
function calculate_gauge_coupling_plaq(
    U::Array{T,1},
    B::Array{T,2},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, B, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end

function calculate_gauge_coupling_clover(
    U::Array{T,1},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end
function calculate_gauge_coupling_clover(
    U::Array{T,1},
    B::Array{T,2},
    temp_UμνTA,
    temps::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, B, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end

function calc_E(UμνTA, numofloops, U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT
    Vol = NX*NY*NZ*NT

    E = 0.0
    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            Uμν = UμνTA[μ, ν]
            s = tr(Uμν, Uμν)
            E += s / numofloops^2
        end
    end

    return - E / (2Vol)
end

# Energy density
function make_cloverloops(μ,ν;Dim=4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)])
    loop_lefttop = Wilsonline([(ν,1),(μ,-1),(ν,-1),(μ,1)])
    loop_rightbottom = Wilsonline([(ν,-1),(μ,1),(ν,1),(μ,-1)])
    loop_leftbottom= Wilsonline([(μ,-1),(ν,-1),(μ,1),(ν,1)])
    push!(loops,loop_righttop)
    push!(loops,loop_lefttop)
    push!(loops,loop_rightbottom)
    push!(loops,loop_leftbottom)
    return loops
end

function cloverloops(Dim)
    loops_μν= Matrix{Vector{Wilsonline{Dim}}}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            loops_μν[μ,ν] = Wilsonline{Dim}[]
            if ν == μ
                continue
            end
            loops_μν[μ,ν] = make_cloverloops(μ,ν,Dim=Dim)
        end
    end
    return  loops_μν
end

function make_energy_density!(
    Wmat,
    U::Vector{<: AbstractGaugefields{NC,Dim}},
    temps_g::Temporalfields
) where {NC,Dim}
    W_operator = cloverloops(Dim)
    temps, it_temps = get_temp(temps_g, 9)
    calc_wilson_loop!(Wmat,W_operator,U,temps)
    unused!(temps_g, it_temps)
    return 
end
function make_energy_density!(
    Wmat,
    U::Array{T,1},
    B::Array{T,2},
    temps_g::Temporalfields
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    W_operator = cloverloops(Dim)
    temps, it_temps = get_temp(temps_g, 9)
    calc_wilson_loop!(Wmat,W_operator,U,B,temps)
    unused!(temps_g, it_temps)
    return 
end

function calc_wilson_loop!(W,W_operator,U::Vector{<: AbstractGaugefields{NC,Dim}},temps) where {NC,Dim}
    for μ=1:Dim
        for ν=1:Dim
            if μ == ν
                continue
            end
            evaluate_gaugelinks!(W[μ,ν],W_operator[μ,ν],U,temps)
            W[μ,ν] = Traceless_antihermitian(W[μ,ν])
        end
    end
    return 
end
function calc_wilson_loop!(W,W_operator,U::Array{T,1},B::Array{T,2},temps
                           ) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    for μ=1:Dim
        for ν=1:Dim
            if μ == ν
                continue
            end
            evaluate_gaugelinks!(W[μ,ν],W_operator[μ,ν],U,B,temps)
            W[μ,ν] = Traceless_antihermitian(W[μ,ν])
        end
    end
    return 
end

function  make_energy_density_core(Wmat::Matrix{<: AbstractGaugefields{NC,Dim}}) where {NC,Dim}
    @assert Dim == 4
    W = 0.0 + 0.0im
    for μ=1:Dim # all directions
        for ν=1:Dim
            if μ == ν
                continue
            end
            W += -tr(Wmat[μ,ν],Wmat[μ,ν])/2
        end
    end
    return W
end

function calculate_energy_density(
    U::Array{T,1},
    Wmat,
    temps::Temporalfields
) where T <: AbstractGaugefields
    # Making a ( Ls × Lt) Wilson loop operator for potential calculations
    WL = 0.0+0.0im
    NV = U[1].NV
    NC = U[1].NC
    make_energy_density!(Wmat,U,temps) # make wilon loop operator and evaluate as a field, not traced.
    WL =  make_energy_density_core(Wmat) # tracing over color and average over spacetime and x,y,z.
    NDir = 4.0*3.0/2 # choice of 2 axis from 4.
    return real(WL)/(NV*4^2)
end
function calculate_energy_density(
    U::Array{T,1},
    B::Array{T,2},
    Wmat,
    temps::Temporalfields
) where T <: AbstractGaugefields
    WL = 0.0+0.0im
    NV = U[1].NV
    NC = U[1].NC
    make_energy_density!(Wmat,U,B,temps)
    WL =  make_energy_density_core(Wmat)
    NDir = 4.0*3.0/2
    return real(WL)/(NV*4^2)
end



function calc_Q_gradflow!(
    U_copy,
    U,
    temp_UμνTA,
    W_temp,
    temps;
    Δt = 0.1,
    tstep = 10,
    meas_step = 10,
    displayon = true,
    conditions = ["Qimproved"],
)
    Dim = 4

    flow_number = tstep

    dt = Δt

    #    flow_times = dt:dt:flow_number*dt

    substitute_U!(U_copy,U)

    numofobs=0
    if "Qplaq" in conditions
        topo_values_plaq = []
        Qplaq = 0.0
        numofobs+=1
    end
    if "Qclover" in conditions
        topo_values_clover = []
        Qclover = 0.0
        numofobs+=1
    end
    if "Qimproved" in conditions
        topo_values_improved = []
        Qclover = 0.0
        Qimproved = 0.0
        numofobs+=1
    end
    if "Eplaq" in conditions
        gauge_values_plaq = []
        Eplaq = 0.0
        numofobs+=1
    end
    if "Eclover" in conditions
        gauge_values_clover = []
        Eclover = 0.0
        numofobs+=1
    end
    if "Energydensity" in conditions
        energy_density_values = []
        E = 0.0
        numofobs+=1
    end

    if !(numofobs == length(conditions))
        println("Not matched condition name!")
        return
    elseif numofobs==0
        return
    end

    for μ=1:Dim
        for ν=1:Dim
            temp_UμνTA[μ,ν] = similar(U_copy[1])
        end
    end

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)
        end
    end

    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]

    g = Gradientflow_general(U_copy,listloops,listvalues,eps = dt)

    measurement_step = meas_step

    for iflow=1:flow_number
        flow!(U_copy, g)
        if iflow % measurement_step == 0
            println("Flowtime $(iflow*dt)")

            for i=1:numofobs
                if conditions[i]=="Qplaq"
                    Qplaq = calculate_topological_charge_plaq(U_copy,temp_UμνTA,temps)
                    push!(topo_values_plaq, Qplaq)
                    if displayon
                        println("Qplaq:         $Qplaq")
                    end
                elseif conditions[i]=="Qclover"
                    Qclover = calculate_topological_charge_clover(U_copy,temp_UμνTA,temps)
                    push!(topo_values_clover, Qclover)
                    if displayon
                        println("Qclover:       $Qclover")
                    end
                elseif conditions[i]=="Qimproved"
                    Qclover = calculate_topological_charge_clover(U_copy,temp_UμνTA,temps)
                    Qimproved= calculate_topological_charge_improved(U_copy,temp_UμνTA,Qclover,temps)
                    push!(topo_values_improved, Qimproved)
                    if displayon
                        println("Qimproved:     $Qimproved")
                    end
                elseif conditions[i]=="Eplaq"
                    Eplaq = calculate_gauge_coupling_plaq(U_copy,temp_UμνTA,temps)
                    push!(gauge_values_plaq, Eplaq)
                    if displayon
                        println("Eplaq:         $Eplaq")
                    end
                elseif conditions[i]=="Eclover"
                    Eclover = calculate_gauge_coupling_clover(U_copy,temp_UμνTA,temps)
                    push!(gauge_values_clover, Eclover)
                    if displayon
                        println("Eclover:       $Eclover")
                    end
                elseif conditions[i]=="Energydensity"
                    E = calculate_energy_density(U_copy,W_temp,temps)
                    push!(energy_density_values, E)
                    if displayon
                        println("Energydensity: $E")
                    end
                end
            end

        end
    end

    values = []
    for i=1:numofobs
        if conditions[i]=="Qplaq"
            push!(values, topo_values_plaq)
        elseif conditions[i]=="Qclover"
            push!(values, topo_values_clover)
        elseif conditions[i]=="Qimproved"
            push!(values, topo_values_improved)
        elseif conditions[i]=="Eplaq"
            push!(values, gauge_values_plaq)
        elseif conditions[i]=="Eclover"
            push!(values, gauge_values_clover)
        elseif conditions[i]=="Energydensity"
            push!(values, energy_density_values)
        end
    end

    return Tuple(values[i] for i=1:numofobs)
end
function calc_Q_gradflow!(
    U_copy,
    B_copy,
    U,
    B,
    temp_UμνTA,
    W_temp,
    temps;
    Δt = 0.1,
    tstep = 10,
    meas_step = 10,
    displayon = true,
    conditions = ["Qimproved"],
)
    Dim = 4
    NC = U[1].NC

    flow_number = tstep

    dt = Δt

    #    flow_times = dt:dt:flow_number*dt

    substitute_U!(U_copy,U)
    substitute_U!(B_copy,B)

    numofobs=0
    if "Qplaq" in conditions
        topo_values_plaq = []
        Qplaq = 0.0
        numofobs+=1
    end
    if "Qclover" in conditions
        topo_values_clover = []
        Qclover = 0.0
        numofobs+=1
    end
    if "Qimproved" in conditions
        topo_values_improved = []
        Qclover = 0.0
        Qimproved = 0.0
        numofobs+=1
    end
    if "Eplaq" in conditions
        gauge_values_plaq = []
        Eplaq = 0.0
        numofobs+=1
    end
    if "Eclover" in conditions
        gauge_values_clover = []
        Eclover = 0.0
        numofobs+=1
    end
    if "Energydensity" in conditions
        energy_density_values = []
        E = 0.0
        numofobs+=1
    end

    if !(numofobs == length(conditions))
        println("Not matched condition name!")
        return
    elseif numofobs==0
        return
    end

    for μ=1:Dim
        for ν=1:Dim
            temp_UμνTA[μ,ν] = similar(U_copy[1])
        end
    end

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)
        end
    end

    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]

    g = Gradientflow_general(U_copy,B_copy,listloops,listvalues,eps = dt)

    measurement_step = meas_step

    for iflow=1:flow_number
        flow!(U_copy, B_copy, g)
        if iflow % measurement_step == 0
            println("Flowtime $(iflow*dt)")

            for i=1:numofobs
                if conditions[i]=="Qplaq"
                    Qplaq = calculate_topological_charge_plaq(U_copy,B_copy,temp_UμνTA,temps)
                    push!(topo_values_plaq, Qplaq)
                    if displayon
                        println("Qplaq:         $Qplaq")
                    end
                elseif conditions[i]=="Qclover"
                    Qclover = calculate_topological_charge_clover(U_copy,B_copy,temp_UμνTA,temps)
                    push!(topo_values_clover, Qclover)
                    if displayon
                        println("Qclover:       $Qclover")
                    end
                elseif conditions[i]=="Qimproved"
                    Qclover = calculate_topological_charge_clover(U_copy,B_copy,temp_UμνTA,temps)
                    Qimproved= calculate_topological_charge_improved(U_copy,B_copy,temp_UμνTA,Qclover,temps)
                    push!(topo_values_improved, Qimproved)
                    if displayon
                        println("Qimproved:     $Qimproved")
                    end
                elseif conditions[i]=="Eplaq"
                    Eplaq = calculate_gauge_coupling_plaq(U_copy,B_copy,temp_UμνTA,temps)
                    push!(gauge_values_plaq, Eplaq)
                    if displayon
                        println("Eplaq:         $Eplaq")
                    end
                elseif conditions[i]=="Eclover"
                    Eclover = calculate_gauge_coupling_clover(U_copy,B_copy,temp_UμνTA,temps)
                    push!(gauge_values_clover, Eclover)
                    if displayon
                        println("Eclover:       $Eclover")
                    end
                elseif conditions[i]=="Energydensity"
                    E = calculate_energy_density(U_copy,B_copy,W_temp,temps)
                    push!(energy_density_values, E)
                    if displayon
                        println("Energydensity: $E")
                    end
                end
            end

        end
    end

    values = []
    for i=1:numofobs
        if conditions[i]=="Qplaq"
            push!(values, topo_values_plaq)
        elseif conditions[i]=="Qclover"
            push!(values, topo_values_clover)
        elseif conditions[i]=="Qimproved"
            push!(values, topo_values_improved)
        elseif conditions[i]=="Eplaq"
            push!(values, gauge_values_plaq)
        elseif conditions[i]=="Eclover"
            push!(values, gauge_values_clover)
        elseif conditions[i]=="Energydensity"
            push!(values, energy_density_values)
        end
    end

    return Tuple(values[i] for i=1:numofobs)
end




end
