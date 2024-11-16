#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

########################################################################

# mutable struct Topological_charge_measurement{Dim,TG} <: AbstractMeasurement
#     filename::String
#     _temporary_gaugefields::Vector{TG}
#     temp_UμνTA::Matrix{TG}
#     Dim::Int8
#     #factor::Float64
#     verbose_print::Union{Verbose_print,Nothing}
#     printvalues::Bool
#     TC_methods::Vector{String}

#     function Topological_charge_measurement(
#         U::Vector{T};
#         filename = nothing,
#         verbose_level = 2,
#         printvalues = true,
#         TC_methods = ["plaquette"],
#     ) where {T}
#         myrank = get_myrank(U)

#         if printvalues
#             verbose_print = Verbose_print(verbose_level, myid = myrank, filename = filename)
#         else
#             verbose_print = nothing
#         end
#         Dim = length(U)

#         temp_UμνTA = Array{T,2}(undef, Dim, Dim)

#         for μ = 1:Dim
#             for ν = 1:Dim
#                 temp_UμνTA[ν, μ] = similar(U[1])
#             end
#         end


#         numg = 3
#         _temporary_gaugefields = Vector{T}(undef, numg)
#         _temporary_gaugefields[1] = similar(U[1])
#         for i = 2:numg
#             _temporary_gaugefields[i] = similar(U[1])
#         end



#         return new{Dim,T}(
#             filename,
#             _temporary_gaugefields,
#             temp_UμνTA,
#             Dim,
#             verbose_print,
#             printvalues,
#             TC_methods,
#         )

#     end



# end

# function Topological_charge_measurement(
#     U::Vector{T},
#     params::TopologicalCharge_parameters,
#     filename,
# ) where {T}

#     return Topological_charge_measurement(
#         U,
#         filename = filename,
#         verbose_level = params.verbose_level,
#         printvalues = params.printvalues,
#         TC_methods = params.kinds_of_topological_charge, #["plaquette"]
#     )

# end

# function measure(
#     m::M,
#     itrj,
#     U::Array{<:AbstractGaugefields{NC,Dim},1};
#     additional_string = "",
# ) where {M<:Topological_charge_measurement,NC,Dim}
#     temps = get_temporary_gaugefields(m)
#     temp1 = temps[1]
#     temp2 = temps[2]
#     measurestring = ""

#     nummethod = length(m.TC_methods)
#     values = Float64[]
#     printstring = "$itrj " * additional_string
#     for i = 1:nummethod
#         methodname = m.TC_methods[i]
#         if methodname == "plaquette"
#             Qplaq = calculate_topological_charge_plaq(U, m.temp_UμνTA, temps)
#             push!(values, real(Qplaq))
#         elseif methodname == "clover"
#             Qclover = calculate_topological_charge_clover(U, m.temp_UμνTA, temps)
#             push!(values, real(Qclover))
#             Qimproved =
#                 calculate_topological_charge_improved(U, m.temp_UμνTA, Qclover, temps)
#             push!(values, real(Qimproved))
#         else
#             error("method $methodname is not supported in topological charge measurement")
#         end
#         #printstring *= "$(values[i]) "
#     end
#     for value in values
#         printstring *= "$(value) "
#     end
#     printstring *= "# itrj "

#     for i = 1:nummethod
#         methodname = m.TC_methods[i]
#         if methodname == "plaquette"
#             printstring *= "Qplaq "
#         elseif methodname == "clover"
#             printstring *= "Qclover Qimproved "
#         else
#             error("method $methodname is not supported in topological charge measurement")
#         end
#     end

#     if m.printvalues
#         #println_verbose_level2(U[1],"-----------------")
#         measurestring = printstring
#         println_verbose_level2(m.verbose_print, printstring)
#         #println_verbose_level2(U[1],"-----------------")
#     end

#     return values, measurestring
# end

function calculate_topological_charge_plaq(U::Array{T,1}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end
function calculate_topological_charge_plaq(U::Array{T,1}, B::Array{T,2}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, B, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end

function calculate_topological_charge_clover(U::Array{T,1}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end
function calculate_topological_charge_clover(U::Array{T,1}, B::Array{T,2}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, B, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end

function calculate_topological_charge_improved(
    U::Array{T,1},
    temp_UμνTA,
    Qclover,
    temps,
) where {T}
    UμνTA = temp_UμνTA
    #numofloops = calc_UμνTA!(UμνTA,"clover",U)
    #Qclover = calc_Q(UμνTA,numofloops,U)

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
    temps,
) where {T}
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
    temps,
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
    temps,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    loops_μν, numofloops = calc_loopset_μν_name(name, Dim)
    calc_UμνTA!(temp_UμνTA, loops_μν, U, B, temps)
    return numofloops
end


function calc_UμνTA!(
    temp_UμνTA,
    loops_μν,
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps,
) where {NC,Dim}
    UμνTA = temp_UμνTA
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end

            evaluate_gaugelinks!(temps[1], loops_μν[μ, ν], U, temps[2:3])
            Traceless_antihermitian!(UμνTA[μ, ν], temps[1])
            #loopset = Loops(U,loops_μν[μ,ν])
            #UμνTA[μ,ν] = evaluate_loops(loopset,U)

            #UμνTA[μ,ν] = Traceless_antihermitian(UμνTA[μ,ν])
        end
    end
    return
end
function calc_UμνTA!(
    temp_UμνTA,
    loops_μν,
    U::Array{T,1},
    B::Array{T,2},
    temps,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end
            evaluate_gaugelinks!(temps[1], loops_μν[μ, ν], U, B, temps[2:6])
            Traceless_antihermitian!(UμνTA[μ, ν], temps[1])
        end
    end
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
function calculate_gauge_coupling_plaq(U::Array{T,1}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end
function calculate_gauge_coupling_plaq(U::Array{T,1}, B::Array{T,2}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, B, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end

function calculate_gauge_coupling_clover(U::Array{T,1}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, temps)
    E = calc_E(UμνTA, numofloops, U)
    return E
end
function calculate_gauge_coupling_clover(U::Array{T,1}, B::Array{T,2}, temp_UμνTA, temps) where {T}
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

    #UU = similar(UμνTA[1,1])

    E = 0.0
    for μ = 1:Dim
        for ν = 1:Dim
            if μ == ν
                continue
            end
            Uμν = UμνTA[μ, ν]
            #mul!(UU, Uμν, Uμν)
            #s = tr(UU)
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

function make_energy_density!(Wmat,U::Vector{<: AbstractGaugefields{NC,Dim}},temps) where {NC,Dim}
    W_operator = cloverloops(Dim)
    calc_wilson_loop!(Wmat,W_operator,U,temps)
    return 
end
function make_energy_density!(Wmat,U::Array{T,1},B::Array{T,2},temps
                              ) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    W_operator = cloverloops(Dim)
    calc_wilson_loop!(Wmat,W_operator,U,B,temps)
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

function calculate_energy_density(U::Array{T,1}, Wmat,temps) where T <: AbstractGaugefields
    # Making a ( Ls × Lt) Wilson loop operator for potential calculations
    WL = 0.0+0.0im
    NV = U[1].NV
    NC = U[1].NC
    make_energy_density!(Wmat,U,temps) # make wilon loop operator and evaluate as a field, not traced.
    WL =  make_energy_density_core(Wmat) # tracing over color and average over spacetime and x,y,z.
    NDir = 4.0*3.0/2 # choice of 2 axis from 4.
    return real(WL)/(NV*4^2)
end
function calculate_energy_density(U::Array{T,1},B::Array{T,2}, Wmat,temps) where T <: AbstractGaugefields
    # Making a ( Ls × Lt) Wilson loop operator for potential calculations
    WL = 0.0+0.0im
    NV = U[1].NV
    NC = U[1].NC
    make_energy_density!(Wmat,U,B,temps) # make wilon loop operator and evaluate as a field, not traced.
    WL =  make_energy_density_core(Wmat) # tracing over color and average over spacetime and x,y,z.
    NDir = 4.0*3.0/2 # choice of 2 axis from 4.
    return real(WL)/(NV*4^2)
end

#################################################################################################


function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end
function calc_action(gauge_action,U,B,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U,B)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,p)
#    println("Sold = $Sold, Snew = $Snew")
#    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold)) # a bug is fixed!
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end
function MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,B,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,B,p)
#    println("Sold = $Sold, Snew = $Snew")
#    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
    end
end
function P_update!(U,B,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U,B)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
    end
end

function HMC_test_4D(NX,NY,NZ,NT,NC,β)

    Dim = 4
    Nwing = 0

    Random.seed!(123)


    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    #"Reproducible"
#    println(typeof(U))

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    ## for gradient flow
    temp3 = similar(U[1])
    temp_UμνTA= Matrix{typeof(U[1])}(undef,Dim,Dim)

    U_copy = similar(U)

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
#    poly = calculate_Polyakov_loop(U,temp1,temp2) 
#    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 400

    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temp1,temp2)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
#            poly = calculate_Polyakov_loop(U,temp1,temp2) 
#            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

        if itrj % 20 == 0
            calc_Q_gradflow!(U_copy,U,temp_UμνTA,[temp1,temp2,temp3])
        end

    end
    return plaq_t,numaccepted/numtrj

end


function HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)

    Dim = 4
    Nwing = 0

    flux = Flux

    println("Flux : ", flux)

    #Random.seed!(123)
    t0 = Dates.DateTime(2024,1,1,16,10,7)
    t  = Dates.now()
    Random.seed!(Dates.value(t-t0))

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
    #B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tloop",tloop_dis=2)

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    ## for gradien flow
    temp3 = similar(U[1])
    temp4 = similar(U[1])
    temp5 = similar(U[1])
    temp6 = similar(U[1])
    temp_UμνTA= Matrix{typeof(U[1])}(undef,Dim,Dim)

    U_copy = similar(U)
    B_copy = similar(B)

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    # for calc energy density
    W_temp = Matrix{typeof(U[1])}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            W_temp[μ,ν] = similar(U[1])
        end
    end

    @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
#    poly = calculate_Polyakov_loop(U,temp1,temp2) 
#    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 50
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 400

    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temp1,temp2)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
#            poly = calculate_Polyakov_loop(U,temp1,temp2) 
#            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

        if itrj % 50 == 0
            #res1,res2,res3=
            calc_Q_gradflow!(U_copy,B_copy,U,B,temp_UμνTA,W_temp,
                             [temp1,temp2,temp3,temp4,temp5,temp6],
                             conditions=["Qclover","Qimproved","Eclover","Energydensity"])
        end

    end
    return plaq_t,numaccepted/numtrj

end


function calc_Q_gradflow!(
    U_copy,
    U,
    temp_UμνTA,
    W_temp,
    temps;
    Δt = 0.1,
    tstep = 100,
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

    for iflow=1:flow_number
        flow!(U_copy, g)
        if iflow % 10 == 0
            println("Flowtime $(iflow*dt)")

            for i=1:numofobs
                if conditions[i]=="Qplaq"
                    Qplaq = calculate_topological_charge_plaq(U_copy,temp_UμνTA,temps)
                    push!(topo_values_plaq, Qplaq)
                    println("Qplaq:         $Qplaq")
                elseif conditions[i]=="Qclover"
                    Qclover = calculate_topological_charge_clover(U_copy,temp_UμνTA,temps)
                    push!(topo_values_clover, Qclover)
                    println("Qclover:       $Qclover")
                elseif conditions[i]=="Qimproved"
                    Qclover = calculate_topological_charge_clover(U_copy,temp_UμνTA,temps)
                    Qimproved= calculate_topological_charge_improved(U_copy,temp_UμνTA,Qclover,temps)
                    push!(topo_values_improved, Qimproved)
                    println("Qimproved:     $Qimproved")
                elseif conditions[i]=="Eplaq"
                    Eplaq = calculate_gauge_coupling_plaq(U_copy,temp_UμνTA,temps)
                    push!(gauge_values_plaq, Eplaq)
                    println("Eplaq:         $Eplaq")
                elseif conditions[i]=="Eclover"
                    Eclover = calculate_gauge_coupling_clover(U_copy,temp_UμνTA,temps)
                    push!(gauge_values_clover, Eclover)
                    println("Eclover:       $Eclover")
                elseif conditions[i]=="Energydensity"
                    E = calculate_energy_density(U_copy,W_temp,temps)
                    push!(energy_density_values, E)
                    println("Energydensity: $E")
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

    for iflow=1:flow_number
        flow!(U_copy, B_copy, g)
        if iflow % 10 == 0
            println("Flow time t $(iflow*dt)")

            for i=1:numofobs
                if conditions[i]=="Qplaq"
                    Qplaq = calculate_topological_charge_plaq(U_copy,B_copy,temp_UμνTA,temps)
                    push!(topo_values_plaq, Qplaq)
                    println("Qplaq:         $Qplaq")
                elseif conditions[i]=="Qclover"
                    Qclover = calculate_topological_charge_clover(U_copy,B_copy,temp_UμνTA,temps)
                    push!(topo_values_clover, Qclover)
                    println("Qclover:       $Qclover")
                elseif conditions[i]=="Qimproved"
                    Qclover = calculate_topological_charge_clover(U_copy,B_copy,temp_UμνTA,temps)
                    Qimproved= calculate_topological_charge_improved(U_copy,B_copy,temp_UμνTA,Qclover,temps)
                    push!(topo_values_improved, Qimproved)
                    println("Qimproved:     $Qimproved")
                elseif conditions[i]=="Eplaq"
                    Eplaq = calculate_gauge_coupling_plaq(U_copy,B_copy,temp_UμνTA,temps)
                    push!(gauge_values_plaq, Eplaq)
                    println("Eplaq:         $Eplaq")
                elseif conditions[i]=="Eclover"
                    Eclover = calculate_gauge_coupling_clover(U_copy,B_copy,temp_UμνTA,temps)
                    push!(gauge_values_clover, Eclover)
                    println("Eclover:       $Eclover")
                elseif conditions[i]=="Energydensity"
                    E = calculate_energy_density(U_copy,B_copy,W_temp,temps)
                    push!(energy_density_values, E)
                    println("Energydensity: $E")
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


function main()
    β = 2.45
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 2
    Flux = [0,0,0,1,0,0]
    #HMC_test_4D(NX,NY,NZ,NT,NC,β)
    HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
end
main()
