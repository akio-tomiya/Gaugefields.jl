#module HMC_MDstep_module

using Random
using LinearAlgebra

import ..AbstractGaugefields_module:
    gauss_distribution!,
    substitute_U!,
    exptU!,
    mul!,
    Traceless_antihermitian_add!
import ..GaugeAction_module: evaluate_GaugeAction, calc_dSdUμ!
import ..Temporalfields_module: Temporalfields, get_temp, unused!

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end
function calc_action(gauge_action, U, B, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U, B) / NC
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end

function MDstep_core!(gauge_action, U, p, MDsteps, Dim, Uold, temps; displayon=true)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end
function MDstep_core!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps; displayon=true)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, B, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, B, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, B, p)
    if displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, gauge_action, temps)
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)#[1]
    temp2, it_temp2 = get_temp(temps)#temps[2]
    expU, it_expU = get_temp(temps)#[3]
    W, it_W = get_temp(temps)#[4]

    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end


function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, temps) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)#[end]
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end
function P_update!(U, B, p, ϵ, Δτ, Dim, gauge_action, temps)
    NC = U[1].NC
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)#[end]
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U, B)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end


#end
