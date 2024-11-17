module HMC_module

using Random
using LinearAlgebra

import ..AbstractGaugefields_module:
    Initialize_Bfields,
    gauss_distribution!,
    substitute_U!,
    exptU!,
    mul!,
    Traceless_antihermitian_add!
import ..GaugeAction_module: evaluate_GaugeAction, calc_dSdUμ!
import ..Abstractsmearing_module: calc_smearedU, back_prop
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
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U, B)
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end


function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, dSdU, nn, temps)
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    temp1, it_temp1 = get_temp(temps)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)

    for μ = 1:Dim
        calc_dSdUμ!(dSdU[μ], gauge_action, μ, Uout)
    end

    dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_temp1)
end

function Flux_update!(B,flux)

    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

    i = rand(1:6)
#    flux[i] += rand(-1:1)
#    flux[i] %= NC
#    flux[i] += (flux[i] < 0) ? NC : 0
    flux[:] = rand(0:NC-1,6)
    B = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

end

function set_comb(U, Dim)
    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end
    
    factor = 1 / (comb * U[1].NV * U[1].NC)

    return comb, factor
end

end
