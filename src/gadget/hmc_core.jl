module HMC_core_module

using Requires

import ..AbstractGaugefields_module: Initialize_Bfields
import ..GaugeAction_module
import ..Temporalfields_module

include("./hmc_mdstep.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        include("./hmc_mdstep_mpi.jl")
    end
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                 displayon=false, mpi=false)
    if mpi
        MDstep_core_mpi!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, p, MDsteps, Dim, Uold, temps;
                     displayon=displayon)
    end
end
function MDstep!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                 displayon=true, mpi=false)
    if mpi
        MDstep_core_mpi!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                         displayon=displayon)
    else
        MDstep_core!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps;
                     displayon=displayon)
    end
end
function MDstep_dynB!(
    gauge_action,
    U,
    B,
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold,
    Bold,
    flux_old,
    temps
) # Halfway-updating HMC
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temps)

        U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,flux)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end

function MDstep_dynB!(
    gauge_action,
    U,
    B,
    flux,
    p,
    MDsteps,
    num_HMC,
    Dim,
    Uold1,
    Uold2,
    Bold,
    flux_old,
    temps
) # Double-tesing HMC
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for ihmc=1:num_HMC
        MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temps)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    #println("Sold = $Sold, Snew = $Snew")
    #println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold1)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
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
