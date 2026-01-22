
using Random
using Gaugefields
using LinearAlgebra
using PreallocatedArrays
using Enzyme
import JACC
JACC.@init_backend

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end

function _calc_action_step!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(E, D, Uν')
    S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(E, D, Uμ')
    S += realtrace(E)

    return S
end


function calc_action(U1, U2, U3, U4, β, NC, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            S += _calc_action_step!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)
        end
    end

    return -S * β / NC
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, tempvec, β)
    NC = U[1].NC
    Δτ = 1.0 / MDsteps
    temp1, it_temp1 = get_block(tempvec)
    temp2, it_temp2 = get_block(tempvec)
    temp, its_temp = get_block(tempvec, 3)
    dtemp, its_dtemp = get_block(tempvec, 3)


    gauss_distribution!(p)
    #Sold = calc_action(gauge_action, U, p)
    Sold = calc_action(U..., β, NC, temp) + p * p / 2
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temp1, temp, dtemp, tempvec, β)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end
    Snew = calc_action(gauge_action, U, p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))

    unused!(tempvec, it_temp1)
    unused!(tempvec, it_temp2)
    unused!(tempvec, its_temp)
    unused!(tempvec, its_dtemp)


    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)

    end
end

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, temp1, temp, dtemp, temps, β) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    factor_ad = ϵ * Δτ / 2

    dSdU, it_dSdU = get_block(temps, 4)#temps[end]
    Gaugefields.clear_U!(dSdU)

    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]
    Enzyme_derivative!(
        calc_action,
        U1, U2, U3, U4,
        dSdU[1], dSdU[2], dSdU[3], dSdU[4], nodiff(β), nodiff(NC);
        temp,
        dtemp
    )

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdU[μ]')
        Traceless_antihermitian_add!(p[μ], factor_ad, temp1)
        #calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        #mul!(temp, U[μ], dSdUμ) # U*dSdUμ
        #Traceless_antihermitian_add!(p[μ], factor, temp)
    end

    unused!(temps, it_dSdU)
end


function HMC_test_4D(NX, NY, NZ, NT, NC, β)
    Dim = 4
    Nwing = 1

    Random.seed!(123)


    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice=true)
    #"Reproducible"
    println(typeof(U))

    temp1 = similar(U[1])
    temp2 = similar(U[1])

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
    tempvec = PreallocatedArray(U[1]; num=30, haslabel=false)


    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β / 2
    push!(gauge_action, β, plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 200
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, tempvec, β)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted, 1, 0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end
    return plaq_t, numaccepted / numtrj

end


function main()
    β = 6
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    HMC_test_4D(NX, NY, NZ, NT, NC, β)
end
main()