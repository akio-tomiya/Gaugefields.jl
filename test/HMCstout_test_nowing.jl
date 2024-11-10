using Gaugefields
using LinearAlgebra

function MDtest!(gauge_action, U, Dim, nn)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)

    substitute_U!(Uold, U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0


    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temp1, temp2)
        numaccepted += ifelse(accepted, 1, 0)

        plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ", numaccepted / itrj)
    end
end

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temp1, temp2)


    Δτ = 1 / MDsteps
    gauss_distribution!(p)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Sold = calc_action(gauge_action, Uout, p)

    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, temp1, temp2)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Snew = calc_action(gauge_action, Uout, p)

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    if accept != true #rand() > ratio
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

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, dSdU, nn, temp1, temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    temp = temp1
    Uout, Uout_multi, _ = calc_smearedU(U, nn)

    for μ = 1:Dim
        calc_dSdUμ!(dSdU[μ], gauge_action, μ, Uout)
    end

    dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

    for μ = 1:Dim
        mul!(temp, U[μ], dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp)
    end
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    Dim = 4
    NC = 3

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 5.7 / 2
    push!(gauge_action, β, plaqloop)

    show(gauge_action)

    L = [NX, NY, NZ, NT]
    nn = CovNeuralnet()
    ρ = [0.1] #* 1e-10
    layername = ["plaquette"]

    st = STOUT_Layer(layername, ρ, U)

    #return
    #st = STOUT_Layer(layername, ρ, L)
    push!(nn, st)

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("plaq_t $plaq_t")

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    plaq_t = calculate_Plaquette(Uout, temp1, temp2) * factor
    println("plaq_t s $plaq_t")

    #return

    MDtest!(gauge_action, U, Dim, nn)

end


test1()
