
using Random
using Gaugefields
using LinearAlgebra

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temp1, temp2)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temp1, temp2)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end
    Snew = calc_action(gauge_action, U, p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))
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

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, temp1, temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp = temp1
    dSdUμ = temp2
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temp, U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp)
    end
end


function HMC_test_4D(NX, NY, NZ, NT, NC, β)
    Dim = 4
    Nwing = 1
    # Nwing = 0

    Random.seed!(123)
    #isMPILattice = false
    isMPILattice = true

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice)
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
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temp1, temp2)
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
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    HMC_test_4D(NX, NY, NZ, NT, NC, β)
end
main()