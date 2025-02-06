using Test
using CUDA
using Gaugefields


using LinearAlgebra
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!



function MDtest!(gauge_action, U, Dim, nn,gauge_actioncpu, Ucpu, nncpu)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    pcpu = initialize_TA_Gaugefields(Ucpu) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 

    Uold = similar(U)
    dSdU = similar(U)

    dSdUcpu = similar(Ucpu)

    substitute_U!(Uold, U)
    MDsteps = 100
    temps = Temporalfields(U[1]; num=10)


    tempscpu = Temporalfields(Ucpu[1]; num=10)
    temp1cpu,it_temp1cpu = get_temp(tempscpu)#[1]#similar(U[1])
    temp2cpu,it_temp2cpu = get_temp(tempscpu) #similar(U[1])
    #temp1cpu = tempscpu[1]#similar(U[1])
    #temp2cpu = temps[2] #similar(U[1])


    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0


    numtrj = 10
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps,
                gauge_actioncpu, Ucpu,pcpu,tempscpu,nncpu , dSdUcpu)
        numaccepted += ifelse(accepted, 1, 0)

        temp1,it_temp1 = get_temp(temps)#[1]#similar(U[1])
        temp2,it_temp2 = get_temp(temps) #similar(U[1])
        plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        unused!(temps,it_temp1)
        unused!(temps,it_temp2)

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


function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps,
    gauge_actioncpu, Ucpu,pcpu,tempscpu,nncpu , dSdUcpu)


    Δτ = 1 / MDsteps
    gauss_distribution!(p)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    Sold = calc_action(gauge_action, Uout, p)

    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps,gauge_actioncpu, Ucpu,pcpu,tempscpu )

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, temps,gauge_actioncpu, Ucpu,pcpu,tempscpu,nncpu, dSdUcpu )

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps,gauge_actioncpu, Ucpu,pcpu,tempscpu )
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

function U_update!(U, p, ϵ, Δτ, Dim, gauge_action, temps,gauge_actioncpu, Ucpu,pcpu,tempscpu )
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    expU, it_expU = get_temp(temps)
    W, it_W = get_temp(temps)


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

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, dSdU, nn, temps,
    gauge_actioncpu, Ucpu,pcpu,tempscpu,nncpu, dSdUcpu ) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    #temps = get_temporary_gaugefields(gauge_action)
    temp1, it_temp1 = get_temp(temps)

    #temp1cpu, it_temp1cpu = get_temp(tempscpu)

    #substitute_U!(Ucpu,U)

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    #Uoutcpu, Uout_multicpu, _ = calc_smearedU(Ucpu, nncpu)

    for μ = 1:Dim
        calc_dSdUμ!(dSdU[μ], gauge_action, μ, Uout)
        #println("cuda")
        #display(Array(dSdU[μ].U)[:,:,1,1])

        #calc_dSdUμ!(dSdUcpu[μ], gauge_actioncpu, μ, Uoutcpu)
        #println("cpu")
        #display(Array(dSdUcpu[μ].U)[:,:,1,1,1,1])
    end
    #error("dd")

    dSdUbare, its_dSdUbare = get_temp(gauge_action._temp_U, Dim)
    back_prop!(dSdUbare, dSdU, nn, Uout_multi, U)

    #back_prop!(dSdU, nn, Uout_multi, U)
    #dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temp1)
    end
    unused!(temps, it_temp1)
    unused!(gauge_action._temp_U, its_dSdUbare)
end

function test1()
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0
    Dim = 4
    NC = 3
    blocks = [4, 4, 4, 4]

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold", cuda=true,
    blocks = blocks)
    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")


    Ucpu = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot")
    substitute_U!(Ucpu,U)



    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 5.7 / 2
    push!(gauge_action, β, plaqloop)
    show(gauge_action)

    gauge_actioncpu = GaugeAction(Ucpu)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = 5.7 / 2
    push!(gauge_actioncpu, β, plaqloop)

    L = [NX, NY, NZ, NT]
    nn = CovNeuralnet(U)
    ρ = [0.01] #* 1e-10
    layername = ["plaquette"]

    st = STOUT_Layer(layername, ρ, U)


    #return
    #st = STOUT_Layer(layername, ρ, L)
    push!(nn, st)


    nncpu = CovNeuralnet(Ucpu)
    #ρ = [0.1] #* 1e-10
    layername = ["plaquette"]
    stcpu = STOUT_Layer(layername, ρ, Ucpu)
    push!(nncpu, stcpu)

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("plaq_t $plaq_t")

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    plaq_t = calculate_Plaquette(Uout, temp1, temp2) * factor
    println("plaq_t s $plaq_t")


    temp1cpu = similar(Ucpu[1])
    temp2cpu = similar(Ucpu[1])
    substitute_U!(Ucpu,U)
    plaq_t = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
    println("plaq_t cpu $plaq_t")

    Uoutcpu, Uout_multicpu, _ = calc_smearedU(Ucpu, nncpu)
    plaq_t = calculate_Plaquette(Uoutcpu, temp1cpu, temp2cpu) * factor
    println("plaq_t s cpu $plaq_t")


    #return

    MDtest!(gauge_action, U, Dim, nn,gauge_actioncpu, Ucpu, nncpu)

end


test1()