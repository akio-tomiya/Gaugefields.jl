using Gaugefields
using LinearAlgebra
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!



function MDtest!(gauge_action, U, Dim, nn)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)

    substitute_U!(Uold, U)
    MDsteps = 100

    temps = Temporalfields(U[1], num=10)
    comb, factor = set_comb(U,Dim)

    numaccepted = 0


    numtrj = 10
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, temps)
        numaccepted += ifelse(accepted, 1, 0)

        plaq_t = calculate_Plaquette(U, temps) * factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ", numaccepted / itrj)
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

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    plaq_t = calculate_Plaquette(U, temps) * factor
    println("plaq_t $plaq_t")

    Uout, Uout_multi, _ = calc_smearedU(U, nn)
    plaq_t = calculate_Plaquette(Uout, temps) * factor
    println("plaq_t s $plaq_t")

    #return

    MDtest!(gauge_action, U, Dim, nn)

end


test1()
