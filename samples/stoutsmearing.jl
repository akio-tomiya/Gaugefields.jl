using Gaugefields
using Wilsonloop

function stoutsmearing(NX,NY,NZ,NT,NC)
    Nwing = 1
    Dim = 4
    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    L = [NX,NY,NZ,NT]

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    plaq_t = calculate_Plaquette(U,temps)*factor
    println(" plaq_t = $plaq_t")


    nn = CovNeuralnet()
    ρ = [0.1]
    layername = ["plaquette"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)

    show(nn)

    
    @time Uout,Uout_multi,_ = calc_smearedU(U,nn)
    plaq_t = calculate_Plaquette(Uout,temps)*factor
    println("plaq_t = $plaq_t")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)

    μ = 1
    dSdUμ = similar(U)
    for μ=1:Dim
        dSdUμ[μ] = calc_dSdUμ(gauge_action,μ,U)
    end

    @time dSdUbareμ = back_prop(dSdUμ,nn,Uout_multi,U) 

end

NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
stoutsmearing(NX,NY,NZ,NT,NC)
