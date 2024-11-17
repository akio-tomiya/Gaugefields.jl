using Gaugefields
using MPI
using LinearAlgebra

function MDtest!(snet,U,Dim,mpi=false)
    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U,Dim)

    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        @time accepted = MDstep!(snet,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temps)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(snet,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(snet,U)/NC 
    #calc_scalar(snet,U)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(snet,U,p,MDsteps,Dim,Uold)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(snet,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,snet)

        P_update!(U,p,1.0,Δτ,Dim,snet)

        U_update!(U,p,0.5,Δτ,Dim,snet)

    end
    Snew = calc_action(snet,U,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(Snew-Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,snet)
    temps = get_temp(snet._temp_U)
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    expU, it_expU = get_temp(temps)
    W, it_W = get_temp(temps)

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end

function P_update!(U,p,ϵ,Δτ,Dim,snet) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temp(snet._temp_U)
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,snet,μ,U)
        mul!(temp1,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
end



function test1()
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 1
    Dim = 4
    NC = 3

    mpi = true
    if mpi
        PEs = (1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",mpi=true,PEs = PEs,mpiinit = false)
    else

        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    end


    snet = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(snet,β,plaqloop)
    
    #show(snet)


    MDtest!(snet,U,Dim,mpi)

end


test1()
