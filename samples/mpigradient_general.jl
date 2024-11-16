using Gaugefields
using MPI
using LinearAlgebra
using Random

const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])




function MDtest!(snet,U,Dim,mpi=false)
    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 200

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    numaccepted = 0

    plaq_t = calculate_Plaquette(U,temps)*factor

    #poly = calculate_Polyakov_loop(U,temps)
    if get_myrank(U) == 0
        println("0 plaq_t = $plaq_t")
        #println("polyakov loop = $(real(poly)) $(imag(poly))")
    end


    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(snet,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #poly = calculate_Polyakov_loop(U,temp1,temp2) 
        
        if get_myrank(U) == 0
            println("$itrj plaq_t = $plaq_t")
            println("acceptance ratio ",numaccepted/itrj)
            #println("polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
end

function calc_action(snet,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(snet,U)/NC
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
        #println(getvalue(U[1],1,1,1,1,1,1))

        P_update!(U,p,1.0,Δτ,Dim,snet)
        #if get_myrank(U) == 0
        #    println(getvalue(U[1],1,1,1,1,1,1))
        #    println("p = ",p[1][1,1,1,1,1])
        #    if isnan(p[1][1,1,1,1,1])
        #        error("p")
        #    end
        #end


        U_update!(U,p,0.5,Δτ,Dim,snet)
        #error("dd")
    end
    #error("end")
    
    Snew = calc_action(snet,U,p)
    if get_myrank(U) == 0
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(Snew-Sold))
    r = rand()
    if mpi
        r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    end
    #println(r,"\t",ratio)

    if r > ratio
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
        #println("dSdU = ",getvalue(dSdUμ,1,1,1,1,1,1))
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
    Nwing = 0
    Dim = 4
    NC = 3


    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",mpi=true,PEs = PEs,mpiinit = false)
        
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    end

    Random.seed!(123+get_myrank(U[1]))    



    snet = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(snet,β,plaqloop)
    
    #show(snet)


    MDtest!(snet,U,Dim,mpi)

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    listnames = [plaqloop]
    listvalues = [1]
    g = Gradientflow_general(U,listnames,listvalues,eps = 0.01)


    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println_verbose_level1(U[1],"$itrj plaq_t = $plaq_t")
            #poly = calculate_Polyakov_loop(U,temps) 
            #println_verbose_level1(U[1],"$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    
    #show(snet)


    #MDtest!(snet,U,Dim,mpi)

end


test1()
