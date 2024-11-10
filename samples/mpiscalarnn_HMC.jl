using Gaugefields
using MPI
using LinearAlgebra

npe_arg = [1 1 1 1]
npe_arg[1] = parse(Int,　ARGS[1]) # npe_x
npe_arg[2] = parse(Int,　ARGS[2]) # npe_y
npe_arg[3] = parse(Int,　ARGS[3]) # npe_z
npe_arg[4] = parse(Int,　ARGS[4]) # npe_t

function MDtest!(snet,U,Dim,mpi=false)
    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        @time accepted = MDstep!(snet,U,p,MDsteps,Dim,Uold,temp1,temp2)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(snet,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(snet,U)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(snet,U,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(snet,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,snet)

        P_update!(U,p,1.0,Δτ,Dim,snet,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,snet)

    end
    Snew = calc_action(snet,U,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,snet)
    temps = get_temporary_gaugefields(snet)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,snet,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,snet,μ,U)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
    end
end



function test1()
    NX = 24
    NY = 24
    NZ = 24
    NT = 48
    Nwing = 1
    Dim = 4
    NC = 3

    mpi = true
    #mpi = false
    if mpi
        println("mpi=$mpi npe=$(npe_arg)")
    else
	println("mpi=$mpi")
    end		  
    if mpi
        PEs = npe_arg # (1,1,1,2)

        u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
        U = Array{typeof(u1),1}(undef,Dim)
        U[1] = u1
        for μ=2:Dim
            U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
        end
    else

        u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
        U = Array{typeof(u1),1}(undef,Dim)
        U[1] = u1
        for μ=2:Dim
            U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
        end
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
