## Parallel computation with MPI
Gaugefields.jl uses MPI.jl to do the parallel computations. 
The function ```println_verbose_level1``` is println function with MPI. Please see the example codes. 

We show the example codes. 

### Gradient flow

```julia
using Gaugefields
using Wilsonloop
using MPI


const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])

function test()
    NX = 8*2
    NY = 8*2
    NZ = 8*2
    NT = 8*2
    Nwing = 0
    Dim = 4
    NC = 3


    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",mpi=true,PEs = PEs,mpiinit = false)
        
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    end

    #println(typeof(U))
    println_verbose_level1(U[1],typeof(U))


    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)

    plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println_verbose_level1(U[1]," plaq_t = $plaq_t")
    #println(" plaq_t = $plaq_t")

    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end

            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)

            push!(loops_p,loop1)
            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)
        end
    end


    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
        end
    end


    listnames = [loops_p,loops]
    listvalues = [1+im,0.1]
    g = Gradientflow_general(U,listnames,listvalues,eps = 0.01)

    for itrj=1:100
        @time flow!(U,g)
        #if itrj % 10 == 0
            plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            
            println_verbose_level1(U[1],"$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println_verbose_level1(U[1],"$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        #end
    end


end
test()
```

We can do the parallel computation like

```
mpirun -np 4 julia --project=../ mpigradientflow.jl 1 1 2 2 true
```

### HMC
This is the sample code:

```julia
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
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    plaq_t = calculate_Plaquette(U,temp1,temp2)*factor

    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    if get_myrank(U) == 0
        println("0 plaq_t = $plaq_t")
        println("polyakov loop = $(real(poly)) $(imag(poly))")
    end


    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(snet,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        poly = calculate_Polyakov_loop(U,temp1,temp2) 
        
        if get_myrank(U) == 0
            println("$itrj plaq_t = $plaq_t")
            println("acceptance ratio ",numaccepted/itrj)
            println("polyakov loop = $(real(poly)) $(imag(poly))")
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

function P_update!(U,p,ϵ,Δτ,Dim,snet) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(snet)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,snet,μ,U)
        #println("dSdU = ",getvalue(dSdUμ,1,1,1,1,1,1))
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
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

end


test1()

```