using Gaugefields
using MPI

function test()
    NX = 8*2
    NY = 8*2
    NZ = 8*2
    NT = 8*2
    Nwing = 1
    NC = 3
    
    Dim = 4
    
    #=
    u1 = RandomGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = RandomGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
    end
    =#

    mpi = true
    if mpi
        PEs = (1,1,1,2)

        u1 = RandomGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
        U = Array{typeof(u1),1}(undef,Dim)
        U[1] = u1
        for μ=2:Dim
            U[μ] = RandomGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
        end
    else

        u1 = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
        U = Array{typeof(u1),1}(undef,Dim)
        U[1] = u1
        for μ=2:Dim
            U[μ] = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
        end
    end


    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    β = 5.7

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println(" plaq_t = $plaq_t")

    g = Gradientflow(U,eps = 0.01,mpi=mpi)

    for itrj=1:100
        @time flow!(U,g)
        if itrj % 10 == 0
            plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            #poly = calculate_Polyakov_loop(U,temp1,temp2) 
            #println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end




end
test()