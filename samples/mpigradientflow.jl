using Gaugefields
using MPI

const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])

function test()
    NX = 8*2
    NY = 8*2
    NZ = 8*2
    NT = 8*2
    Nwing = 0
    NC = 3
    
    Dim = 4
    
    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",mpi=true,PEs = PEs,mpiinit = false)
        
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")

    end


    temps = Temporalfields(U[1], num=3)
    comb, factor = set_comb(U,Dim)

    β = 5.7

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println(" plaq_t = $plaq_t")

    g = Gradientflow(U,eps = 0.01,mpi=mpi)

    for itrj=1:100
        @time flow!(U,g)
        if itrj % 10 == 0
            plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            #poly = calculate_Polyakov_loop(U,temps)
            #println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end




end
test()
