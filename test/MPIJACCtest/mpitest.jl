import JACC
JACC.@init_backend
using Random
using Gaugefields
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!
using MPI
using InteractiveUtils

function main()
    Dim = 4
    Nwing = 1
    Random.seed!(123)
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    β = 6.0
    NV = NX * NY * NZ * NT

    Random.seed!(123)
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)


    docpu = true
    donewJACC = true
    dooldJACC = true
    docuda = false

    #Ucpu = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="one instanton")
    #"Reproducible"
    if donewJACC
        U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
            condition="cold"; isMPILattice=true)
    
    end

    if docpu
        Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
            condition="hot")
        
    end
    if dooldJACC
        Uj = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
            condition="cold",
            accelerator="JACC")
    end
    if docuda
        Uc = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
            condition="cold",
            accelerator="cuda", blocks=(4, 4, 4, 4))
    end
    #for i=1:4
    #    MPI.Bcast(Ucpu[i].U,0,MPI.COMM_WORLD)
    #end

    if docpu
        if donewJACC
            substitute_U!(U, Ucpu)
        end
        if dooldJACC
            substitute_U!(Uj, Ucpu)
        end
        if docuda
            substitute_U!(Uc, Ucpu)
        end
    end


    if donewJACC
        if myrank == 0
            println(typeof(U))
        end
        temps = Temporalfields(U[1]; num=10)
        temp1, it_temp1 = get_temp(temps)#similar(U[1])
        temp2, it_temp2 = get_temp(temps)
    end

    if docpu
        tempscpu = Temporalfields(Ucpu[1]; num=10)
        temp1cpu, it_temp1cpu = get_temp(tempscpu)#similar(U[1])
        temp2cpu, it_temp2cpu = get_temp(tempscpu)
    end

    if dooldJACC
        if myrank == 0
            println(typeof(Uj))
        end
        tempsj = Temporalfields(Uj[1]; num=10)
        temp1j, it_temp1j = get_temp(tempsj)#similar(U[1])
        temp2j, it_temp2j = get_temp(tempsj)
    end

    if docuda
        if myrank == 0
            println(typeof(Uc))
        end
        tempsc = Temporalfields(Uc[1]; num=10)
        temp1c, it_temp1c = get_temp(tempsc)#similar(U[1])
        temp2c, it_temp2c = get_temp(tempsc)
    end


    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1 / (comb * NV * NC)

    if donewJACC
        @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        if myrank == 0
            println("JACC: 0 plaq_t = $plaq_t")
        end
        poly_t = calculate_Polyakov_loop(U, temp1, temp2)
        if myrank == 0
            println("JACC: 0 polyakov loop = $(real(poly_t)) $(imag(poly_t))")
        end
    end

    if docpu
        @time plaq_tcpu = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
        if myrank == 0
            println("CPU: 0 plaq_tcpu = $plaq_tcpu")
        end
        poly_tcpu = calculate_Polyakov_loop(Ucpu, temp1cpu, temp2cpu)
        if myrank == 0
            println("CPU: 0 polyakov loop = $(real(poly_tcpu)) $(imag(poly_tcpu))")
        end
    end

    if dooldJACC
        @time plaq_tj = calculate_Plaquette(Uj, temp1j, temp2j) * factor
        if myrank == 0
            println("old JACC: 0 plaq_t = $plaq_tj")
        end
        poly_tj = calculate_Polyakov_loop(Uj, temp1j, temp2j)
        if myrank == 0
            println("old JACC: 0 polyakov loop = $(real(poly_tj)) $(imag(poly_tj))")
        end

    end
    #polycpu = calculate_Polyakov_loop(Ucpu, temp1cpu, temp2cpu)
    #println("CPU: 0 polyakov loop = $(real(polycpu)) $(imag(polycpu))")
    if docuda
        @time plaq_tc = calculate_Plaquette(Uc, temp1c, temp2c) * factor
        if myrank == 0
            println("CUDA: 0 plaq_t = $plaq_tc")
        end
        poly_tc = calculate_Polyakov_loop(Uc, temp1c, temp2c)
        if myrank == 0
            println("CUDA: 0 polyakov loop = $(real(poly_tc)) $(imag(poly_tc))")
        end
    end

    println("--------------2nd time-----------------")

    if donewJACC
        @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor

        if myrank == 0
            println("JACC: 0 plaq_t = $plaq_t")
        end
        poly_t = calculate_Polyakov_loop(U, temp1, temp2)
        if myrank == 0
            println("JACC: 0 polyakov loop = $(real(poly_t)) $(imag(poly_t))")
        end
    end

    if docpu
        @time plaq_tcpu = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
        if myrank == 0
            println("CPU: 0 plaq_tcpu = $plaq_tcpu")
        end
        poly_tcpu = calculate_Polyakov_loop(Ucpu, temp1cpu, temp2cpu)
        if myrank == 0
            println("CPU: 0 polyakov loop = $(real(poly_tcpu)) $(imag(poly_tcpu))")
        end
    end

    if dooldJACC
        @time plaq_tj = calculate_Plaquette(Uj, temp1j, temp2j) * factor
        if myrank == 0
            println("old JACC: 0 plaq_t = $plaq_tj")
        end
        poly_tj = calculate_Polyakov_loop(Uj, temp1j, temp2j)
        if myrank == 0
            println("old JACC: 0 polyakov loop = $(real(poly_tj)) $(imag(poly_tj))")
        end

    end
    #polycpu = calculate_Polyakov_loop(Ucpu, temp1cpu, temp2cpu)
    #println("CPU: 0 polyakov loop = $(real(polycpu)) $(imag(polycpu))")
    if docuda
        @time plaq_tc = calculate_Plaquette(Uc, temp1c, temp2c) * factor
        if myrank == 0
            println("CUDA: 0 plaq_t = $plaq_tc")
        end
        poly_tc = calculate_Polyakov_loop(Uc, temp1c, temp2c)
        if myrank == 0
            println("CUDA: 0 polyakov loop = $(real(poly_tc)) $(imag(poly_tc))")
        end
    end
end
main()