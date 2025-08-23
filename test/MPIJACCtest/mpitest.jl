import JACC
JACC.@init_backend
using Random
using Gaugefields
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!


function main()
    Dim = 4
    Nwing = 1
    Random.seed!(123)
    NX = 32
    NY = 32
    NZ = 32
    NT = 32
    NC = 3
    β = 6.0

    Random.seed!(123)


    #Ucpu = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="one instanton")
    #"Reproducible"
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="cold"; isMPILattice=true)

    Ucpu = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
        condition="hot")

    Uj = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
        condition="cold",
        accelerator="JACC")

    Uc = Initialize_Gaugefields(NC, 0, NX, NY, NZ, NT,
        condition="cold",
        accelerator="cuda",blocks=(4,4,4,4))

    substitute_U!(U, Ucpu)
    substitute_U!(Uj, Ucpu)
    substitute_U!(Uc, Ucpu)


    println(typeof(U))
    temps = Temporalfields(U[1]; num=10)
    temp1, it_temp1 = get_temp(temps)#similar(U[1])
    temp2, it_temp2 = get_temp(temps)

    tempscpu = Temporalfields(Ucpu[1]; num=10)
    temp1cpu, it_temp1cpu = get_temp(tempscpu)#similar(U[1])
    temp2cpu, it_temp2cpu = get_temp(tempscpu)

    println(typeof(Uj))
    tempsj = Temporalfields(Uj[1]; num=10)
    temp1j, it_temp1j = get_temp(tempsj)#similar(U[1])
    temp2j, it_temp2j = get_temp(tempsj)

        println(typeof(Uc))
    tempsc = Temporalfields(Uc[1]; num=10)
    temp1c, it_temp1c = get_temp(tempsc)#similar(U[1])
    temp2c, it_temp2c = get_temp(tempsc)


    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1 / (comb * U[1].NV * U[1].NC)


    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("JACC: 0 plaq_t = $plaq_t")
    #poly = calculate_Polyakov_loop(U, temp1, temp2)
    #println("JACC: 0 polyakov loop = $(real(poly)) $(imag(poly))")

    @time plaq_tcpu = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
    println("CPU: 0 plaq_tcpu = $plaq_tcpu")

    @time plaq_tj = calculate_Plaquette(Uj, temp1j, temp2j) * factor
    println("JACC0: 0 plaq_t = $plaq_tj")
    #polycpu = calculate_Polyakov_loop(Ucpu, temp1cpu, temp2cpu)
    #println("CPU: 0 polyakov loop = $(real(polycpu)) $(imag(polycpu))")
    @time plaq_tc = calculate_Plaquette(Uc, temp1c, temp2c) * factor
    println("CUDA: 0 plaq_t = $plaq_tc")

    println("--------------2nd time-----------------")

    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("JACC: 0 plaq_t = $plaq_t")
    #poly = calculate_Polyakov_loop(U, temp1, temp2)
    #println("JACC: 0 polyakov loop = $(real(poly)) $(imag(poly))")

    @time plaq_tcpu = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
    println("CPU: 0 plaq_tcpu = $plaq_tcpu")

    @time plaq_tj = calculate_Plaquette(Uj, temp1j, temp2j) * factor
    println("JACC0: 0 plaq_t = $plaq_tj")

    @time plaq_tc = calculate_Plaquette(Uc, temp1c, temp2c) * factor
    println("CUDA: 0 plaq_t = $plaq_tc")
end
main()