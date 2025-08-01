using JACC
JACC.@init_backend
using Random
using Test
using CUDA
using Gaugefields
using Wilsonloop



function gradientflow_test_4D(NX, NY, NZ, NT, NC)
    Dim = 4
    Nwing = 0

    Random.seed!(123)

    #U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    Ucpu = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold", accelerator="JACC")
    substitute_U!(U, Ucpu)

    println(typeof(U))


    temps = Temporalfields(U[1]; num=10)
    temp1 = temps[1]#similar(U[1])
    temp2 = temps[2] #similar(U[1])
    #temp1 = similar(U[1])
    #temp2 = similar(U[1])

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
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ = 1:Dim
        for ν = μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)], Dim=Dim)
            push!(loops_p, loop1)
            #loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            #push!(loops_p,loop1)
        end
    end


    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ = 1:Dim
        for ν = μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ, 1), (ν, 2), (μ, -1), (ν, -2)], Dim=Dim)
            push!(loops, loop1)
            loop1 = Wilsonline([(μ, 2), (ν, 1), (μ, -2), (ν, -1)], Dim=Dim)

            push!(loops, loop1)
        end
    end

    listloops = [loops_p, loops]
    listvalues = [1 + im, 0.1]
    g = Gradientflow_general(U, listloops, listvalues, eps=0.01)

    @time for itrj = 1:100
        flow!(U, g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end



println("4D system")
@testset "4D" begin
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0

    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =0.6414596466929057
        #val = 0.5920897445000382
        val = 0.9440125563836135
        @time plaq_t = gradientflow_test_4D(NX, NY, NZ, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        #val  =0.9440125563836135
        #val = 0.5385142466966718
        val = 0.8786515255315753
        @time plaq_t = gradientflow_test_4D(NX, NY, NZ, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        #val = 0.1904815857904191
        val = 0.7301232810349298
        @time plaq_t = gradientflow_test_4D(NX, NY, NZ, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end






end