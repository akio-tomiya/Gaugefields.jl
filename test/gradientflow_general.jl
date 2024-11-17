
using Random
using Test
using Gaugefields
using Wilsonloop

function gradientflow_test_4D(NX, NY, NZ, NT, NC)
    Dim = 4
    Nwing = 1

    Random.seed!(123)

    #U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U, temps) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temps)
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

    for itrj = 1:100
        flow!(U, g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temps) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end


function gradientflow_test_2D(NX, NT, NC)
    Dim = 2
    Nwing = 1


    U = Initialize_Gaugefields(NC, Nwing, NX, NT, condition="hot", randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U, temps) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temps)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #g = Gradientflow(U,eps = 0.01)
    #listnames = ["plaquette"]
    #listvalues = [1]
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
    listvalues = [1 + im, 0.1] #complex coefficient
    g = Gradientflow_general(U, listloops, listvalues, eps=0.01)
    #g = Gradientflow(U,eps = 0.01)

    for itrj = 1:100
        flow!(U, g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temps) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end

    return plaq_t

end



#const eps = 0.1


println("2D system")
@testset "2D" begin
    NX = 4
    #NY = 4
    #NZ = 4
    NT = 4
    Nwing = 0

    @testset "NC=1" begin
        β = 2.3
        NC = 1
        println("NC = $NC")
        #val =0.6414596466929057
        val = 0.9993254431181984
        @time plaq_t = gradientflow_test_2D(NX, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end
    #error("d")

    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =0.6414596466929057
        val = 0.9768786716327604
        @time plaq_t = gradientflow_test_2D(NX, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        val = 0.9656356864814539
        @time plaq_t = gradientflow_test_2D(NX, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        val = 0.8138836242603148
        @time plaq_t = gradientflow_test_2D(NX, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end
end

println("4D system")
@testset "4D" begin
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
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
