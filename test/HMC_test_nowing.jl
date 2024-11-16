
using Random
using Gaugefields
using LinearAlgebra
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!


function HMC_test_4D(NX, NY, NZ, NT, NC, β)
    Dim = 4
    Nwing = 0

    Random.seed!(123)

    #=
    u1 = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
    end
    =#

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    #"Reproducible"
    println(typeof(U))

    temps = Temporalfields(U[1], num=10)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U, temps) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temps)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β / 2
    push!(gauge_action, β, plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    MDsteps = 100

    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps)
        numaccepted += ifelse(accepted, 1, 0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temps) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end


    return plaq_t, numaccepted / numtrj

end


function HMC_test_2D(NX, NT, NC)
    Dim = 2
    Nwing = 0

    #=
    u1 = RandomGauges(NC,Nwing,NX,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = RandomGauges(NC,Nwing,NX,NT)
    end
    =#

    U = Initialize_Gaugefields(NC, Nwing, NX, NT, condition="hot", randomnumber="Reproducible")

    println(typeof(U))

    temps = Temporalfields(U[1], num=10)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U, temps) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temps)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette", Dim=Dim)
    append!(plaqloop, plaqloop')
    β = 5.7 * (NC / 3) / 2
    push!(gauge_action, β, plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    MDsteps = 100

    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps)
        numaccepted += ifelse(accepted, 1, 0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temps) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end
    return plaq_t, numaccepted / numtrj

end



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
        val = 0.9768786716327604
        @time plaq_t, ratio = HMC_test_2D(NX, NT, NC)
        @test ratio > 0.5
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =0.6414596466929057
        val = 0.9768786716327604
        @time plaq_t, ratio = HMC_test_2D(NX, NT, NC)
        @test ratio > 0.5
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        val = 0.9656356864814539
        @time plaq_t, ratio = HMC_test_2D(NX, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
        @test ratio > 0.5
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        val = 0.8138836242603148
        @time plaq_t, ratio = HMC_test_2D(NX, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
        @test ratio > 0.5
    end


end


println("4D system")
@testset "4D" begin
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0

    @testset "NC=1" begin
        β = 2.3
        NC = 1
        println("NC = $NC")
        #val =0.6414596466929057
        #val = 0.5920897445000382
        val = 0.7887418522553702
        @time plaq_t, ratio = HMC_test_4D(NX, NY, NZ, NT, NC, β)
        #@test abs(plaq_t-val)/abs(val) < eps
        @test ratio > 0.5
    end

    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =0.6414596466929057
        #val = 0.5920897445000382
        val = 0.9440125563836135
        @time plaq_t, ratio = HMC_test_4D(NX, NY, NZ, NT, NC, β)
        #@test abs(plaq_t-val)/abs(val) < eps
        @test ratio > 0.5
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        #val  =0.9440125563836135
        #val = 0.5385142466966718
        val = 0.8786515255315753
        @time plaq_t, ratio = HMC_test_4D(NX, NY, NZ, NT, NC, β)
        #@test abs(plaq_t-val)/abs(val) < eps
        @test ratio > 0.5
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        #val = 0.1904815857904191
        val = 0.7301232810349298
        @time plaq_t, ratio = HMC_test_4D(NX, NY, NZ, NT, NC, β)
        #@test abs(plaq_t-val)/abs(val) < eps
        @test ratio > 0.5
    end



end


