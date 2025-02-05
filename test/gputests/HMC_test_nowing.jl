
using Random
using Test
using CUDA
using Gaugefields
using LinearAlgebra
import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!



function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_Gauge_action(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps,Ucpu,tempscpu,pcpu,gauge_actioncpu)
    Δτ = 1 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)
    substitute_U!(Ucpu, U)
    for i=1:4
        substitute_U!(pcpu[i],p[i])
    end


    @time for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)#,Ucpu,tempscpu,pcpu)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temps)#,Ucpu,tempscpu,pcpu)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)#,Ucpu,tempscpu,pcpu)
        #error("dd")
    end
    #error("dd")

    
    @time for itrj = 1:MDsteps
        U_update!(Ucpu, pcpu, 0.5, Δτ, Dim, gauge_actioncpu ,tempscpu)#,Ucpu,tempscpu,pcpu)

        P_update!(Ucpu, pcpu, 1.0, Δτ, Dim, gauge_actioncpu, tempscpu)#,Ucpu,tempscpu,pcpu)

        U_update!(Ucpu, pcpu, 0.5, Δτ, Dim, gauge_actioncpu, tempscpu)#,Ucpu,tempscpu,pcpu)
    end
    
    Snew = calc_action(gauge_action, U, p)
    Snewcpu = calc_action(gauge_actioncpu, Ucpu, pcpu)
    println("Sold = $Sold, Snew = $Snew Snewcpu = $Snewcpu")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, gauge_action, temps)#,Ucpu,tempscpu,pcpu)
    #temps = get_temporary_gaugefields(gauge_action)
    
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    temp3, it_temp3 = get_temp(temps)
    expU, it_expU = get_temp(temps)
    W, it_W = get_temp(temps)

    #=
    temp1cpu, it_temp1cpu = get_temp(tempscpu)#similar(U[1])
    temp2cpu, it_temp2cpu = get_temp(tempscpu)
    temp3cpu, it_temp3cpu = get_temp(tempscpu)
    expUcpu, it_expUcpu = get_temp(tempscpu)
    Wcpu, it_Wcpu = get_temp(tempscpu)


    =#

    for μ = 1:Dim


        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2,temp3])
        #substitute_U!(Ucpu,U)
        #substitute_U!(pcpu[μ],p[μ])

        #exptU!(expUcpu, ϵ * Δτ, pcpu[μ], [temp1cpu, temp2cpu, temp3cpu])
        #mul!(Wcpu, expUcpu, Ucpu[μ])
        #display(Array(p[μ].a)[:,1,1])
        #display(Array(pcpu[μ].a)[:,1,1,1,1])
        #println("cuda")
        #display(Array(expU.U)[:,:,1,1])
        #println("cpu")
        #display(Array(expUcpu.U)[:,:,1,1,1,1])
        #println("diff")
        #display(Array(expU.U)[:,:,1,1] .- Array(expUcpu.U)[:,:,1,1,1,1])
        

        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)

        
        #println("cuda")
        ##display(Array(W.U)[:,:,1,1])
        #println("cpu")
        #display(Array(Wcpu.U)[:,:,1,1,1,1])
        #error("d")

    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_temp3)
    unused!(temps, it_expU)
    unused!(temps, it_W)
    
    #=
    unused!(tempscpu, it_temp1cpu)
    unused!(tempscpu, it_temp2cpu)
    unused!(tempscpu, it_temp3cpu)
    unused!(tempscpu, it_expUcpu)
    unused!(tempscpu, it_Wcpu)
    =#
   
    
end

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action, temps)#,Ucpu,tempscpu,pcpu) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    #temps = get_temporary_gaugefields(gauge_action)
    dSdUμ, it_dSdUμ = get_temp(temps)#temps[end]
    temp1, it_temp1 = get_temp(temps)
    #dSdUμ = temps[end]
    factor = -ϵ * Δτ / (NC)

    #=
    dSdUμcpu, it_dSdUμcpu = get_temp(tempscpu)#temps[end]
    temp1cpu, it_temp1cpu = get_temp(tempscpu)
    =#

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        #display(Array(dSdUμ.U)[:,:,1,1])
        #calc_dSdUμ!(dSdUμcpu, gauge_action, μ, U)
        #display(Array(dSdUμcpu.U)[:,:,1,1,1,1])
        mul!(temp1, U[μ], dSdUμ) # U*dSdUμ
        #display(Array(temp1.U)[:,:,1,1])
        #substitute_U!(pcpu[μ],p[μ])

        #pcpu[μ].a .= 0
        #substitute_U!(p[μ],pcpu[μ])
        #display(Array(p[μ].a)[:,1,1])

        Traceless_antihermitian_add!(p[μ], factor, temp1)
        #display(Array(p[μ].a)[:,1,1])
        #substitute_U!(temp1cpu,temp1)
        #display(pcpu[μ].a[:,1,1,1,1])
        #Traceless_antihermitian_add!(pcpu[μ], factor, temp1cpu)
        #display(pcpu[μ].a[:,1,1,1,1])
        #error("dd")
    end
    #error("d")
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)

    #=
    unused!(tempscpu, it_dSdUμcpu)
    unused!(tempscpu, it_temp1cpu)
    =#
end


function HMC_test_4D(NX, NY, NZ, NT, NC, β0)
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

    blocks = [4, 4, 4, 4]

    #=
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot",
        cuda=true, blocks=blocks)
        =#

                
    Ucpu = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
    condition="hot")

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="cold",
        cuda=true, blocks=blocks)
    

    #"Reproducible"
    println(typeof(U))

    temps = Temporalfields(U[1]; num=10)
    temp1, it_temp1 = get_temp(temps)#similar(U[1])
    temp2, it_temp2 = get_temp(temps)

    tempscpu = Temporalfields(Ucpu[1]; num=10)
    temp1cpu, it_temp1cpu = get_temp(tempscpu)#similar(U[1])
    temp2cpu, it_temp2cpu = get_temp(tempscpu)

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

    substitute_U!(U,Ucpu)
    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")
    @time plaq_t = calculate_Plaquette(Ucpu, temp1cpu, temp2cpu) * factor
    println("0 plaq_t cpu = $plaq_t")
 


    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")
    poly = calculate_Polyakov_loop(Ucpu, temp1cpu, temp2cpu)
    println("0 polyakov cpu loop = $(real(poly)) $(imag(poly))")

    #error("dd")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β0 / 2
    push!(gauge_action, β, plaqloop)

    gauge_actioncpu = GaugeAction(Ucpu)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β0 / 2
    push!(gauge_actioncpu, β, plaqloop)


    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    println(typeof(p))
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 50
    #temp1 = similar(U[1])
    #temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0

    pcpu = initialize_TA_Gaugefields(Ucpu) 
    for i=1:4
        substitute_U!(pcpu[i],p[i])
    end

    numtrj = 20
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps,Ucpu,tempscpu,pcpu,gauge_actioncpu)
        numaccepted += ifelse(accepted, 1, 0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end


    return plaq_t, numaccepted / numtrj

end





println("4D system")
@testset "4D" begin
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0

    #=
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
    =#

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





end


