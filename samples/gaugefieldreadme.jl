using Gaugefields
function readme1()

using Gaugefields

    function heatbath_SU3!(U,NC,temps_g,β)
        Dim = 4
        V, it_V = get_temp(temps_g)
        ITERATION_MAX = 10^5
        
        temps, it_temps = get_temp(temps_g, 5)
        
        temps2 = Array{Matrix{ComplexF64},1}(undef,5) 
        temps3 = Array{Matrix{ComplexF64},1}(undef,5) 
        for i=1:5
            temps2[i] = zeros(ComplexF64,2,2)
            temps3[i] = zeros(ComplexF64,NC,NC)
        end
        
        
        mapfunc!(A,B) = SU3update_matrix!(A,B,β,NC,temps2,temps3,ITERATION_MAX)
        
        for μ=1:Dim
            
            loops = loops_staple[(Dim,μ)]
            iseven = true
            
            evaluate_gaugelinks_evenodd!(V,loops,U,temps,iseven)
            map_U!(U[μ],mapfunc!,V,iseven) 
            
            iseven = false
            evaluate_gaugelinks_evenodd!(V,loops,U,temps,iseven)
            map_U!(U[μ],mapfunc!,V,iseven) 
        end
        
        unused!(temps_g, it_V)
        unused!(temps_g, it_temps)
    end
    
    function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        Dim = 4
        Nwing = 1
        
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
        
        
        temps = Temporalfields(U[1], num=5)
        comb, factor = set_comb(U,Dim)
        
        @time plaq_t = calculate_Plaquette(U,temps)*factor
        println("plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("polyakov loop = $(real(poly)) $(imag(poly))")
        
        numhb = 40
        for itrj = 1:numhb
            heatbath_SU3!(U,NC,temps,β)
            
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps) 
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            end
        end
        
        
        return plaq_t
        
    end
    
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1

    β = 5.7
    NC = 3
    @time plaq_t = heatbathtest_4D(NX, NY, NZ, NT, β, NC)
end


function readme2()

    function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        Dim = 4
        Nwing = 1
        
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
        println(typeof(U))
        
        gauge_action = GaugeAction(U)
        plaqloop = make_loops_fromname("plaquette",Dim=Dim)
        append!(plaqloop,plaqloop')
        βinp = β/2
        push!(gauge_action,βinp,plaqloop)
        
        rectloop = make_loops_fromname("rectangular",Dim=Dim)
        append!(rectloop,rectloop')
        βinp = β/2
        push!(gauge_action,βinp,rectloop)
        
        hnew = Heatbath_update(U,gauge_action)
        
        show(gauge_action)
        
        temps = Temporalfields(U[1], num=9)
        comb, factor = set_comb(U,Dim)
        
        @time plaq_t = calculate_Plaquette(U,temps)*factor
        println("plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps)
        println("polyakov loop = $(real(poly)) $(imag(poly))")
        
        numhb = 1000
        for itrj = 1:numhb
            
            heatbath!(U,hnew)
            
            plaq_t = calculate_Plaquette(U,temps)*factor
            poly = calculate_Polyakov_loop(U,temps) 
            
            if itrj % 40 == 0
                println("$itrj plaq_t = $plaq_t")
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            end
        end
        
        return plaq_t
        
    end

    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    β = 5.7
    heatbathtest_4D(NX, NY, NZ, NT, β, NC)
end

function readme3()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    NC = 3

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot")

    temps = Temporalfields(U[1], num=3)
    comb, factor = set_comb(U,Dim)

    g = Gradientflow(U)
    for itrj = 1:100
        flow!(U, g)
        @time plaq_t = calculate_Plaquette(U, temps) * factor
        println("$itrj plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U, temps)
        println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
    end

end

using Random
using Gaugefields
using LinearAlgebra

function readme4()

    function HMC_test_4D(NX,NY,NZ,NT,NC,β)
        Dim = 4
        Nwing = 0

        Random.seed!(123)

        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
        #"Reproducible"
        println(typeof(U))

        temps = Temporalfields(U[1], num=6)
        comb, factor = set_comb(U, Dim)

        @time plaq_t = calculate_Plaquette(U,temps)*factor
        println("0 plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")

        gauge_action = GaugeAction(U)
        plaqloop = make_loops_fromname("plaquette")
        append!(plaqloop,plaqloop')
        β = β/2
        push!(gauge_action,β,plaqloop)
        
        #show(gauge_action)

        p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
        Uold = similar(U)
        MDsteps = 100
        numaccepted = 0

        numtrj = 10
        for itrj = 1:numtrj
            t = @timed begin
                accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temps)
            end
            if get_myrank(U) == 0
                println("elapsed time for MDsteps: $(t.time) [s]")
            end
            numaccepted += ifelse(accepted,1,0)

            #plaq_t = calculate_Plaquette(U,temps)*factor
            #println("$itrj plaq_t = $plaq_t")
            
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps) 
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
                println("acceptance ratio ",numaccepted/itrj)
            end
        end
        return plaq_t,numaccepted/numtrj

    end


    function main()
        β = 5.7
        NX = 8
        NY = 8
        NZ = 8
        NT = 8
        NC = 3
        HMC_test_4D(NX,NY,NZ,NT,NC,β)
    end
    main()
end

using Random
using Gaugefields
using LinearAlgebra

function readme5()

    function HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
        Dim = 4
        Nwing = 0

        flux = Flux
        println("Flux : ", flux)

        Random.seed!(123)


        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

        temps = Temporalfields(U[1], num=9)
        comb, factor = set_comb(U, Dim)

        @time plaq_t = calculate_Plaquette(U,B,temps)*factor
        println("0 plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")

        gauge_action = GaugeAction(U,B)
        plaqloop = make_loops_fromname("plaquette")
        append!(plaqloop,plaqloop')
        β = β/2
        push!(gauge_action,β,plaqloop)
        
        #show(gauge_action)

        p = initialize_TA_Gaugefields(U)
        Uold = similar(U)
        MDsteps = 50
        numaccepted = 0

        numtrj = 100
        for itrj = 1:numtrj
            t = @timed begin
                accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temps)
            end
            if get_myrank(U) == 0
                #            println("elapsed time for MDsteps: $(t.time) [s]")
            end
            numaccepted += ifelse(accepted,1,0)

            #plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
            #println("$itrj plaq_t = $plaq_t")
            
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,B,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps) 
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
                println("acceptance ratio ",numaccepted/itrj)
            end

        end
        return plaq_t,numaccepted/numtrj

    end


    function main()
        β = 5.7
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        NC = 3
        Flux = [0,0,1,1,0,0]
        #HMC_test_4D(NX,NY,NZ,NT,NC,β)
        HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
    end
    main()
end

using Random
using Gaugefields
using Wilsonloop
using LinearAlgebra

function readme6()


    function MDstep_dynB!(
        gauge_action,
        U,
        B,
        flux,
        p,
        MDsteps, # MDsteps should be an even integer
        Dim,
        Uold,
        Bold,
        flux_old,
        temps
    ) # Halfway-updating HMC
        Δτ = 1.0/MDsteps
        gauss_distribution!(p)

        Sold = calc_action(gauge_action,U,B,p)

        substitute_U!(Uold,U)
        substitute_U!(Bold,B)
        flux_old[:] = flux[:]

        for itrj=1:MDsteps
            U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

            P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temps)

            U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

            if itrj == Int(MDsteps/2)
                Flux_update!(B,flux)
            end
        end

        Snew = calc_action(gauge_action,U,B,p)
        ratio = min(1,exp(-Snew+Sold))
        if rand() > ratio
            println("rejected! flux = ", flux_old)
            substitute_U!(U,Uold)
            substitute_U!(B,Bold)
            flux[:] = flux_old[:]
            return false
        else
            println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
            return true
        end
    end

    function MDstep_dynB!(
        gauge_action,
        U,
        B,
        flux,
        p,
        MDsteps,
        num_HMC,
        Dim,
        Uold1,
        Uold2,
        Bold,
        flux_old,
        temps
    ) # Double-tesing HMC
        p0 = initialize_TA_Gaugefields(U)
        Sold = calc_action(gauge_action,U,B,p0)

        substitute_U!(Uold1,U)
        substitute_U!(Bold, B)
        flux_old[:] = flux[:]

        Flux_update!(B,flux)

        for ihmc=1:num_HMC
            MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temps)
        end

        Snew = calc_action(gauge_action,U,B,p0)
        #println("Sold = $Sold, Snew = $Snew")
        #println("Snew - Sold = $(Snew-Sold)")
        ratio = min(1,exp(-Snew+Sold))
        if rand() > ratio
            println("rejected! flux = ", flux_old)
            substitute_U!(U,Uold1)
            substitute_U!(B,Bold)
            flux[:] = flux_old[:]
            return false
        else
            println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
            return true
        end
    end

    function Flux_update!(B,flux)
        NC  = B[1,2].NC
        NDW = B[1,2].NDW
        NX  = B[1,2].NX
        NY  = B[1,2].NY
        NZ  = B[1,2].NZ
        NT  = B[1,2].NT

        i = rand(1:6)
        flux[i] += rand(-1:1)
        flux[i] %= NC
        flux[i] += (flux[i] < 0) ? NC : 0
        #    flux[:] = rand(0:NC-1,6)
        B = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

    end

    function HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)
        Dim = 4
        Nwing = 0

        Random.seed!(123)

        flux = [1,1,1,1,2,0]

        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

        L = [NX,NY,NZ,NT]
        filename = "test/confs/U_beta6.0_L8_F111120_4000.txt"
        load_BridgeText!(filename,U,L,NC)

        temps = Temporalfields(U[1], num=9)
        comb, factor = set_comb(U, Dim)

        @time plaq_t = calculate_Plaquette(U,B,temps)*factor
        println("0 plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")

        gauge_action = GaugeAction(U,B)
        plaqloop = make_loops_fromname("plaquette")
        append!(plaqloop,plaqloop')
        β = β/2
        push!(gauge_action,β,plaqloop)
        
        #show(gauge_action)

        p = initialize_TA_Gaugefields(U)
        Uold  = similar(U)
        Bold = similar(B)
        flux_old = zeros(Int, 6)

        MDsteps = 50 # even integer!!!
        numaccepted = 0

        numtrj = 100
        for itrj = 1:numtrj
            t = @timed begin
                accepted = MDstep_dynB!(
                    gauge_action,
                    U,
                    B,
                    flux,
                    p,
                    MDsteps,
                    Dim,
                    Uold,
                    Bold,
                    flux_old,
                    temps
                )
            end
            if get_myrank(U) == 0
                println("Flux : ", flux)
                #            println("elapsed time for MDsteps: $(t.time) [s]")
            end
            numaccepted += ifelse(accepted,1,0)

            #plaq_t = calculate_Plaquette(U,B,temps)*factor
            #println("$itrj plaq_t = $plaq_t")
            
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,B,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps)
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
                println("acceptance ratio ",numaccepted/itrj)
            end

        end
        return plaq_t,numaccepted/numtrj

    end


function main()
    β = 6.0
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)
end
main()
end



using Random
using Test
using Gaugefields
using Wilsonloop
const eps = 0.1


function readme7()



    function gradientflow_test_4D(NX,NY,NZ,NT,NC)
        Dim = 4
        Nwing = 1

        Random.seed!(123)

        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")

        temps = Temporalfields(U[1], num=2)
        comb, factor = set_comb(U, Dim)

        @time plaq_t = calculate_Plaquette(U,temps)*factor
        println("0 plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")

        #Plaquette term
        loops_p = Wilsonline{Dim}[]
        for μ=1:Dim
            for ν=μ:Dim
                if ν == μ
                    continue
                end
                loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
                push!(loops_p,loop1)
            end
        end

        #Rectangular term
        loops = Wilsonline{Dim}[]
        for μ=1:Dim
            for ν=μ:Dim
                if ν == μ
                    continue
                end
                loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
                push!(loops,loop1)
                loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
                
                push!(loops,loop1)
            end
        end

        listloops = [loops_p,loops]
        listvalues = [1+im,0.1]
        g = Gradientflow_general(U,listloops,listvalues,eps = 0.01)

        for itrj=1:100
            flow!(U,g)
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps) 
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            end
        end
        return plaq_t

    end


    function gradientflow_test_2D(NX,NT,NC)
        Dim = 2
        Nwing = 1
        U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")

        temps = Temporalfields(U[1], num=2)
        comb, factor = set_comb(U, Dim)

        @time plaq_t = calculate_Plaquette(U,temps)*factor
        println("0 plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")

        #g = Gradientflow(U,eps = 0.01)
        #listnames = ["plaquette"]
        #listvalues = [1]
        loops_p = Wilsonline{Dim}[]
        for μ=1:Dim
            for ν=μ:Dim
                if ν == μ
                    continue
                end

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
                loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
                
                push!(loops,loop1)
            end
        end

        listloops = [loops_p,loops]
        listvalues = [1+im,0.1]
        g = Gradientflow_general(U,listloops,listvalues,eps = 0.01)

        for itrj=1:100
            flow!(U,g)
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps)
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            end
        end

        return plaq_t

    end



    const eps = 0.1


    println("2D system")
    @testset "2D" begin
        NX = 4
        #NY = 4
        #NZ = 4
        NT = 4
        Nwing = 1

        @testset "NC=1" begin
            β = 2.3
            NC = 1
            println("NC = $NC")
            @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        end
        #error("d")
        
        @testset "NC=2" begin
            β = 2.3
            NC = 2
            println("NC = $NC")
            @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        end

        @testset "NC=3" begin
            β = 5.7
            NC = 3
            println("NC = $NC")
            @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        end

        @testset "NC=4" begin
            β = 5.7
            NC = 4
            println("NC = $NC")
            @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        end
    end

    println("4D system")
    @testset "4D" begin
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        Nwing = 1


        
        @testset "NC=2" begin
            β = 2.3
            NC = 2
            println("NC = $NC")
            @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
        end

        @testset "NC=3" begin
            β = 5.7
            NC = 3
            println("NC = $NC")
            @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
        end

        @testset "NC=4" begin
            β = 5.7
            NC = 4
            println("NC = $NC")

            val = 0.7301232810349298
            @time plaq_t =gradientflow_test_4D(NX,NY,NZ,NT,NC)
        end


    end
end

using Random
using Test
using Gaugefields
using Wilsonloop

function readme8()



    using Random
    using Test
    using Gaugefields
    using Wilsonloop

    function gradientflow_test_4D(NX,NY,NZ,NT,NC)
        Dim = 4
        Nwing = 0

        flux = [0,0,1,1,0,0]

        Random.seed!(123)

        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

        temps = Temporalfields(U[1], num=2)
        comb, factor = set_comb(U, Dim)

        @time plaq_t = calculate_Plaquette(U,B,temps)*factor
        println("0 plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")

        #Plaquette term
        loops_p = Wilsonline{Dim}[]
        for μ=1:Dim
            for ν=μ:Dim
                if ν == μ
                    continue
                end
                loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
                push!(loops_p,loop1)
            end
        end

        #Rectangular term
        loops = Wilsonline{Dim}[]
        for μ=1:Dim
            for ν=μ:Dim
                if ν == μ
                    continue
                end
                loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
                push!(loops,loop1)
                loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
                
                push!(loops,loop1)
            end
        end

        listloops = [loops_p,loops]
        listvalues = [1+im,0.1]
        g = Gradientflow_general(U,B,listloops,listvalues,eps = 0.1)

        for itrj=1:10
            flow!(U,B,g)
            if itrj % 10 == 0
                @time plaq_t = calculate_Plaquette(U,B,temps)*factor
                println("$itrj plaq_t = $plaq_t")
                poly = calculate_Polyakov_loop(U,temps) 
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            end
        end
        return plaq_t

    end

    const eps = 0.1

    println("4D system")
    @testset "4D" begin
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        Nwing = 1

        
        @testset "NC=2" begin
            β = 2.3
            NC = 2
            println("NC = $NC")
            @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
        end

        @testset "NC=3" begin
            β = 5.7
            NC = 3
            println("NC = $NC")
            @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
        end

        @testset "NC=4" begin
            β = 5.7
            NC = 4
            println("NC = $NC")

            val = 0.7301232810349298
            @time plaq_t =gradientflow_test_4D(NX,NY,NZ,NT,NC)
        end


    end
end

#readme1()
#readme2()
#readme3()
#readme4()
#readme5()
#readme6()
#readme7()
readme8()

