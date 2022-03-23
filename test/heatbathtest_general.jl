

function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 0

    #=
    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end
    =#

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    h = Heatbath(U,β)

    
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
    

    
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")
    #hnew = Heatbath_update(U,gauge_action)

    numhb = 200
    numOR = 3
    plaq_ave = 0.0
    for itrj = 1:numhb
        heatbath!(U,hnew)
        for ior=1:numOR
            overrelaxation!(U,hnew)
        end
        #heatbath!(U,h)
        #=
        if NC == 2
            heatbath_SU2!(U,NC,[temp1,temp2,temp3],β)
        elseif NC == 3
            heatbath_SU3!(U,NC,[temp1,temp2,temp3],β)
        else
            heatbath_SUN!(U,NC,[temp1,temp2,temp3],β)
        end
        =#
        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        plaq_ave += plaq_t

        if itrj % 40 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t average: $(plaq_ave/itrj)")
            #println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    

    return  plaq_ave/numhb #plaq_t

end

function heatbathtest_2D(NX,NT,β,NC)
    Dim = 2
    Nwing = 1

    #=
    u1 = RandomGauges(NC,Nwing,NX,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = RandomGauges(NC,Nwing,NX,NT)
    end
    =#

    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    #comb = 6
    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")

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
    

    numhb = 1000
    numOR = 3
    plaq_ave = 0.0
    for itrj = 1:numhb
        heatbath!(U,hnew)
        for ior=1:numOR
            overrelaxation!(U,hnew)
        end
        #heatbath!(U,[temp1,temp2,temp3],β)
        #=
        if NC == 2
            heatbath_SU2!(U,NC,[temp1,temp2,temp3],β,Dim)
        elseif NC == 3
            heatbath_SU3!(U,NC,[temp1,temp2,temp3],β,Dim)
        else
            heatbath_SUN!(U,NC,[temp1,temp2,temp3],β,Dim)
        end
        =#
        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        plaq_ave += plaq_t

        if itrj % 200 == 0
            #@time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t average: $(plaq_ave/itrj)")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    

    return plaq_ave/numhb

end



println("2D system")
@testset "2D" begin
    NX = 4
    #NY = 4
    #NZ = 4
    NT = 4
    Nwing = 1
    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =  0.8275345965584693
        #val =0.6414596466929057
        val = 0.8237678929985711
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        val = 0.8070960744356934
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        val = 0.38564608114572807
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
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
        #val =0.9042749888688881
        val = 0.9040744318206326
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        
        NC = 3
        β = 2.3*NC #5.7
        println("NC = $NC")
        #val = 0.9148612416401057
        val = 0.9148494229319207
        #@time plaq_t = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.791346082492281
        val = 0.792025880016283
        #@time plaq_t = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end



end

