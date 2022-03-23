function heatbath_SU2!(U,NC,temps,β,Dim=4)

    temp1 = temps[1]
    temp2 = temps[2]
    V = temps[3]
    ITERATION_MAX = 10^5

    temps2 = Array{Matrix{ComplexF64},1}(undef,5) 
    for i=1:5
        temps2[i] = zeros(ComplexF64,2,2)
    end

    mapfunc!(A,B) = SU2update_KP!(A,B,β,NC,temps2,ITERATION_MAX)

    for μ=1:Dim

        loops = loops_staple[(Dim,μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 
    end
    
end

function heatbath_SU3!(U,NC,temps,β,Dim=4)
    temp1 = temps[1]
    temp2 = temps[2]
    V = temps[3]
    ITERATION_MAX = 10^5

    temps2 = Array{Matrix{ComplexF64},1}(undef,5) 
    temps3 = Array{Matrix{ComplexF64},1}(undef,5) 
    for i=1:5
        temps2[i] = zeros(ComplexF64,2,2)
        temps3[i] = zeros(ComplexF64,3,3)
    end

    mapfunc!(A,B) = SU3update_matrix!(A,B,β,NC,temps2,temps3,ITERATION_MAX)

    for μ=1:Dim

        loops = loops_staple[(Dim,μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 
    end
    
end


function heatbath_SUN!(U,NC,temps,β,Dim = 4)
    #Dim = 4
    temp1 = temps[1]
    temp2 = temps[2]
    V = temps[3]
    ITERATION_MAX = 10^5

    temps2 = Array{Matrix{ComplexF64},1}(undef,5) 
    temps3 = Array{Matrix{ComplexF64},1}(undef,5) 
    for i=1:5
        temps2[i] = zeros(ComplexF64,2,2)
        temps3[i] = zeros(ComplexF64,NC,NC)
    end

    mapfunc!(A,B) = SUNupdate_matrix!(A,B,β,NC,temps2,temps3,ITERATION_MAX)

    for μ=1:Dim

        loops = loops_staple[(Dim,μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 
    end
    
end


function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 1

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

    #=
    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette",Dim=Dim)
    append!(plaqloop,plaqloop')
    βinp = β/2
    push!(gauge_action,βinp,plaqloop)
    =#

    
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

    plaq_ave = 0.0
    numhb = 200
    for itrj = 1:numhb
        #heatbath!(U,hnew)
        heatbath!(U,h)
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
            #@time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t average: $(plaq_ave/itrj)")
            #println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
        #=

        if itrj % 40 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
        =#
    end
    

    return plaq_ave/numhb

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

    #=
    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette",Dim=Dim)
    append!(plaqloop,plaqloop')
    βinp = β/2
    push!(gauge_action,βinp,plaqloop)
    hnew = Heatbath_update(U,gauge_action)

    =#

    numhb = 200
    plaq_ave = 0.0
    for itrj = 1:numhb
        #heatbath!(U,hnew)
        heatbath!(U,[temp1,temp2,temp3],β)
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

        if itrj % 40 == 0
            #@time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t average: $(plaq_ave/itrj)")
            #println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end

        #=
        if itrj % 40 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
        =#
    end
    

    return plaq_ave/numhb

end

#eps = 1e-1

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
        #val =0.6414596466929057
        val = 0.6042874193905048
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps

    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        val = 0.5616169101071591
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        val = 0.1967198548214144
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end



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
        val = 0.47007878197368624
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        val = 0.40073421896125794
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        val = 0.17476796328668975
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end


end