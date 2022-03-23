

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
    
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 200
    numOR = 3

    plaq_ave = 0.0
    for itrj = 1:numhb
        heatbath!(U,h)
        for ior=1:numOR
            overrelaxation!(U,h)
        end

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        plaq_ave += plaq_t

        if itrj % 40 == 0
            #@time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t average: $(plaq_ave/itrj)")
            #println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    

    return plaq_ave/numhb

end

function heatbathtest_2D(NX,NT,β,NC)
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

    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")
    h = Heatbath(U,β)

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

    numhb = 2000
    numOR = 10
    plaq_ave = 0.0
    for itrj = 1:numhb
        heatbath!(U,h)
        #heatbath!(U,[temp1,temp2,temp3],β)
        for ior=1:numOR
            overrelaxation!(U,h)
        end


        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        plaq_ave += plaq_t

        if itrj % 200 == 0
            #@time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t average: $(plaq_ave/itrj)")
            #println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    

    return plaq_ave/numhb

end

#eps = 1e-1

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
        #val = 0.5767979418826605
        #val =0.6414596466929057
        #val =  0.32876152559048755
        val = 0.47686969885505276
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        val = 0.40215145054471996
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        val = 0.1763164690262148
        #@time plaq_t = heatbathtest_2D(NX,NT,β,NC)
        @time plaq_ave = heatbathtest_2D(NX,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
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
        #val = 0.6023531251990353
        val = 0.60876803568248
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        val = 0.5618833345986648
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        val = 0.19494730324923296
        @time plaq_ave = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
        @test abs(plaq_ave-val)/abs(val) < eps
        #@test abs(plaq_t-val)/abs(val) < eps
    end



end


