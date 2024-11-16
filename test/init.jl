function Init_cold_4D(NX,NY,NZ,NT,Nwing,NC)
    Dim = 4

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    #factor = 2/(U[1].NV*4*3*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")
    return plaq_t
    

end

function Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
    Random.seed!(123)
    Dim = 4

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot") #for debug
    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
    

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    #factor = 2/(U[1].NV*4*3*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps)
    println("polyakov loop = $(real(poly)) $(imag(poly))")
    return plaq_t

end

function Init_ildg_4D(NX,NY,NZ,NT,Nwing,NC,filename)
    Random.seed!(123)
    Dim = 4

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    save_binarydata(U,filename)

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    ildg = ILDG(filename)
    i = 1
    L = [NX,NY,NZ,NT]
    load_gaugefield!(U,i,ildg,L,NC)

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    poly = calculate_Polyakov_loop(U,temps) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")
    return plaq_t

end

function Init_cold_2D(NX,NT,Nwing,NC)
    Dim = 2

    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "cold")
    
    
    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    #factor = 2/(U[1].NV*4*3*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps)
    println("polyakov loop = $(real(poly)) $(imag(poly))")
    return plaq_t
    

end

function Init_hot_2D(NX,NT,Nwing,NC)
    Random.seed!(123)
    Dim = 2

    #U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot")    
    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    #factor = 2/(U[1].NV*4*3*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps)
    println("polyakov loop = $(real(poly)) $(imag(poly))")
    return plaq_t

end


@testset "cold start" begin
    println("cold start")
    println("4D system")
    @testset "4D" begin
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        Nwing = 1
        
        @testset "NC=2" begin
            NC = 2
            println("NC = $NC")
            plaq_t = Init_cold_4D(NX,NY,NZ,NT,Nwing,NC)

            @test plaq_t == one(plaq_t)
        end

        @testset "NC=3" begin
            NC = 3
            println("NC = $NC")
            plaq_t = Init_cold_4D(NX,NY,NZ,NT,Nwing,NC)
            @test plaq_t == one(plaq_t)
        end

        @testset "NC=4" begin
            NC = 4
            println("NC = $NC")
            plaq_t = Init_cold_4D(NX,NY,NZ,NT,Nwing,NC)
            @test plaq_t == one(plaq_t)
        end

        @testset "NC=5" begin
            NC = 5
            println("NC = $NC")
            plaq_t = Init_cold_4D(NX,NY,NZ,NT,Nwing,NC)
            @test plaq_t == one(plaq_t)
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
            NC = 2
            println("NC = $NC")
            plaq_t = Init_cold_2D(NX,NT,Nwing,NC)

            @test plaq_t == one(plaq_t)
        end

        @testset "NC=3" begin
            NC = 3
            println("NC = $NC")
            plaq_t = Init_cold_2D(NX,NT,Nwing,NC)
            @test plaq_t == one(plaq_t)
        end

        @testset "NC=4" begin
            NC = 4
            println("NC = $NC")
            plaq_t = Init_cold_2D(NX,NT,Nwing,NC)
            @test plaq_t == one(plaq_t)
        end

        @testset "NC=5" begin
            NC = 5
            println("NC = $NC")
            plaq_t = Init_cold_2D(NX,NT,Nwing,NC)
            @test plaq_t == one(plaq_t)
        end
    end
end

eps = 1e-8

@testset "hot start" begin
    println("hot start")
    println("4D system")
    @testset "4D" begin
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        Nwing = 1

        @testset "NC=2" begin
            NC = 2
            println("NC = $NC")
            plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
            #val = -0.007853743153861802
            val = 0.0002565633289190507
            @test abs(plaq_t-val)/abs(val) < eps
        end

        @testset "NC=3" begin
            NC = 3
            println("NC = $NC")
            plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
            #val = 0.0015014233929744197
            val = 0.008449494077606137
            @test abs(plaq_t-val)/abs(val) < eps
        end

        @testset "NC=4" begin
            NC = 4
            println("NC = $NC")
            plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
            #val = -0.004597227507817238
            val =  0.0029731967994215515
            @test abs(plaq_t-val)/abs(val) < eps
        end

        @testset "NC=5" begin
            NC = 5
            println("NC = $NC")
            plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
            #val = 0.0037580826460029506
            val = -0.0005506443879866896
            @test abs(plaq_t-val)/abs(val) < eps
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
            NC = 2
            println("NC = $NC")
            plaq_t = Init_hot_2D(NX,NT,Nwing,NC)
            #val = 0.022601163616639157
            val  =-0.09978762099361757
            @test abs(plaq_t-val)/abs(val) < eps
        end

        @testset "NC=3" begin
            NC = 3
            println("NC = $NC")
            plaq_t = Init_hot_2D(NX,NT,Nwing,NC)
            #val = -0.04647124293538649
            val = 0.046263787757668186
            @test abs(plaq_t-val)/abs(val) < eps
        end

        @testset "NC=4" begin
            NC = 4
            println("NC = $NC")
            plaq_t = Init_hot_2D(NX,NT,Nwing,NC)
            #val = 0.07457370324173362
            val = 0.03150732915218983
            @test abs(plaq_t-val)/abs(val) < eps
        end

        @testset "NC=5" begin
            NC = 5
            println("NC = $NC")
            plaq_t = Init_hot_2D(NX,NT,Nwing,NC)
            #val = -0.013511504030861661
            val = -0.03564420387393341
            @test abs(plaq_t-val)/abs(val) < eps
        end
    end

end

@testset "one instanton" begin
    @testset "4D" begin
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        NC = 2
        Nwing = 1
        U = Oneinstanton(NC,NX,NY,NZ,NT,Nwing)

        comb = 6
        factor = 1/(comb*U[1].NV*U[1].NC)

        temp1 = similar(U[1])
        temp2 = similar(U[1])

        @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("plaq_t = $plaq_t")
        val =  0.9796864531099871
        @test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "2D" begin
        Dim = 2
        NX = 4
        #NY = 4
        #NZ = 4
        NT = 4
        NC = 2
        Nwing = 1
        U = Oneinstanton(NC,NX,NT,Nwing)

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

        temp1 = similar(U[1])
        temp2 = similar(U[1])

        @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("plaq_t = $plaq_t")
        val = 0.9300052284270868
        @test abs(plaq_t-val)/abs(val) < eps
    end
end

#=
@testset "File start" begin
    @testset "4444 SU(2)" begin
        Dim = 4
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        NC = 2
        Nwing = 1
        #U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")

        filename = "./data/conf_00000100_4444_test.ildg"
        plaq_t = Init_ildg_4D(NX,NY,NZ,NT,Nwing,NC,filename)
        
        println("plaq_t = $plaq_t")
        val = -0.007853743153861798
        @test abs(plaq_t-val)/abs(val) < eps
    end
end
=#
