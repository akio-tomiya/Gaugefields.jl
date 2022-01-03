function Init_cold_4D(NX,NY,NZ,NT,Nwing,NC)
    Dim = 4

    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end
    

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)

    #factor = 2/(U[1].NV*4*3*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    return plaq_t
    

end

function Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
    Random.seed!(123)
    Dim = 4

    u1 = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
    end
    

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)

    #factor = 2/(U[1].NV*4*3*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    return plaq_t

end

function Init_ildg_4D(NX,NY,NZ,NT,Nwing,NC,filename)
    Dim = 4
    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)

    ildg = ILDG(filename)
    i = 1
    for μ=1:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
        #U[μ] = IdentityGauges(NC,NX,NY,NZ,NT,Nwing)
    end
    L = [NX,NY,NZ,NT]
    load_gaugefield!(U,i,ildg,L,NC)

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    return plaq_t

end

@testset "cold start" begin
    println("cold start")
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
end

@testset "hot start" begin
    println("hot start")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1

    @testset "NC=2" begin
        NC = 2
        println("NC = $NC")
        plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
        @test plaq_t == -0.05371251492617195
    end

    @testset "NC=3" begin
        NC = 3
        println("NC = $NC")
        plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
        @test plaq_t == 0.0015014233929744197
    end

    @testset "NC=4" begin
        NC = 4
        println("NC = $NC")
        plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
        @test plaq_t == -0.004597227507817238
    end

    @testset "NC=5" begin
        NC = 5
        println("NC = $NC")
        plaq_t = Init_hot_4D(NX,NY,NZ,NT,Nwing,NC)
        @test plaq_t == 0.0037580826460029506
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
        @test plaq_t == 0.2791463211708036
    end
end

@testset "File start" begin
    @testset "4444 SU(2)" begin
        Dim = 4
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        NC = 2
        Nwing = 1
        filename = "./data/conf_00000100_4444nc2.ildg"
        plaq_t = Init_ildg_4D(NX,NY,NZ,NT,Nwing,NC,filename)
        
        println("plaq_t = $plaq_t")
        @test plaq_t == 0.6684748868359871
    end
end