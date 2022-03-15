
using Random
function gradientflow_test_4D(NX,NY,NZ,NT,NC)
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

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
    #"Reproducible"

    temp1 = similar(U[1])
    temp2 = similar(U[1])

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
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    g = Gradientflow(U,eps = 0.01)

    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end

    return plaq_t

end


function gradientflow_test_2D(NX,NT,NC)
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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

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
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    g = Gradientflow(U,eps = 0.01)

    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end

    return plaq_t

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
        val  =0.9440125563836135
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
        @test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        #val  =0.9440125563836135
        #val = 0.5385142466966718
        val = 0.8786515255315753
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
        @test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        #val = 0.1904815857904191
        val = 0.7301232810349298
        @time plaq_t =gradientflow_test_4D(NX,NY,NZ,NT,NC)
        @test abs(plaq_t-val)/abs(val) < eps
    end



end


println("2D system")
@testset "2D" begin
    NX = 4
    #NY = 4
    #NZ = 4
    NT = 4
    Nwing = 0
    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =0.6414596466929057
        val =  0.9768786716327604
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        @test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        val = 0.9656356864814539
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        @test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        val = 0.8138836242603148
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
        @test abs(plaq_t-val)/abs(val) < eps
    end


end