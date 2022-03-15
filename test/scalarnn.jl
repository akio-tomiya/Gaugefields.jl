using LinearAlgebra
function test1(Nwing)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    #Nwing = 1
    Dim = 4
    NC = 3

    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 1
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    Uout = evaluate_GaugeAction_untraced(gauge_action,U)
    println(tr(Uout))

end

@testset "initialization" begin
    println("with Nwing")
    Nwing = 1
    test1(Nwing)
    println("without Nwing")
    Nwing = 0
    test1(Nwing)
end