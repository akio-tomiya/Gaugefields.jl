using Gaugefields
function test()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    L = [NX,NY,NZ,NT]

    nn = CovNeuralnet()
    ρ = [0.1]
    layername = ["plaquette"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)

    ρ = [0.3]
    layername = ["plaquette"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)
    show(nn)

    Dim = 4
    Nwing = 1
    NC = 3
    u1 = RandomGauges(NC,Nwing,L...)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = RandomGauges(NC,Nwing,L...)
    end

    @time Uout,Uout_multi,_ = calc_smearedU(U,nn)
end
test()