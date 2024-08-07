using Gaugefields

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

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 1000
    for itrj = 1:numhb

        heatbath!(U,hnew)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        poly = calculate_Polyakov_loop(U,temp1,temp2)

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
heatbathtest_4D(NX,NY,NZ,NT,β,NC)
