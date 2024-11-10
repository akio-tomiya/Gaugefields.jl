using Revise
#using LatticeQCD
using Wilsonloop
using Gaugefields
using LinearAlgebra

function main()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    NC = 3

    flux=[1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]

    println("Flux is ", flux)

    U1 = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    B1 = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")

    println("Initial conf of B at [1,2][2,2,:,:,NZ,NT]")
    display(B1[1,2][2,2,:,:,NZ,NT])

    temps = typeof(U1[1])[]
    for i=1:10
        push!(temps,similar(U1[1]))
    end
    Uloop = similar(U1[1])
    UloopB = similar(U1[1])

    #loop = [(1,+1),(2,+1),(1,-1),(2,-1)]
    loop = [(2, -1), (1, -1), (2, 1), (1, 1)]
    #loop = [(1, -1), (2, 1), (1, 1), (2, -1)]
    #loop = [(2, -1), (1, 1), (2, 1), (1, -1)]
    #loop = [(1, -1), (2, -1), (1, 1), (2, 1)]
    #loop = [(1, 1), (1, -1), (3, -1), (2, -1), (1, -1), (2, 1), (1, 1), (3, 1)]
    println(loop)
    w = Wilsonline(loop)
    println("P: ")
    show(w)


    #println("Evaluate gaugelinks")
    #Gaugefields.evaluate_gaugelinks!(Uloop, w, U1, temps)
    #display(Uloop[2,2,:,:,NZ,NT])
    #println("Evaluate gaugelinks with Bfield")
    #Gaugefields.evaluate_gaugelinks!(UloopB, w, U1, B1, temps)
    #display(UloopB[2,2,:,:,NZ,NT])
    #println(det(UloopB[:,:,NX,NY,NZ,NT]))

    #println("gaugelinks==gaugeliksB : ", Uloop[:,:,:,:,:,:]==UloopB[:,:,:,:,:,:])

    println("Evaluate Bplaquettes for P")
    Gaugefields.evaluate_Bplaquettes!(UloopB,  w, B1, temps)
    display(UloopB[2,2,:,:,NZ,NT])

    println("Identical to original B? :")
    println(B1[1,2][:,:,:,:,:,:]==UloopB[:,:,:,:,:,:])


    println("Staples :")
    for i = 1:4
        println(i, "th staple")
        V = make_staple(w,i)
        show(V)
        Gaugefields.evaluate_gaugelinks!(UloopB,  V, U1, B1, temps)
        display(UloopB[2,2,:,:,NZ,NT])
    end


end
main()

