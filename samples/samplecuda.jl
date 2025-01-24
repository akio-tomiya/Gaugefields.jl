using CUDA
using Gaugefields
using LinearAlgebra

function main()
    Nx = 16
    Ny = 16
    Nz = 16
    Nt = 16
    #L = [Nx,Ny,Nz,Nt]
    Nwing = 0
    NC = 3
    NN = [Nx,Ny,Nz,Nt]
    blocks = [4,4,4,4]

    U  =Initialize_Gaugefields(
        NC,
        Nwing,
        NN...;
        condition="cold",
        cuda=true,
        blocks)

    return

    Ucpu  =Initialize_Gaugefields(
        NC,
        Nwing,
        NN...;
        condition="hot")



    tempcpu = Temporalfields(Ucpu[1]; num=5)

    volume = prod(NN)
    dim = length(NN)
    factor = 1 / (binomial(dim, 2) * NC * volume)

    nsteps = 2
    numOR = 3
    β = 6.0
    for istep = 1:nsteps
        println("# istep = $istep")

        t = @timed begin

            heatbath!(Ucpu, tempcpu, β)
            unused!(tempcpu)

            for _ = 1:numOR

                overrelaxation!(Ucpu, tempcpu, β)
                unused!(tempcpu)

            end
        end
        plaq = calculate_Plaquette(Ucpu, tempcpu[1], tempcpu[2]) * factor
        unused!(tempcpu)
        println("$istep $plaq # plaq")

        unused!(tempcpu)
    end

    substitute_U!(U,Ucpu)

    temp = Temporalfields(U[1]; num=5)
    plaqgpu = calculate_Plaquette(U, temp[1], temp[2]) * factor
    println("$plaqgpu # plaqgpu")
    return

    
    substitute_U!(U,Ucpu)
    return

    println(typeof(U))

    


    plaq = calculate_Plaquette(U, temp[1], temp[2]) * factor
    println(plaq)
    return

    Ut = similar(U[1])
    mul!(Ut,U[1],shift_U(U[1],1)')
    display(Ut.U[:,:,1,1])
    return


    display(U[1].U[:,:,1,1]*U[1].U[:,:,1,1]')
    Ut = similar(U[1])
    mul!(Ut,U[1],U[1]')
    display(Ut.U[:,:,1,1])

    display(U[1].U[:,:,1,1]'*U[1].U[:,:,1,1]')
    Ut = similar(U[1])
    mul!(Ut,U[1]',U[1]')
    display(Ut.U[:,:,1,1])

    display(U[1].U[:,:,1,1]'*U[1].U[:,:,1,1])
    Ut = similar(U[1])
    mul!(Ut,U[1]',U[1])
    display(Ut.U[:,:,1,1])

    Ut = similar(U[1])
    mul!(Ut,U[1],shift_U(U[1],1))
    display(Ut.U[:,:,1,1])

    return
    display(U[1].U[:,:,1,1]'*U[1].U[:,:,1,1])

    display(U[2].U[:,:,1,1]'*U[2].U[:,:,1,1])
    #display(U)

    Ut = similar(U)

    substitute_U!(Ut,U)
    display(Ut[1].U[:,:,1,1]'*U[1].U[:,:,1,1])
    return

    U  =Initialize_Gaugefields(
        NC,
        Nwing,
        NN...;
        condition="hot")

    println(typeof(U))
    display(U[1].U[:,:,1,1,1,1]'*U[1].U[:,:,1,1,1,1])
    return

    U  =Initialize_Gaugefields(
        NC,
        Nwing,
        NN...;
        condition="hot",
        cuda=true,
        blocks)

    println(typeof(U))
    display(U)



end

main()