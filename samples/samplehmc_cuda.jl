using CUDA
using Gaugefields
using LinearAlgebra
using Random


function MDtest!(gauge_action,U,Dim,Ucpu,gauge_actioncpu)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    tempcpu1 = similar(Ucpu[1])
    tempcpu2 = similar(Ucpu[1])

    pcpu = initialize_TA_Gaugefields(Ucpu) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 


    Random.seed!(123)

    plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("initial plaq: $plaq_t")

    #substitute_U!(Ucpu,U)
    #plaqcpu_t = calculate_Plaquette(Ucpu,tempcpu1,tempcpu2)*factor
    #println("initial plaq in cpu: $plaqcpu_t")
    println("MDsteps $MDsteps")

    numtrj = 50
    for itrj = 1:numtrj
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,Ucpu,gauge_actioncpu,pcpu)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,Ucpu,gauge_actioncpu,pcpu)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    for μ=1:Dim
        pworkcpu = gauss_distribution(prod(NN)*(NC^2-1)) 
        pwork =CUDA.CuArray(pworkcpu)

        substitute_U!(p[μ],pwork)
        #substitute_U!(pcpu[μ],pworkcpu)
    end

    #gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    #println("Sold $Sold")
    #substitute_U!(Ucpu,U)
    #Soldcpu = calc_action(gauge_actioncpu,Ucpu,p)
    #println("Soldcpu $Soldcpu")

    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        #println(itrj,"/$MDsteps")
        U_update!(U,p,0.5,Δτ,Dim,gauge_action,Ucpu,gauge_actioncpu,pcpu)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action,Ucpu,gauge_actioncpu,pcpu)

        #substitute_U!(Ucpu,U)
        #P_update!(Ucpu,pcpu,1.0,Δτ,Dim,gauge_actioncpu,Ucpu,gauge_actioncpu,pcpu)
        #display(p[1].a[:,1,1])
        #display(pcpu[1].a[:,1,1,1,1])
        #error("p")

        U_update!(U,p,0.5,Δτ,Dim,gauge_action,Ucpu,gauge_actioncpu,pcpu)

        #error("1step")
    end
    Snew = calc_action(gauge_action,U,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action,Ucpu,gauge_actioncpu,pcpu)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    #tempscpu = get_temporary_gaugefields(gauge_actioncpu)
    #temp1cpu = tempscpu[1]
    #temp2cpu = tempscpu[2]
    #expUcpu = tempscpu[3]
    #Wcpu = tempscpu[4]

    for μ=1:Dim
        #display(p[μ].a[:,1,1])
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        #println("expU")
        #display(expU.U[:,:,1,1])

        #display(pcpu[μ].a[:,1,1,1,1])
        #exptU!(expUcpu,ϵ*Δτ,pcpu[μ],[temp1cpu,temp2cpu])
        #println("expUcpu")
        #display(expUcpu.U[:,:,1,1,1,1])

        #error("di")


        #println("U[μ]")
        #display(U[μ].U[:,:,1,1])
        mul!(W,expU,U[μ])

        #mul!(Wcpu,expUcpu,Ucpu[μ])
        #println("W")
        #display(W.U[:,:,1,1])
        #println("U")
        substitute_U!(U[μ],W)
        #substitute_U!(Ucpu[μ],Wcpu)
        #display(U[μ].U[:,:,1,1])
        #display(Ucpu[μ].U[:,:,1,1,1,1])
        
    end
    #error("upu")
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action,Ucpu,gauge_actioncpu,pcpu) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    #tempscpu = get_temporary_gaugefields(gauge_actioncpu)
    #dSdUμcpu = tempscpu[end]

    #substitute_U!(Ucpu,U)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        #display(dSdUμ.U[:,:,1,1])
        #calc_dSdUμ!(dSdUμcpu,gauge_actioncpu,μ,Ucpu)
        ##display(dSdUμcpu.U[:,:,1,1,1,1])
        #error("p")
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end

function test1()
    NX = 16
    NY = 16
    NZ = 16
    NT = 16
    Nwing = 0
    Dim = 4
    NC = 3

    #U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    NN = [NX,NY,NZ,NT]
    blocks = [4,4,8,8]
    #blocks = [8,8,8,8]

    U  =Initialize_Gaugefields(
        NC,
        Nwing,
        NN...;
        condition="cold",
        cuda=true,
        blocks)

    Ucpu  =Initialize_Gaugefields(
        NC,
        Nwing,
        NN...;
        condition="hot")


        #=
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


    substitute_U!(Ucpu,U)
    plaq = calculate_Plaquette(Ucpu, tempcpu[1], tempcpu[2]) * factor
    unused!(tempcpu)
    println("$plaq # plaq")


    return
    =#

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 6.0/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    gauge_actioncpu = GaugeAction(Ucpu)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 6.0/2
    push!(gauge_actioncpu,β,plaqloop)

    MDtest!(gauge_action,U,Dim,Ucpu,gauge_actioncpu)

end

function gauss_distribution(nv) 
    variance = 1
    nvh = div(nv,2)
    granf = zeros(Float64,nv)
    for i=1:nvh
        rho = sqrt(-2*log(rand())*variance)
        theta = 2pi*rand()
        granf[i] = rho*cos(theta)
        granf[i+nvh] = rho*sin(theta)
    end
    if 2*nvh == nv
        return granf
    end

    granf[nv] = sqrt(-2*log(rand())*variance) * cos(2pi*rand())
    return granf
end


test1()

