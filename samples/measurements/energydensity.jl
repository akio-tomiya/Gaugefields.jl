using Gaugefields
using Wilsonloop
using LinearAlgebra

function make_cloverloops(μ,ν;Dim=4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)])
    loop_lefttop = Wilsonline([(ν,1),(μ,-1),(ν,-1),(μ,1)])
    loop_rightbottom = Wilsonline([(ν,-1),(μ,1),(ν,1),(μ,-1)])
    loop_leftbottom= Wilsonline([(μ,-1),(ν,-1),(μ,1),(ν,1)])
    push!(loops,loop_righttop)
    push!(loops,loop_lefttop)
    push!(loops,loop_rightbottom)
    push!(loops,loop_leftbottom)
    return loops
end

function cloverloops(Dim)
    loops_μν= Matrix{Vector{Wilsonline{Dim}}}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            loops_μν[μ,ν] = Wilsonline{Dim}[]
            if ν == μ
                continue
            end
            loops_μν[μ,ν] = make_cloverloops(μ,ν,Dim=Dim)
        end
    end
    return  loops_μν
end

function make_energy_density!(Wmat,U::Vector{<: AbstractGaugefields{NC,Dim}},temps) where {NC,Dim}
    W_operator = cloverloops(Dim)
    calc_wilson_loop!(Wmat,W_operator,U,temps)
    return 
end


function calc_wilson_loop!(W,W_operator,U::Vector{<: AbstractGaugefields{NC,Dim}},temps) where {NC,Dim}
    for μ=1:Dim
        for ν=1:Dim
            if μ == ν
                continue
            end
            evaluate_gaugelinks!(W[μ,ν],W_operator[μ,ν],U,temps)
            W[μ,ν] = Traceless_antihermitian(W[μ,ν])
        end
    end
    return 
end

function  make_energy_density_core(Wmat::Matrix{<: AbstractGaugefields{NC,Dim}}) where {NC,Dim}
    @assert Dim == 4
    W = 0.0 + 0.0im
    for μ=1:Dim # all directions
        for ν=1:Dim
            if μ == ν
                continue
            end
            W += -tr(Wmat[μ,ν],Wmat[μ,ν])/2
        end
    end
    return W
end


function calculate_energy_density(U::Array{T,1}, Wmat,temps) where T <: AbstractGaugefields
    # Making a ( Ls × Lt) Wilson loop operator for potential calculations
    WL = 0.0+0.0im
    NV = U[1].NV
    NC = U[1].NC
    make_energy_density!(Wmat,U,temps) # make wilon loop operator and evaluate as a field, not traced.
    WL =  make_energy_density_core(Wmat) # tracing over color and average over spacetime and x,y,z.
    NDir = 4.0*3.0/2 # choice of 2 axis from 4.
    return real(WL)/(NV*4^2)
end

function test()
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0
    NC = 3
    
    Dim = 4
    
    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    filename="./conf_08080808.ildg"
    
    ildg = ILDG(filename)
    i = 1
    L = [NX,NY,NZ,NT]
    load_gaugefield!(U,i,ildg,L,NC)

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    β = 5.7
    W_temp = Matrix{typeof(U[1])}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            W_temp[μ,ν] = similar(U[1])
        end
    end

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println(" plaq_t = $plaq_t")

    dt = 0.01

    g = Gradientflow(U,eps = dt)

    for itrj=1:10
        flow!(U,g)
        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        e = calculate_energy_density(U, W_temp,[temp1,temp2])
        println("$itrj $dt $plaq_t $e # itrj dt plaq energy")
    end




end
test()