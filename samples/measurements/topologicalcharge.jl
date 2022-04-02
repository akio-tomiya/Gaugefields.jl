using Gaugefields
using Wilsonloop
using LinearAlgebra

function calculate_topological_charge_plaq(U::Array{T,1},temp_UμνTA,temps) where T
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA,"plaq",U,temps)
    Q = calc_Q(UμνTA,numofloops,U)
    return Q
end

function calculate_topological_charge_clover(U::Array{T,1},temp_UμνTA,temps) where T 
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA,"clover",U,temps)
    Q = calc_Q(UμνTA,numofloops,U)
    return Q
end

function calculate_topological_charge_improved(U::Array{T,1},temp_UμνTA,Qclover,temps) where T 
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA,"rect",U,temps)
    Qrect = 2*calc_Q(UμνTA,numofloops,U)
    c1 = -1/12
    c0 = 5/3
    Q = c0*Qclover + c1*Qrect
    return Q
end

function calc_Q(UμνTA,numofloops,U::Array{<: AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    Q = 0.0
    if Dim == 4
        ε(μ,ν,ρ,σ) = epsilon_tensor(μ,ν,ρ,σ)  
    else
        error("Dimension $Dim is not supported")
    end
    for μ=1:Dim
        for ν=1:Dim
            if ν == μ
                continue
            end
            Uμν = UμνTA[μ,ν]                 
            for ρ =1:Dim
                for σ=1:Dim
                    if ρ == σ
                        continue
                    end
                    Uρσ = UμνTA[ρ,σ]
                    s = tr(Uμν,Uρσ)
                    Q += ε(μ,ν,ρ,σ)*s/numofloops^2
                end
            end
        end
    end

    return -real(Q)/(32*(π^2))
end


function calc_UμνTA!(temp_UμνTA,name::String,U::Array{<: AbstractGaugefields{NC,Dim},1},temps) where {NC,Dim}
    loops_μν,numofloops = calc_loopset_μν_name(name,Dim)
    calc_UμνTA!(temp_UμνTA,loops_μν,U,temps)
    return numofloops
end


function calc_UμνTA!(UμνTA,loops_μν,U::Array{<: AbstractGaugefields{NC,Dim},1},temps) where {NC,Dim}
    for μ=1:Dim
        for ν=1:Dim
            if ν == μ
                continue
            end
            evaluate_gaugelinks!(temps[1],loops_μν[μ,ν],U,temps[2:3])
            Traceless_antihermitian!(UμνTA[μ,ν],temps[1])
        end
    end
    return 
end

function calc_loopset_μν_name(name,Dim)
    loops_μν= Array{Vector{Wilsonline{Dim}},2}(undef,Dim,Dim)
    if name == "plaq"
        numofloops = 1
        for μ=1:Dim
            for ν=1:Dim
                loops_μν[μ,ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                plaq = make_plaq(μ,ν,Dim=Dim)
                push!(loops_μν[μ,ν],plaq)
            end
        end
    elseif name == "clover"
        numofloops = 4
        for μ=1:Dim
            for ν=1:Dim
                loops_μν[μ,ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                loops_μν[μ,ν] = make_cloverloops(μ,ν,Dim=Dim)
            end
        end
    elseif name == "rect"
        numofloops = 8
        for μ=1:4
            for ν=1:4
                if ν == μ
                    continue
                end
                loops = Wilsonline{Dim}[]
                loop_righttop = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)])
                loop_lefttop = Wilsonline([(ν,1),(μ,-2),(ν,-1),(μ,2)])
                loop_rightbottom = Wilsonline([(ν,-1),(μ,2),(ν,1),(μ,-2)])
                loop_leftbottom= Wilsonline([(μ,-2),(ν,-1),(μ,2),(ν,1)])
                push!(loops,loop_righttop)
                push!(loops,loop_lefttop)
                push!(loops,loop_rightbottom)
                push!(loops,loop_leftbottom)

                loop_righttop = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)])
                loop_lefttop = Wilsonline([(ν,2),(μ,-1),(ν,-2),(μ,1)])
                loop_rightbottom = Wilsonline([(ν,-2),(μ,1),(ν,2),(μ,-1)])
                loop_leftbottom= Wilsonline([(μ,-1),(ν,-2),(μ,1),(ν,2)])
                push!(loops,loop_righttop)
                push!(loops,loop_lefttop)
                push!(loops,loop_rightbottom)
                push!(loops,loop_leftbottom)

                loops_μν[μ,ν] = loops
            end
        end
    else
        error("$name is not supported")
    end
    return loops_μν,numofloops
end

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


#topological charge
function epsilon_tensor(mu::Int,nu::Int,rho::Int,sigma::Int) 
    sign=1 # (3) 1710.09474 extended epsilon tensor
    if mu < 0
        sign*=-1
        mu=-mu
    end
    if nu < 0
        sign*=-1
        nu=-nu
    end
    if rho < 0
        sign*=-1
        rh=-rho
    end
    if sigma < 0
        sign*=-1
        sigma=-sigma
    end
    epsilon = zeros(Int,4,4,4,4)
    epsilon[ 1, 2, 3, 4 ] = 1
    epsilon[ 1, 2, 4, 3 ] = -1
    epsilon[ 1, 3, 2, 4 ] = -1
    epsilon[ 1, 3, 4, 2 ] = 1
    epsilon[ 1, 4, 2, 3 ] = 1
    epsilon[ 1, 4, 3, 2 ] = -1
    epsilon[ 2, 1, 3, 4 ] = -1
    epsilon[ 2, 1, 4, 3 ] = 1
    epsilon[ 2, 3, 1, 4 ] = 1
    epsilon[ 2, 3, 4, 1 ] = -1
    epsilon[ 2, 4, 1, 3 ] = -1
    epsilon[ 2, 4, 3, 1 ] = 1
    epsilon[ 3, 1, 2, 4 ] = 1
    epsilon[ 3, 1, 4, 2 ] = -1
    epsilon[ 3, 2, 1, 4 ] = -1
    epsilon[ 3, 2, 4, 1 ] = 1
    epsilon[ 3, 4, 1, 2 ] = 1
    epsilon[ 3, 4, 2, 1 ] = -1
    epsilon[ 4, 1, 2, 3 ] = -1
    epsilon[ 4, 1, 3, 2 ] = 1
    epsilon[ 4, 2, 1, 3 ] = 1
    epsilon[ 4, 2, 3, 1 ] = -1
    epsilon[ 4, 3, 1, 2 ] = -1
    epsilon[ 4, 3, 2, 1 ] = 1
    return epsilon[mu,nu,rho,sigma]*sign
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
    temp_UμνTA= Matrix{typeof(U[1])}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            temp_UμνTA[μ,ν] = similar(U[1])
        end
    end

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println(" plaq_t = $plaq_t")

    dt = 0.01

    g = Gradientflow(U,eps = dt)

    for itrj=1:10
        flow!(U,g)
        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        Qplaq = calculate_topological_charge_plaq(U,temp_UμνTA,[temp1,temp2,temp3])
        Qclover = calculate_topological_charge_clover(U,temp_UμνTA,[temp1,temp2,temp3])
        Qimproved= calculate_topological_charge_improved(U,temp_UμνTA,Qclover,[temp1,temp2,temp3])
        println("$itrj $dt $plaq_t $Qplaq $Qclover $Qimproved # itrj dt plaq Qplaq Qclover Qimproved ")
    end




end
test()