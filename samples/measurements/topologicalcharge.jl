# Topological charge (plaquette, clover, and improved definitions) along the
# Wilson gradient flow of the stored 8^4 SU(3) configuration conf_08080808.ildg,
# built from Wilson loops (Gaugefields v1).
# Run from any directory: julia -t 4 --project=. topologicalcharge.jl
# For production measurements, QCDMeasurements.jl provides these observables;
# see docs/src/measurements.md.
import JACC
JACC.@init_backend

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
            evaluate_gaugelinks!(temps[1],loops_μν[μ,ν],U,temps[2:5]) # needs four work fields
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
        rho=-rho
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

function main()
    Dim = 4
    U = gauge_configuration((8, 8, 8, 8); colors=3)
    filename = joinpath(@__DIR__, "conf_08080808.ildg")
    load_configuration!(U, filename; format=:ildg)

    temps = [similar(U[1]) for _ in 1:5]
    temp_UμνTA = Matrix{typeof(U[1])}(undef, Dim, Dim)
    for μ in 1:Dim
        for ν in 1:Dim
            temp_UμνTA[μ, ν] = similar(U[1])
        end
    end

    println("plaquette = ", measure_plaquette(U))

    dt = 0.01
    flow = gradient_flow(U; steps=1, step_size=dt)
    for itrj in 1:10
        flow!(U, flow)
        plaq = measure_plaquette(U)
        Qplaq = calculate_topological_charge_plaq(U, temp_UμνTA, temps)
        Qclover = calculate_topological_charge_clover(U, temp_UμνTA, temps)
        Qimproved = calculate_topological_charge_improved(U, temp_UμνTA, Qclover, temps)
        println("$itrj $dt $plaq $Qplaq $Qclover $Qimproved # itrj dt plaq Qplaq Qclover Qimproved")
    end
end

main()
