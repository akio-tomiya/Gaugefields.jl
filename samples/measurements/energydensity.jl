# Clover energy density along the Wilson gradient flow of the stored 8^4 SU(3)
# configuration conf_08080808.ildg, built from Wilson loops (Gaugefields v1).
# Run from any directory: julia -t 4 --project=. energydensity.jl
# For production measurements, QCDMeasurements.jl provides these observables;
# see docs/src/measurements.md.
import JACC
JACC.@init_backend

using Gaugefields
using Wilsonloop
using LinearAlgebra

function make_cloverloops(μ, ν; Dim=4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)])
    loop_lefttop = Wilsonline([(ν, 1), (μ, -1), (ν, -1), (μ, 1)])
    loop_rightbottom = Wilsonline([(ν, -1), (μ, 1), (ν, 1), (μ, -1)])
    loop_leftbottom = Wilsonline([(μ, -1), (ν, -1), (μ, 1), (ν, 1)])
    push!(loops, loop_righttop)
    push!(loops, loop_lefttop)
    push!(loops, loop_rightbottom)
    push!(loops, loop_leftbottom)
    return loops
end

function cloverloops(Dim)
    loops_μν = Matrix{Vector{Wilsonline{Dim}}}(undef, Dim, Dim)
    for μ in 1:Dim
        for ν in 1:Dim
            loops_μν[μ, ν] = Wilsonline{Dim}[]
            if ν == μ
                continue
            end
            loops_μν[μ, ν] = make_cloverloops(μ, ν, Dim=Dim)
        end
    end
    return loops_μν
end

# Wmat[μ,ν] = traceless anti-Hermitian part of the clover sum, an untraced field.
function make_energy_density!(Wmat, loops_μν, U, temps)
    Dim = length(U)
    for μ in 1:Dim
        for ν in 1:Dim
            if μ == ν
                continue
            end
            evaluate_gaugelinks!(temps[1], loops_μν[μ, ν], U, temps[2:5])
            Traceless_antihermitian!(Wmat[μ, ν], temps[1])
        end
    end
    return
end

function make_energy_density_core(Wmat, Dim)
    W = 0.0 + 0.0im
    for μ in 1:Dim
        for ν in 1:Dim
            if μ == ν
                continue
            end
            W += -tr(Wmat[μ, ν], Wmat[μ, ν]) / 2
        end
    end
    return W
end

# E = (1/V) sum_x (1/2) sum_{μν} tr(F_μν F_μν), with F_μν from the clover
# (the factor 4^2 is the square of the number of leaves).
function calculate_energy_density(U, Wmat, loops_μν, temps)
    volume = prod(gauge_lattice_size(U))
    make_energy_density!(Wmat, loops_μν, U, temps)
    WL = make_energy_density_core(Wmat, length(U))
    return real(WL) / (volume * 4^2)
end

function main()
    Dim = 4
    U = gauge_configuration((8, 8, 8, 8); colors=3)
    filename = joinpath(@__DIR__, "conf_08080808.ildg")
    load_configuration!(U, filename; format=:ildg)

    # evaluate_gaugelinks! with a vector of loops needs four work fields.
    temps = [similar(U[1]) for _ in 1:5]
    Wmat = Matrix{typeof(U[1])}(undef, Dim, Dim)
    for μ in 1:Dim
        for ν in 1:Dim
            Wmat[μ, ν] = similar(U[1])
        end
    end
    loops_μν = cloverloops(Dim)

    println("plaquette = ", measure_plaquette(U))

    dt = 0.01
    flow = gradient_flow(U; steps=1, step_size=dt)
    for itrj in 1:10
        flow!(U, flow)
        plaq = measure_plaquette(U)
        e = calculate_energy_density(U, Wmat, loops_μν, temps)
        println("$itrj $dt $plaq $e # itrj dt plaq energy")
    end
end

main()
