module heatbath_module
using LinearAlgebra
using MPI
using StaticArrays: MMatrix
import JACC
import LatticeMatrices:
    Philox4x32,
    RNGStreamKey,
    SiteRNG,
    SiteRNGAlgorithm,
    delinearize,
    global_site_coordinates,
    global_site_id,
    mark_halo_dirty!,
    rand_bounded,
    rand_uniform,
    rand_uniform_open,
    set_halo!,
    site_rng

import ..AbstractGaugefields_module:
    normalize3!,
    normalizeN!,
    AbstractGaugefields,
    evaluate_gaugelinks_evenodd!,
    Gaugefields_2D_MPILattice,
    Gaugefields_3D_MPILattice,
    Gaugefields_4D_MPILattice,
    map_U!,
    map_U_sequential!
import Wilsonloop: get_direction, get_position, loops_staple
import ..GaugeAction_module:
    GaugeAction,
    calc_dSdUμ!,
    evaluate_staple_eachindex!
using InteractiveUtils
import ..Temporalfields_module: Temporalfields, unused!, get_temp

include("portable/rng_protocol.jl")
include("portable/kernels.jl")
include("rng_adapters.jl")


"""
    Heatbath(U, beta; ITERATION_MAX=10^5, seed=0, sweep=0,
        overrelaxation_sweep=sweep, rng_algorithm=Philox4x32())

Reusable storage and parameters for heatbath sweeps. On an SU(N)
LatticeMatrices-backed field in 2D, 3D, or 4D, `seed`, `sweep`, and
`rng_algorithm` select the device-safe site streams. Successful `heatbath!`
and `overrelaxation!` calls advance `h.sweep` and
`h.overrelaxation_sweep`, respectively. The serial Gaugefields
implementations retain their legacy global-RNG behavior.
"""
mutable struct Heatbath{T,R<:SiteRNGAlgorithm}
    #_tempotal_gauges::Vector{T}
    _tempotal_gauges::Temporalfields{T}
    β::Float64
    ITERATION_MAX::Int64
    seed::UInt64
    sweep::UInt64
    overrelaxation_sweep::UInt64
    rng_algorithm::R


    function Heatbath(
        U::Array{T,1},
        β;
        ITERATION_MAX=10^5,
        seed::Integer=0,
        sweep::Integer=0,
        overrelaxation_sweep::Integer=sweep,
        rng_algorithm::R=Philox4x32(),
    ) where {T<:AbstractGaugefields,R<:SiteRNGAlgorithm}
        ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))
        _tempotal_gauges = Temporalfields(U[1], num=5)
        #_tempotal_gauges = Array{T,1}(undef, 5) # length >= 5
        #for i = 1:5
        #    _tempotal_gauges[i] = similar(U[1])
        #end
        return new{T,R}(
            _tempotal_gauges,
            β,
            ITERATION_MAX,
            UInt64(seed),
            UInt64(sweep),
            UInt64(overrelaxation_sweep),
            rng_algorithm,
        )
    end

end

"""
    HeatbathColoring(ncolors, coefficients)

Translation-invariant site coloring used by parallel general-action heatbath
and overrelaxation updates.  A global site `x` has color
`mod(sum(coefficients[d] * (x[d]-1)), ncolors)`.
"""
struct HeatbathColoring{Dim}
    ncolors::Int64
    coefficients::NTuple{Dim,Int64}
end

@inline function heatbath_site_color(
    coloring::HeatbathColoring{Dim},
    global_indices,
) where Dim
    value = Int64(0)
    @inbounds for d in 1:Dim
        value += coloring.coefficients[d] * (global_indices[d] - 1)
    end
    return mod(value, coloring.ncolors)
end

struct _CheckerboardHeatbathTarget
    target_even::Bool
end

struct _LinearHeatbathTarget{Dim}
    coloring::HeatbathColoring{Dim}
    target_color::Int64
end

@inline function _heatbath_site_selected(
    global_indices,
    target::_CheckerboardHeatbathTarget,
)
    coordinate_sum = 0
    @inbounds for d in eachindex(global_indices)
        coordinate_sum += global_indices[d]
    end
    return iseven(coordinate_sum) == target.target_even
end

@inline function _heatbath_site_selected(
    global_indices,
    target::_LinearHeatbathTarget,
)
    return heatbath_site_color(target.coloring, global_indices) ==
           target.target_color
end

"""
    Heatbath_update(U, gauge_action; seed=0, sweep=0,
                    overrelaxation_sweep=sweep,
                    rng_algorithm=Philox4x32(), coloring=:auto,
                    max_colors=256, ITERATION_MAX=10^5)

Construct a heatbath/overrelaxation updater for `gauge_action`. For a 2D, 3D,
or 4D LatticeMatrices-backed field, `coloring=:auto` derives a safe periodic
site coloring from the Wilson-loop staples. `coloring=:sequential` assigns a
distinct color to every global site and is an exact, deliberately slow
fallback. Parallel general-action updates support SU(N) for `N >= 2`.
"""
mutable struct Heatbath_update{Dim,T,R<:SiteRNGAlgorithm,C}
    _temporary_gaugefields::Temporalfields{T}# Vector{T}
    gauge_action::GaugeAction{Dim,T}
    ITERATION_MAX::Int64
    seed::UInt64
    sweep::UInt64
    overrelaxation_sweep::UInt64
    rng_algorithm::R
    colorings::C

    function Heatbath_update(
        U::Array{T,1},
        gauge_action;
        ITERATION_MAX=10^5,
        seed::Integer=0,
        sweep::Integer=0,
        overrelaxation_sweep::Integer=sweep,
        rng_algorithm::R=Philox4x32(),
        coloring=:auto,
        max_colors::Integer=256,
    ) where {T<:AbstractGaugefields,R<:SiteRNGAlgorithm}
        ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))
        max_colors > 0 || throw(ArgumentError("max_colors must be positive"))
        _temporary_gaugefields = Temporalfields(U[1], num=5)#Array{T,1}(undef, 5) # length >= 5
        Dim = length(U)
        colorings = _heatbath_action_colorings(
            U, gauge_action, coloring, Int(max_colors)
        )
        #for i = 1:5
        #    _temporary_gaugefields[i] = similar(U[1])
        #end
        C = typeof(colorings)
        return new{Dim,T,R,C}(
            _temporary_gaugefields,
            gauge_action,
            ITERATION_MAX,
            UInt64(seed),
            UInt64(sweep),
            UInt64(overrelaxation_sweep),
            rng_algorithm,
            colorings,
        )
    end
end

function _heatbath_action_colorings(U, gauge_action, coloring, max_colors)
    return nothing
end

function _heatbath_action_colorings(
    U::Array{<:Union{Gaugefields_2D_MPILattice,Gaugefields_3D_MPILattice,Gaugefields_4D_MPILattice},1},
    gauge_action::GaugeAction{Dim},
    coloring,
    max_colors,
) where Dim
    global_size = ntuple(d -> Int(U[1].U.gsize[d]), Dim)
    if coloring === :auto
        return heatbath_colorings(
            gauge_action, global_size; max_colors=max_colors
        )
    elseif coloring === :sequential
        strides = ntuple(
            d -> d == 1 ? 1 : prod(global_size[1:(d-1)]),
            Dim,
        )
        sequential = HeatbathColoring(prod(global_size), strides)
        return ntuple(_ -> sequential, Dim)
    elseif coloring isa NTuple{Dim,HeatbathColoring{Dim}}
        _validate_heatbath_colorings(
            gauge_action, global_size, coloring
        )
        return coloring
    end
    throw(ArgumentError(
        "coloring must be :auto, :sequential, or an NTuple{$Dim,HeatbathColoring{$Dim}}"
    ))
end

function _validate_heatbath_colorings(
    gauge_action::GaugeAction{Dim},
    global_size::NTuple{Dim,Int},
    colorings::NTuple{Dim,HeatbathColoring{Dim}},
) where Dim
    for direction in 1:Dim
        coloring = colorings[direction]
        coloring.ncolors > 0 || throw(ArgumentError(
            "direction-$direction coloring must have at least one color"
        ))
        for d in 1:Dim
            mod(
                coloring.coefficients[d] * global_size[d],
                coloring.ncolors,
            ) == 0 || throw(ArgumentError(
                "direction-$direction coloring is not periodic in dimension $d"
            ))
        end
        dependencies = _same_direction_action_dependencies(
            gauge_action, direction, global_size
        )
        for displacement in dependencies
            dot_value = sum(
                coloring.coefficients[d] * displacement[d]
                for d in 1:Dim
            )
            mod(dot_value, coloring.ncolors) != 0 || throw(ArgumentError(
                "direction-$direction coloring assigns dependent displacement " *
                "$displacement to the same color"
            ))
        end
    end
    return nothing
end

function _same_direction_action_dependencies(
    gauge_action::GaugeAction{Dim},
    direction,
    global_size,
) where Dim
    dependencies = Set{NTuple{Dim,Int64}}()
    for dataset in gauge_action.dataset
        iszero(dataset.β) && continue
        iszero(imag(dataset.β)) || throw(ArgumentError(
            "heatbath requires real GaugeAction coefficients; got $(dataset.β)"
        ))
        for staple in dataset.staples[direction]
            for link_index in 1:length(staple)
                link = staple[link_index]
                get_direction(link) == direction || continue
                displacement = get_position(link)
                periodic_displacement = ntuple(
                    d -> Int64(mod(displacement[d], global_size[d])),
                    Dim,
                )
                all(iszero, periodic_displacement) && throw(ArgumentError(
                    "the direction-$direction staple contains its target link " *
                    "at periodic displacement $displacement; this action/lattice " *
                    "combination has no single-link heatbath conditional"
                ))
                push!(dependencies, periodic_displacement)
            end
        end
    end
    return sort!(collect(dependencies))
end

function _find_linear_heatbath_coloring(
    dependencies::Vector{NTuple{Dim,Int64}},
    global_size::NTuple{Dim,Int},
    max_colors,
) where Dim
    isempty(dependencies) && return HeatbathColoring(1, ntuple(_ -> 0, Dim))

    for ncolors in 2:max_colors
        allowed_coefficients = ntuple(Dim) do d
            [
                coefficient for coefficient in 0:(ncolors-1)
                if mod(coefficient * global_size[d], ncolors) == 0
            ]
        end
        for coefficient_product in Iterators.product(allowed_coefficients...)
            coefficients = ntuple(
                d -> Int64(coefficient_product[d]), Dim
            )
            all(iszero, coefficients) && continue
            valid = true
            for displacement in dependencies
                dot_value = Int64(0)
                @inbounds for d in 1:Dim
                    dot_value += coefficients[d] * displacement[d]
                end
                if mod(dot_value, ncolors) == 0
                    valid = false
                    break
                end
            end
            valid && return HeatbathColoring(ncolors, coefficients)
        end
    end
    return nothing
end

"""
    heatbath_colorings(gauge_action, global_size; max_colors=256)

Construct a safe periodic linear coloring for each link direction by examining
same-direction link displacements in the action staples.  The method throws if
the target link reappears in its own staple through a periodic alias, or if no
compact coloring is found.  `coloring=:sequential` in [`Heatbath_update`](@ref)
is an exact but deliberately slow fallback.
"""
function heatbath_colorings(
    gauge_action::GaugeAction{Dim},
    global_size::NTuple{Dim,<:Integer};
    max_colors::Integer=256,
) where Dim
    gauge_action.hascovnet && throw(ArgumentError(
        "CovNeuralnet GaugeAction heatbath is not supported"
    ))
    sizes = ntuple(d -> Int(global_size[d]), Dim)
    all(>(0), sizes) || throw(ArgumentError("global lattice sizes must be positive"))

    return ntuple(Dim) do direction
        dependencies = _same_direction_action_dependencies(
            gauge_action, direction, sizes
        )
        result = _find_linear_heatbath_coloring(
            dependencies, sizes, Int(max_colors)
        )
        isnothing(result) && throw(ArgumentError(
            "no periodic linear heatbath coloring with at most $max_colors " *
            "colors was found for direction $direction; use " *
            "coloring=:sequential or increase max_colors"
        ))
        result
    end
end

const heatbath_factor = 2

function heatbath_update_eachsite_SU2!(
    A,
    μ,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
    h::Heatbath_update{Dim,T},
    mat_temps,
    indices...,
) where {NC,Dim,T}
    @assert NC == 2
    V = mat_temps[1]
    evaluate_staple_eachindex!(V, μ, h.gauge_action, U, view(mat_temps, 2:6), indices...) # length >= 5
    SU2update_KP!(A, V, heatbath_factor, NC, view(mat_temps, 7:8), h.ITERATION_MAX)
end

function heatbath!(
    U::Array{<:AbstractGaugefields{2,Dim},1},
    h::Heatbath_update{Dim,T},
) where {Dim,T}
    NC = 2
    nt = 8
    temps = Vector{Matrix{ComplexF64}}(undef, nt)
    for i = 1:nt
        temps[i] = zeros(ComplexF64, NC, NC)
    end

    for μ = 1:Dim
        mapfunc!(A, U, indices...) =
            heatbath_update_eachsite_SU2!(A, μ, U, h, temps, indices...)
        map_U_sequential!(U[μ], mapfunc!, U)
    end


end


function heatbath_update_eachsite_SU3!(
    A,
    μ,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
    h::Heatbath_update{Dim,T},
    mat_temps1,
    mat_temps2,
    indices...,
) where {NC,Dim,T}
    @assert NC == 3
    V = mat_temps1[1]
    evaluate_staple_eachindex!(V, μ, h.gauge_action, U, view(mat_temps1, 2:6), indices...) # length >= 5
    SU3update_matrix!(
        A,
        V,
        heatbath_factor,
        NC,
        view(mat_temps1, 7:11),
        mat_temps2,
        h.ITERATION_MAX,
    )
    #SU2update_KP!(A,V,1,NC,view(mat_temps,7:8),h.ITERATION_MAX)
end

function heatbath_update_eachsite_SUN!(
    A,
    μ,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
    h::Heatbath_update{Dim,T},
    mat_temps1,
    mat_temps2,
    indices...,
) where {NC,Dim,T}
    V = mat_temps1[1]
    evaluate_staple_eachindex!(V, μ, h.gauge_action, U, view(mat_temps1, 2:6), indices...) # length >= 5
    SUNupdate_matrix!(
        A,
        V,
        heatbath_factor,
        NC,
        view(mat_temps1, 7:11),
        mat_temps2,
        h.ITERATION_MAX,
    )
    #SU2update_KP!(A,V,1,NC,view(mat_temps,7:8),h.ITERATION_MAX)
end

function overrelaxation_update_eachsite_SUN!(
    A,
    μ,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
    h::Heatbath_update{Dim,T},
    mat_temps1,
    mat_temps2,
    indices...,
) where {NC,Dim,T}
    V = mat_temps1[1]
    evaluate_staple_eachindex!(V, μ, h.gauge_action, U, view(mat_temps1, 2:6), indices...) # length >= 5
    SUN_overrelaxation!(
        A,
        V,
        heatbath_factor,
        NC,
        view(mat_temps1, 7:11),
        mat_temps2,
        h.ITERATION_MAX,
    )
    #SU2update_KP!(A,V,1,NC,view(mat_temps,7:8),h.ITERATION_MAX)
end

function heatbath!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    h::Heatbath_update{Dim,T},
) where {Dim,T,NC}
    nt = 11
    temps = Vector{Matrix{ComplexF64}}(undef, nt)
    for i = 1:nt
        temps[i] = zeros(ComplexF64, NC, NC)
    end

    if NC != 2
        temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
        for i = 1:5
            temps3[i] = zeros(ComplexF64, NC, NC)
        end
    end

    for μ = 1:Dim
        mapfunc!(A, U, indices...) =
            heatbath_update_eachsite_SUN!(A, μ, U, h, temps, temps3, indices...)
        map_U_sequential!(U[μ], mapfunc!, U)
    end
end

function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},1},
    h::Heatbath_update{Dim},
) where Dim
    staple, staple_index = get_temp(h._temporary_gaugefields)
    current_sweep = h.sweep
    try
        for direction in eachindex(U)
            coloring = h.colorings[direction]
            for target_color in 0:(coloring.ncolors-1)
                # The preceding color changes same-direction links appearing
                # in this staple, so the weighted staple must be rebuilt.
                calc_dSdUμ!(staple, h.gauge_action, direction, U)
                heatbath_su2_sites!(
                    U[direction],
                    staple,
                    heatbath_factor,
                    coloring,
                    target_color;
                    seed=h.seed,
                    sweep=current_sweep,
                    direction,
                    rng_algorithm=h.rng_algorithm,
                    iteration_max=h.ITERATION_MAX,
                )
            end
        end
    finally
        unused!(h._temporary_gaugefields, staple_index)
    end
    h.sweep = current_sweep + one(current_sweep)
    return nothing
end

function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},1},
    h::Heatbath_update{Dim},
) where Dim
    staple, staple_index = get_temp(h._temporary_gaugefields)
    current_sweep = h.sweep
    try
        for direction in eachindex(U)
            coloring = h.colorings[direction]
            for target_color in 0:(coloring.ncolors-1)
                calc_dSdUμ!(staple, h.gauge_action, direction, U)
                heatbath_su3_sites!(
                    U[direction],
                    staple,
                    heatbath_factor,
                    coloring,
                    target_color;
                    seed=h.seed,
                    sweep=current_sweep,
                    direction,
                    rng_algorithm=h.rng_algorithm,
                    iteration_max=h.ITERATION_MAX,
                )
            end
        end
    finally
        unused!(h._temporary_gaugefields, staple_index)
    end
    h.sweep = current_sweep + one(current_sweep)
    return nothing
end

function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},1},
    h::Heatbath_update{Dim},
) where {NC,Dim}
    NC >= 2 || throw(ArgumentError("heatbath requires NC >= 2; got NC=$NC"))
    staple, staple_index = get_temp(h._temporary_gaugefields)
    current_sweep = h.sweep
    try
        for direction in eachindex(U)
            coloring = h.colorings[direction]
            for target_color in 0:(coloring.ncolors-1)
                calc_dSdUμ!(staple, h.gauge_action, direction, U)
                _heatbath_sun_sites!(
                    U[direction],
                    staple,
                    heatbath_factor,
                    coloring,
                    target_color;
                    seed=h.seed,
                    sweep=current_sweep,
                    direction,
                    rng_algorithm=h.rng_algorithm,
                    iteration_max=h.ITERATION_MAX,
                )
            end
        end
    finally
        unused!(h._temporary_gaugefields, staple_index)
    end
    h.sweep = current_sweep + one(current_sweep)
    return nothing
end

function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},1},
    temps_g::Temporalfields,
    beta;
    ITERATION_MAX::Integer=10^5,
    seed::Integer=0,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where NC
    NC >= 2 || throw(ArgumentError("heatbath requires NC >= 2; got NC=$NC"))
    Dim = length(U)
    Dim in (2, 3, 4) || throw(DimensionMismatch(
        "MPILattice heatbath requires two, three, or four links"
    ))
    ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))

    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(Dim, direction)]
            for target_even in (true, false)
                color = target_even ? 0 : 1
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                _heatbath_sun_sites!(
                    U[direction],
                    staple,
                    beta,
                    _CheckerboardHeatbathTarget(target_even),
                    color;
                    seed,
                    sweep,
                    direction,
                    rng_algorithm,
                    iteration_max=ITERATION_MAX,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},1},
    h::Heatbath_update{Dim},
) where Dim
    staple, staple_index = get_temp(h._temporary_gaugefields)
    current_sweep = h.overrelaxation_sweep
    try
        for direction in eachindex(U)
            coloring = h.colorings[direction]
            for target_color in 0:(coloring.ncolors-1)
                calc_dSdUμ!(staple, h.gauge_action, direction, U)
                overrelaxation_su2_sites!(
                    U[direction],
                    staple,
                    coloring,
                    target_color;
                    seed=h.seed,
                    sweep=current_sweep,
                    direction,
                    rng_algorithm=h.rng_algorithm,
                )
            end
        end
    finally
        unused!(h._temporary_gaugefields, staple_index)
    end
    h.overrelaxation_sweep = current_sweep + one(current_sweep)
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},1},
    h::Heatbath_update{Dim},
) where Dim
    staple, staple_index = get_temp(h._temporary_gaugefields)
    current_sweep = h.overrelaxation_sweep
    try
        for direction in eachindex(U)
            coloring = h.colorings[direction]
            for target_color in 0:(coloring.ncolors-1)
                calc_dSdUμ!(staple, h.gauge_action, direction, U)
                overrelaxation_su3_sites!(
                    U[direction],
                    staple,
                    coloring,
                    target_color;
                    seed=h.seed,
                    sweep=current_sweep,
                    direction,
                    rng_algorithm=h.rng_algorithm,
                )
            end
        end
    finally
        unused!(h._temporary_gaugefields, staple_index)
    end
    h.overrelaxation_sweep = current_sweep + one(current_sweep)
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},1},
    temps_g::Temporalfields,
    beta;
    ITERATION_MAX::Integer=10^5,
    seed::Integer=0,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where NC
    NC >= 2 || throw(ArgumentError(
        "overrelaxation requires NC >= 2; got NC=$NC"
    ))
    Dim = length(U)
    Dim in (2, 3, 4) || throw(DimensionMismatch(
        "MPILattice overrelaxation requires two, three, or four links"
    ))
    ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))

    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(Dim, direction)]
            for target_even in (true, false)
                color = target_even ? 0 : 1
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                _overrelaxation_sites!(
                    U[direction],
                    staple,
                    _CheckerboardHeatbathTarget(target_even),
                    color;
                    seed,
                    sweep,
                    direction,
                    rng_algorithm,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},1},
    h::Heatbath_update{Dim},
) where {NC,Dim}
    NC >= 2 || throw(ArgumentError("overrelaxation requires NC >= 2; got NC=$NC"))
    staple, staple_index = get_temp(h._temporary_gaugefields)
    current_sweep = h.overrelaxation_sweep
    try
        for direction in eachindex(U)
            coloring = h.colorings[direction]
            for target_color in 0:(coloring.ncolors-1)
                calc_dSdUμ!(staple, h.gauge_action, direction, U)
                _overrelaxation_sites!(
                    U[direction],
                    staple,
                    _LinearHeatbathTarget(coloring, Int64(target_color)),
                    target_color;
                    seed=h.seed,
                    sweep=current_sweep,
                    direction,
                    rng_algorithm=h.rng_algorithm,
                )
            end
        end
    finally
        unused!(h._temporary_gaugefields, staple_index)
    end
    h.overrelaxation_sweep = current_sweep + one(current_sweep)
    return nothing
end

function overrelaxation!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    h::Heatbath_update{Dim,T},
) where {Dim,T,NC}
    nt = 11
    temps = Vector{Matrix{ComplexF64}}(undef, nt)
    for i = 1:nt
        temps[i] = zeros(ComplexF64, NC, NC)
    end

    #if NC != 2
    temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps3[i] = zeros(ComplexF64, NC, NC)
    end
    #end

    for μ = 1:Dim
        mapfunc!(A, U, indices...) =
            overrelaxation_update_eachsite_SUN!(A, μ, U, h, temps, temps3, indices...)
        map_U_sequential!(U[μ], mapfunc!, U)
    end


end


function heatbath!(
    U::Array{<:AbstractGaugefields{3,Dim},1},
    h::Heatbath_update{Dim,T},
) where {Dim,T}
    NC = 3
    nt = 11
    temps = Vector{Matrix{ComplexF64}}(undef, nt)
    for i = 1:nt
        temps[i] = zeros(ComplexF64, NC, NC)
    end

    if NC != 2
        temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
        for i = 1:5
            temps3[i] = zeros(ComplexF64, NC, NC)
        end
    end

    for μ = 1:Dim
        mapfunc!(A, U, indices...) =
            heatbath_update_eachsite_SU3!(A, μ, U, h, temps, temps3, indices...)
        map_U_sequential!(U[μ], mapfunc!, U)
    end


end

function heatbath!(U::Array{<:AbstractGaugefields{NC,Dim},1}, h::Heatbath) where {Dim,NC}
    heatbath!(U, h._tempotal_gauges, h.β; ITERATION_MAX=h.ITERATION_MAX)
end

function overrelaxation!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    h::Heatbath,
) where {Dim,NC}
    overrelaxation!(U, h._tempotal_gauges, h.β; ITERATION_MAX=h.ITERATION_MAX)
end

function heatbath!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps_g, # length >= 5
    β;
    ITERATION_MAX=10^5,
) where {Dim,NC}
    error("now heatbath!(U,temp_g::Temporalfields,β) is supported. Use temp=Temporalfields(U[1]; num=10)")
end


function heatbath!(
    U::Array{<:AbstractGaugefields{2,Dim},1},
    temps_g::Temporalfields{TG}, # length >= 5
    β;
    ITERATION_MAX=10^5,
) where {Dim,TG}
    NC = 2
    #temp1 = temps[1]
    #temp2 = temps[2]
    V, it_V = get_temp(temps_g)# temps[5]
    temps, its_temps = get_temp(temps_g, 4)

    temps2 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps2[i] = zeros(ComplexF64, 2, 2)
    end

    if NC != 2
        temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
        for i = 1:5
            temps3[i] = zeros(ComplexF64, NC, NC)
        end
    end



    mapfunc!(A, B) = SU2update_KP!(A, B, β, NC, temps2, ITERATION_MAX)

    for μ = 1:Dim

        loops = loops_staple[(Dim, μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven) # length >~ 3,4
        map_U!(U[μ], mapfunc!, V, iseven)

        iseven = false
        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven) # length >~ 3,4
        map_U!(U[μ], mapfunc!, V, iseven)
    end
    unused!(temps_g, it_V)
    unused!(temps_g, its_temps)

end

function heatbath!(
    U::Array{<:AbstractGaugefields{3,Dim},1},
    temps_g::Temporalfields{TG}, # length >= 5
    β;
    ITERATION_MAX=10^5,
) where {Dim,TG}
    NC = 3
    #temp1 = temps[1]
    #temp2 = temps[2]
    #V = temps[5]
    V, it_V = get_temp(temps_g)# temps[5]
    temps, its_temps = get_temp(temps_g, 4)


    temps2 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps2[i] = zeros(ComplexF64, 2, 2)
    end

    if NC != 2
        temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
        for i = 1:5
            temps3[i] = zeros(ComplexF64, NC, NC)
        end
    end



    mapfunc!(A, B) = SU3update_matrix!(A, B, β, NC, temps2, temps3, ITERATION_MAX)


    for μ = 1:Dim

        loops = loops_staple[(Dim, μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)

        iseven = false
        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)
    end

    unused!(temps_g, it_V)
    unused!(temps_g, its_temps)

end

function heatbath!(
    U::Array{<:AbstractGaugefields{3,Dim},1},
    temps_g::Temporalfields{TG}, # length >= 5
    β,
    gauge_action;
    ITERATION_MAX=10^5,
) where {Dim,TG} #This function is for debugging
    NC = 3
    #temp1 = temps[1]
    #temp2 = temps[2]
    V, it_V = get_temp(temps_g)# temps[5]
    temps, its_temps = get_temp(temps_g, 4)

    #V = temps[5]

    temps2 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps2[i] = zeros(ComplexF64, 2, 2)
    end

    if NC != 2
        temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
        for i = 1:5
            temps3[i] = zeros(ComplexF64, NC, NC)
        end
    end



    mapfunc!(A, B) = SU3update_matrix!(A, B, β, NC, temps2, temps3, ITERATION_MAX)


    for μ = 1:Dim

        #loops = loops_staple[(Dim,μ)]
        loops = gauge_action.dataset[1].staples[μ]
        iseven = true

        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)

        iseven = false
        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)
    end

    unused!(temps_g, it_V)
    unused!(temps_g, its_temps)

end


function heatbath!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps_g::Temporalfields{TG}, # length >= 5
    β;
    ITERATION_MAX=10^5,
) where {Dim,NC,TG}

    #temp1 = temps[1]
    #temp2 = temps[2]
    #V = temps[5]
    V, it_V = get_temp(temps_g)# temps[5]
    temps, its_temps = get_temp(temps_g, 4)


    temps2 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps2[i] = zeros(ComplexF64, 2, 2)
    end

    if NC != 2
        temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
        for i = 1:5
            temps3[i] = zeros(ComplexF64, NC, NC)
        end
    end


    mapfunc!(A, B) = SUNupdate_matrix!(A, B, β, NC, temps2, temps3, ITERATION_MAX)


    for μ = 1:Dim

        loops = loops_staple[(Dim, μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)

        iseven = false
        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)
    end

    unused!(temps_g, it_V)
    unused!(temps_g, its_temps)

end

function overrelaxation!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps_g, # length >= 5
    β;
    ITERATION_MAX=10^5,
) where {Dim,NC}
    error("now overrelaxation!(U,temp_g::Temporalfields,β) is supported. Use temp=Temporalfields(U[1]; num=10)")
end

function overrelaxation!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps_g::Temporalfields{TG}, # length >= 5
    β;
    ITERATION_MAX=10^5,
) where {Dim,NC,TG}

    #temp1 = temps[1]
    #temp2 = temps[2]
    #V = temps[3]
    V, it_V = get_temp(temps_g)# temps[5]
    temps, its_temps = get_temp(temps_g, 4)


    temps2 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps2[i] = zeros(ComplexF64, 2, 2)
    end

    #if NC != 2
    temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
    for i = 1:5
        temps3[i] = zeros(ComplexF64, NC, NC)
    end
    #end


    mapfunc!(A, B) = SUN_overrelaxation!(A, B, β, NC, temps2, temps3, ITERATION_MAX) #SUNupdate_matrix!(A,B,β,NC,temps2,temps3,ITERATION_MAX)


    for μ = 1:Dim

        loops = loops_staple[(Dim, μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)

        iseven = false
        evaluate_gaugelinks_evenodd!(V, loops, U, temps, iseven)
        map_U!(U[μ], mapfunc!, V, iseven)
    end
    unused!(temps_g, it_V)
    unused!(temps_g, its_temps)

end



function heatbath!(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    h::Heatbath_update,
) where {Dim,NC}
    heatbath!(U, h._tempotal_gauges, h.gauge_action; ITERATION_MAX=h.ITERATION_MAX)
end

# function heatbath!(
#     U::Array{<:AbstractGaugefields{2,Dim},1},
#     temps,
#     S::GaugeAction;
#     ITERATION_MAX = 10^5,
# ) where {Dim}
#     NC = 2
#
#
#     temps2 = Array{Matrix{ComplexF64},1}(undef, 5)
#     for i = 1:5
#         temps2[i] = zeros(ComplexF64, 2, 2)
#     end
#
#     if NC != 2
#         temps3 = Array{Matrix{ComplexF64},1}(undef, 5)
#         for i = 1:5
#             temps3[i] = zeros(ComplexF64, NC, NC)
#         end
#     end
#
#     mapfunc!(A, B) = SU2update_KP!(A, B, 1, NC, temps2, ITERATION_MAX)
#
#     numterm = length(S.dataset)
#     temp1 = S._temp_U[1]
#     temp2 = S._temp_U[2]
#     temp3 = S._temp_U[3]
#
#     error("error")
#
#
#     for μ = 1:Dim
#
#         loops = loops_staple[(Dim, μ)]
#         iseven = true
#
#         evaluate_gaugelinks_evenodd!(V, loops, U, [temp1, temp2], iseven)
#         map_U!(U[μ], mapfunc!, V, iseven)
#
#         iseven = false
#         evaluate_gaugelinks_evenodd!(V, loops, U, [temp1, temp2], iseven)
#         map_U!(U[μ], mapfunc!, V, iseven)
#     end
#
# end


"""
    SUNupdate_matrix_rng!(u, V, beta, temps2, tempsN, rng, Val(NC),
        ITERATION_MAX=10^5) -> updated_rng, accepted, failed_subgroup

Device-safe Cabibbo--Marinari update for general SU(N).  It follows the
legacy `SUNupdate_matrix!` scheme: `NC` randomly selected SU(2) subgroups are
updated and the result is reunitarized.  All subgroup selection and heatbath
draws come from the supplied site-local RNG.
"""
@inline function SUNupdate_matrix_rng!(
    u,
    V,
    beta,
    temps2,
    tempsN,
    rng::SiteRNG,
    ::Val{NC},
    ITERATION_MAX=10^5,
) where NC
    NC >= 2 || return rng, false, 0
    V0, temp, S, K = temps2
    UV, A, AU = tempsN

    for subgroup in 1:NC
        rng, n_offset = rand_bounded(rng, UInt32(NC - 1))
        n = Int(n_offset) + 1
        rng, m_offset = rand_bounded(rng, UInt32(NC - n))
        m = n + Int(m_offset) + 1

        mul!(UV, u, V)
        _make_su2_submatrix!(S, UV, n, m)
        _project_onto_su2!(S)
        rng, accepted, _ = SU2update_KP_rng!(
            K, S, beta, NC, (V0, temp), rng, ITERATION_MAX
        )
        accepted || return rng, false, subgroup

        _make_embedded_su2_matrix!(A, K, n, m, Val(NC))
        mul!(AU, A, u)
        _copy_square_matrix!(u, AU, Val(NC))
    end

    _copy_square_matrix!(AU, u, Val(NC))
    _normalize_columns_allocationfree!(AU, Val(NC)) ||
        return rng, false, NC + 1
    _copy_square_matrix!(u, AU, Val(NC))
    return rng, true, 0
end

const _OVERRELAXATION_RNG_TAG = typemax(UInt32)

@inline function _normalize_columns_allocationfree!(u, ::Val{N}) where N
    T = eltype(u)
    RT = typeof(real(zero(T)))
    @inbounds for column in 1:N
        for previous in 1:(column-1)
            projection = zero(T)
            for row in 1:N
                projection += conj(u[row, previous]) * u[row, column]
            end
            for row in 1:N
                u[row, column] -= projection * u[row, previous]
            end
        end

        norm2 = zero(RT)
        for row in 1:N
            norm2 += abs2(u[row, column])
        end
        iszero(norm2) && return false
        inverse_norm = inv(sqrt(norm2))
        for row in 1:N
            u[row, column] *= inverse_norm
        end
    end
    return true
end

@inline function _overrelaxation_subgroup!(
    u,
    V,
    w,
    h,
    UV,
    A,
    AU,
    n,
    m,
    ::Val{NC},
) where NC
    mul!(UV, u, V)
    _make_su2_submatrix!(w, UV, n, m)
    _project_onto_su2!(w)

    T = eltype(h)
    @inbounds for jc in 1:2
        for ic in 1:2
            value = zero(T)
            for kc in 1:2
                value += conj(w[kc, ic]) * conj(w[jc, kc])
            end
            h[ic, jc] = value
        end
    end
    _normalize_columns_allocationfree!(h, Val(2)) || return false

    if NC == 2
        _copy_square_matrix!(A, h, Val(2))
    else
        _make_embedded_su2_matrix!(A, h, n, m, Val(NC))
    end
    mul!(AU, A, u)
    _copy_square_matrix!(u, AU, Val(NC))
    return true
end

"""
    SUN_overrelaxation_rng!(u, V, temps2, tempsN, rng, Val(NC))
        -> updated_rng, success

Allocation-free SU(N) overrelaxation driven by a site-local RNG.
`temps2` contains two 2x2 matrices and `tempsN` three `NC`x`NC` matrices.
The legacy subgroup-selection distribution is retained without device-side
allocation, global `rand()`, or exceptions.
"""
@inline function SUN_overrelaxation_rng!(
    u,
    V,
    temps2,
    tempsN,
    rng::SiteRNG,
    ::Val{NC},
) where NC
    NC >= 2 || return rng, false
    w, h = temps2
    UV, A, AU = tempsN

    for _ in 1:NC
        rng, n_offset = rand_bounded(rng, UInt32(NC - 1))
        n = Int(n_offset) + 1
        rng, m_offset = rand_bounded(rng, UInt32(NC - n))
        m = n + Int(m_offset) + 1
        success = _overrelaxation_subgroup!(
            u, V, w, h, UV, A, AU, n, m, Val(NC)
        )
        success || return rng, false
    end

    _copy_square_matrix!(AU, u, Val(NC))
    _normalize_columns_allocationfree!(AU, Val(NC)) || return rng, false
    _copy_square_matrix!(u, AU, Val(NC))
    return rng, true
end

@inline function kernel_overrelaxation_sites!(
    i,
    u,
    staple,
    dindexer,
    ::Val{nw},
    coords,
    local_size,
    global_size,
    key,
    algorithm,
    target,
    ::Val{NC},
    failures,
) where {nw,NC}
    local_indices = delinearize(dindexer, i, 0)
    global_indices = global_site_coordinates(local_indices, coords, local_size)
    if _heatbath_site_selected(global_indices, target)
        indices = delinearize(dindexer, i, nw)
        T = eltype(u)
        u_local = MMatrix{NC,NC,T}(undef)
        staple_local = MMatrix{NC,NC,T}(undef)
        w = MMatrix{2,2,T}(undef)
        h = MMatrix{2,2,T}(undef)
        UV = MMatrix{NC,NC,T}(undef)
        A = MMatrix{NC,NC,T}(undef)
        AU = MMatrix{NC,NC,T}(undef)

        @inbounds for jc in 1:NC
            for ic in 1:NC
                u_local[ic, jc] = u[ic, jc, indices...]
                staple_local[ic, jc] = staple[ic, jc, indices...]
            end
        end

        global_site = global_site_id(global_indices, global_size)
        rng = site_rng(key, global_site, algorithm)
        _, success = SUN_overrelaxation_rng!(
            u_local,
            staple_local,
            (w, h),
            (UV, A, AU),
            rng,
            Val(NC),
        )

        if success
            @inbounds for jc in 1:NC
                for ic in 1:NC
                    u[ic, jc, indices...] = u_local[ic, jc]
                end
            end
        else
            JACC.@atomic failures[1] += Int32(1)
        end
    end
    return nothing
end

function _overrelaxation_sites!(
    U::Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},
    staple::Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},
    target,
    color::Integer;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    rng_algorithm::SiteRNGAlgorithm,
) where NC
    U.U.PN == staple.U.PN || throw(
        DimensionMismatch("U and staple local sizes differ")
    )
    U.U.gsize == staple.U.gsize || throw(
        DimensionMismatch("U and staple global sizes differ")
    )
    NC >= 2 || throw(ArgumentError(
        "parallel overrelaxation requires NC >= 2; got NC=$NC"
    ))

    key = RNGStreamKey(
        seed,
        sweep,
        direction,
        color,
        _OVERRELAXATION_RNG_TAG,
    )
    failures = JACC.zeros(Int32, 1)
    mark_halo_dirty!(U.U)
    JACC.parallel_for(
        prod(U.U.PN),
        kernel_overrelaxation_sites!,
        U.U.A,
        staple.U.A,
        U.U.indexer,
        Val(U.U.nw),
        U.U.coords,
        U.U.PN,
        U.U.gsize,
        key,
        rng_algorithm,
        target,
        Val(NC),
        failures,
    )

    set_halo!(U.U)
    local_failures = Int(JACC.to_host(failures)[1])
    total_failures = MPI.Allreduce(local_failures, MPI.SUM, U.U.comm)
    total_failures == 0 || error(
        "overrelaxation normalization failed at $total_failures site(s)"
    )
    return nothing
end

"""
    overrelaxation_su2_sites!(U, staple, target_even; seed, sweep,
        direction, color, rng_algorithm=Philox4x32())

Update one checkerboard color of an SU(2) MPILattice link field using
allocation-free, site-RNG overrelaxation and synchronize its halo.
"""
function overrelaxation_su2_sites!(
    U::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    staple::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    target_even::Bool;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    color::Integer=(target_even ? 0 : 1),
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    return _overrelaxation_sites!(
        U,
        staple,
        _CheckerboardHeatbathTarget(target_even),
        color;
        seed,
        sweep,
        direction,
        rng_algorithm,
    )
end

function overrelaxation_su2_sites!(
    U::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    staple::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    coloring::HeatbathColoring{Dim},
    target_color::Integer;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where Dim
    0 <= target_color < coloring.ncolors || throw(ArgumentError(
        "target_color must be in 0:$(coloring.ncolors-1)"
    ))
    return _overrelaxation_sites!(
        U,
        staple,
        _LinearHeatbathTarget(coloring, Int64(target_color)),
        target_color;
        seed,
        sweep,
        direction,
        rng_algorithm,
    )
end

"""
    overrelaxation_su3_sites!(U, staple, target_even; seed, sweep,
        direction, color, rng_algorithm=Philox4x32())

Update one checkerboard color of an SU(3) MPILattice link field using the
legacy random subgroup-selection distribution and synchronize its halo.
"""
function overrelaxation_su3_sites!(
    U::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    staple::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    target_even::Bool;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    color::Integer=(target_even ? 0 : 1),
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    return _overrelaxation_sites!(
        U,
        staple,
        _CheckerboardHeatbathTarget(target_even),
        color;
        seed,
        sweep,
        direction,
        rng_algorithm,
    )
end

function overrelaxation_su3_sites!(
    U::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    staple::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    coloring::HeatbathColoring{Dim},
    target_color::Integer;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where Dim
    0 <= target_color < coloring.ncolors || throw(ArgumentError(
        "target_color must be in 0:$(coloring.ncolors-1)"
    ))
    return _overrelaxation_sites!(
        U,
        staple,
        _LinearHeatbathTarget(coloring, Int64(target_color)),
        target_color;
        seed,
        sweep,
        direction,
        rng_algorithm,
    )
end

@inline function kernel_heatbath_su2_sites!(
    i,
    u,
    staple,
    dindexer,
    ::Val{nw},
    coords,
    local_size,
    global_size,
    key,
    algorithm,
    target,
    beta,
    iteration_max,
    failures,
) where nw
    local_indices = delinearize(dindexer, i, 0)
    global_indices = global_site_coordinates(local_indices, coords, local_size)
    if _heatbath_site_selected(global_indices, target)
        indices = delinearize(dindexer, i, nw)
        T = eltype(u)
        u_local = MMatrix{2,2,T}(undef)
        staple_local = MMatrix{2,2,T}(undef)
        V0 = MMatrix{2,2,T}(undef)
        temp = MMatrix{2,2,T}(undef)

        @inbounds for jc in 1:2
            for ic in 1:2
                u_local[ic, jc] = u[ic, jc, indices...]
                staple_local[ic, jc] = staple[ic, jc, indices...]
            end
        end

        global_site = global_site_id(global_indices, global_size)
        rng = site_rng(key, global_site, algorithm)
        rng, accepted, _ = SU2update_KP_rng!(
            u_local,
            staple_local,
            beta,
            2,
            (V0, temp),
            rng,
            iteration_max,
        )

        if accepted
            @inbounds for jc in 1:2
                for ic in 1:2
                    u[ic, jc, indices...] = u_local[ic, jc]
                end
            end
        else
            JACC.@atomic failures[1] += Int32(1)
        end
    end
    return nothing
end

"""
    heatbath_su2_sites!(U, staple, beta, target_even;
        seed, sweep, direction, color, subgroup=0,
        rng_algorithm=Philox4x32(), iteration_max=10^5)

Update one checkerboard color of an SU(2) `Gaugefields_4D_MPILattice`.  Each
site constructs its RNG from global coordinates and the supplied stream tags,
so results do not depend on MPI decomposition or parallel scheduling.  Philox
is the default for this parallel kernel; `PCG32()` and
`Xoshiro256PlusPlus()` are accepted through the same API.
"""
function heatbath_su2_sites!(
    U::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    staple::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    beta,
    target_even::Bool;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    color::Integer=(target_even ? 0 : 1),
    subgroup::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
    iteration_max::Integer=10^5,
)
    U.U.PN == staple.U.PN || throw(DimensionMismatch("U and staple local sizes differ"))
    U.U.gsize == staple.U.gsize || throw(DimensionMismatch("U and staple global sizes differ"))
    iteration_max > 0 || throw(ArgumentError("iteration_max must be positive"))

    key = RNGStreamKey(seed, sweep, direction, color, subgroup)
    failures = JACC.zeros(Int32, 1)
    mark_halo_dirty!(U.U)
    JACC.parallel_for(
        prod(U.U.PN),
        kernel_heatbath_su2_sites!,
        U.U.A,
        staple.U.A,
        U.U.indexer,
        Val(U.U.nw),
        U.U.coords,
        U.U.PN,
        U.U.gsize,
        key,
        rng_algorithm,
        _CheckerboardHeatbathTarget(target_even),
        beta,
        Int(iteration_max),
        failures,
    )

    # All ranks finish communication and agree on failure status before the
    # host throws, preventing one failed rank from stranding its peers.
    set_halo!(U.U)
    local_failures = Int(JACC.to_host(failures)[1])
    total_failures = MPI.Allreduce(local_failures, MPI.SUM, U.U.comm)
    total_failures == 0 || error(
        "KP heatbath failed at $total_failures site(s) after $iteration_max tries"
    )
    return nothing
end

"""
    heatbath_su2_sites!(U, staple, beta, coloring, target_color;
        seed, sweep, direction, rng_algorithm=Philox4x32(),
        iteration_max=10^5)

Update one action-derived color of an SU(2) MPILattice field.  `target_color`
is also the RNG color stream tag.
"""
function heatbath_su2_sites!(
    U::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    staple::Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},
    beta,
    coloring::HeatbathColoring{Dim},
    target_color::Integer;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
    iteration_max::Integer=10^5,
) where Dim
    U.U.PN == staple.U.PN || throw(DimensionMismatch("U and staple local sizes differ"))
    U.U.gsize == staple.U.gsize || throw(DimensionMismatch("U and staple global sizes differ"))
    0 <= target_color < coloring.ncolors || throw(ArgumentError(
        "target_color must be in 0:$(coloring.ncolors-1)"
    ))
    iteration_max > 0 || throw(ArgumentError("iteration_max must be positive"))

    key = RNGStreamKey(seed, sweep, direction, target_color, 0)
    failures = JACC.zeros(Int32, 1)
    target = _LinearHeatbathTarget(coloring, Int64(target_color))
    mark_halo_dirty!(U.U)
    JACC.parallel_for(
        prod(U.U.PN),
        kernel_heatbath_su2_sites!,
        U.U.A,
        staple.U.A,
        U.U.indexer,
        Val(U.U.nw),
        U.U.coords,
        U.U.PN,
        U.U.gsize,
        key,
        rng_algorithm,
        target,
        beta,
        Int(iteration_max),
        failures,
    )

    set_halo!(U.U)
    local_failures = Int(JACC.to_host(failures)[1])
    total_failures = MPI.Allreduce(local_failures, MPI.SUM, U.U.comm)
    total_failures == 0 || error(
        "KP heatbath failed at $total_failures site(s) after $iteration_max tries"
    )
    return nothing
end

@inline function kernel_heatbath_su3_sites!(
    i,
    u,
    staple,
    dindexer,
    ::Val{nw},
    coords,
    local_size,
    global_size,
    keys,
    algorithm,
    target,
    beta,
    iteration_max,
    failures,
) where nw
    local_indices = delinearize(dindexer, i, 0)
    global_indices = global_site_coordinates(local_indices, coords, local_size)
    if _heatbath_site_selected(global_indices, target)
        indices = delinearize(dindexer, i, nw)
        T = eltype(u)
        u_local = MMatrix{3,3,T}(undef)
        staple_local = MMatrix{3,3,T}(undef)
        V0 = MMatrix{2,2,T}(undef)
        temp = MMatrix{2,2,T}(undef)
        S = MMatrix{2,2,T}(undef)
        K = MMatrix{2,2,T}(undef)
        UV = MMatrix{3,3,T}(undef)
        A = MMatrix{3,3,T}(undef)
        AU = MMatrix{3,3,T}(undef)

        @inbounds for jc in 1:3
            for ic in 1:3
                u_local[ic, jc] = u[ic, jc, indices...]
                staple_local[ic, jc] = staple[ic, jc, indices...]
            end
        end

        global_site = global_site_id(global_indices, global_size)
        rngs = (
            site_rng(keys[1], global_site, algorithm),
            site_rng(keys[2], global_site, algorithm),
            site_rng(keys[3], global_site, algorithm),
        )
        _, accepted, _ = SU3update_matrix_rng!(
            u_local,
            staple_local,
            beta,
            (V0, temp, S, K),
            (UV, A, AU),
            rngs,
            iteration_max,
        )

        if accepted
            @inbounds for jc in 1:3
                for ic in 1:3
                    u[ic, jc, indices...] = u_local[ic, jc]
                end
            end
        else
            JACC.@atomic failures[1] += Int32(1)
        end
    end
    return nothing
end

"""
    heatbath_su3_sites!(U, staple, beta, target_even;
        seed, sweep, direction, color,
        rng_algorithm=Philox4x32(), iteration_max=10^5)

Update one checkerboard color of an SU(3) `Gaugefields_4D_MPILattice` through
the fixed `(1,2)`, `(2,3)`, `(1,3)` Cabibbo--Marinari subgroup sequence.  Each
subgroup receives a distinct global-site stream tag.  The updated link halo is
synchronized before the function returns.
"""
function heatbath_su3_sites!(
    U::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    staple::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    beta,
    target_even::Bool;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    color::Integer=(target_even ? 0 : 1),
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
    iteration_max::Integer=10^5,
)
    U.U.PN == staple.U.PN || throw(DimensionMismatch("U and staple local sizes differ"))
    U.U.gsize == staple.U.gsize || throw(DimensionMismatch("U and staple global sizes differ"))
    iteration_max > 0 || throw(ArgumentError("iteration_max must be positive"))

    keys = ntuple(
        subgroup -> RNGStreamKey(seed, sweep, direction, color, subgroup),
        3,
    )
    failures = JACC.zeros(Int32, 1)
    mark_halo_dirty!(U.U)
    JACC.parallel_for(
        prod(U.U.PN),
        kernel_heatbath_su3_sites!,
        U.U.A,
        staple.U.A,
        U.U.indexer,
        Val(U.U.nw),
        U.U.coords,
        U.U.PN,
        U.U.gsize,
        keys,
        rng_algorithm,
        _CheckerboardHeatbathTarget(target_even),
        beta,
        Int(iteration_max),
        failures,
    )

    set_halo!(U.U)
    local_failures = Int(JACC.to_host(failures)[1])
    total_failures = MPI.Allreduce(local_failures, MPI.SUM, U.U.comm)
    total_failures == 0 || error(
        "SU(3) heatbath failed at $total_failures site(s) after $iteration_max tries"
    )
    return nothing
end

@inline function kernel_heatbath_sun_sites!(
    i,
    u,
    staple,
    dindexer,
    ::Val{nw},
    coords,
    local_size,
    global_size,
    key,
    algorithm,
    target,
    beta,
    iteration_max,
    ::Val{NC},
    failures,
) where {nw,NC}
    local_indices = delinearize(dindexer, i, 0)
    global_indices = global_site_coordinates(local_indices, coords, local_size)
    if _heatbath_site_selected(global_indices, target)
        indices = delinearize(dindexer, i, nw)
        T = eltype(u)
        u_local = MMatrix{NC,NC,T}(undef)
        staple_local = MMatrix{NC,NC,T}(undef)
        V0 = MMatrix{2,2,T}(undef)
        temp = MMatrix{2,2,T}(undef)
        S = MMatrix{2,2,T}(undef)
        K = MMatrix{2,2,T}(undef)
        UV = MMatrix{NC,NC,T}(undef)
        A = MMatrix{NC,NC,T}(undef)
        AU = MMatrix{NC,NC,T}(undef)

        @inbounds for jc in 1:NC
            for ic in 1:NC
                u_local[ic, jc] = u[ic, jc, indices...]
                staple_local[ic, jc] = staple[ic, jc, indices...]
            end
        end

        global_site = global_site_id(global_indices, global_size)
        rng = site_rng(key, global_site, algorithm)
        _, accepted, _ = SUNupdate_matrix_rng!(
            u_local,
            staple_local,
            beta,
            (V0, temp, S, K),
            (UV, A, AU),
            rng,
            Val(NC),
            iteration_max,
        )

        if accepted
            @inbounds for jc in 1:NC
                for ic in 1:NC
                    u[ic, jc, indices...] = u_local[ic, jc]
                end
            end
        else
            JACC.@atomic failures[1] += Int32(1)
        end
    end
    return nothing
end

function _heatbath_sun_sites!(
    U::Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},
    staple::Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},
    beta,
    target,
    color::Integer;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
    iteration_max::Integer=10^5,
) where NC
    NC >= 2 || throw(ArgumentError("heatbath requires NC >= 2; got NC=$NC"))
    U.U.PN == staple.U.PN || throw(
        DimensionMismatch("U and staple local sizes differ")
    )
    U.U.gsize == staple.U.gsize || throw(
        DimensionMismatch("U and staple global sizes differ")
    )
    iteration_max > 0 || throw(ArgumentError("iteration_max must be positive"))

    key = RNGStreamKey(seed, sweep, direction, color, 0)
    failures = JACC.zeros(Int32, 1)
    mark_halo_dirty!(U.U)
    JACC.parallel_for(
        prod(U.U.PN),
        kernel_heatbath_sun_sites!,
        U.U.A,
        staple.U.A,
        U.U.indexer,
        Val(U.U.nw),
        U.U.coords,
        U.U.PN,
        U.U.gsize,
        key,
        rng_algorithm,
        target,
        beta,
        Int(iteration_max),
        Val(NC),
        failures,
    )

    set_halo!(U.U)
    local_failures = Int(JACC.to_host(failures)[1])
    total_failures = MPI.Allreduce(local_failures, MPI.SUM, U.U.comm)
    total_failures == 0 || error(
        "SU($NC) heatbath failed at $total_failures site(s) after " *
        "$iteration_max tries"
    )
    return nothing
end

function _heatbath_sun_sites!(
    U::Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},
    staple::Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},
    beta,
    coloring::HeatbathColoring{Dim},
    target_color::Integer;
    kwargs...,
) where {NC,Dim}
    0 <= target_color < coloring.ncolors || throw(ArgumentError(
        "target_color must be in 0:$(coloring.ncolors-1)"
    ))
    return _heatbath_sun_sites!(
        U,
        staple,
        beta,
        _LinearHeatbathTarget(coloring, Int64(target_color)),
        target_color;
        kwargs...,
    )
end

"""
    heatbath_su3_sites!(U, staple, beta, coloring, target_color;
        seed, sweep, direction, rng_algorithm=Philox4x32(),
        iteration_max=10^5)

Update one action-derived color of an SU(3) MPILattice field.  The three
Cabibbo--Marinari subgroups receive distinct subgroup stream tags.
"""
function heatbath_su3_sites!(
    U::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    staple::Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},
    beta,
    coloring::HeatbathColoring{Dim},
    target_color::Integer;
    seed::Integer,
    sweep::Integer,
    direction::Integer,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
    iteration_max::Integer=10^5,
) where Dim
    U.U.PN == staple.U.PN || throw(DimensionMismatch("U and staple local sizes differ"))
    U.U.gsize == staple.U.gsize || throw(DimensionMismatch("U and staple global sizes differ"))
    0 <= target_color < coloring.ncolors || throw(ArgumentError(
        "target_color must be in 0:$(coloring.ncolors-1)"
    ))
    iteration_max > 0 || throw(ArgumentError("iteration_max must be positive"))

    keys = ntuple(
        subgroup -> RNGStreamKey(
            seed, sweep, direction, target_color, subgroup
        ),
        3,
    )
    failures = JACC.zeros(Int32, 1)
    target = _LinearHeatbathTarget(coloring, Int64(target_color))
    mark_halo_dirty!(U.U)
    JACC.parallel_for(
        prod(U.U.PN),
        kernel_heatbath_su3_sites!,
        U.U.A,
        staple.U.A,
        U.U.indexer,
        Val(U.U.nw),
        U.U.coords,
        U.U.PN,
        U.U.gsize,
        keys,
        rng_algorithm,
        target,
        beta,
        Int(iteration_max),
        failures,
    )

    set_halo!(U.U)
    local_failures = Int(JACC.to_host(failures)[1])
    total_failures = MPI.Allreduce(local_failures, MPI.SUM, U.U.comm)
    total_failures == 0 || error(
        "SU(3) heatbath failed at $total_failures site(s) after $iteration_max tries"
    )
    return nothing
end

"""
    heatbath!(U::Vector{<:Gaugefields_4D_MPILattice{2}}, temps, beta;
        seed=0, sweep=0, rng_algorithm=Philox4x32(), ITERATION_MAX=10^5)

Perform one checkerboard SU(2) plaquette heatbath sweep.  For every direction,
the staple and update are evaluated first on even global sites and then on odd
global sites.  `heatbath_su2_sites!` synchronizes the updated link halo after
each color before the next staple is constructed.

This low-level overload uses exactly the supplied `sweep` stream.  Prefer
constructing a [`Heatbath`](@ref) and calling `heatbath!(U, h)` for repeated
sweeps; that overload advances `h.sweep` after each successfully completed
sweep.
"""
function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},1},
    temps_g::Temporalfields,
    beta;
    ITERATION_MAX::Integer=10^5,
    seed::Integer=0,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    Dim = length(U)
    Dim in (2, 3, 4) || throw(DimensionMismatch(
        "MPILattice heatbath requires two, three, or four links"
    ))
    ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))

    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(Dim, direction)]
            for target_even in (true, false)
                color = target_even ? 0 : 1
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                heatbath_su2_sites!(
                    U[direction],
                    staple,
                    beta,
                    target_even;
                    seed,
                    sweep,
                    direction,
                    color,
                    subgroup=0,
                    rng_algorithm,
                    iteration_max=ITERATION_MAX,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

"""
    heatbath!(U::Vector{<:Gaugefields_4D_MPILattice{3}}, temps, beta;
        seed=0, sweep=0, rng_algorithm=Philox4x32(), ITERATION_MAX=10^5)

Perform one checkerboard SU(3) plaquette heatbath sweep.  The low-level method
uses exactly the supplied `sweep`; repeated simulations should normally use a
`Heatbath` object so the counter advances after a successful sweep.
"""
function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},1},
    temps_g::Temporalfields,
    beta;
    ITERATION_MAX::Integer=10^5,
    seed::Integer=0,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    Dim = length(U)
    Dim in (2, 3, 4) || throw(DimensionMismatch(
        "MPILattice heatbath requires two, three, or four links"
    ))
    ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))

    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(Dim, direction)]
            for target_even in (true, false)
                color = target_even ? 0 : 1
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                heatbath_su3_sites!(
                    U[direction],
                    staple,
                    beta,
                    target_even;
                    seed,
                    sweep,
                    direction,
                    color,
                    rng_algorithm,
                    iteration_max=ITERATION_MAX,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},1},
    h::Heatbath,
) where NC
    current_sweep = h.sweep
    heatbath!(
        U,
        h._tempotal_gauges,
        h.β;
        ITERATION_MAX=h.ITERATION_MAX,
        seed=h.seed,
        sweep=current_sweep,
        rng_algorithm=h.rng_algorithm,
    )
    h.sweep = current_sweep + one(current_sweep)
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},1},
    temps_g::Temporalfields,
    beta;
    ITERATION_MAX::Integer=10^5,
    seed::Integer=0,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    Dim = length(U)
    Dim in (2, 3, 4) || throw(DimensionMismatch(
        "MPILattice overrelaxation requires two, three, or four links"
    ))
    ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))

    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(Dim, direction)]
            for target_even in (true, false)
                color = target_even ? 0 : 1
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                overrelaxation_su2_sites!(
                    U[direction],
                    staple,
                    target_even;
                    seed,
                    sweep,
                    direction,
                    color,
                    rng_algorithm,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},1},
    temps_g::Temporalfields,
    beta;
    ITERATION_MAX::Integer=10^5,
    seed::Integer=0,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    Dim = length(U)
    Dim in (2, 3, 4) || throw(DimensionMismatch(
        "MPILattice overrelaxation requires two, three, or four links"
    ))
    ITERATION_MAX > 0 || throw(ArgumentError("ITERATION_MAX must be positive"))

    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(Dim, direction)]
            for target_even in (true, false)
                color = target_even ? 0 : 1
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                overrelaxation_su3_sites!(
                    U[direction],
                    staple,
                    target_even;
                    seed,
                    sweep,
                    direction,
                    color,
                    rng_algorithm,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

"""
    heatbath!(U::Vector{<:Gaugefields_4D_MPILattice{2}}, h::Heatbath)

Run one parallel SU(2) plaquette heatbath sweep using the device-safe RNG
configuration stored in `h`.  The sweep counter is incremented only after all
directions and both checkerboard colors finish successfully.
"""
function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},1},
    h::Heatbath,
)
    current_sweep = h.sweep
    heatbath!(
        U,
        h._tempotal_gauges,
        h.β;
        ITERATION_MAX=h.ITERATION_MAX,
        seed=h.seed,
        sweep=current_sweep,
        rng_algorithm=h.rng_algorithm,
    )
    h.sweep = current_sweep + one(current_sweep)
    return nothing
end

function heatbath!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},1},
    h::Heatbath,
)
    current_sweep = h.sweep
    heatbath!(
        U,
        h._tempotal_gauges,
        h.β;
        ITERATION_MAX=h.ITERATION_MAX,
        seed=h.seed,
        sweep=current_sweep,
        rng_algorithm=h.rng_algorithm,
    )
    h.sweep = current_sweep + one(current_sweep)
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{2},Gaugefields_3D_MPILattice{2},Gaugefields_4D_MPILattice{2}},1},
    h::Heatbath,
)
    current_sweep = h.overrelaxation_sweep
    overrelaxation!(
        U,
        h._tempotal_gauges,
        h.β;
        ITERATION_MAX=h.ITERATION_MAX,
        seed=h.seed,
        sweep=current_sweep,
        rng_algorithm=h.rng_algorithm,
    )
    h.overrelaxation_sweep = current_sweep + one(current_sweep)
    return nothing
end


function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{NC},Gaugefields_3D_MPILattice{NC},Gaugefields_4D_MPILattice{NC}},1},
    h::Heatbath,
) where NC
    current_sweep = h.overrelaxation_sweep
    overrelaxation!(
        U,
        h._tempotal_gauges,
        h.β;
        ITERATION_MAX=h.ITERATION_MAX,
        seed=h.seed,
        sweep=current_sweep,
        rng_algorithm=h.rng_algorithm,
    )
    h.overrelaxation_sweep = current_sweep + one(current_sweep)
    return nothing
end

function overrelaxation!(
    U::Array{<:Union{Gaugefields_2D_MPILattice{3},Gaugefields_3D_MPILattice{3},Gaugefields_4D_MPILattice{3}},1},
    h::Heatbath,
)
    current_sweep = h.overrelaxation_sweep
    overrelaxation!(
        U,
        h._tempotal_gauges,
        h.β;
        ITERATION_MAX=h.ITERATION_MAX,
        seed=h.seed,
        sweep=current_sweep,
        rng_algorithm=h.rng_algorithm,
    )
    h.overrelaxation_sweep = current_sweep + one(current_sweep)
    return nothing
end

function SU2update_KP(V, beta, NC, temps, ITERATION_MAX=10^5)
    #println("V = ",V)
    Unew = zero(V)
    SU2update_KP!(Unew, V, beta, NC, temps, ITERATION_MAX)
    return Unew
end



function SUNupdate_matrix!(u, V, beta, NC, temps2, temps3, ITERATION_MAX)
    UV = temps3[1]
    A = temps3[2]
    AU = temps3[3]
    temp1 = temps2[1]
    temp2 = temps2[2]
    S = temps2[3]
    K = temps2[4]

    for l = 1:NC
        #for l=1:2NC

        mul!(UV, u, V)
        #UV = u[:,:]*V

        n = rand(1:NC-1)#l
        m = rand(n:NC)
        while (n == m)
            m = rand(n:NC)
        end

        #=
        if l < NC
            n = l
            m = l+1
        else
            n = rand(1:NC)#l
            m = rand(1:NC)
            while(n==m)
                m = rand(1:NC)
            end
        end
        =#


        make_submatrix!(S, UV, n, m)
        #S = make_submatrix(UV,n,m)

        #gramschmidt_special!(S)
        project_onto_SU2!(S)

        SU2update_KP!(K, S, beta, NC, [temp1, temp2], ITERATION_MAX)


        make_largematrix!(A, K, n, m, NC)

        mul!(AU, A, u)

        #AU = A*u[:,:]

        u[:, :] .= AU
        #println("det U ",det(AU))

    end

    AU[:, :] .= u #u[:,:]
    normalizeN!(AU)
    u[:, :] .= AU
end

function SUN_overrelaxation!(u, V, beta, NC, temps2, temps3, ITERATION_MAX)
    UV = temps3[1]
    A = temps3[2]
    AU = temps3[3]
    temp1 = temps2[1]
    temp2 = temps2[2]
    w = temps2[3]
    h = temps2[4]
    #K = temps2[5]

    for l = 1:NC
        #for l=1:2NC

        mul!(UV, u, V)
        #UV = u[:,:]*V

        n = rand(1:NC-1)#l
        m = rand(n:NC)
        while (n == m)
            m = rand(n:NC)
        end

        # we emplay DeGrand's textbook notation
        make_submatrix!(w, UV, n, m)
        #S = make_submatrix(UV,n,m)

        #gramschmidt_special!(S)
        project_onto_SU2!(w)

        # following two lines are only difference to HB
        #SU2update_KP!(K,S,beta,NC,[temp1,temp2],ITERATION_MAX)
        for j = 1:2
            for i = 1:2
                h[i, j] = 0
                for k = 1:2
                    h[i, j] += w'[i, k] * w'[k, j]
                end
            end
        end
        normalizeN!(h)

        make_largematrix!(A, h, n, m, NC)

        mul!(AU, A, u)

        #AU = A*u[:,:]

        u[:, :] .= AU
        #println("det U ",det(AU))

    end

    AU[:, :] .= u #u[:,:]
    normalizeN!(AU)
    u[:, :] .= AU
end


function make_submatrix(UV, i, j)
    S = zeros(ComplexF64, 2, 2)
    S[1, 1] = UV[i, i]
    S[1, 2] = UV[i, j]
    S[2, 1] = UV[j, i]
    S[2, 2] = UV[j, j]
    return S
end

function make_largematrix(K, i, j, NC)
    A = zeros(ComplexF64, NC, NC)
    for n = 1:NC
        A[n, n] = 1
    end
    #K = project_onto_su2(K)
    A[i, i] = K[1, 1]
    A[i, j] = K[1, 2]
    A[j, i] = K[2, 1]
    A[j, j] = K[2, 2]
    return A
end

const nhit = 6
const rwidth = 0.4


"""
-------------------------------------------------c
 su2-submatrix(c) in su3 matrix(x)
        su2            su3
 k=1         <-    1-2 elements
 k=2         <-    2-3 elements
 k=3         <-    1-3 elements
 k=4          ->   1-2 elements
 k=5          ->   2-3 elements
 k=6          ->   1-3 elements
-------------------------------------------------c
"""
function submat!(x, c, n, k, id)

    if k == 1
        for i = 1:n
            c[1, i] = real(x[1, 1, i] + x[2, 2, i]) * 0.5
            c[2, i] = imag(x[1, 2, i] + x[2, 1, i]) * 0.5
            c[3, i] = real(x[1, 2, i] - x[2, 1, i]) * 0.5
            c[4, i] = imag(x[1, 1, i] - x[2, 2, i]) * 0.5
        end
    elseif k == 2
        for i = 1:n
            c[1, i] = real(x[2, 2, i] + x[3, 3, i]) * 0.5
            c[2, i] = imag(x[3, 2, i] + x[2, 3, i]) * 0.5
            c[3, i] = real(x[3, 2, i] - x[2, 3, i]) * 0.5
            c[4, i] = imag(x[2, 2, i] - x[3, 3, i]) * 0.5
        end

    elseif k == 3
        for i = 1:n
            c[1, i] = real(x[1, 1, i] + x[3, 3, i]) * 0.5
            c[2, i] = imag(x[3, 1, i] + x[1, 3, i]) * 0.5
            c[3, i] = real(x[1, 3, i] - x[3, 1, i]) * 0.5
            c[4, i] = imag(x[1, 1, i] - x[3, 3, i]) * 0.5
        end
    elseif k == 4

        for i = 1:n
            #println("i = $i")
            #println(c[:,i])
            if id[i] == 1
                x[1, 1, i] = c[1, i] + im * c[4, i]
                x[1, 2, i] = c[3, i] + im * c[2, i]
                x[1, 3, i] = 0
                x[2, 1, i] = -c[3, i] + im * c[2, i]
                x[2, 2, i] = c[1, i] - im * c[4, i]
                x[2, 3, i] = 0
                x[3, 1, i] = 0
                x[3, 2, i] = 0
                x[3, 3, i] = 1

            elseif id[i] == 0
                x[1, 1, i] = 1
                x[1, 2, i] = 0
                x[1, 3, i] = 0
                x[2, 1, i] = 0
                x[2, 2, i] = 1
                x[2, 3, i] = 0
                x[3, 1, i] = 0
                x[3, 2, i] = 0
                x[3, 3, i] = 1
            end
        end
    elseif k == 5
        for i = 1:n
            if id[i] == 1
                x[1, 1, i] = 1
                x[1, 2, i] = 0
                x[1, 3, i] = 0
                x[2, 1, i] = 0
                x[2, 2, i] = c[1, i] + im * c[4, i]
                x[2, 3, i] = -c[3, i] + im * c[2, i]
                x[3, 1, i] = 0
                x[3, 2, i] = c[3, i] + im * c[2, i]
                x[3, 3, i] = c[1, i] - im * c[4, i]

            elseif id[i] == 0
                x[1, 1, i] = 1
                x[1, 2, i] = 0
                x[1, 3, i] = 0
                x[2, 1, i] = 0
                x[2, 2, i] = 1
                x[2, 3, i] = 0
                x[3, 1, i] = 0
                x[3, 2, i] = 0
                x[3, 3, i] = 1
            end
        end

    elseif k == 6
        for i = 1:n
            if id[i] == 1
                x[1, 1, i] = c[1, i] + im * c[4, i]
                x[1, 2, i] = 0
                x[1, 3, i] = c[3, i] + im * c[2, i]
                x[2, 1, i] = 0
                x[2, 2, i] = 1
                x[2, 3, i] = 0
                x[3, 1, i] = -c[3, i] + im * c[2, i]
                x[3, 2, i] = 0
                x[3, 3, i] = c[1, i] - im * c[4, i]

            elseif id[i] == 0
                x[1, 1, i] = 1
                x[1, 2, i] = 0
                x[1, 3, i] = 0
                x[2, 1, i] = 0
                x[2, 2, i] = 1
                x[2, 3, i] = 0
                x[3, 1, i] = 0
                x[3, 2, i] = 0
                x[3, 3, i] = 1
            end
        end
    end
end

function rndprd!(ranf, n)
    rn = zeros(Float64, n)
    rndprd!(ranf, rn, n)
    return rn
end

function rndprd!(ranf, rn, n)
    for i = 1:n
        rn[i] = ranf()
    end
    return rn
end

function rndprd2!(ranf, n)
    xrn = zeros(Float64, 3, n)
    rndprd2!(ranf, xrn, n)
    return xrn
end

function rndprd2!(ranf, xrn, n)
    for j = 1:n
        for i = 1:3
            xrn[i, j] = ranf()
        end
    end
    return
end

end
