import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

@inline function _general_host_field_value(
    storage,
    field,
    ic,
    jc,
    local_indices,
)
    array_indices = ntuple(
        d -> local_indices[d] + field.U.nw,
        length(local_indices),
    )
    return storage[ic, jc, array_indices...]
end

function _general_test_action(U, nc)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, adjoint(plaquettes))
    push!(action, nc == 2 ? 1.15 : 2.85, plaquettes)

    rectangles = make_loops_fromname("rectangular", Dim=4)
    append!(rectangles, adjoint(rectangles))
    push!(action, -0.05, rectangles)
    return action
end

function _reference_general_color!(
    link,
    staple,
    nc,
    coloring,
    target_color;
    seed,
    sweep,
    direction,
    algorithm,
    iteration_max,
)
    global_size = (link.NX, link.NY, link.NZ, link.NT)
    element_type = eltype(link.U)
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        heatbath_site_color(coloring, global_indices) == target_color || continue
        global_site = global_site_id(global_indices, global_size)

        if nc == 2
            link_local = LatticeMatrices.MMatrix{2,2,element_type}(undef)
            staple_local = LatticeMatrices.MMatrix{2,2,element_type}(undef)
            temps = ntuple(
                _ -> LatticeMatrices.MMatrix{2,2,element_type}(undef), 2
            )
            for jc in 1:2, ic in 1:2
                link_local[ic, jc] = link[ic, jc, global_indices...]
                staple_local[ic, jc] = staple[ic, jc, global_indices...]
            end
            key = RNGStreamKey(seed, sweep, direction, target_color, 0)
            rng = site_rng(key, global_site, algorithm)
            _, accepted, _ = SU2update_KP_rng!(
                link_local,
                staple_local,
                2,
                2,
                temps,
                rng,
                iteration_max,
            )
            @test accepted
            for jc in 1:2, ic in 1:2
                link[ic, jc, global_indices...] = link_local[ic, jc]
            end
        else
            link_local = LatticeMatrices.MMatrix{3,3,element_type}(undef)
            staple_local = LatticeMatrices.MMatrix{3,3,element_type}(undef)
            temps2 = ntuple(
                _ -> LatticeMatrices.MMatrix{2,2,element_type}(undef), 4
            )
            temps3 = ntuple(
                _ -> LatticeMatrices.MMatrix{3,3,element_type}(undef), 3
            )
            for jc in 1:3, ic in 1:3
                link_local[ic, jc] = link[ic, jc, global_indices...]
                staple_local[ic, jc] = staple[ic, jc, global_indices...]
            end
            rngs = ntuple(3) do subgroup
                key = RNGStreamKey(
                    seed, sweep, direction, target_color, subgroup
                )
                site_rng(key, global_site, algorithm)
            end
            _, accepted, failed_subgroup = SU3update_matrix_rng!(
                link_local,
                staple_local,
                2,
                temps2,
                temps3,
                rngs,
                iteration_max,
            )
            @test accepted
            @test failed_subgroup == 0
            for jc in 1:3, ic in 1:3
                link[ic, jc, global_indices...] = link_local[ic, jc]
            end
        end
    end
    return nothing
end

"""
Reference one general-action sweep.  Weighted staples, Wilson-line shifts,
checkerboard-independent field storage, and updates between colors all use the
serial `Gaugefields_4D_nowing` backend.  Only the public local RNG/update APIs
are shared with the accelerator implementation.
"""
function _reference_general_action_sweep!(
    U,
    action,
    colorings,
    nc;
    seed,
    sweep,
    algorithm,
    iteration_max=100_000,
)
    staple = similar(U[1])
    for direction in eachindex(U)
        coloring = colorings[direction]
        for target_color in 0:(coloring.ncolors-1)
            calc_dSdUμ!(staple, action, direction, U)
            _reference_general_color!(
                U[direction],
                staple,
                nc,
                coloring,
                target_color;
                seed,
                sweep,
                direction,
                algorithm,
                iteration_max,
            )
        end
    end
    return nothing
end

function _compare_general_action_fields(U, reference, nc)
    tolerance = nc == 2 ? 2e-11 : 5e-11
    for direction in eachindex(U)
        field = U[direction]
        storage = JACC.to_host(field.U.A)
        for site in CartesianIndices(field.U.PN)
            local_indices = Tuple(site)
            global_indices = global_site_coordinates(field.U, local_indices)
            for jc in 1:nc, ic in 1:nc
                @test _general_host_field_value(
                    storage, field, ic, jc, local_indices
                ) ≈ reference[direction][ic, jc, global_indices...] atol = tolerance rtol = tolerance
            end
        end
    end
    return nothing
end

@testset "MPILattice plaquette+rectangle heatbath" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs <= 2 || error("this comparison test supports at most two MPI ranks")
    global_size = (4, 4, 4, 4)
    process_grid = (nprocs, 1, 1, 1)
    seed = 0xa54ff53a5f1d36f1
    initial_sweep = 13
    algorithm = Philox4x32()

    for nc in (2, 3)
        U = Initialize_Gaugefields(
            nc,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        reference = Initialize_Gaugefields(
            nc,
            0,
            global_size...;
            condition="cold",
            verbose_level=0,
        )
        action = _general_test_action(U, nc)
        reference_action = _general_test_action(reference, nc)
        h = Heatbath_update(
            U,
            action;
            seed,
            sweep=initial_sweep,
            rng_algorithm=algorithm,
        )

        @test all(coloring -> coloring.ncolors == 4, h.colorings)
        _reference_general_action_sweep!(
            reference,
            reference_action,
            h.colorings,
            nc;
            seed,
            sweep=initial_sweep,
            algorithm,
        )
        heatbath!(U, h)

        @test h.sweep == UInt64(initial_sweep + 1)
        _compare_general_action_fields(U, reference, nc)
    end
end
