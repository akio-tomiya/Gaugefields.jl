import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

function _overrelaxation_action(U, nc)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, adjoint(plaquettes))
    push!(action, nc == 2 ? 1.15 : 2.85, plaquettes)

    rectangles = make_loops_fromname("rectangular", Dim=4)
    append!(rectangles, adjoint(rectangles))
    push!(action, -0.05, rectangles)
    return action
end

function _reference_overrelaxation_color!(
    link,
    staple,
    nc,
    coloring,
    target_color;
    seed,
    sweep,
    direction,
    algorithm,
)
    global_size = (link.NX, link.NY, link.NZ, link.NT)
    T = eltype(link.U)
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        heatbath_site_color(coloring, global_indices) == target_color || continue

        u_local = zeros(T, nc, nc)
        staple_local = zeros(T, nc, nc)
        for jc in 1:nc, ic in 1:nc
            u_local[ic, jc] = link[ic, jc, global_indices...]
            staple_local[ic, jc] = staple[ic, jc, global_indices...]
        end
        temps2 = ntuple(_ -> zeros(T, 2, 2), 2)
        tempsN = ntuple(_ -> zeros(T, nc, nc), 3)
        global_site = global_site_id(global_indices, global_size)
        key = RNGStreamKey(
            seed,
            sweep,
            direction,
            target_color,
            typemax(UInt32),
        )
        rng = site_rng(key, global_site, algorithm)
        _, success = SUN_overrelaxation_rng!(
            u_local,
            staple_local,
            temps2,
            tempsN,
            rng,
            Val(nc),
        )
        @test success
        for jc in 1:nc, ic in 1:nc
            link[ic, jc, global_indices...] = u_local[ic, jc]
        end
    end
    return nothing
end

function _reference_overrelaxation_sweep!(
    U,
    action,
    colorings,
    nc;
    seed,
    sweep,
    algorithm,
)
    staple = similar(U[1])
    for direction in eachindex(U)
        coloring = colorings[direction]
        for target_color in 0:(coloring.ncolors-1)
            calc_dSdUμ!(staple, action, direction, U)
            _reference_overrelaxation_color!(
                U[direction],
                staple,
                nc,
                coloring,
                target_color;
                seed,
                sweep,
                direction,
                algorithm,
            )
        end
    end
    return nothing
end

@inline function _overrelaxation_host_value(
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

function _compare_overrelaxation_fields(U, reference, nc)
    tolerance = nc == 2 ? 3e-11 : 8e-11
    for direction in eachindex(U)
        field = U[direction]
        storage = JACC.to_host(field.U.A)
        for site in CartesianIndices(field.U.PN)
            local_indices = Tuple(site)
            global_indices = global_site_coordinates(field.U, local_indices)
            for jc in 1:nc, ic in 1:nc
                @test _overrelaxation_host_value(
                    storage, field, ic, jc, local_indices
                ) ≈ reference[direction][ic, jc, global_indices...] atol = tolerance rtol = tolerance
            end
        end
    end
    return nothing
end

@testset "MPILattice plaquette+rectangle overrelaxation" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs <= 2 || error("this comparison test supports at most two MPI ranks")
    global_size = (4, 4, 4, 4)
    process_grid = (nprocs, 1, 1, 1)
    seed = 0x510e527fade682d1
    initial_sweep = 17
    heatbath_sweep = 11
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
            condition="hot",
            randomnumber="Reproducible",
            verbose_level=0,
        )
        substitute_U!(U, reference)

        action = _overrelaxation_action(U, nc)
        reference_action = _overrelaxation_action(reference, nc)
        h = Heatbath_update(
            U,
            action;
            seed,
            sweep=heatbath_sweep,
            overrelaxation_sweep=initial_sweep,
            rng_algorithm=algorithm,
        )
        @test all(coloring -> coloring.ncolors == 4, h.colorings)

        initial_action = real(evaluate_GaugeAction(reference_action, reference))
        _reference_overrelaxation_sweep!(
            reference,
            reference_action,
            h.colorings,
            nc;
            seed,
            sweep=initial_sweep,
            algorithm,
        )
        final_action = real(evaluate_GaugeAction(reference_action, reference))
        @test final_action ≈ initial_action atol = 2e-8 rtol = 2e-11

        overrelaxation!(U, h)
        @test h.sweep == UInt64(heatbath_sweep)
        @test h.overrelaxation_sweep == UInt64(initial_sweep + 1)
        _compare_overrelaxation_fields(U, reference, nc)
    end
end

@testset "MPILattice plaquette Heatbath overrelaxation" begin
    global_size = (4, 4, 4, 4)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1, 1, 1)
    seed = 0x1f83d9abfb41bd6b
    initial_sweep = 23
    heatbath_sweep = 19
    algorithm = Philox4x32()
    checkerboard = ntuple(
        _ -> HeatbathColoring(2, (1, 1, 1, 1)), 4
    )

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
            condition="hot",
            randomnumber="Reproducible",
            verbose_level=0,
        )
        substitute_U!(U, reference)

        reference_action = GaugeAction(reference)
        plaquettes = make_loops_fromname("plaquette", Dim=4)
        append!(plaquettes, adjoint(plaquettes))
        push!(reference_action, 1.0, plaquettes)
        _reference_overrelaxation_sweep!(
            reference,
            reference_action,
            checkerboard,
            nc;
            seed,
            sweep=initial_sweep,
            algorithm,
        )

        h = Heatbath(
            U,
            6.0;
            seed,
            sweep=heatbath_sweep,
            overrelaxation_sweep=initial_sweep,
            rng_algorithm=algorithm,
        )
        overrelaxation!(U, h)
        @test h.sweep == UInt64(heatbath_sweep)
        @test h.overrelaxation_sweep == UInt64(initial_sweep + 1)
        _compare_overrelaxation_fields(U, reference, nc)
    end
end
