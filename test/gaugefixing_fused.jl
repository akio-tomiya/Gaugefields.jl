module FusedGaugeFixingTests

using Gaugefields, LatticeMatrices, LinearAlgebra, Random, Test, JACC
JACC.@init_backend

const M = Gaugefields.AbstractGaugefixing_module
const comm = LatticeMatrices._default_communicator()
const ranks = LatticeMatrices._comm_size(comm)
if get(ENV, "LDO_TEST_MPI", "false") == "true"
    @assert ranks > 1 "MPI validation must use a distributed field communicator"
end
const snapshots = Dict{String,Vector{Float64}}()
const hot_snapshots = Dict{String,Vector{Float64}}()
pack(x) = Float64.(vcat(vec(real.(x)), vec(imag.(x))))
global_links(U) = [gather_and_bcast_matrix(u.U) for u in U]

# Generate well-conditioned SU(N) links on the host, then copy the same values
# to every backend. Backend-specific hot-start orthogonalization has rounding
# differences that can be amplified by gauge-fixing on rough configurations.
function common_links(dims, nc, T)
    rng = MersenneTwister(189)
    links = [zeros(T, nc, nc, dims...) for _ in 1:4]
    for link in links, site in CartesianIndices(dims)
        a = randn(rng, ComplexF64, nc, nc)
        a = (a - a') / 2
        a -= tr(a) / nc * I
        link[:, :, Tuple(site)...] = exp(0.15a)
    end
    return links
end

function copy_links(U)
    V = [similar(u) for u in U]
    for μ in eachindex(U)
        Gaugefields.substitute_U!(V[μ], U[μ])
        Gaugefields.set_wing_U!(V[μ])
    end
    return V
end

# Independent periodic, dense-matrix divergence; no field shifts or projections.
function dense_delta(links, dims, D_fix)
    nc = size(links[1], 1)
    result = zeros(eltype(links[1]), size(links[1]))
    for site in CartesianIndices(dims)
        pos = Tuple(site)
        v = zeros(ComplexF64, nc, nc)
        for μ in 1:D_fix
            back = ntuple(d -> d == μ ? mod1(pos[d] - 1, dims[d]) : pos[d], 4)
            v += links[μ][:, :, pos...] - links[μ][:, :, back...]
        end
        q = (v - v') / 2
        q -= tr(q) / nc * I
        result[:, :, pos...] = q
    end
    return result
end

function check_halos(U)
    for u in U
        @test !LatticeMatrices.halo_is_dirty(u.U)
        before = Array(u.U.A)
        Gaugefields.set_wing_U!(u)
        @test Array(u.U.A) == before
    end
end

@testset "Fused gauge fixing across precision and rank boundaries" begin
    # A two-rank decomposition has an odd local extent of three, testing the
    # global checkerboard offset. Also split time, outside Coulomb's D_fix.
    dims = (6, 4, 4, 6)
    for fixture in (:hot, :common),
        (axis, halo, colors) in ((1, 1, (2, 3, 4)), (4, 2, (3,)), (1, 0, (3,)))
        grid = ntuple(d -> d == axis ? ranks : 1, 4)
        for T in (ComplexF32, ComplexF64), nc in colors, D_fix in (3, 4)
            @testset "$fixture axis=$axis halo=$halo $T NC=$nc D_fix=$D_fix" begin
                @info "Gauge-fixing validation case" fixture axis halo T nc D_fix ranks
                tol = T == ComplexF32 ? 5e-5 : 5e-12
                U = gauge_configuration(dims; colors=nc, halo,
                    start=fixture == :hot ? :hot : :cold,
                    seed=UInt64(0x189), eltype=T, process_grid=grid, comm, verbose=0)
                if fixture == :common
                    for (u, link) in zip(U, common_links(dims, nc, T))
                        source = LatticeMatrix(link, 4, grid; nw=halo, comm0=comm)
                        LatticeMatrices.substitute!(u.U, source)
                        Gaugefields.set_wing_U!(u)
                    end
                end
                initial = global_links(U)
                initial_plaquette = measure_plaquette(U)
                reference = copy_links(U)
                g = similar(U[1]); gr = similar(U[1])
                delta = similar(U[1]); dr = similar(U[1])
                temps = [similar(U[1]) for _ in 1:6]
                trf = [similar(U[1]) for _ in 1:6]
                key = "$T-nc$nc-d$D_fix-axis$axis-halo$halo"
                output = fixture == :common ? snapshots : hot_snapshots
                for (μ, link) in enumerate(initial)
                    output["$key-initial$μ"] = pack(link)
                end

                M._gaugefixing_delta!(delta, U, temps, D_fix)
                actual = gather_and_bcast_matrix(delta.U)
                @test actual ≈ dense_delta(initial, dims, D_fix) rtol=tol atol=tol
                M.get_Δ!(dr, reference, trf, D_fix)
                @test actual ≈ gather_and_bcast_matrix(dr.U) rtol=tol atol=tol

                # Compare each complete sweep with the unfused implementation.
                for parity in (0, 1)
                    M.make_g_los_alamos!(reference, gr, trf[1], parity,
                        1.7, 0.595, -0.0595, D_fix)
                    M.gUgshift!(reference, gr, trf[1])
                end
                M.gaugefixing_step_la!(U, g, 1.7, 0.595, -0.0595, temps[1]; D_fix)
                for μ in 1:4
                    @test gather_and_bcast_matrix(U[μ].U) ≈
                          gather_and_bcast_matrix(reference[μ].U) rtol=tol atol=tol
                end
                @test gather_and_bcast_matrix(g.U) ≈ gather_and_bcast_matrix(gr.U) rtol=tol atol=tol

                for parity in (0, 1)
                    M.make_g_steepest_descent!(reference, gr, dr, parity,
                        1.5, trf, D_fix)
                    M.gUgshift!(reference, gr, trf[1])
                end
                M.gaugefixing_step_sd!(U, g, delta, 1.5, temps; D_fix)
                for μ in 1:4
                    @test gather_and_bcast_matrix(U[μ].U) ≈
                          gather_and_bcast_matrix(reference[μ].U) rtol=tol atol=tol
                end

                # Full entry point, fixed iterations and convergence exit.
                gaugefixing!(U, g, 1.7, 1, 1.5, 2, 0.0, 189, temps;
                    D_fix, min_iterations=0)
                check_halos(U)
                @test measure_plaquette(U) ≈ initial_plaquette rtol=tol atol=tol
                for (μ, link) in enumerate(global_links(U))
                    @test all(isfinite, link)
                    max_unitarity = maximum(norm(link[:, :, Tuple(site)...]' *
                        link[:, :, Tuple(site)...] - I) for site in CartesianIndices(dims))
                    @test max_unitarity < tol
                    output["$key-link$μ"] = pack(link)
                end
                output["$key-g"] = pack(gather_and_bcast_matrix(g.U))
                M._gaugefixing_delta!(delta, U, temps, D_fix)
                output["$key-delta"] = pack(gather_and_bcast_matrix(delta.U))
                output["$key-observables"] = Float64[
                    measure_plaquette(U), M.validate_training(U, temps; D_fix)...]
                gaugefixing!(U, g, 1.7, 3, 1.5, 3, 1e30, 189, temps;
                    D_fix, min_iterations=0)
                check_halos(U)
                @test measure_plaquette(U) ≈ initial_plaquette rtol=tol atol=tol
            end
        end
    end
end
end
