module MILCIOTests
using Gaugefields, LatticeMatrices, Test, LinearAlgebra, JACC
JACC.@init_backend
const comm = LatticeMatrices._default_communicator()
const rank = LatticeMatrices._comm_rank(comm)
const ranks = LatticeMatrices._comm_size(comm)
root(f) = Gaugefields._portable_jld2_root_operation(f, comm)

# test/data/milc_l2468.lat was written by the MILC code (github.com/milc-qcd/milc_qcd,
# commit 551dc47, pure_gauge/su3_ora, beta = 6.0, 5 overrelaxed heatbath
# trajectories from a fresh start, saved with save_serial on x86-64). The
# reference values below are printed by MILC when it reloads the file with
# reload_serial: "Checksums b1e0dc31 b167c647 OK", "CHECK PLAQ", "CHECK NERSC
# LINKTR ... CKSUM", and the Polyakov loop in "GMES".
const fixture = joinpath(@__DIR__, "data", "milc_l2468.lat")
const fixture_dims = (2, 4, 6, 8)
const fixture_stamp = "Sat Nov 14 23:18:50 2026"
const milc_ssplaq = 1.8079896238585031
const milc_stplaq = 1.7613458892199576
const milc_linktr = -4.2750023222137340e-03
const milc_nersc_checksum = 0x9b321e3b
const milc_polyakov = complex(-5.324294e-02, 4.673888e-02)

# MILC ploop: sum over spatial sites of tr(U_t(x,0) U_t(x,1) ... U_t(x,nt-1)),
# divided by the spatial volume (not by the number of colors).
function milc_polyakov_loop(links)
    nx, ny, nz, nt = size(links[4])[3:end]
    total = zero(ComplexF64)
    for x in 1:nx, y in 1:ny, z in 1:nz
        p = Matrix{ComplexF64}(I, 3, 3)
        for t in 1:nt
            p *= ComplexF64.(links[4][:, :, x, y, z, t])
        end
        total += tr(p)
    end
    return total / (nx * ny * nz)
end

function header_value(file, key)
    for line in eachline(file)
        startswith(line, key) && return strip(split(line, '='; limit=2)[2], [' ', '"'])
    end
    error("missing $key in $file")
end

# Rewrite a MILC file in node-dump order: a site list (order = 1) followed by
# the checksum and the links in the order of the list.
function write_site_list_order(src, dst, permutation)
    header, words = open(src, "r") do io
        header = read(io, Gaugefields._MILC_HEADER_BYTES)
        read(io, 8)
        payload = Vector{UInt32}(undef, Gaugefields._MILC_WORDS_PER_SITE * prod(fixture_dims))
        read!(io, payload)
        (header, ltoh.(payload))
    end
    n = Gaugefields._MILC_WORDS_PER_SITE
    reordered = similar(words)
    for (record, site) in enumerate(permutation)
        reordered[(record-1)*n+1:record*n] = words[site*n+1:(site+1)*n]
    end
    sum29, sum31 = Gaugefields._milc_checksums(reordered)
    header[end-3:end] = reinterpret(UInt8, [htol(UInt32(1))])
    open(dst, "w") do io
        write(io, header)
        write(io, htol.(UInt32.(permutation)))
        write(io, htol(sum29), htol(sum31))
        write(io, htol.(reordered))
    end
end

@testset "MILC I/O" begin
    directory = root(() -> mktempdir())
    directory = Gaugefields.broadcast(directory, 0, comm)
    try
        grid = (1, 1, 1, ranks)
        V = load_configuration(fixture; format=:milc, eltype=ComplexF64, process_grid=grid, comm)
        @test gauge_lattice_size(V) == fixture_dims
        @test eltype(V[1].U.A) == ComplexF64
        links = gather_and_bcast_matrix.(getproperty.(V, :U))

        @testset "values printed by MILC" begin
            obs = Gaugefields._milc_observables(links)
            @test obs.ssplaq ≈ milc_ssplaq atol = 1e-12
            @test obs.stplaq ≈ milc_stplaq atol = 1e-12
            @test obs.linktrace ≈ milc_linktr atol = 1e-12
            @test Gaugefields._milc_nersc_checksum(links) == milc_nersc_checksum
            @test milc_polyakov_loop(links) ≈ milc_polyakov atol = 1e-8
            @test measure_plaquette(V) ≈ (milc_ssplaq + milc_stplaq) / 6 atol = 1e-12
            @test 3measure_polyakov_loop(V) ≈ milc_polyakov atol = 1e-8
        end

        @testset "cross-format: NERSC checksum" begin
            # MILC's nersc_cksum sums the Float32 bits of the first two rows,
            # which is the NERSC checksum of a 32-bit two-row file.
            nersc = joinpath(directory, "same.nersc")
            save_configuration(nersc, V; format=:nersc, precision=32, compressed=true)
            root() do
                @test parse(UInt32, header_value(nersc, "CHECKSUM"); base=16) == milc_nersc_checksum
            end
            W = load_configuration(nersc; format=:nersc, eltype=ComplexF64, process_grid=grid, comm)
            @test measure_plaquette(W) ≈ measure_plaquette(V) atol = 1e-6
        end

        @testset "byte-identical rewrite" begin
            rewritten = joinpath(directory, "copy.lat")
            save_configuration(rewritten, V; format=:milc, time_stamp=fixture_stamp)
            root() do
                @test read(rewritten) == read(fixture)
                info = rewritten * ".info"
                @test header_value(info, "checksums") == header_value(fixture * ".info", "checksums")
                @test header_value(info, "gauge.nersc_checksum") == header_value(fixture * ".info", "gauge.nersc_checksum")
                for key in ("gauge.ssplaq", "gauge.stplaq", "gauge.nersc_linktr")
                    @test parse(Float64, header_value(info, key)) ≈
                          parse(Float64, header_value(fixture * ".info", key)) atol = 1.1e-6
                end
            end
        end

        @testset "precision, byte order, and site order" begin
            default = load_configuration(fixture; format=:milc, process_grid=grid, comm)
            @test eltype(default[1].U.A) == ComplexF64
            single = load_configuration(fixture; format=:milc, eltype=ComplexF32, process_grid=grid, comm)
            @test eltype(single[1].U.A) == ComplexF32
            for mu in 1:4
                @test ComplexF64.(gather_and_bcast_matrix(single[mu].U)) == links[mu]
            end
            bigfile = joinpath(directory, "big.lat")
            save_configuration(bigfile, V; format=:milc, endian=:big, info=false)
            root() do
                @test !isfile(bigfile * ".info")
                @test read(bigfile, UInt32) == hton(UInt32(20103))
            end
            B = load_configuration(bigfile; format=:milc, eltype=ComplexF64, process_grid=grid, comm)
            @test all(gather_and_bcast_matrix(B[mu].U) == links[mu] for mu in 1:4)

            listed = joinpath(directory, "listed.lat")
            root(() -> write_site_list_order(fixture, listed, reverse(0:prod(fixture_dims)-1)))
            L = load_configuration(listed; format=:milc, eltype=ComplexF64, process_grid=grid, comm)
            @test all(gather_and_bcast_matrix(L[mu].U) == links[mu] for mu in 1:4)
        end

        @testset "experimental warning and reunitarization" begin
            warnings(f) = [string(l.message) for l in first(Test.collect_test_logs(f))
                           if l.level == Base.CoreLogging.Warn]
            shown = warnings(() -> load_configuration!(V, fixture; format=:milc))
            if rank == 0
                @test any(m -> occursin("experimental", m) && occursin("reunitarize=false", m), shown)
            else
                @test isempty(shown)
            end
            @test all(gather_and_bcast_matrix(V[mu].U) == links[mu] for mu in 1:4)

            R = nothing
            shown = warnings() do
                R = load_configuration(fixture; format=:milc, reunitarize=true, process_grid=grid, comm)
            end
            rank == 0 && @test any(m -> occursin("experimental", m) && occursin("reunitarize=true", m), shown)
            projected = gather_and_bcast_matrix.(getproperty.(R, :U))
            # normalize_U! agrees with MILC reunit_su3 (rows 1 and 2 by
            # Gram-Schmidt, row 3 as the conjugate cross product).
            reference_difference = 0.0
            for mu in 1:4, s in CartesianIndices(fixture_dims)
                u = links[mu][:, :, Tuple(s)...]
                r1 = u[1, :] / norm(u[1, :])
                r2 = u[2, :] - dot(r1, u[2, :]) * r1
                r2 /= norm(r2)
                r3 = conj(cross(r1, r2))
                reference = permutedims(hcat(r1, r2, r3))
                reference_difference = max(reference_difference,
                    maximum(abs, projected[mu][:, :, Tuple(s)...] - reference))
            end
            @test reference_difference < 1e-14
            @test Gaugefields._milc_unitarity_deviation(links) > 1e-8
            @test Gaugefields._milc_unitarity_deviation(projected) < 1e-14
            @test all(abs(det(projected[mu][:, :, Tuple(s)...]) - 1) < 1e-14
                      for mu in 1:4 for s in CartesianIndices(fixture_dims))
            @test all(maximum(abs, projected[mu] - links[mu]) < 1e-6 for mu in 1:4)
            obs = Gaugefields._milc_observables(projected)
            @test obs.ssplaq ≈ milc_ssplaq atol = 1e-6
            @test obs.stplaq ≈ milc_stplaq atol = 1e-6

            shown = warnings(() -> save_configuration(joinpath(directory, "w.lat"), V; format=:milc))
            rank == 0 && @test any(m -> occursin("experimental", m), shown)
            @test_throws ArgumentError load_configuration!(V, fixture; format=:jld2, reunitarize=true)
            @test_throws ArgumentError load_configuration(fixture; format=:nersc, reunitarize=true)

            # Links far from SU(3) are refused, as MILC does above 1e-4.
            scaled = joinpath(directory, "scaled.lat")
            root(() -> Gaugefields._write_milc(scaled, [1.01 .* a for a in links]; info=false))
            @test_throws ErrorException load_configuration!(V, scaled; format=:milc, reunitarize=true)
            @test all(gather_and_bcast_matrix(V[mu].U) == links[mu] for mu in 1:4)
            shown = warnings(() -> load_configuration!(V, scaled; format=:milc))
            rank == 0 && @test any(m -> occursin("reunitarize=false", m), shown)
            load_configuration!(V, fixture; format=:milc)
        end

        @testset "legacy backend" begin
            if ranks == 1
                legacy = gauge_configuration(fixture_dims; backend=LegacyBackend(), colors=3,
                    start=:cold, verbose=0)
                load_configuration!(legacy, fixture; format=:milc)
                @test measure_plaquette(legacy) ≈ measure_plaquette(V) atol = 1e-12
            end
        end

        @testset "hot configuration round trip" begin
            dims = (2, 2, 2, 4)
            U = gauge_configuration(dims; colors=3, halo=1, start=:hot, seed=UInt64(20103),
                process_grid=(1, 1, 1, ranks), comm, verbose=0)
            originals = gather_and_bcast_matrix.(getproperty.(U, :U))
            file = joinpath(directory, "hot.lat")
            for endian in (:little, :big)
                save_configuration(file, U; format=:milc, endian)
                for T in (ComplexF32, ComplexF64)
                    R = load_configuration(file; format=:milc, eltype=T, process_grid=(1, 1, 1, ranks), comm)
                    @test eltype(R[1].U.A) == T
                    for mu in 1:4
                        actual = gather_and_bcast_matrix(R[mu].U)
                        @test actual ≈ originals[mu] rtol = 3e-7 atol = 3e-7
                        @test actual == ComplexF32.(originals[mu])
                        @test !LatticeMatrices.halo_is_dirty(R[mu].U)
                    end
                end
            end
        end

        @testset "input validation" begin
            before = [copy(a) for a in links]
            bytes = root(() -> read(fixture))
            bad = joinpath(directory, "bad.lat")
            for (label, mutate) in (
                ("flipped data bit", b -> (b[end-100] ⊻= 0x01; b)),
                ("flipped checksum", b -> (b[90] ⊻= 0x01; b)),
                ("truncated", b -> b[1:end-1]),
                ("extra byte", b -> vcat(b, UInt8[0])),
                ("unknown magic", b -> (b[1] = 0x00; b)),
                ("zero dimension", b -> (b[5:8] .= 0x00; b)),
            )
                root(() -> write(bad, mutate(copy(bytes))))
                @test_throws ErrorException load_configuration!(V, bad; format=:milc)
                @test all(gather_and_bcast_matrix(V[mu].U) == before[mu] for mu in 1:4)
            end
            nersc = joinpath(directory, "same.nersc")
            err = try
                load_configuration!(V, nersc; format=:milc)
                nothing
            catch e
                e
            end
            @test err isa ErrorException && occursin("format=:nersc", err.msg)
            other = gauge_configuration((2, 2, 2, 4); colors=3, halo=1, start=:cold,
                process_grid=(1, 1, 1, ranks), comm, verbose=0)
            @test_throws DimensionMismatch load_configuration!(other, fixture; format=:milc)
            @test_throws ErrorException save_configuration(joinpath(directory, "x.lat"), V;
                format=:milc, endian=:middle)
        end
    finally
        root(() -> rm(directory; recursive=true, force=true))
    end
end
end
