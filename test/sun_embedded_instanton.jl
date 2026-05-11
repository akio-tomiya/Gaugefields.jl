using Gaugefields
using Test
using LinearAlgebra
using Wilsonloop: Wilsonline

const AG = Gaugefields.AbstractGaugefields_module

function sample_su2_matrix()
    c = cos(0.37)
    s = sin(0.37)
    return ComplexF64[
        c im*s
        im*s c
    ]
end

function check_su2_embedding(NC, block)
    u2 = sample_su2_matrix()
    U = fill(99.0 + 99.0im, NC, NC)

    @test AG._embed_su2_matrix_in_sun!(U, u2, block) === U
    i, j = block
    spectator = setdiff(1:NC, block)

    @test U[i, i] == u2[1, 1]
    @test U[i, j] == u2[1, 2]
    @test U[j, i] == u2[2, 1]
    @test U[j, j] == u2[2, 2]

    for c in spectator
        @test U[c, c] == 1
        for d = 1:NC
            if d != c
                @test U[c, d] == 0
                @test U[d, c] == 0
            end
        end
    end

    @test U' * U ≈ Matrix{ComplexF64}(I, NC, NC)
    @test det(U) ≈ 1
end

@testset "SU(2) matrix embedding into SU(Nc)" begin
    @test AG._validate_su2_embedding_block(3, (1, 2)) == (1, 2)
    @test AG._validate_su2_embedding_block(4, [2, 4]) == (2, 4)

    check_su2_embedding(3, (1, 2))
    check_su2_embedding(3, (1, 3))
    check_su2_embedding(3, (2, 3))
    check_su2_embedding(4, (2, 4))
    check_su2_embedding(5, (2, 5))

    @test_throws ArgumentError AG._validate_su2_embedding_block(1, (1, 2))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (1, 1))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (0, 2))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (1, 4))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (2, 1))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (1, 2, 3))
    @test_throws ArgumentError AG._embed_su2_matrix_in_sun!(zeros(3, 3), zeros(3, 3), (1, 2))
    @test_throws ArgumentError AG._embed_su2_matrix_in_sun!(zeros(3, 2), sample_su2_matrix(), (1, 2))
end

function check_su2_instanton_links(U)
    L = (U[1].NX, U[1].NY, U[1].NZ, U[1].NT)
    center = (L[1] / 2 + 0.5, L[2] / 2 + 0.5, L[3] / 2 + 0.5, L[4] / 2 + 0.5)
    radius = div(L[1], 2)

    for μ = 1:4
        for it = 1:L[4], iz = 1:L[3], iy = 1:L[2], ix = 1:L[1]
            link = AG._su2_instanton_link(μ, ix, iy, iz, it, L; center, radius)
            measured = ComplexF64[
                U[μ][1, 1, ix, iy, iz, it] U[μ][1, 2, ix, iy, iz, it]
                U[μ][2, 1, ix, iy, iz, it] U[μ][2, 2, ix, iy, iz, it]
            ]

            @test measured ≈ link
            @test link' * link ≈ Matrix{ComplexF64}(I, 2, 2)
            @test det(link) ≈ 1
        end
    end
end

@testset "SU(2) instanton link helper" begin
    check_su2_instanton_links(Oneinstanton(2, 0, 4, 4, 4, 4))
    check_su2_instanton_links(Oneinstanton(2, 1, 4, 4, 4, 4))

    L = (4, 4, 4, 4)
    anti_link = AG._su2_instanton_link(1, 1, 1, 1, 1, L; sign=-1)
    @test anti_link' * anti_link ≈ Matrix{ComplexF64}(I, 2, 2)
    @test det(anti_link) ≈ 1

    @test_throws ArgumentError AG._su2_instanton_link(0, 1, 1, 1, 1, L)
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, (4, 4, 4))
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, L; radius=0)
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, L; sign=0)
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, L; center=(1, 2, 3))
    singular_center_error = try
        AG._su2_instanton_link(1, 1, 1, 1, 1, L; center=(0, 0, 0, 0))
    catch err
        err
    end
    @test singular_center_error isa ArgumentError
    @test occursin("center must not coincide", sprint(showerror, singular_center_error))
end

function normalized_plaquette(U)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    return calculate_Plaquette(U, temp1, temp2) / (6 * U[1].NV * U[1].NC)
end

function check_sun_embedded_instanton(NC, block; NDW=0)
    L = (4, 4, 4, 4)
    U2 = Oneinstanton(2, NDW, L...)
    U = Oneinstanton_SUN_embedded(NC, L...; NDW, block)
    spectator = setdiff(1:NC, block)

    for μ = 1:4
        for it = 1:L[4], iz = 1:L[3], iy = 1:L[2], ix = 1:L[1]
            link = Matrix(U[μ][:, :, ix, iy, iz, it])
            u2 = ComplexF64[
                U2[μ][1, 1, ix, iy, iz, it] U2[μ][1, 2, ix, iy, iz, it]
                U2[μ][2, 1, ix, iy, iz, it] U2[μ][2, 2, ix, iy, iz, it]
            ]

            for b = 1:2, a = 1:2
                @test link[block[a], block[b]] ≈ u2[a, b]
            end
            for c in spectator
                @test link[c, c] == 1
                for d = 1:NC
                    if d != c
                        @test link[c, d] == 0
                        @test link[d, c] == 0
                    end
                end
            end

            @test link' * link ≈ Matrix{ComplexF64}(I, NC, NC)
            @test det(link) ≈ 1
        end
    end

    @test normalized_plaquette(U) ≈ (2 * normalized_plaquette(U2) + (NC - 2)) / NC
end

@testset "SUN embedded instanton public API" begin
    check_sun_embedded_instanton(3, (1, 2))
    check_sun_embedded_instanton(3, (1, 3))
    check_sun_embedded_instanton(3, (2, 3))
    check_sun_embedded_instanton(4, (2, 4))
    check_sun_embedded_instanton(5, (2, 5))
    check_sun_embedded_instanton(3, (1, 3); NDW=1)

    @test_throws ArgumentError Oneinstanton_SUN_embedded(3, 4, 4, 4, 4; block=(1, 1))
    @test_throws ArgumentError Oneinstanton_SUN_embedded(3, 4, 4, 4, 4; block=(1, 4))
    @test_throws ArgumentError Oneinstanton_SUN_embedded(3, 4, 4, 4, 4; NDW=-1)
    singular_center_error = try
        Oneinstanton_SUN_embedded(3, 5, 5, 5, 5)
    catch err
        err
    end
    @test singular_center_error isa ArgumentError
    @test occursin("center must not coincide", sprint(showerror, singular_center_error))
end

plaquette_topological_charge(U) = AG._plaquette_topological_charge(U)
plaquette_topological_charge_density(U) = AG._plaquette_topological_charge_density(U)
clover_topological_charge(U) = AG._clover_topological_charge(U)
clover_topological_charge_density(U) = AG._clover_topological_charge_density(U)
rectangle_topological_charge(U) = AG._rectangle_topological_charge(U)
rectangle_topological_charge_density(U) = AG._rectangle_topological_charge_density(U)
improved_topological_charge(U) = AG._improved_topological_charge(U)
improved_topological_charge_density(U) = AG._improved_topological_charge_density(U)

const PUBLIC_TOPOLOGICAL_CHARGE_METHODS = (:plaquette, :clover, :improved)

function test_topological_charge_method_guard(U)
    density_error = try
        topological_charge_density(U; method=:rect)
    catch err
        err
    end
    @test density_error isa ArgumentError
    @test occursin("supported topological_charge_density methods", sprint(showerror, density_error))

    charge_error = try
        topological_charge(U; method=:rect)
    catch err
        err
    end
    @test charge_error isa ArgumentError
    @test occursin("supported topological_charge methods", sprint(showerror, charge_error))
end

function test_topological_charge_storage_guard(U)
    for method in PUBLIC_TOPOLOGICAL_CHARGE_METHODS
        density_error = try
            topological_charge_density(U; method)
        catch err
            err
        end
        @test density_error isa ArgumentError
        @test occursin("topological charge only supports serial 4D gauge fields",
            sprint(showerror, density_error))

        charge_error = try
            topological_charge(U; method)
        catch err
            err
        end
        @test charge_error isa ArgumentError
        @test occursin("topological charge only supports serial 4D gauge fields",
            sprint(showerror, charge_error))
    end
end

function test_topological_charge_density_contract(U)
    physical_size = (U[1].NX, U[1].NY, U[1].NZ, U[1].NT)
    storage_size = Tuple(size(U[1].U)[3:6])
    for method in PUBLIC_TOPOLOGICAL_CHARGE_METHODS
        q = topological_charge_density(U; method)
        Q = topological_charge(U; method)
        @test q isa Array{Float64,4}
        @test size(q) == physical_size
        @test axes(q) == map(Base.OneTo, physical_size)
        U[1].NDW > 0 && @test size(q) != storage_size
        @test isapprox(sum(q), Q; rtol=1e-12, atol=1e-12)
    end
end

function clover_reference_topological_charge(U)
    temps = [similar(U[1]), similar(U[1]), similar(U[1]), similar(U[1]), similar(U[1])]
    F = Matrix{eltype(U)}(undef, 4, 4)
    for μ = 1:4
        for ν = 1:4
            F[μ, ν] = similar(U[1])
            if μ != ν
                evaluate_gaugelinks!(temps[1], AG.make_cloverloops(μ, ν, Dim=4), U, temps[2:end])
                Traceless_antihermitian!(F[μ, ν], temps[1])
            end
        end
    end

    Q = 0.0
    numofloops = 4
    for μ = 1:4
        for ν = 1:4
            μ == ν && continue
            for ρ = 1:4
                for σ = 1:4
                    ρ == σ && continue
                    Q += AG._epsilon_tensor_4d(μ, ν, ρ, σ) * tr(F[μ, ν], F[ρ, σ]) / numofloops^2
                end
            end
        end
    end
    return -real(Q) / (32 * pi^2)
end

function rectangle_reference_loops(μ, ν; Dim=4)
    loops = Wilsonline{Dim}[]

    push!(loops, Wilsonline([(μ, 2), (ν, 1), (μ, -2), (ν, -1)], Dim=Dim))
    push!(loops, Wilsonline([(ν, 1), (μ, -2), (ν, -1), (μ, 2)], Dim=Dim))
    push!(loops, Wilsonline([(ν, -1), (μ, 2), (ν, 1), (μ, -2)], Dim=Dim))
    push!(loops, Wilsonline([(μ, -2), (ν, -1), (μ, 2), (ν, 1)], Dim=Dim))

    push!(loops, Wilsonline([(μ, 1), (ν, 2), (μ, -1), (ν, -2)], Dim=Dim))
    push!(loops, Wilsonline([(ν, 2), (μ, -1), (ν, -2), (μ, 1)], Dim=Dim))
    push!(loops, Wilsonline([(ν, -2), (μ, 1), (ν, 2), (μ, -1)], Dim=Dim))
    push!(loops, Wilsonline([(μ, -1), (ν, -2), (μ, 1), (ν, 2)], Dim=Dim))

    return loops
end

function rectangle_reference_topological_charge(U)
    temps = [similar(U[1]), similar(U[1]), similar(U[1]), similar(U[1]), similar(U[1])]
    F = Matrix{eltype(U)}(undef, 4, 4)
    for μ = 1:4
        for ν = 1:4
            F[μ, ν] = similar(U[1])
            if μ != ν
                evaluate_gaugelinks!(temps[1], rectangle_reference_loops(μ, ν, Dim=4), U, temps[2:end])
                Traceless_antihermitian!(F[μ, ν], temps[1])
            end
        end
    end

    Q = 0.0
    numofloops = 8
    for μ = 1:4
        for ν = 1:4
            μ == ν && continue
            for ρ = 1:4
                for σ = 1:4
                    ρ == σ && continue
                    Q += AG._epsilon_tensor_4d(μ, ν, ρ, σ) * tr(F[μ, ν], F[ρ, σ]) / numofloops^2
                end
            end
        end
    end
    return 2 * (-real(Q) / (32 * pi^2))
end

function improved_reference_topological_charge(U)
    c0 = 5 / 3
    c1 = -1 / 12
    return c0 * clover_reference_topological_charge(U) + c1 * rectangle_reference_topological_charge(U)
end

function wilson_action_over_8pi2(U)
    NC = size(U[1].U, 1)
    volume = U[1].NX * U[1].NY * U[1].NZ * U[1].NT
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    plaquette = calculate_Plaquette(U, temp1, temp2) / (NC * 6 * volume)
    return NC * 6 * volume * (1 - plaquette) / (8 * pi^2)
end

@testset "SUN embedded instanton topological charge" begin
    L = (4, 4, 4, 4)
    cold = Initialize_Gaugefields(3, 0, L..., condition="cold")
    test_topological_charge_density_contract(cold)
    cold_density = plaquette_topological_charge_density(cold)
    @test size(cold_density) == L
    @test all(isapprox.(cold_density, 0; atol=1e-12))
    @test isapprox(sum(cold_density), plaquette_topological_charge(cold); atol=1e-12)
    cold_clover_density = clover_topological_charge_density(cold)
    @test size(cold_clover_density) == L
    @test all(isapprox.(cold_clover_density, 0; atol=1e-12))
    @test isapprox(sum(cold_clover_density), clover_topological_charge(cold); atol=1e-12)
    @test isapprox(clover_topological_charge(cold), clover_reference_topological_charge(cold); atol=1e-12)
    cold_rectangle_density = rectangle_topological_charge_density(cold)
    @test size(cold_rectangle_density) == L
    @test all(isapprox.(cold_rectangle_density, 0; atol=1e-12))
    @test isapprox(sum(cold_rectangle_density), rectangle_topological_charge(cold); atol=1e-12)
    @test isapprox(rectangle_topological_charge(cold), rectangle_reference_topological_charge(cold); atol=1e-12)
    cold_improved_density = improved_topological_charge_density(cold)
    @test size(cold_improved_density) == L
    @test all(isapprox.(cold_improved_density, 0; atol=1e-12))
    @test isapprox(sum(cold_improved_density), improved_topological_charge(cold); atol=1e-12)
    @test isapprox(improved_topological_charge(cold), improved_reference_topological_charge(cold); atol=1e-12)

    U2 = Oneinstanton(2, 0, L...)
    test_topological_charge_density_contract(U2)
    q2 = plaquette_topological_charge_density(U2)
    Q2 = plaquette_topological_charge(U2)
    @test abs(Q2) > 0.1
    @test size(q2) == L
    @test isapprox(sum(q2), Q2; rtol=1e-12, atol=1e-12)
    @test topological_charge_density(U2) ≈ q2
    @test topological_charge(U2) ≈ Q2
    q2_clover = clover_topological_charge_density(U2)
    Q2_clover = clover_topological_charge(U2)
    @test size(q2_clover) == L
    @test isapprox(sum(q2_clover), Q2_clover; rtol=1e-12, atol=1e-12)
    @test isapprox(Q2_clover, clover_reference_topological_charge(U2); rtol=1e-12, atol=1e-12)
    @test topological_charge_density(U2; method=:clover) ≈ q2_clover
    @test topological_charge(U2; method=:clover) ≈ Q2_clover
    q2_rectangle = rectangle_topological_charge_density(U2)
    Q2_rectangle = rectangle_topological_charge(U2)
    @test size(q2_rectangle) == L
    @test isapprox(sum(q2_rectangle), Q2_rectangle; rtol=1e-12, atol=1e-12)
    @test isapprox(Q2_rectangle, rectangle_reference_topological_charge(U2); rtol=1e-12, atol=1e-12)
    q2_improved = improved_topological_charge_density(U2)
    Q2_improved = improved_topological_charge(U2)
    @test size(q2_improved) == L
    @test isapprox(sum(q2_improved), Q2_improved; rtol=1e-12, atol=1e-12)
    @test isapprox(q2_improved, (5 / 3) .* q2_clover .- (1 / 12) .* q2_rectangle; rtol=1e-12, atol=1e-12)
    @test isapprox(Q2_improved, improved_reference_topological_charge(U2); rtol=1e-12, atol=1e-12)
    @test topological_charge_density(U2; method=:improved) ≈ q2_improved
    @test topological_charge(U2; method=:improved) ≈ Q2_improved
    test_topological_charge_method_guard(U2)

    accelerator_field = Initialize_Gaugefields(3, 0, L...; condition="cold", cuda=true)
    test_topological_charge_storage_guard(accelerator_field)

    mpi_field = Initialize_Gaugefields(3, 0, L...; condition="cold", mpi=true, PEs=(1, 1, 1, 1), mpiinit=false)
    test_topological_charge_storage_guard(mpi_field)

    cold_wing = Initialize_Gaugefields(3, 1, 2, 3, 4, 5; condition="cold")
    test_topological_charge_density_contract(cold_wing)

    U3 = Oneinstanton_SUN_embedded(3, L...; block=(1, 2))
    U3_alt = Oneinstanton_SUN_embedded(3, L...; block=(2, 3))
    U4 = Oneinstanton_SUN_embedded(4, L...; block=(2, 4))
    U5 = Oneinstanton_SUN_embedded(5, L...; block=(2, 5))

    @test plaquette_topological_charge(U3) ≈ Q2
    @test plaquette_topological_charge(U3_alt) ≈ Q2
    @test plaquette_topological_charge(U4) ≈ Q2
    @test plaquette_topological_charge(U5) ≈ Q2
    @test topological_charge(U3) ≈ Q2
    @test isapprox(plaquette_topological_charge_density(U3), q2; rtol=1e-12, atol=1e-12)
    @test isapprox(plaquette_topological_charge_density(U3_alt), q2; rtol=1e-12, atol=1e-12)
    @test isapprox(plaquette_topological_charge_density(U4), q2; rtol=1e-12, atol=1e-12)
    @test isapprox(plaquette_topological_charge_density(U5), q2; rtol=1e-12, atol=1e-12)
    @test clover_topological_charge(U3) ≈ Q2_clover
    @test clover_topological_charge(U3_alt) ≈ Q2_clover
    @test clover_topological_charge(U4) ≈ Q2_clover
    @test clover_topological_charge(U5) ≈ Q2_clover
    @test isapprox(clover_topological_charge_density(U3), q2_clover; rtol=1e-12, atol=1e-12)
    @test isapprox(clover_topological_charge_density(U3_alt), q2_clover; rtol=1e-12, atol=1e-12)
    @test topological_charge(U3; method=:clover) ≈ Q2_clover
    @test isapprox(topological_charge_density(U3; method=:clover), q2_clover; rtol=1e-12, atol=1e-12)
    @test rectangle_topological_charge(U3) ≈ Q2_rectangle
    @test rectangle_topological_charge(U3_alt) ≈ Q2_rectangle
    @test rectangle_topological_charge(U4) ≈ Q2_rectangle
    @test rectangle_topological_charge(U5) ≈ Q2_rectangle
    @test isapprox(rectangle_topological_charge_density(U3), q2_rectangle; rtol=1e-12, atol=1e-12)
    @test isapprox(rectangle_topological_charge_density(U3_alt), q2_rectangle; rtol=1e-12, atol=1e-12)
    @test improved_topological_charge(U3) ≈ Q2_improved
    @test improved_topological_charge(U3_alt) ≈ Q2_improved
    @test improved_topological_charge(U4) ≈ Q2_improved
    @test improved_topological_charge(U5) ≈ Q2_improved
    @test isapprox(improved_topological_charge_density(U3), q2_improved; rtol=1e-12, atol=1e-12)
    @test isapprox(improved_topological_charge_density(U3_alt), q2_improved; rtol=1e-12, atol=1e-12)
    @test topological_charge(U3; method=:improved) ≈ Q2_improved
    @test isapprox(topological_charge_density(U3; method=:improved), q2_improved; rtol=1e-12, atol=1e-12)

    U3_anti = Oneinstanton_SUN_embedded(3, L...; block=(1, 2), sign=-1)
    @test plaquette_topological_charge(U3_anti) ≈ -Q2
    q3_anti = plaquette_topological_charge_density(U3_anti)
    @test isapprox(sum(q3_anti), -Q2; rtol=1e-12, atol=1e-12)
    @test maximum(abs.(q3_anti .+ q2)) < 2e-5
    q3_anti_clover = clover_topological_charge_density(U3_anti)
    @test isapprox(sum(q3_anti_clover), -Q2_clover; rtol=1e-12, atol=1e-12)
    q3_anti_rectangle = rectangle_topological_charge_density(U3_anti)
    @test isapprox(sum(q3_anti_rectangle), -Q2_rectangle; rtol=1e-12, atol=1e-12)
    q3_anti_improved = improved_topological_charge_density(U3_anti)
    @test isapprox(sum(q3_anti_improved), -Q2_improved; rtol=1e-12, atol=1e-12)
end

@testset "SUN embedded unit-charge instanton regression" begin
    L = (12, 12, 12, 12)
    U2 = Oneinstanton_SUN_embedded(2, L...; NDW=0, radius=3.0, verbose_level=0)
    Qplaq = topological_charge(U2; method=:plaquette)
    Qclover = topological_charge(U2; method=:clover)
    Qimproved = topological_charge(U2; method=:improved)

    @test isapprox(Qplaq, 0.911527333054419; rtol=1e-12, atol=1e-12)
    @test isapprox(Qclover, 0.852911335839915; rtol=1e-12, atol=1e-12)
    @test isapprox(Qimproved, 0.930245472598196; rtol=1e-12, atol=1e-12)
    @test abs(Qimproved - 1) < 0.1
    S_over_8pi2 = wilson_action_over_8pi2(U2)
    @test abs(S_over_8pi2 - 1) < 0.08
    @test abs(S_over_8pi2 - abs(Qimproved)) < 0.12

    q_improved = topological_charge_density(U2; method=:improved)
    @test size(q_improved) == L
    @test isapprox(sum(q_improved), Qimproved; rtol=1e-12, atol=1e-12)

    U3 = Oneinstanton_SUN_embedded(3, L...; NDW=0, radius=3.0, block=(1, 2), verbose_level=0)
    @test isapprox(topological_charge(U3; method=:plaquette), Qplaq; rtol=1e-12, atol=1e-12)
    @test isapprox(topological_charge(U3; method=:clover), Qclover; rtol=1e-12, atol=1e-12)
    @test isapprox(topological_charge(U3; method=:improved), Qimproved; rtol=1e-12, atol=1e-12)
    @test isapprox(sum(topological_charge_density(U3; method=:improved)), Qimproved; rtol=1e-12, atol=1e-12)

    U3_anti = Oneinstanton_SUN_embedded(3, L...; NDW=0, radius=3.0, block=(1, 2), sign=-1, verbose_level=0)
    @test isapprox(topological_charge(U3_anti; method=:improved), -Qimproved; rtol=1e-12, atol=1e-12)
end
