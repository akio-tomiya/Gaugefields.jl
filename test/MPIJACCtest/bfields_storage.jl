using Gaugefields, LatticeMatrices, LinearAlgebra, Test

@testset "compact B storage and matrix compatibility" begin
    nprocs = LatticeMatrices._comm_size(LatticeMatrices._default_communicator())
    lattice = (2nprocs, 2, 2, 2)
    grid = (nprocs, 1, 1, 1)
    flux = [1, 0, 1, 1, 0, 1]
    planes = ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
    native(B) = Gaugefields.Bfield_module._uses_scalar_storage(B.u)
    for nc in (2, 3), nw in (0, 1), T in (ComplexF32, ComplexF64), condition in ("tflux", "tloop")
        tol = T == ComplexF32 ? 5e-4 : 2e-10
        args = (; isMPILattice=true, PEs=grid, elementtype=T, condition, verbose_level=0)
        B = Initialize_Bfields(nc, flux, nw, lattice...; args...)
        M = Initialize_Bfields(nc, flux, nw, lattice...; args..., bfield_storage=:matrix)
        @test native(B)
        @test !native(M)
        @test all(isnothing, B.u.matrices)
        @test sum(length(B.u.phases[mu, nu].scalar.A) for (mu, nu) in planes) * (2nc^2) ==
              sum(length(M[mu, nu].U.A) + length(M[nu, mu].U.A) for (mu, nu) in planes)
        U = Initialize_Gaugefields(nc, nw, lattice...;
            isMPILattice=true, PEs=grid, elementtype=T,
            condition="hot", randomnumber="Reproducible", verbose_level=0)
        loops = make_loops_fromname("plaquette"; Dim=4)
        append!(loops, loops')
        action = GaugeAction(U, B)
        push!(action, 1.0, loops)
        value = evaluate_GaugeAction(action, U, B)
        @test value ≈ evaluate_GaugeAction(action, U, M) atol=tol rtol=tol
        @test calculate_Plaquette(U, B, similar(U[1]), similar(U[1])) ≈ real(value) / 2 atol=tol rtol=tol
        @test native(B)

        C = similar(B)
        substitute_U!(C, B)
        substitute_U!(C, B, false)
        @test native(C)
        @test evaluate_GaugeAction(action, U, C) ≈ value atol=tol rtol=tol
        phase = get_Bphase(B, 1, 2)
        @test get_Bplane(B, 1, 2) isa ScaledIdentityLattice
        mul!(phase, -one(T))
        @test !isapprox(evaluate_GaugeAction(action, U, B), value; atol=tol, rtol=tol)
        @test_throws ArgumentError B[1, 2]
        # Matrix conversion must not mutate or materialize the native source.
        substitute_U!(M, B)
        @test native(B)
        @test evaluate_GaugeAction(action, U, B) ≈ evaluate_GaugeAction(action, U, M) atol=tol rtol=tol

        # Existing B[mu,nu] returns an authoritative, mutable matrix field.
        exposed = C[1, 2]
        @test !native(C)
        @test C[1, 2] === exposed
        @test_throws ArgumentError get_Bphase(C, 1, 2)
        mul!(exposed.U, -one(T))
        set_wing_U!(exposed)
        @test evaluate_GaugeAction(action, U, C) ≈ evaluate_GaugeAction(action, U, B) atol=tol rtol=tol
        @test typeof(C[2, 1]) == typeof(M[2, 1])
    end
end
