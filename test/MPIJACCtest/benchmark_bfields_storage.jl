using Gaugefields, LatticeMatrices, Random
import JACC
JACC.@init_backend

function compare_us(functions; samples=31)
    for f in functions, _ in 1:4
        f()
    end
    JACC.synchronize()
    GC.gc()
    times = [Float64[] for _ in functions]
    rng = MersenneTwister(92426)
    for _ in 1:samples, k in randperm(rng, length(functions))
        start = time_ns()
        functions[k]()
        JACC.synchronize()
        push!(times[k], (time_ns() - start) / 1e3)
    end
    return map(t -> sort(t)[cld(samples, 2)], times)
end

function benchmark_B_storage()
    println((gaugefields=Base.pkgversion(Gaugefields),
        latticematrices=Base.pkgversion(LatticeMatrices), jacc=Base.pkgversion(JACC),
        julia=VERSION, threads=Threads.nthreads()))
    n = isempty(ARGS) ? 4 : parse(Int, ARGS[1])
    nc, lattice = 3, (n,n,n,n)
    args = (; isMPILattice=true, PEs=(1,1,1,1), verbose_level=0)
    U = Initialize_Gaugefields(nc, 1, lattice...; args..., condition="hot", randomnumber="Reproducible")
    B = Initialize_Bfields(nc, [1,0,1,1,0,1], 1, lattice...; args...)
    full = Initialize_Bfields(nc, [1,0,1,1,0,1], 1, lattice...; args..., bfield_storage=:matrix)
    action = GaugeAction(U, B)
    plaq = make_loops_fromname("plaquette"; Dim=4); append!(plaq, plaq')
    rect = make_loops_fromname("rectangular"; Dim=4); append!(rect, rect')
    push!(action, 0.7, plaq); push!(action, -0.13, rect)
    temps = [similar(U[1]) for _ in 1:4]
    out = similar(U[1])
    @assert isapprox(evaluate_GaugeAction(action,U,B),evaluate_GaugeAction(action,U,full); atol=1e-10)
    for (name, f, matrix_output) in (
        ("B plaquette", b -> evaluate_Bplaquettes!(out,plaq[1],b,temps), true),
        ("B rectangle", b -> evaluate_Bplaquettes!(out,rect[1],b,temps), true),
        ("mixed action", b -> evaluate_GaugeAction(action,U,b), false),
        ("mixed force mu1", b -> calc_dSdUμ!(out,action,1,U,b), true),
    )
        full_value = f(full)
        full_value = matrix_output ? gather_and_bcast_matrix(out.U) : full_value
        compact_value = f(B)
        compact_value = matrix_output ? gather_and_bcast_matrix(out.U) : compact_value
        difference = matrix_output ? maximum(abs, full_value - compact_value) : abs(full_value - compact_value)
        @assert difference < 1e-10
        full_us, compact_us = compare_us((() -> f(full), () -> f(B)))
        println((name=name, full_us=full_us, compact_us=compact_us,
            speedup=full_us/compact_us, maxdiff=difference))
    end
end

benchmark_B_storage()
