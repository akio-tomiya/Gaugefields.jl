module BFlowRegressionTests

using Gaugefields, LatticeMatrices, LinearAlgebra, Test

const BF = Gaugefields.Bfield_module
const PLANES = ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
const NPROCS = LatticeMatrices._comm_size(LatticeMatrices._default_communicator())
const LATTICE = (max(4, 2NPROCS), 4, 4, 4)
const GRID = (NPROCS, 1, 1, 1)

copylinks(U) = (V = similar(U); substitute_U!(V, U); V)
function fieldarray(u)
    if u isa Gaugefields.AbstractGaugefields_module.Gaugefields_4D_MPILattice
        return gather_and_bcast_matrix(u.U)
    end
    return [u[i,j,x,y,z,t] for i=1:u.NC, j=1:u.NC,
        x=1:LATTICE[1], y=1:LATTICE[2], z=1:LATTICE[3], t=1:LATTICE[4]]
end
arrays(U) = fieldarray.(U)
maxdiff(A, B) = maximum(maximum(abs, a-b) for (a,b) in zip(A,B))

function serialcopy(data, nc, nw)
    U = Initialize_Gaugefields(nc, nw, LATTICE...; condition="cold", verbose_level=0)
    for (u, a) in zip(U, data), site in CartesianIndices(LATTICE), j=1:nc, i=1:nc
        u[i,j,Tuple(site)...] = a[i,j,Tuple(site)...]
    end
    set_wing_U!(U)
    return U
end

generalflow(U, B) = Gradientflow_general_Bfields(
    U, B, ["plaquette", "rectangular"], [1.0, 0.1]; eps=0.005, Nflow=1)

# Independent variation of the links, not a second staple evaluator.
function check_finite_difference(U, B, action)
    nc = U[1].NC
    generator = zeros(ComplexF64, nc, nc)
    generator[1,1], generator[2,2] = im, -im
    epsilon = 1e-6
    for mu in (1,2), site in (LATTICE, (1,1,1,1))
        link = [U[mu][i,j,site...] for i=1:nc, j=1:nc]
        d = calc_dSdUμ(action, mu, U, B)
        derivative = [d[i,j,site...] for i=1:nc, j=1:nc]
        analytic = 2real(tr(generator * link * derivative))
        function varied(eps)
            V = copylinks(U)
            value = exp(eps * generator) * link
            for j=1:nc, i=1:nc
                V[mu][i,j,site...] = value[i,j]
            end
            set_wing_U!(V[mu])
            return real(evaluate_GaugeAction(action, V, B))
        end
        numerical = (varied(epsilon) - varied(-epsilon)) / (2epsilon)
        @test numerical ≈ analytic atol=2e-6 rtol=2e-6
    end
end

@testset "B rectangle flow: backend and physical regressions" begin
    for nc in (2,3)
        @testset "SU($nc)" begin
            seed = Initialize_Gaugefields(nc, 0, LATTICE...;
                condition="hot", randomnumber="Reproducible", verbose_level=0)
            initial = arrays(seed)
            for nonzero in (false,true)
                flux = nonzero ? [1,0,1,1,0,1] : zeros(Int,6)
                reference = copylinks(seed)
                reference_B = Initialize_Bfields(nc, flux, 0, LATTICE...; verbose_level=0)
                reference_flow = generalflow(reference, reference_B)
                check_finite_difference(reference, reference_B, reference_flow.gaugeaction)
                derivatives = [fieldarray(calc_dSdUμ(reference_flow.gaugeaction, mu,
                    reference, reference_B)) for mu=1:4]
                flow!(reference, reference_B, reference_flow)
                expected = arrays(reference)
                if !nonzero
                    plain = copylinks(seed)
                    g = Gradientflow_general(plain, ["plaquette","rectangular"],
                        [1.0,0.1]; eps=0.005, Nflow=1)
                    flow!(plain, g)
                    @test maxdiff(expected, arrays(plain)) < 5e-11
                end

                for backend in (:serial, :lattice), nw in (0,1), mode in (:optimized,:matrix,:legacy)
                    backend === :serial && mode === :matrix && continue
                    @testset "$backend nw=$nw mode=$mode nonzero=$nonzero" begin
                        lm = backend === :lattice
                        opts = lm ? (; isMPILattice=true, PEs=GRID) : (;)
                        U = lm ? Initialize_Gaugefields(nc, nw, LATTICE...; opts...,
                            condition="cold", verbose_level=0) : serialcopy(initial,nc,nw)
                        if lm
                            # Public cross-backend copy uses global site coordinates.
                            substitute_U!(U, seed)
                        end
                        @test maxdiff(arrays(U), initial) == 0
                        B = Initialize_Bfields(nc, flux, nw, LATTICE...; opts...,
                            bfield_evaluation=mode === :legacy ? :legacy : :optimized,
                            bfield_storage=lm && mode === :optimized ? :scalar : :matrix,
                            verbose_level=0)
                        before_B = lm && mode === :optimized ?
                            [gather_and_bcast_matrix(get_Bphase(B,m,n)) for (m,n) in PLANES] :
                            [fieldarray(B[m,n]) for (m,n) in PLANES]
                        g = generalflow(U, B)
                        for mu=1:4
                            d = calc_dSdUμ(g.gaugeaction,mu,U,B)
                            @test maximum(abs, fieldarray(d)-derivatives[mu]) < 5e-11
                        end
                        flow!(U,B,g)
                        @test maxdiff(arrays(U),expected) < 5e-11
                        after_B = lm && mode === :optimized ?
                            [gather_and_bcast_matrix(get_Bphase(B,m,n)) for (m,n) in PLANES] :
                            [fieldarray(B[m,n]) for (m,n) in PLANES]
                        @test all(isequal(a,b) for (a,b) in zip(before_B,after_B))
                        if lm && mode === :optimized
                            @test BF._uses_scalar_storage(B.u)
                        end

                        # Reusing the flow must read updated B, not cached values.
                        updated = Initialize_Bfields(nc,[0,1,0,0,1,0],nw,LATTICE...;
                            opts..., verbose_level=0,
                            bfield_storage=lm && mode === :optimized ? :scalar : :matrix)
                        substitute_U!(B,updated)
                        fresh_U = copylinks(U)
                        fresh = generalflow(fresh_U,B)
                        flow!(U,B,g)
                        flow!(fresh_U,B,fresh)
                        @test maxdiff(arrays(U),arrays(fresh_U)) < 5e-11
                        mode === :legacy && @test isempty(B.pathplans)
                    end
                end
            end
        end
    end
end

@testset "B path evaluation returns shifted scratch buffers" begin
    for nc in (2,3), mode in (:optimized,:matrix,:legacy)
        U = Initialize_Gaugefields(nc,0,LATTICE...; isMPILattice=true, PEs=GRID,
            condition="hot", randomnumber="Reproducible", verbose_level=0)
        B = Initialize_Bfields(nc,zeros(Int,6),0,LATTICE...;
            isMPILattice=true, PEs=GRID, verbose_level=0,
            bfield_evaluation=mode === :legacy ? :legacy : :optimized,
            bfield_storage=mode === :optimized ? :scalar : :matrix)
        g = generalflow(U,B)
        path = g.gaugeaction.dataset[2].staples[1][1]
        out = similar(U[1])
        work = [similar(out) for _=1:4]
        # A tiny capacity turns missing releases into a deterministic failure,
        # independent of when the garbage collector runs.
        fields = vcat(U,[out],work)
        pools = LatticeMatrices.LatticeScratchPool[u.U.temps for u in fields]
        append!(pools, mode === :optimized ?
            [get_Bphase(B,m,n).temps for (m,n) in PLANES] :
            [B[m,n].U.temps for (m,n) in PLANES])
        foreach(pool -> (pool.Nmax=8), pools)
        for _=1:12
            evaluate_gaugelinks!(out,path,U,B,work)
            @test all(LatticeMatrices.scratch_inuse(pool) == 0 for pool in pools)
            substitute_U!(out,U[1])
            BF.multiply_Bplaquettes!(out,path,B,work)
            @test fieldarray(out) ≈ fieldarray(U[1]) atol=2e-12 rtol=2e-12
            @test all(LatticeMatrices.scratch_inuse(pool) == 0 for pool in pools)
            @test isfinite(calculate_Plaquette(U,B,work[1],work[2]))
            @test all(LatticeMatrices.scratch_inuse(pool) == 0 for pool in pools)
        end
    end
end

end # module
