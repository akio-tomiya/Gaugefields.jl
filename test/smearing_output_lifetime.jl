using Gaugefields, Test, LinearAlgebra
import LatticeMatrices

function test_smearing_output_lifetime(; process_grid=(1, 1, 1, 1))
    snapshot(U) = [LatticeMatrices.gather_and_bcast_matrix(u.U) for u in U]
    U = gauge_configuration((4, 4, 4, 4); colors=3, start=:hot, seed=1234,
                            process_grid, verbose=0)
    input = snapshot(U)
    action = GaugeAction(U)
    loops = make_loops_fromname("plaquette"; Dim=4)
    append!(loops, loops')
    push!(action, 5.7 / 2, loops)
    diagnostics = Float64[]
    @testset "Owned legacy smearing outputs" begin
        for layers in (1, 2)
            net = CovNeuralnet(U)
            for _ in 1:layers
                push!(net, STOUT_Layer(["plaquette"], [0.1], U))
            end
            V, history, _ = calc_smearedU(U, net)
            saved = snapshot.(history)
            before = measure_plaquette(V)
            raw = similar(V)
            for mu in 1:4
                calc_dSdUμ!(raw[mu], action, mu, V)
            end
            raw_saved = snapshot(raw)
            derivative = back_prop(raw, net, history, U)
            @test snapshot.(history) == saved
            @test snapshot(U) == input
            @test snapshot(raw) == raw_saved
            @test measure_plaquette(V) == before
            repeated = similar(derivative)
            back_prop!(repeated, raw, net, history, U;
                       calculate_parameter_derivatives=false)
            @test snapshot.(history) == saved
            @test all(isapprox.(snapshot(repeated), snapshot(derivative); atol=5e-12, rtol=5e-12))
            # A second forward call may reuse scratch storage, but neither
            # old outputs nor its own outputs may alias the pool.
            other, other_history, _ = calc_smearedU(U, net)
            @test snapshot.(history) == saved
            @test all(isapprox.(snapshot(other), saved[end]; atol=5e-12, rtol=5e-12))
            back_prop(raw, net, other_history, U)
            @test snapshot.(other_history) == saved
            @test snapshot.(history) == saved

            newV, cache = link_smear(U, StoutSmearing(rho=0.1, iterations=layers))
            @test all(isapprox.(snapshot(newV), saved[end]; atol=5e-12, rtol=5e-12))
            cotangent = similar(raw)
            for mu in 1:4
                clear_U!(cotangent[mu]); add_U!(cotangent[mu], raw[mu]')
            end
            set_wing_U!(cotangent)
            modern = link_smear_pullback(cotangent, U, cache)
            for mu in 1:4
                clear_U!(cotangent[mu]); add_U!(cotangent[mu], modern[mu]')
            end
            @test all(isapprox.(snapshot(cotangent), snapshot(derivative); atol=5e-11, rtol=5e-11))
            push!(diagnostics, before, sum(sum(abs2, a) for a in snapshot(derivative)))
            for configuration in (V, derivative), values in snapshot(configuration)
                append!(diagnostics, vec(real.(values)), vec(imag.(values)))
            end
        end
    end
    return diagnostics
end
