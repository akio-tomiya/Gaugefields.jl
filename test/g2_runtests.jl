using Test

const G2_TEST_FILES = (
    "g2_algebra_test.jl",
    "g2_gaugefields_test.jl",
    "g2_momentum_test.jl",
    "g2_interface_test.jl",
    "g2_projection_update_test.jl",
    "g2_gauge_action_test.jl",
    "g2_force_test.jl",
    "g2_hmc_test.jl",
)

@testset "G2 Gaugefields" begin
    for test_file in G2_TEST_FILES
        @testset "$test_file" begin
            include(test_file)
        end
    end
end
