using Gaugefields
using Test
using Random
import Wilsonloop: loops_staple


const eps = 1e-1

@testset "G2 algebra" begin
    println("G2 algebra")
    include("g2_algebra_test.jl")
end

@testset "G2 gaugefields" begin
    println("G2 gaugefields")
    include("g2_gaugefields_test.jl")
end

@testset "G2 momentum" begin
    println("G2 momentum")
    include("g2_momentum_test.jl")
end

@testset "G2 projection and update" begin
    println("G2 projection and update")
    include("g2_projection_update_test.jl")
end

@testset "G2 interface" begin
    println("G2 interface")
    include("g2_interface_test.jl")
end

@testset "G2 gauge action" begin
    println("G2 gauge action")
    include("g2_gauge_action_test.jl")
end

@testset "G2 force" begin
    println("G2 force")
    include("g2_force_test.jl")
end

#=
@testset "Ising field" begin
    println("Ising field")
    include("Isingtest.jl")
end
=#

@testset "Bfield HMC" begin
    println("Bfield HMC")
    include("Btest/sample_dynB.jl")
end

@testset "Bfield Gradient flow" begin
    println("Bfield Gradient flow")
    include("Btest/gradientflow_general_B.jl")
end

@testset "Initialization" begin
    println("Initialization")
    include("init.jl")
end

@testset "SUN embedded instanton helpers" begin
    println("SUN embedded instanton helpers")
    include("sun_embedded_instanton.jl")
end



@testset "HMC nowing" begin
    println("HMC nowing")
    include("HMC_test_nowing.jl")
end

@testset "HMCstout nowing" begin
    println("HMCstout nowing")
    include("HMCstout_test_nowing.jl")
end


@testset "gradientflow_general" begin
    println("gradientflow with general action")
    include("gradientflow_general.jl")
end

@testset "gradientflow nowing" begin
    println("gradientflow nowing")
    include("gradientflow_test_nowing.jl")
end


@testset "gradientflow" begin
    println("gradientflow")
    include("gradientflow_test.jl")
end




@testset "HMC" begin
    println("HMC")
    include("HMC_test.jl")
end



@testset "heatbath" begin
    println("heatbath")
    include("heatbathtest.jl")
end


@testset "heatbath nowing" begin
    println("heatbath nowing")
    include("heatbathtest_nowing.jl")
end

@testset "heatbath_bare" begin
    println("heatbath_bare")
    include("heatbathtest_bare.jl")
end


@testset "heatbath with plaq and rect actions" begin
    println("heatbath with plaq and rect actions")
    include("heatbathtest_general.jl")
end











@testset "ScalarNN" begin
    println("Scalar neural networks")
    include("scalarnn.jl")
end



#@testset "Accel test" begin
#    include("./gputests/runtests.jl")
#end


@testset "Gaugefields.jl" begin
    # Write your tests here.
end
