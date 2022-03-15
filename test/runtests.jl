using Gaugefields
using Test
using Random
import Wilsonloop:loops_staple



@testset "HMC nowing" begin
    println("HMC nowing")
    include("HMC_test_nowing.jl")
end


@testset "HMC" begin
    println("HMC")
    include("HMC_test.jl")
end




@testset "ScalarNN" begin
    println("Scalar neural networks")
    include("scalarnn.jl")
end

@testset "Initialization" begin
    println("Initialization")
    include("init.jl")
end

@testset "heatbath nowing" begin
    println("heatbath nowing")
    include("heatbathtest_nowing.jl")
end


@testset "heatbath" begin
    println("heatbath")
    include("heatbathtest.jl")
end




@testset "gradientflow nowing" begin
    println("gradientflow nowing")
    include("gradientflow_test_nowing.jl")
end


@testset "gradientflow" begin
    println("gradientflow")
    include("gradientflow_test.jl")
end



@testset "Gaugefields.jl" begin
    # Write your tests here.
end




