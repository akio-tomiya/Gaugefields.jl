using Gaugefields
using Test
using Random
import Wilsonloop:loops_staple

@testset "ScalarNN" begin
    println("Scalar neural networks")
    include("scalarnn.jl")
end

@testset "Initialization" begin
    println("Initialization")
    include("init.jl")
end



@testset "heatbath" begin
    println("heatbath")
    include("heatbathtest.jl")
end

@testset "gradientflow" begin
    println("gradientflow")
    include("gradientflow_test.jl")
end

@testset "Gaugefields.jl" begin
    # Write your tests here.
end




