using Gaugefields
using Test
using Random
import Wilsonloop:loops_staple



@testset "Initialization" begin
    println("Initialization")
    include("init.jl")
end

@testset "heatbath" begin
    println("heatbath")
    include("heatbathtest.jl")
end

@testset "Gaugefields.jl" begin
    # Write your tests here.
end




