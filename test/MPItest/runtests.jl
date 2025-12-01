using Gaugefields
using Test
using Random
using MPI
import Wilsonloop: loops_staple
MPI.Init()

const PEs = (1, 1, 1, MPI.Comm_size(MPI.COMM_WORLD))
const PEs2 = (1, MPI.Comm_size(MPI.COMM_WORLD))


const eps = 1e-1

#=
@testset "Ising field" begin
    println("Ising field")
    include("Isingtest.jl")
end
=#

#@testset "Bfield HMC" begin
#    println("Bfield HMC")
#    include("Btest/sample_dynB.jl")
#end

#@testset "Bfield Gradient flow" begin
#    println("Bfield Gradient flow")
#    include("Btest/gradientflow_general_B.jl")
#end

@testset "HMCstout nowing" begin
    println("HMCstout nowing")
    include("HMCstout_test_nowing.jl")
end

@testset "HMC nowing" begin
    println("HMC nowing")
    include("HMC_test_nowing.jl")
end


@testset "Initialization" begin
    println("Initialization")
    include("init.jl")
end

#=

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




=#