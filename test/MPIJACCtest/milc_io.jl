using MPI
MPI.Initialized() || MPI.Init()
include(joinpath(@__DIR__, "..", "milc_io.jl"))
