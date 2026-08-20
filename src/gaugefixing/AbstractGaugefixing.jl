module AbstractGaugefixing_module

using JACC
using LinearAlgebra
using Requires
using StaticArrays

using ..AbstractGaugefields_module:
    AbstractGaugefields,
    Gaugefields_4D_accelerator,
    Gaugefields_4D_MPILattice,
    Gaugefields_4D_nowing,
    Gaugefields_4D_nowing_mpi,
    Gaugefields_4D_wing_mpi,
    Traceless_antihermitian!,
    add_U!,
    clear_U!,
    getvalue,
    normalize3!,
    println_verbose_level1,
    println_verbose_level3,
    set_halo!,
    set_wing_U!,
    shift_U,
    setvalue!,
    substitute_U!,
    normalize_U!

import ..AbstractGaugefields_module:
    fourdim_cordinate,
    shiftedindex

import ..AbstractGaugefields_module: unit_U!

import LatticeMatrices:
    LatticeMatrix,
    delinearize,
    normalize_matrix!

# The common layer contains only control flow and operations expressed through
# the AbstractGaugefields interface.  Concrete kernels are supplied below by
# the LatticeMatrices/JACC implementation.
include("gaugefixing_utility.jl")
include("gaugefixing_common.jl")
include("gaugefixing_local_matrix.jl")
include("gaugefixing_utility_4D_MPILattice.jl")
include("gaugefixing_utility_4D_nowing.jl")
include("gaugefixing_utility_4D_legacy_mpi.jl")
include("gaugefixing_utility_4D_nowing_accelerator.jl")

function __init__()
    # Keep the original direct-CUDA accelerator backend available without
    # making CUDA a hard dependency.  The portable `accelerator="JACC"` path
    # above remains backend independent.
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("gaugefixing_utility_4D_nowing_accelerator_cuda.jl")
    end
end

end # module AbstractGaugefixing_module
