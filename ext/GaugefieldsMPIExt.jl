module GaugefieldsMPIExt
using MPI
#using Gaugefields
using LinearAlgebra

comm = MPI.COMM_WORLD
export comm

import Gaugefields.AbstractGaugefields_module: calc_rank_and_indices,
    get_myrank,
    get_nprocs, barrier, getvalue, setvalue!,
    identityGaugefields_4D_nowing_mpi,
    randomGaugefields_4D_nowing_mpi,
    clear_U!, add_U!,
    substitute_U!, map_U!,
    map_U_sequential!,
    shifted_U_improved_zeroshift!,
    update_sent_data!,
    shifted_U_improved_xshift!,
    shifted_U_improved_yshift!,
    shifted_U_improved_zshift!,
    shifted_U_improved_tshift!,
    mpi_updates_U_1data!,
    mpi_updates_U_moredata!,
    mpi_updates_U!,
    shifted_U_improved!,
    shifted_U!,
    shift_U,
    normalize_U!,
    exptU!,
    Traceless_antihermitian!,
    Antihermitian!,
    calculate_Polyakov_loop,
    partial_tr,
    mul_skiplastindex!,
    set_wing_U!,
    lambda_k_mul!,
    minusidentityGaugefields_4D_nowing_mpi,
    construct_Λmatrix_forSTOUT!,
    unit_U!,
    Gaugefields_4D,
    Data_sent,
    Adjoint_Gaugefields,
    Abstractfields,
    Shifted_Gaugefields,
    Gaugefields_2D,
    TA_Gaugefields_4D,
    Generator,
    TA_Gaugefields_2D,
    AbstractGaugefields,
    gramschmidt!,
    check_outside,
    sr3i2,
    lie2matrix!,
    matrix2lie!

import Gaugefields.Verbose_print

include("./4D/gaugefields_4D_mpi_nowing.jl")
export Gaugefields_4D_nowing_mpi,
    Shifted_Gaugefields_4D_mpi_nowing

import Gaugefields.AbstractGaugefields_module: get_myrank_xyzt,
    identityGaugefields_4D_wing_mpi,
    randomGaugefields_4D_wing_mpi,
    minusidentityGaugefields_4D_wing_mpi

include("./4D/gaugefields_4D_mpi.jl")
export Gaugefields_4D_wing_mpi,
    Shifted_Gaugefields_4D_mpi

import Gaugefields.AbstractGaugefields_module: thooftFlux_4D_B_at_bndry_nowing_mpi
include("./4D/gaugefields_4D_mpi_nowing_Bfields.jl")

import Gaugefields.AbstractGaugefields_module: thooftFlux_4D_B_at_bndry_wing_mpi
include("./4D/gaugefields_4D_mpi_Bfields.jl")

import Gaugefields.AbstractGaugefields_module: gauss_distribution!,
    Traceless_antihermitian_add!

import Gaugefields.AbstractGaugefields_module: identityGaugefields_2D_nowing_mpi,
    randomGaugefields_2D_nowing_mpi,
    mpi_updates_U_moredata!,
    mpi_updates_U!
include("./2D/gaugefields_2D_mpi_nowing.jl")
export Gaugefields_2D_nowing_mpi, Shifted_Gaugefields_2D_mpi_nowing

using Distributions
import Gaugefields.AbstractGaugefields_module: init_TA_Gaugefields_4D_mpi
include("./4D/TA_gaugefields_4D_mpi.jl")
export TA_Gaugefields_4D_mpi


import Gaugefields.AbstractGaugefields_module: init_TA_Gaugefields_2D_mpi
include("./2D/TA_gaugefields_2D_mpi.jl")
export TA_Gaugefields_2D_mpi

import Gaugefields.ILDG_format: load_binarydata!, save_binarydata
include("./output/ildg_format.jl")

import Gaugefields.Abstractsmearing_module: CdexpQdQ!, calc_coefficients_Q,
    construct_B1B2!, construct_trCB1B2,
    construct_CdeQdQ_3!, calc_Bmatrix!
include("./smearing/stout_fast.jl")

#import Gaugefields.Bridge_format: save_binarydata

end # module