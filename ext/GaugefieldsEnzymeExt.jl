module GaugefieldsEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices: Wiltinger_derivative!, toann, DiffArg, NoDiffArg, Enzyme_derivative!, fold_halo_to_core_grad!

using Gaugefields
#import LatticeMatrices: diff, nodiff, toann, Wiltinger_derivative!, Wiltinger!
#export diff, nodiff, Wiltinger_derivative!
import LatticeMatrices: Wiltinger!
import Gaugefields: Wiltinger_U!, diff, nodiff, Wiltinger_derivative!
import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice

function Wiltinger_U!(U::T) where {NC,T<:Gaugefields_4D_MPILattice{NC}}
    Wiltinger!(U.U)
end

function Wiltinger_derivative!(
    func,
    U::Vector{T},
    dfdU::Vector{T}, args...;
    temp=nothing,
    dtemp=nothing
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}

    # Primary variable: always differentiated
    annU = Enzyme.Duplicated(U, dfdU)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU,
            ann_args...
        )
    else
        Gaugefields.clear_U!.(temp)
        Gaugefields.clear_U!.(dtemp)
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU,
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
            ann_args..., Duplicated(temp, dtemp)
        )
    end

    # Convert real/imaginary gradients to Wirtinger derivatives
    Wiltinger_U!.(dfdU)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U1::T,
    U2::T,
    U3::T,
    U4::T,
    dfdU1::T,
    dfdU2::T,
    dfdU3::T,
    dfdU4::T, args...;
    temp=nothing,
    dtemp=nothing
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}

    #println("Enzyme_derivative! in Gaugefields.jl")
    # Primary variable: always differentiated
    annU1 = Enzyme.Duplicated(U1, dfdU1)
    annU2 = Enzyme.Duplicated(U2, dfdU2)
    annU3 = Enzyme.Duplicated(U3, dfdU3)
    annU4 = Enzyme.Duplicated(U4, dfdU4)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...
        )
    else
        Gaugefields.clear_U!.(temp)
        Gaugefields.clear_U!.(dtemp)

        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args..., Duplicated(temp, dtemp)
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end

    fold_halo_to_core_grad!(dfdU1.U)
    fold_halo_to_core_grad!(dfdU2.U)
    fold_halo_to_core_grad!(dfdU3.U)
    fold_halo_to_core_grad!(dfdU4.U)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U::Vector{T},
    dfdU::Vector{T}, args...;
    temp=nothing,
    dtemp=nothing
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}

    error("Enzyme_derivative! does not support Vector U input. Please define a function that takes U1, U2, U3, U4 as separate arguments and run autodiff on that.")

end
export Enzyme_derivative!

end