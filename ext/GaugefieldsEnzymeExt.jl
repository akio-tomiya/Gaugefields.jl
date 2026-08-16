module GaugefieldsEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices:
    Wiltinger_derivative!,
    toann,
    DiffArg,
    NoDiffArg,
    Enzyme_derivative!,
    enzyme_duplicated,
    fold_halo_dim_to_core_grad!,
    zero_halo_region!

using Gaugefields
#import LatticeMatrices: diff, nodiff, toann, Wiltinger_derivative!, Wiltinger!
#export diff, nodiff, Wiltinger_derivative!
import LatticeMatrices: Wiltinger!
import Gaugefields:
    Wiltinger_U!,
    diff,
    nodiff,
    Wiltinger_derivative!,
    enzyme_md_action,
    md_action_workspace,
    md_potential,
    md_force!
import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice

@inline _enzyme_workspace(x) = x
@static if VERSION >= v"1.12"
    @inline _enzyme_workspace(x::AbstractVector) = Tuple(x)
end

function _fold_and_zero_gradient!(field::Gaugefields_4D_MPILattice)
    for dimension in length(field.U.PN):-1:1
        fold_halo_dim_to_core_grad!(field.U, dimension)
    end
    zero_halo_region!(field.U)
    return nothing
end

"""An Enzyme-differentiated scalar potential used by the MD driver."""
struct EnzymeMDAction{F,A}
    potential::F
    arguments::A
    num_temps::Int
end

struct EnzymeMDWorkspace{G,T,DT,P}
    gradient::G
    temps::T
    dtemps::DT
    projection::P
end

function enzyme_md_action(potential, arguments...; num_temps::Integer=0)
    num_temps >= 0 || throw(ArgumentError(
        "num_temps must be nonnegative; got $num_temps",
    ))
    return EnzymeMDAction(potential, arguments, Int(num_temps))
end

function md_action_workspace(
    action::EnzymeMDAction,
    U::Vector{T},
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}
    length(U) == 4 || throw(ArgumentError(
        "Enzyme MD currently supports four-dimensional configurations only",
    ))
    Gaugefields.gauge_halo_width(U) >= 1 || throw(ArgumentError(
        "Enzyme MD requires halo >= 1",
    ))

    gradient = [similar(U[1]) for _ in 1:4]
    if iszero(action.num_temps)
        temps = nothing
        dtemps = nothing
    else
        temps = [similar(U[1]) for _ in 1:action.num_temps]
        dtemps = [similar(U[1]) for _ in 1:action.num_temps]
    end
    return EnzymeMDWorkspace(gradient, temps, dtemps, similar(U[1]))
end

function md_action_workspace(action::EnzymeMDAction, U)
    throw(ArgumentError(
        "Enzyme MD currently supports four-dimensional LatticeMatrices " *
        "configurations only; got $(typeof(U))",
    ))
end

function md_potential(action::EnzymeMDAction, U, workspace::EnzymeMDWorkspace)
    Gaugefields.set_wing_U!.(U)
    value = if workspace.temps === nothing
        action.potential(U..., action.arguments...)
    else
        Gaugefields.clear_U!.(workspace.temps)
        action.potential(U..., action.arguments..., workspace.temps)
    end
    value isa Real || throw(ArgumentError(
        "an Enzyme MD potential must return a real scalar; got $(typeof(value))",
    ))
    return value
end

function md_force!(
    force,
    action::EnzymeMDAction,
    U,
    workspace::EnzymeMDWorkspace,
)
    length(force) == 4 == length(U) || throw(ArgumentError(
        "Enzyme MD requires four gauge links and four force fields",
    ))
    Gaugefields.set_wing_U!.(U)
    Gaugefields.clear_U!.(workspace.gradient)
    constant_arguments = map(nodiff, action.arguments)
    Enzyme_derivative!(
        action.potential,
        U...,
        workspace.gradient...,
        constant_arguments...;
        temp=workspace.temps,
        dtemp=workspace.dtemps,
    )

    for direction in eachindex(U)
        mul!(
            workspace.projection,
            U[direction],
            workspace.gradient[direction]',
        )
        Gaugefields.clear_U!(force[direction])
        Gaugefields.Traceless_antihermitian_add!(
            force[direction],
            0.5,
            workspace.projection,
        )
    end
    return nothing
end

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

    Enzyme.API.strictAliasing!(false)

    # Primary variable: always differentiated
    annU = enzyme_duplicated(
        _enzyme_workspace(U),
        _enzyme_workspace(dfdU),
    )

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
            ann_args...,
            enzyme_duplicated(
                _enzyme_workspace(temp),
                _enzyme_workspace(dtemp),
            )
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
    Enzyme.API.strictAliasing!(false)

    # Primary variables: always differentiated
    annU1 = enzyme_duplicated(U1, dfdU1)
    annU2 = enzyme_duplicated(U2, dfdU2)
    annU3 = enzyme_duplicated(U3, dfdU3)
    annU4 = enzyme_duplicated(U4, dfdU4)

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
            ann_args...,
            enzyme_duplicated(
                _enzyme_workspace(temp),
                _enzyme_workspace(dtemp),
            )
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end


    _fold_and_zero_gradient!(dfdU1)
    _fold_and_zero_gradient!(dfdU2)
    _fold_and_zero_gradient!(dfdU3)
    _fold_and_zero_gradient!(dfdU4)


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
