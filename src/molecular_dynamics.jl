using LinearAlgebra: mul!

"""Supertype for molecular-dynamics integration schemes."""
abstract type AbstractMDIntegrator end

"""
    md_action_workspace(action, U)

Allocate reusable work fields for an MD action provider. Custom action
providers implement this function together with [`md_potential`](@ref) and
[`md_force!`](@ref).
"""
function md_action_workspace(action, U)
    throw(ArgumentError(
        "md_action_workspace is not implemented for $(typeof(action))",
    ))
end

"""
    md_potential(action, U, workspace)

Return the potential energy represented by an MD action provider. The result
must use the same normalization and sign convention as its [`md_force!`](@ref)
method.
"""
function md_potential(action, U, workspace)
    throw(ArgumentError(
        "md_potential is not implemented for $(typeof(action))",
    ))
end

"""
    md_force!(force, action, U, workspace)

Write the traceless anti-Hermitian molecular-dynamics force `dp/dτ` to
`force`. Custom action providers own the conversion from their derivative
representation to this common force convention.
"""
function md_force!(force, action, U, workspace)
    throw(ArgumentError(
        "md_force! is not implemented for $(typeof(action))",
    ))
end

"""
    MDActionSet(; name=action, ...)
    MDActionSet(actions::NamedTuple)

Collect independently scheduled MD action providers in a type-stable named
tuple. Every member implements [`md_action_workspace`](@ref),
[`md_potential`](@ref), and [`md_force!`](@ref). The combined potential and
force are the sums of the member contributions.

Names are also used by [`MDForceGroup`](@ref) and
[`SextonWeingarten`](@ref) to select time scales.
"""
struct MDActionSet{T<:NamedTuple}
    terms::T

    function MDActionSet(terms::T) where {T<:NamedTuple}
        isempty(terms) && throw(ArgumentError(
            "an MDActionSet must contain at least one action provider",
        ))
        return new{T}(terms)
    end
end

MDActionSet(; terms...) = MDActionSet((; terms...))

"""
    MDForceGroup(names...)

Select named members of an [`MDActionSet`](@ref) for a momentum update. The
names are stored in the group type so the selected action providers remain
type-stable in the integration loop.
"""
struct MDForceGroup{Names} end

function MDForceGroup(names::Symbol...)
    isempty(names) && throw(ArgumentError(
        "an MDForceGroup must contain at least one action name",
    ))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "an MDForceGroup must not contain duplicate action names: $names",
    ))
    return MDForceGroup{names}()
end

MDForceGroup(names::Tuple{Vararg{Symbol}}) = MDForceGroup(names...)

_md_force_group(group::MDForceGroup) = group
_md_force_group(name::Symbol) = MDForceGroup(name)
_md_force_group(names::Tuple{Vararg{Symbol}}) = MDForceGroup(names)

struct _MDActionSetWorkspace{W,F}
    terms::W
    force::F
end

function md_action_workspace(actions::MDActionSet, U)
    workspaces = map(
        action -> md_action_workspace(action, U),
        actions.terms,
    )
    return _MDActionSetWorkspace(
        workspaces,
        initialize_TA_Gaugefields(U),
    )
end

function md_potential(
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
)
    potentials = map(
        (action, term_workspace) -> md_potential(
            action,
            U,
            term_workspace,
        ),
        actions.terms,
        workspace.terms,
    )
    return sum(values(potentials))
end

function _md_add_named_forces!(
    force,
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
    ::Val{()},
)
    return nothing
end

function _md_add_named_forces!(
    force,
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
    ::Val{Names},
) where {Names}
    name = first(Names)
    md_force!(
        workspace.force,
        getproperty(actions.terms, name),
        U,
        getproperty(workspace.terms, name),
    )
    add_U!(force, 1, workspace.force)
    return _md_add_named_forces!(
        force,
        actions,
        U,
        workspace,
        Val(Base.tail(Names)),
    )
end

function _md_write_named_forces!(
    force,
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
    ::Val{Names},
) where {Names}
    name = first(Names)
    md_force!(
        force,
        getproperty(actions.terms, name),
        U,
        getproperty(workspace.terms, name),
    )
    return _md_add_named_forces!(
        force,
        actions,
        U,
        workspace,
        Val(Base.tail(Names)),
    )
end

function _md_accumulate_forces!(
    force,
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
    names,
)
    return _md_write_named_forces!(
        force,
        actions,
        U,
        workspace,
        Val(names),
    )
end

function md_force!(
    force,
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
)
    return _md_accumulate_forces!(
        force,
        actions,
        U,
        workspace,
        keys(actions.terms),
    )
end

"""
    md_force!(force, actions, U, workspace, group::MDForceGroup)

Write the sum of only the selected action forces to `force`. This is the
force-selection operation used by multiple-time-scale integrators.
"""
function md_force!(
    force,
    actions::MDActionSet,
    U,
    workspace::_MDActionSetWorkspace,
    ::MDForceGroup{Names},
) where {Names}
    return _md_accumulate_forces!(
        force,
        actions,
        U,
        workspace,
        Names,
    )
end

struct _GaugeActionMDWorkspace{T}
    derivative::T
    force_work::T
end

function md_action_workspace(action::GaugeAction, U)
    return _GaugeActionMDWorkspace(similar(U[1]), similar(U[1]))
end

function md_potential(action::GaugeAction, U, workspace)
    return -real(evaluate_GaugeAction(action, U)) / U[1].NC
end

function md_force!(force, action::GaugeAction, U, workspace)
    length(force) == length(U) || throw(ArgumentError(
        "force and U must have the same number of directions",
    ))
    factor = -one(_md_real_scalar_type(U)) / U[1].NC
    for direction in eachindex(U)
        calc_dSdUμ!(workspace.derivative, action, direction, U)
        mul!(workspace.force_work, U[direction], workspace.derivative)
        clear_U!(force[direction])
        Traceless_antihermitian_add!(
            force[direction],
            factor,
            workspace.force_work,
        )
    end
    return nothing
end

"""
    enzyme_md_action(potential, arguments...; num_temps=0)

Construct an Enzyme-backed MD action provider. This method becomes available
after loading Enzyme through the Gaugefields Enzyme extension. `potential`
receives the four underlying LatticeMatrices link matrices separately,
followed by `arguments` and, when `num_temps > 0`, the LM work-field
collection.
"""
function enzyme_md_action end

"""
    PQP()

Second-order symmetric `P(1/2) Q(1) P(1/2)` integrator. Here `P` updates the
momenta and `Q` updates the gauge links.
"""
struct PQP <: AbstractMDIntegrator end

"""
    QPQ()

Second-order symmetric `Q(1/2) P(1) Q(1/2)` integrator. This is the ordering
used by the historical Gaugefields HMC examples and is the default of
[`md_driver`](@ref).
"""
struct QPQ <: AbstractMDIntegrator end

"""
    SextonWeingarten(; slow, fast, n_fast, ordering=QPQ())

Second-order Sexton--Weingarten integrator for two force time scales. `slow`
and `fast` are action names, tuples of action names, or [`MDForceGroup`](@ref)
objects selecting members of an [`MDActionSet`](@ref). `n_fast` is a positive
runtime integer and is deliberately not a type parameter.

With the default `QPQ()` ordering, a half-duration evolution of the fast
Hamiltonian is applied on each side of one full slow-force update. Each half
is divided into `n_fast` QPQ substeps. With `PQP()` ordering, half slow-force
updates surround one full fast evolution divided into `n_fast` QPQ substeps.
"""
struct SextonWeingarten{
    S<:MDForceGroup,
    F<:MDForceGroup,
    O<:AbstractMDIntegrator,
} <: AbstractMDIntegrator
    slow::S
    fast::F
    n_fast::Int
    ordering::O

    function SextonWeingarten(
        slow::S,
        fast::F,
        n_fast::Integer,
        ordering::O,
    ) where {
        S<:MDForceGroup,
        F<:MDForceGroup,
        O<:AbstractMDIntegrator,
    }
        n_fast > 0 || throw(ArgumentError(
            "n_fast must be positive; got $n_fast",
        ))
        (ordering isa PQP || ordering isa QPQ) || throw(ArgumentError(
            "SextonWeingarten ordering must be PQP() or QPQ(); " *
            "got $(typeof(ordering))",
        ))
        return new{S,F,O}(slow, fast, Int(n_fast), ordering)
    end
end

function SextonWeingarten(;
    slow,
    fast,
    n_fast::Integer,
    ordering=QPQ(),
)
    n_fast > 0 || throw(ArgumentError(
        "n_fast must be positive; got $n_fast",
    ))
    (ordering isa PQP || ordering isa QPQ) || throw(ArgumentError(
        "SextonWeingarten ordering must be PQP() or QPQ(); " *
        "got $(typeof(ordering))",
    ))
    return SextonWeingarten(
        _md_force_group(slow),
        _md_force_group(fast),
        Int(n_fast),
        ordering,
    )
end

_md_force_group_names(::MDForceGroup{Names}) where {Names} = Names

function _validate_md_integrator(integrator, action)
    return nothing
end

function _validate_md_integrator(
    integrator::SextonWeingarten,
    actions::MDActionSet,
)
    available = keys(actions.terms)
    slow = _md_force_group_names(integrator.slow)
    fast = _md_force_group_names(integrator.fast)
    selected = (slow..., fast...)

    overlap = intersect(slow, fast)
    isempty(overlap) || throw(ArgumentError(
        "slow and fast force groups overlap: $(Tuple(overlap))",
    ))

    unknown = setdiff(selected, available)
    isempty(unknown) || throw(ArgumentError(
        "SextonWeingarten selects unknown actions $(Tuple(unknown)); " *
        "available names are $available",
    ))

    omitted = setdiff(available, selected)
    isempty(omitted) || throw(ArgumentError(
        "SextonWeingarten does not schedule actions $(Tuple(omitted))",
    ))
    return nothing
end

function _validate_md_integrator(
    ::SextonWeingarten,
    action,
)
    throw(ArgumentError(
        "SextonWeingarten requires an MDActionSet; got $(typeof(action))",
    ))
end

"""
    MDDriver

Preallocated, deterministic molecular-dynamics driver. Construct one with
[`md_driver`](@ref), then reuse it with [`md_trajectory!`](@ref).

The driver does not generate momenta, draw random numbers, or perform an
accept/reject decision.
"""
struct MDDriver{A,I,R<:AbstractFloat,T,W,F}
    action::A
    integrator::I
    trajectory_length::R
    steps::Int
    exponential_temps::Vector{T}
    exponential::T
    link_work::T
    action_workspace::W
    force::F
end

function _md_real_scalar_type(U)
    element_type = eltype(first(U))
    element_type <: Number || return Float64
    return typeof(real(zero(element_type)))
end

"""
    md_driver(U, action; steps, trajectory_length=1.0, integrator=QPQ())

Construct a reusable molecular-dynamics driver for `U` and `action`.

`steps` is required. `trajectory_length` may be negative, which is useful for
reversibility checks, but must be finite and nonzero. `integrator` may be
`PQP()`, `QPQ()`, or any custom object implementing [`md_step!`](@ref).
"""
function md_driver(
    U::Vector{T},
    action;
    steps::Integer,
    trajectory_length::Real=1.0,
    integrator=QPQ(),
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(U) == Dim || throw(ArgumentError(
        "the gauge configuration has length $(length(U)); expected $Dim",
    ))
    steps > 0 || throw(ArgumentError("steps must be positive; got $steps"))
    isfinite(trajectory_length) || throw(ArgumentError(
        "trajectory_length must be finite; got $trajectory_length",
    ))
    iszero(trajectory_length) && throw(ArgumentError(
        "trajectory_length must not be zero",
    ))
    _validate_md_integrator(integrator, action)
    scalar_type = _md_real_scalar_type(U)
    converted_trajectory_length = convert(scalar_type, trajectory_length)
    isfinite(converted_trajectory_length) || throw(ArgumentError(
        "trajectory_length is not finite after conversion to $scalar_type",
    ))
    iszero(converted_trajectory_length) && throw(ArgumentError(
        "trajectory_length becomes zero after conversion to $scalar_type",
    ))

    exponential_temps = [similar(U[1]), similar(U[1])]
    exponential = similar(U[1])
    link_work = similar(U[1])
    action_workspace = md_action_workspace(action, U)
    force = initialize_TA_Gaugefields(U)
    return MDDriver(
        action,
        integrator,
        converted_trajectory_length,
        Int(steps),
        exponential_temps,
        exponential,
        link_work,
        action_workspace,
        force,
    )
end

"""Return the MD step size, `trajectory_length / steps`."""
md_step_size(driver::MDDriver) = driver.trajectory_length / driver.steps

"""
    md_hamiltonian(U, p, driver)

Calculate the Hamiltonian used by the MD driver. The potential comes from the
driver's action provider and the kinetic term is `p*p/2`.
"""
function md_hamiltonian(U, p, driver::MDDriver)
    length(U) == length(p) || throw(ArgumentError(
        "U and p must have the same number of directions",
    ))
    potential = md_potential(
        driver.action,
        U,
        driver.action_workspace,
    )
    kinetic = real(p * p) / 2
    return potential + kinetic
end

"""
    update_gaugefields!(U, P, step_size, driver)

Update only the gauge fields as `Uμ ← exp(step_size * Pμ) Uμ`. The
mutated gauge configuration `U` is the first argument and is also returned.
"""
function update_gaugefields!(U, P, step_size, driver::MDDriver)
    length(U) == length(P) || throw(ArgumentError(
        "U and P must have the same number of directions",
    ))
    isfinite(step_size) || throw(ArgumentError(
        "the gauge-field step size must be finite; got $step_size",
    ))
    for direction in eachindex(U)
        exptU!(
            driver.exponential,
            step_size,
            P[direction],
            driver.exponential_temps,
        )
        mul!(driver.link_work, driver.exponential, U[direction])
        substitute_U!(U[direction], driver.link_work)
    end
    return U
end

"""
    update_momenta!(P, U, step_size, driver)

Update only the conjugate momenta as `P ← P + step_size * force(U)`. The
mutated momenta `P` are the first argument and are also returned.
"""
function update_momenta!(P, U, step_size, driver::MDDriver)
    length(P) == length(U) || throw(ArgumentError(
        "P and U must have the same number of directions",
    ))
    isfinite(step_size) || throw(ArgumentError(
        "the momentum step size must be finite; got $step_size",
    ))
    md_force!(driver.force, driver.action, U, driver.action_workspace)
    for direction in eachindex(P)
        add_U!(P[direction], step_size, driver.force[direction])
    end
    return P
end

"""
    update_momenta!(P, U, step_size, driver, group::MDForceGroup)

Update the momenta with only the named members selected from the driver's
[`MDActionSet`](@ref). This is the elementary kick used by
[`SextonWeingarten`](@ref) and custom multiple-time-scale integrators.
"""
function update_momenta!(
    P,
    U,
    step_size,
    driver::MDDriver,
    group::MDForceGroup,
)
    length(P) == length(U) || throw(ArgumentError(
        "P and U must have the same number of directions",
    ))
    isfinite(step_size) || throw(ArgumentError(
        "the momentum step size must be finite; got $step_size",
    ))
    md_force!(
        driver.force,
        driver.action,
        U,
        driver.action_workspace,
        group,
    )
    for direction in eachindex(P)
        add_U!(P[direction], step_size, driver.force[direction])
    end
    return P
end

"""
    md_step!(integrator, U, P, step_size, driver)

Apply one molecular-dynamics integration step, mutating `U` and `P` in place.
Custom integrators implement this method using [`update_momenta!`](@ref) and
[`update_gaugefields!`](@ref). Subtyping [`AbstractMDIntegrator`](@ref) is
optional.
"""
function md_step!(integrator, U, P, step_size, driver)
    throw(ArgumentError(
        "md_step! is not implemented for $(typeof(integrator))",
    ))
end

function md_step!(integrator::Function, U, P, step_size, driver::MDDriver)
    return integrator(U, P, step_size, driver)
end

function md_step!(::PQP, U, P, step_size, driver::MDDriver)
    update_momenta!(P, U, step_size / 2, driver)
    update_gaugefields!(U, P, step_size, driver)
    update_momenta!(P, U, step_size / 2, driver)
    return nothing
end

function md_step!(::QPQ, U, P, step_size, driver::MDDriver)
    update_gaugefields!(U, P, step_size / 2, driver)
    update_momenta!(P, U, step_size, driver)
    update_gaugefields!(U, P, step_size / 2, driver)
    return nothing
end

function _md_fast_qpq!(
    U,
    P,
    duration,
    nsteps,
    driver,
    fast::MDForceGroup,
)
    step_size = duration / nsteps
    update_gaugefields!(U, P, step_size / 2, driver)
    for step in 1:nsteps
        update_momenta!(P, U, step_size, driver, fast)
        link_step = step == nsteps ? step_size / 2 : step_size
        update_gaugefields!(U, P, link_step, driver)
    end
    return nothing
end

function _md_step_sexton_weingarten!(
    ::QPQ,
    integrator::SextonWeingarten,
    U,
    P,
    step_size,
    driver,
)
    _md_fast_qpq!(
        U,
        P,
        step_size / 2,
        integrator.n_fast,
        driver,
        integrator.fast,
    )
    update_momenta!(P, U, step_size, driver, integrator.slow)
    _md_fast_qpq!(
        U,
        P,
        step_size / 2,
        integrator.n_fast,
        driver,
        integrator.fast,
    )
    return nothing
end

function _md_step_sexton_weingarten!(
    ::PQP,
    integrator::SextonWeingarten,
    U,
    P,
    step_size,
    driver,
)
    update_momenta!(P, U, step_size / 2, driver, integrator.slow)
    _md_fast_qpq!(
        U,
        P,
        step_size,
        integrator.n_fast,
        driver,
        integrator.fast,
    )
    update_momenta!(P, U, step_size / 2, driver, integrator.slow)
    return nothing
end

function md_step!(
    integrator::SextonWeingarten,
    U,
    P,
    step_size,
    driver::MDDriver,
)
    return _md_step_sexton_weingarten!(
        integrator.ordering,
        integrator,
        U,
        P,
        step_size,
        driver,
    )
end

"""
    md_trajectory!(U, p, driver; diagnostics=true)

Evolve `U` and `p` in place through one deterministic MD trajectory. No
momentum refresh or accept/reject step is performed.

With `diagnostics=true`, return a named tuple containing
`initial_hamiltonian`, `final_hamiltonian`, and `delta_hamiltonian`. Set it to
`false` to skip the two Hamiltonian evaluations and return `nothing`.
"""
function md_trajectory!(U, P, driver::MDDriver; diagnostics::Bool=true)
    length(U) == length(P) || throw(ArgumentError(
        "U and P must have the same number of directions",
    ))
    initial_hamiltonian = diagnostics ? md_hamiltonian(U, P, driver) : nothing

    step_size = md_step_size(driver)
    for _ in 1:driver.steps
        md_step!(driver.integrator, U, P, step_size, driver)
    end

    diagnostics || return nothing
    final_hamiltonian = md_hamiltonian(U, P, driver)
    return (
        initial_hamiltonian=initial_hamiltonian,
        final_hamiltonian=final_hamiltonian,
        delta_hamiltonian=final_hamiltonian - initial_hamiltonian,
    )
end

export AbstractMDIntegrator,
    md_action_workspace,
    md_potential,
    md_force!,
    MDActionSet,
    MDForceGroup,
    update_momenta!,
    update_gaugefields!,
    enzyme_md_action,
    PQP,
    QPQ,
    SextonWeingarten,
    md_step!,
    MDDriver,
    md_driver,
    md_step_size,
    md_hamiltonian,
    md_trajectory!
