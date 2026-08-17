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
    factor = -1 / U[1].NC
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
after loading Enzyme through the Gaugefields Enzyme extension.
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
    MDDriver

Preallocated, deterministic molecular-dynamics driver. Construct one with
[`md_driver`](@ref), then reuse it with [`md_trajectory!`](@ref).

The driver does not generate momenta, draw random numbers, or perform an
accept/reject decision.
"""
struct MDDriver{A,I,T,W,F}
    action::A
    integrator::I
    trajectory_length::Float64
    steps::Int
    exponential_temps::Vector{T}
    exponential::T
    link_work::T
    action_workspace::W
    force::F
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

    exponential_temps = [similar(U[1]), similar(U[1])]
    exponential = similar(U[1])
    link_work = similar(U[1])
    action_workspace = md_action_workspace(action, U)
    force = initialize_TA_Gaugefields(U)
    return MDDriver(
        action,
        integrator,
        Float64(trajectory_length),
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
    update_momenta!(P, U, 0.5 * step_size, driver)
    update_gaugefields!(U, P, step_size, driver)
    update_momenta!(P, U, 0.5 * step_size, driver)
    return nothing
end

function md_step!(::QPQ, U, P, step_size, driver::MDDriver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    update_momenta!(P, U, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    return nothing
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
    update_momenta!,
    update_gaugefields!,
    enzyme_md_action,
    PQP,
    QPQ,
    md_step!,
    MDDriver,
    md_driver,
    md_step_size,
    md_hamiltonian,
    md_trajectory!
