"""Backend hook for one Los Alamos checkerboard update."""
function make_g_los_alamos! end

"""Backend hook for one steepest-descent checkerboard update."""
function make_g_steepest_descent! end

"""Return whether a concrete field has supplied the gauge-fixing hooks."""
gaugefixing_backend_supported(::AbstractGaugefields) = false

function _validate_gaugefixing_inputs(
    U,
    g_transform::AbstractGaugefields{NC,Dim},
    LA_iteration,
    SD_iteration,
    min_iterations,
    tol,
    temp_master,
    D_fix,
) where {NC,Dim}
    gaugefixing_backend_supported(g_transform) || throw(ArgumentError(
        "gaugefixing! does not support this gauge-field backend; " *
        "got $(typeof(g_transform))",
    ))
    length(U) == Dim || throw(ArgumentError(
        "expected $Dim gauge links, got $(length(U))",
    ))
    1 <= D_fix <= Dim || throw(ArgumentError(
        "D_fix must lie between 1 and $Dim, got $D_fix",
    ))
    LA_iteration >= 0 || throw(ArgumentError(
        "LA_iteration must be nonnegative, got $LA_iteration",
    ))
    SD_iteration >= 0 || throw(ArgumentError(
        "SD_iteration must be nonnegative, got $SD_iteration",
    ))
    min_iterations >= 0 || throw(ArgumentError(
        "min_iterations must be nonnegative, got $min_iterations",
    ))
    tol >= 0 || throw(ArgumentError("tol must be nonnegative, got $tol"))
    length(temp_master) >= 6 || throw(ArgumentError(
        "gaugefixing! requires at least six temporary gauge fields",
    ))
    return nothing
end

@inline function get_SU2_index(NG::Integer, hit_color::Integer)
    index = 0
    for delta in 1:(NG - 1)
        for i1 in 1:(NG - delta)
            index += 1
            index == hit_color && return i1, i1 + delta
        end
    end
    # Kernel callers always use 1:NG*(NG-1)÷2. Keep this fallback free of
    # exception paths so accelerator compilation cannot introduce a hostcall.
    return 1, min(2, NG)
end

function gaugefixing_step_la!(
    U::Array{T,1},
    g_transform::AbstractGaugefields{NC,Dim},
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    temp::AbstractGaugefields{NC,Dim};
    D_fix=4,
) where {NC,Dim,T<:AbstractGaugefields}
    for parity in (0, 1)
        make_g_los_alamos!(
            U,
            g_transform,
            temp,
            parity,
            overrelax,
            ovr_coeff2,
            ovr_coeff3,
            D_fix,
        )
        gUgshift!(U, g_transform, temp)
    end
    return nothing
end

function gaugefixing_step_sd!(
    U::Array{T,1},
    g_transform::AbstractGaugefields{NC,Dim},
    Δ::AbstractGaugefields{NC,Dim},
    overrelax::Float64,
    temps::Array{T,1};
    D_fix=4,
) where {NC,Dim,T<:AbstractGaugefields}
    for parity in (0, 1)
        make_g_steepest_descent!(
            U, g_transform, Δ, parity, overrelax, temps, D_fix,
        )
        gUgshift!(U, g_transform, temps[1])
    end
    return nothing
end

"""
    gaugefixing!(U, g_transform, LA_overrelax, LA_iteration,
                 SD_overrelax, SD_iteration, tol, config_n, temp_master;
                 D_fix=4, min_iterations=500)

Gauge-fix `U` with Los Alamos and steepest-descent updates. Each stage runs
for at least `min(min_iterations, requested_iterations)` iterations before its
tolerance test may stop it. Set `min_iterations=0` to test convergence after
the first iteration.

On return, `g_transform` contains the most recent checkerboard transformation;
it is not the accumulated transformation from the input configuration.
"""
function gaugefixing!(
    U::Array{T,1},
    g_transform::AbstractGaugefields{NC,Dim},
    LA_overrelax::Float64,
    LA_iteration::Int,
    SD_overrelax::Float64,
    SD_iteration::Int,
    tol::Float64,
    config_n::Int,
    temp_master;
    D_fix::Int=4,
    min_iterations::Int=500,
) where {NC,Dim,T<:AbstractGaugefields}
    _validate_gaugefixing_inputs(
        U,
        g_transform,
        LA_iteration,
        SD_iteration,
        min_iterations,
        tol,
        temp_master,
        D_fix,
    )

    Δ = temp_master[1]
    temps = temp_master[2:6]
    unit_U!(g_transform)

    trace = real(trace_U(U; D_fix))
    norm_g = NC * g_transform.NV
    trace_g = real(tr(g_transform) / norm_g)

    get_Δ!(Δ, U, temps, D_fix)
    trace_dAmu_sqr = real(trace_AAdagger(Δ, temps[1]))

    println_verbose_level1(
        U[1],
        "[GaugeFixing]config #$config_n step...0 tr[U]=$trace " *
        "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr",
    )
    flush(stdout)

    if LA_overrelax > 1.0
        LA_ovr_coeff2 = LA_overrelax * (LA_overrelax - 1) / 2
        LA_ovr_coeff3 = LA_ovr_coeff2 * (LA_overrelax - 2) / 3
    else
        LA_ovr_coeff2 = 0.0
        LA_ovr_coeff3 = 0.0
    end

    for iter in 1:LA_iteration
        gaugefixing_step_la!(
            U,
            g_transform,
            LA_overrelax,
            LA_ovr_coeff2,
            LA_ovr_coeff3,
            temps[1];
            D_fix,
        )

        trace = real(trace_U(U; D_fix))
        trace_g = real(tr(g_transform) / norm_g)
        get_Δ!(Δ, U, temps, D_fix)
        trace_dAmu_sqr = real(trace_AAdagger(Δ, temps[1]))

        if iter >= min(min_iterations, LA_iteration) && trace_dAmu_sqr < tol
            println_verbose_level1(
                U[1],
                "[GaugeFixing]config #$config_n L.A. step...$iter tr[U]=$trace " *
                "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr",
            )
            println_verbose_level1(
                U[1],
                "[GaugeFixing]config #$config_n: tr[dA dA'] < tolerance " *
                "[$tol] -> Gauge fixing DONE!",
            )
            break
        elseif iter == LA_iteration
            println_verbose_level1(
                U[1],
                "[GaugeFixing]config #$config_n L.A. step...$iter tr[U]=$trace " *
                "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr",
            )
        else
            println_verbose_level3(
                U[1],
                "[GaugeFixing]config #$config_n L.A. step...$iter tr[U]=$trace " *
                "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr",
            )
        end
        flush(stdout)
    end

    for iter in 1:SD_iteration
        trace_prev = trace
        gaugefixing_step_sd!(
            U, g_transform, Δ, SD_overrelax, temps; D_fix,
        )

        trace = real(trace_U(U; D_fix))
        trace_g = real(tr(g_transform) / norm_g)
        get_Δ!(Δ, U, temps, D_fix)
        trace_dAmu_sqr = real(trace_AAdagger(Δ, temps[1]))
        diff = abs(trace_prev - trace)

        if iter >= min(min_iterations, SD_iteration) && trace_dAmu_sqr < tol
            println_verbose_level1(
                U[1],
                "[GaugeFixing]config #$config_n SD step...$iter tr[U]=$trace " *
                "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr diff_tr[U]=$diff",
            )
            println_verbose_level1(
                U[1],
                "[GaugeFixing]config #$config_n: tr[dA dA'] < tolerance " *
                "[$tol] -> Gauge fixing DONE!",
            )
            break
        elseif iter == SD_iteration
            println_verbose_level1(
                U[1],
                "[GaugeFixing]config #$config_n SD step...$iter tr[U]=$trace " *
                "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr diff_tr[U]=$diff",
            )
        else
            println_verbose_level3(
                U[1],
                "[GaugeFixing]config #$config_n SD step...$iter tr[U]=$trace " *
                "tr[G]=$trace_g tr[dA dA']=$trace_dAmu_sqr diff_tr[U]=$diff",
            )
        end
        flush(stdout)
    end
    return U
end

function validate_training(
    U::Array{T,1},
    temp_master;
    D_fix::Int=4,
) where {T<:AbstractGaugefields}
    Δ = temp_master[1]
    temps = temp_master[2:6]
    trace = real(trace_U(U; D_fix))
    get_Δ!(Δ, U, temps, D_fix)
    trace_dAmu_sqr = real(trace_AAdagger(Δ, temps[1]))
    return trace, trace_dAmu_sqr
end
