@inline function _heatbath_copy_square_matrix!(
    destination,
    source,
    ::Val{N},
) where N
    @inbounds for jc in 1:N
        for ic in 1:N
            destination[ic, jc] = source[ic, jc]
        end
    end
    return nothing
end

@inline function project_onto_SU2!(S)
    half = typeof(real(S[1, 1]))(0.5)
    alpha = S[1, 1] * half + conj(S[2, 2]) * half
    beta_su2 = S[2, 1] * half - conj(S[1, 2]) * half
    @inbounds begin
        S[1, 1] = alpha
        S[2, 1] = beta_su2
        S[1, 2] = -conj(beta_su2)
        S[2, 2] = conj(alpha)
    end
    return S[2, 2]
end

@inline function make_submatrix!(S, UV, i, j)
    @inbounds begin
        S[1, 1] = UV[i, i]
        S[1, 2] = UV[i, j]
        S[2, 1] = UV[j, i]
        S[2, 2] = UV[j, j]
    end
    return nothing
end

@inline function make_largematrix!(A, K, i, j, NC)
    T = eltype(A)
    @inbounds for jc in 1:NC
        for ic in 1:NC
            A[ic, jc] = ifelse(ic == jc, one(T), zero(T))
        end
    end
    @inbounds begin
        A[i, i] = K[1, 1]
        A[i, j] = K[1, 2]
        A[j, i] = K[2, 1]
        A[j, j] = K[2, 2]
    end
    return A
end

# Compatibility aliases retained for the device-safe overrelaxation kernels.
@inline _copy_square_matrix!(destination, source, size::Val) =
    _heatbath_copy_square_matrix!(destination, source, size)
@inline _make_su2_submatrix!(S, UV, i, j) = make_submatrix!(S, UV, i, j)
@inline _project_onto_su2!(S) = project_onto_SU2!(S)
@inline _make_embedded_su2_matrix!(A, K, i, j) =
    make_largematrix!(A, K, i, j, 3)
@inline _make_embedded_su2_matrix!(A, K, i, j, ::Val{NC}) where NC =
    make_largematrix!(A, K, i, j, NC)

@inline function _su2_update_kp_core!(
    rng,
    Unew,
    V,
    beta,
    NC,
    temps,
    ITERATION_MAX,
)
    V0 = temps[1]
    temp = temps[2]

    rho0 = real(V[1, 1] + V[2, 2]) / 2
    rho1 = -imag(V[1, 2] + V[2, 1]) / 2
    rho2 = real(V[2, 1] - V[1, 2]) / 2
    rho3 = imag(V[2, 2] - V[1, 1]) / 2
    rho = sqrt(rho0^2 + rho1^2 + rho2^2 + rho3^2)

    detV = V[1, 1] * V[2, 2] - V[1, 2] * V[2, 1]
    V0[1, 1] = rho * V[2, 2] / detV
    V0[1, 2] = -rho * V[1, 2] / detV
    V0[2, 1] = -rho * V[2, 1] / detV
    V0[2, 2] = rho * V[1, 1] / detV

    k = 2 * (beta / NC) * rho
    WT = typeof(k)
    one_w = one(WT)
    two = WT(2)
    half = WT(0.5)
    pi_w = WT(pi)

    R = zero(WT)
    Rp = zero(WT)
    Rpp = zero(WT)
    Rppp = zero(WT)
    delta = zero(WT)
    accepted = false
    tries_used = 0

    for tries in 1:ITERATION_MAX
        rng, R = heatbath_log_uniform(rng, WT)
        rng, Rp = heatbath_log_uniform(rng, WT)
        X = -log(R) / k
        Xp = -log(Rp) / k
        rng, Rpp = heatbath_uniform(rng, WT)
        C = cos(two * pi_w * Rpp)^2
        A = X * C
        delta = Xp + A
        rng, Rppp = heatbath_uniform(rng, WT)
        if Rppp^2 <= one_w - half * delta
            accepted = true
            tries_used = tries
            break
        end
        tries_used = tries
    end

    if !accepted
        return rng, false, tries_used, k, rho, delta, R, Rp, Rpp, Rppp
    end

    a1 = one_w - delta
    rr = sqrt(one_w - a1^2)
    rng, random_phi = heatbath_uniform(rng, WT)
    phi = random_phi * pi_w * two
    rng, random_costheta = heatbath_uniform(rng, WT)
    costheta = (random_costheta - half) * two
    sintheta = sqrt(one_w - costheta^2)

    a2 = rr * cos(phi) * sintheta
    a3 = rr * sin(phi) * sintheta
    a4 = rr * costheta
    temp[1, 1] = a1 + im * a4
    temp[1, 2] = a3 + im * a2
    temp[2, 1] = -a3 + im * a2
    temp[2, 2] = a1 - im * a4
    mul!(Unew, temp, V0)

    alpha = Unew[1, 1] * half + conj(Unew[2, 2]) * half
    beta_su2 = Unew[2, 1] * half - conj(Unew[1, 2]) * half
    detU = abs(alpha)^2 + abs(beta_su2)^2
    Unew[1, 1] = alpha / detU
    Unew[2, 1] = beta_su2 / detU
    Unew[1, 2] = -conj(beta_su2) / detU
    Unew[2, 2] = conj(alpha) / detU

    return rng, true, tries_used, k, rho, delta, R, Rp, Rpp, Rppp
end

@inline function _prepare_and_su2_update_kp!(
    rng,
    Unew,
    V,
    beta,
    NC,
    temps,
    ITERATION_MAX,
)
    RT = typeof(real(V[1, 1]))
    rng = heatbath_prepare_su2(rng, RT)
    beta_core = heatbath_beta(rng, beta, RT)
    return _su2_update_kp_core!(
        rng, Unew, V, beta_core, NC, temps, ITERATION_MAX
    )
end

"""
    SU2update_KP!(rng, Unew, V, beta, NC, temps, ITERATION_MAX)
        -> updated_rng, accepted, tries

Apply one allocation-free Kennedy--Pendleton SU(2) update using the portable
RNG protocol.  The clean protocol contains no historical discarded prefix
draws; a host compatibility adapter may add them through
`heatbath_prepare_su2`.
"""
@inline function SU2update_KP!(
    rng,
    Unew,
    V,
    beta,
    NC,
    temps,
    ITERATION_MAX,
)
    rng, accepted, tries, _, _, _, _, _, _, _ =
        _prepare_and_su2_update_kp!(
            rng, Unew, V, beta, NC, temps, ITERATION_MAX
        )
    return rng, accepted, tries
end

@inline function _su3_update_subgroup!(
    rng,
    u,
    V,
    beta,
    NC,
    temps2,
    temps3,
    n,
    m,
    ITERATION_MAX,
)
    V0 = temps2[1]
    temp = temps2[2]
    S = temps2[3]
    K = temps2[4]
    UV = temps3[1]
    A = temps3[2]
    AU = temps3[3]

    mul!(UV, u, V)
    make_submatrix!(S, UV, n, m)
    project_onto_SU2!(S)
    result = _prepare_and_su2_update_kp!(
        rng, K, S, beta, NC, (V0, temp), ITERATION_MAX
    )
    rng, accepted = result[1], result[2]
    accepted || return result

    make_largematrix!(A, K, n, m, NC)
    mul!(AU, A, u)
    _heatbath_copy_square_matrix!(u, AU, Val(3))
    return result
end

@inline function _finish_su3_update!(u, temps3)
    AU = temps3[3]
    _heatbath_copy_square_matrix!(AU, u, Val(3))
    heatbath_normalize3!(AU) || return false
    _heatbath_copy_square_matrix!(u, AU, Val(3))
    return true
end

@inline function _su3_update_single_stream_core!(
    rng,
    u,
    V,
    beta,
    NC,
    temps2,
    temps3,
    ITERATION_MAX,
)
    result = _su3_update_subgroup!(
        rng, u, V, beta, NC, temps2, temps3, 1, 2, ITERATION_MAX
    )
    rng = result[1]
    result[2] || return rng, false, 1, result

    result = _su3_update_subgroup!(
        rng, u, V, beta, NC, temps2, temps3, 2, 3, ITERATION_MAX
    )
    rng = result[1]
    result[2] || return rng, false, 2, result

    result = _su3_update_subgroup!(
        rng, u, V, beta, NC, temps2, temps3, 1, 3, ITERATION_MAX
    )
    rng = result[1]
    result[2] || return rng, false, 3, result

    _finish_su3_update!(u, temps3) || return rng, false, 4, result
    return rng, true, 0, result
end

@inline function _su3_update_three_stream_core!(
    rngs::NTuple{3,R},
    u,
    V,
    beta,
    NC,
    temps2,
    temps3,
    ITERATION_MAX,
) where R
    rng1, rng2, rng3 = rngs
    result = _su3_update_subgroup!(
        rng1, u, V, beta, NC, temps2, temps3, 1, 2, ITERATION_MAX
    )
    rng1 = result[1]
    result[2] || return (rng1, rng2, rng3), false, 1, result

    result = _su3_update_subgroup!(
        rng2, u, V, beta, NC, temps2, temps3, 2, 3, ITERATION_MAX
    )
    rng2 = result[1]
    result[2] || return (rng1, rng2, rng3), false, 2, result

    result = _su3_update_subgroup!(
        rng3, u, V, beta, NC, temps2, temps3, 1, 3, ITERATION_MAX
    )
    rng3 = result[1]
    result[2] || return (rng1, rng2, rng3), false, 3, result

    rngs = (rng1, rng2, rng3)
    _finish_su3_update!(u, temps3) || return rngs, false, 4, result
    return rngs, true, 0, result
end

"""
    SU3update_matrix!(rng, u, V, beta, NC, temps2, temps3, ITERATION_MAX)
        -> updated_rng, accepted, failed_subgroup

Apply the fixed `(1,2)`, `(2,3)`, `(1,3)` SU(3) subgroup sequence using one
sequential RNG stream.
"""
@inline function SU3update_matrix!(
    rng,
    u,
    V,
    beta,
    NC,
    temps2,
    temps3,
    ITERATION_MAX,
)
    NC == 3 || return rng, false, -1
    rng, accepted, failed_subgroup, _ = _su3_update_single_stream_core!(
        rng, u, V, beta, NC, temps2, temps3, ITERATION_MAX
    )
    return rng, accepted, failed_subgroup
end

"""
    SU3update_matrix!(rngs::NTuple{3}, u, V, beta, NC, temps2, temps3,
        ITERATION_MAX) -> updated_rngs, accepted, failed_subgroup

Apply the same SU(3) physics core with one independent stream for each fixed
SU(2) subgroup.
"""
@inline function SU3update_matrix!(
    rngs::NTuple{3,R},
    u,
    V,
    beta,
    NC,
    temps2,
    temps3,
    ITERATION_MAX,
) where R
    NC == 3 || return rngs, false, -1
    rngs, accepted, failed_subgroup, _ = _su3_update_three_stream_core!(
        rngs, u, V, beta, NC, temps2, temps3, ITERATION_MAX
    )
    return rngs, accepted, failed_subgroup
end
