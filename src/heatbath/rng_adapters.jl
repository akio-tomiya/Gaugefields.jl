import Random

@inline function heatbath_uniform(
    rng::Random.AbstractRNG,
    ::Type{T},
) where T
    return rng, rand(rng)
end

@inline function heatbath_log_uniform(
    rng::Random.AbstractRNG,
    ::Type{T},
) where T
    return rng, rand(rng)
end

# Preserve the released global-RNG stream position.  These four draws were
# historically computed and then overwritten before the rejection loop.
@inline function heatbath_prepare_su2(
    rng::Random.AbstractRNG,
    ::Type{T},
) where T
    rng, _ = heatbath_uniform(rng, T)
    rng, _ = heatbath_uniform(rng, T)
    rng, _ = heatbath_uniform(rng, T)
    rng, _ = heatbath_uniform(rng, T)
    return rng
end

@inline heatbath_uniform(rng::SiteRNG, ::Type{T}) where T =
    rand_uniform(rng, T)
@inline heatbath_log_uniform(rng::SiteRNG, ::Type{T}) where T =
    rand_uniform_open(rng, T)
@inline heatbath_beta(rng::SiteRNG, beta, ::Type{T}) where T = T(beta)

"""
    normalize3_allocationfree!(u) -> success

Apply the legacy row-wise SU(3) reunitarization order without temporary heap
storage or host-only failure handling.
"""
@inline function normalize3_allocationfree!(u)
    T = eltype(u)
    RT = typeof(real(zero(T)))
    w1 = zero(T)
    w2 = zero(T)
    @inbounds for ic in 1:3
        w1 += u[2, ic] * conj(u[1, ic])
        w2 += u[1, ic] * conj(u[1, ic])
    end
    iszero(w2) && return false

    w1 = -w1 / w2
    @inbounds begin
        x4 = u[2, 1] + w1 * u[1, 1]
        x5 = u[2, 2] + w1 * u[1, 2]
        x6 = u[2, 3] + w1 * u[1, 3]
        w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)
        iszero(w3) && return false

        u[2, 1] = x4
        u[2, 2] = x5
        u[2, 3] = x6

        w3 = 1 / sqrt(w3)
        w2 = 1 / sqrt(w2)
        u[1, 1] = u[1, 1] * w2
        u[1, 2] = u[1, 2] * w2
        u[1, 3] = u[1, 3] * w2
        u[2, 1] = u[2, 1] * w3
        u[2, 2] = u[2, 2] * w3
        u[2, 3] = u[2, 3] * w3

        aa1 = RT(real(u[1, 1]))
        aa2 = RT(imag(u[1, 1]))
        aa3 = RT(real(u[1, 2]))
        aa4 = RT(imag(u[1, 2]))
        aa5 = RT(real(u[1, 3]))
        aa6 = RT(imag(u[1, 3]))
        aa7 = RT(real(u[2, 1]))
        aa8 = RT(imag(u[2, 1]))
        aa9 = RT(real(u[2, 2]))
        aa10 = RT(imag(u[2, 2]))
        aa11 = RT(real(u[2, 3]))
        aa12 = RT(imag(u[2, 3]))

        aa13 = aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
        aa14 = aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
        aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
        aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
        aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
        aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

        u[3, 1] = aa13 + im * aa14
        u[3, 2] = aa15 + im * aa16
        u[3, 3] = aa17 + im * aa18
    end
    return true
end

@inline heatbath_normalize3!(u) = normalize3_allocationfree!(u)

@noinline function _throw_kp_heatbath_failure(result, ITERATION_MAX)
    _, _, _, k, rho, delta, R, Rp, Rpp, Rppp = result
    error(
        "KP heatbath failed to accept after $ITERATION_MAX tries; " *
        "k=$k, ρ=$rho, delta=$delta R = $R Rp=$Rp Rpp=$Rpp Rppp=$Rppp",
    )
end

function SU2update_KP!(Unew, V, beta, NC, temps, ITERATION_MAX=10^5)
    result = _prepare_and_su2_update_kp!(
        Random.default_rng(), Unew, V, beta, NC, temps, ITERATION_MAX
    )
    result[2] || _throw_kp_heatbath_failure(result, ITERATION_MAX)
    return nothing
end

function SU2update_KP_allocationfree!(
    Unew,
    V,
    beta,
    NC,
    temps,
    ITERATION_MAX=10^5,
)
    return SU2update_KP!(Unew, V, beta, NC, temps, ITERATION_MAX)
end

@inline function SU2update_KP_rng!(
    Unew,
    V,
    beta,
    NC,
    temps,
    rng::SiteRNG,
    ITERATION_MAX=10^5,
)
    return SU2update_KP!(rng, Unew, V, beta, NC, temps, ITERATION_MAX)
end

function SU3update_matrix!(u, V, beta, NC, temps2, temps3, ITERATION_MAX)
    NC == 3 || throw(ArgumentError("SU3update_matrix! requires NC=3"))
    rng, accepted, failed_subgroup, result = _su3_update_single_stream_core!(
        Random.default_rng(),
        u,
        V,
        beta,
        NC,
        temps2,
        temps3,
        ITERATION_MAX,
    )
    if !accepted
        failed_subgroup in 1:3 &&
            _throw_kp_heatbath_failure(result, ITERATION_MAX)
        error("SU(3) normalization failed")
    end
    return nothing
end

function SU3update_matrix_allocationfree!(
    u,
    V,
    beta,
    NC,
    temps2,
    temps3,
    ITERATION_MAX=10^5,
)
    return SU3update_matrix!(
        u, V, beta, NC, temps2, temps3, ITERATION_MAX
    )
end

@inline function SU3update_matrix_rng!(
    u,
    V,
    beta,
    temps2,
    temps3,
    rngs::NTuple{3,R},
    ITERATION_MAX=10^5,
) where {R<:SiteRNG}
    return SU3update_matrix!(
        rngs, u, V, beta, 3, temps2, temps3, ITERATION_MAX
    )
end
