# Portable adapter for the deprecated accelerator storage.  PR #155 used a
# separate CUDA-only implementation; this path deliberately targets the
# storage's JACC mode so the same kernels can run on every JACC backend.
gaugefixing_backend_supported(
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
) where {NC,TU,TUv,TS} = NC in (2, 3)

@inline function _accelerator_coords(i, NX, NY, NZ)
    i0 = i - 1
    ix = i0 % NX + 1
    i0 = i0 ÷ NX
    iy = i0 % NY + 1
    i0 = i0 ÷ NY
    iz = i0 % NZ + 1
    it = i0 ÷ NZ + 1
    return ix, iy, iz, it
end

@inline function _accelerator_index(ix, iy, iz, it, NX, NY, NZ)
    return (((it - 1) * NZ + iz - 1) * NY + iy - 1) * NX + ix
end

@inline function _accelerator_neighbours(i, NX, NY, NZ, NT)
    ix, iy, iz, it = _accelerator_coords(i, NX, NY, NZ)
    xm = ifelse(ix == 1, NX, ix - 1)
    ym = ifelse(iy == 1, NY, iy - 1)
    zm = ifelse(iz == 1, NZ, iz - 1)
    tm = ifelse(it == 1, NT, it - 1)
    xp = ifelse(ix == NX, 1, ix + 1)
    yp = ifelse(iy == NY, 1, iy + 1)
    zp = ifelse(iz == NZ, 1, iz + 1)
    tp = ifelse(it == NT, 1, it + 1)
    minus = (
        _accelerator_index(xm, iy, iz, it, NX, NY, NZ),
        _accelerator_index(ix, ym, iz, it, NX, NY, NZ),
        _accelerator_index(ix, iy, zm, it, NX, NY, NZ),
        _accelerator_index(ix, iy, iz, tm, NX, NY, NZ),
    )
    plus = (
        _accelerator_index(xp, iy, iz, it, NX, NY, NZ),
        _accelerator_index(ix, yp, iz, it, NX, NY, NZ),
        _accelerator_index(ix, iy, zp, it, NX, NY, NZ),
        _accelerator_index(ix, iy, iz, tp, NX, NY, NZ),
    )
    return (ix + iy + iz + it) & 1, minus, plus
end

@inline function _kernel_accelerator_identity!(i, g, ::Val{NC}) where {NC}
    @inbounds for jc in 1:NC
        for ic in 1:NC
            g[ic, jc, i] = ifelse(ic == jc, one(eltype(g)), zero(eltype(g)))
        end
    end
    return nothing
end

function unit_U!(
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
) where {NC,TU,TUv,TS}
    JACC.parallel_for(g.NV, _kernel_accelerator_identity!, g.U, Val(NC))
    return nothing
end

@inline function _kernel_accelerator_transform!(
    i,
    g,
    u1,
    u2,
    u3,
    u4,
    NX,
    NY,
    NZ,
    NT,
    parity,
    overrelax,
    coeff2,
    coeff3,
    D_fix,
    ::Val{NC},
) where {NC}
    site_parity, minus, _ = _accelerator_neighbours(i, NX, NY, NZ, NT)
    T = eltype(g)
    G_tmp = MMatrix{NC,NC,T}(undef)

    if site_parity == parity
        w_x = MMatrix{NC,NC,T}(undef)
        M_tmp = MMatrix{NC,NC,T}(undef)
        A_tmp = MMatrix{NC,NC,T}(undef)
        @inbounds for jc in 1:NC
            for ic in 1:NC
                value = zero(T)
                if D_fix >= 1
                    value += u1[ic, jc, i] + conj(u1[jc, ic, minus[1]])
                end
                if D_fix >= 2
                    value += u2[ic, jc, i] + conj(u2[jc, ic, minus[2]])
                end
                if D_fix >= 3
                    value += u3[ic, jc, i] + conj(u3[jc, ic, minus[3]])
                end
                if D_fix >= 4
                    value += u4[ic, jc, i] + conj(u4[jc, ic, minus[4]])
                end
                w_x[ic, jc] = value
            end
        end
        _su2_subgroup_hit!(
            G_tmp,
            w_x,
            M_tmp,
            A_tmp,
            overrelax,
            coeff2,
            coeff3,
            Val(NC),
        )
    else
        _set_identity!(G_tmp, Val(NC))
    end

    @inbounds for jc in 1:NC
        for ic in 1:NC
            g[ic, jc, i] = G_tmp[ic, jc]
        end
    end
    return nothing
end

function make_g_los_alamos!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    temp::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
) where {NC,TU,TUv,TS,T<:Gaugefields_4D_accelerator}
    RT = typeof(real(zero(eltype(g.U))))
    JACC.parallel_for(
        g.NV,
        _kernel_accelerator_transform!,
        g.U,
        U[1].U,
        U[2].U,
        U[3].U,
        U[4].U,
        g.NX,
        g.NY,
        g.NZ,
        g.NT,
        parity,
        convert(RT, overrelax),
        convert(RT, ovr_coeff2),
        convert(RT, ovr_coeff3),
        D_fix,
        Val(NC),
    )
    return nothing
end

@inline function _kernel_accelerator_delta!(
    i, Δ, u1, u2, u3, u4, NX, NY, NZ, NT, D_fix, ::Val{NC},
) where {NC}
    _, minus, _ = _accelerator_neighbours(i, NX, NY, NZ, NT)
    T = eltype(Δ)
    sum_u = MMatrix{NC,NC,T}(undef)
    trace_imag = zero(typeof(real(zero(T))))

    @inbounds for jc in 1:NC
        for ic in 1:NC
            value = zero(T)
            if D_fix >= 1
                value += u1[ic, jc, i] - u1[ic, jc, minus[1]]
            end
            if D_fix >= 2
                value += u2[ic, jc, i] - u2[ic, jc, minus[2]]
            end
            if D_fix >= 3
                value += u3[ic, jc, i] - u3[ic, jc, minus[3]]
            end
            if D_fix >= 4
                value += u4[ic, jc, i] - u4[ic, jc, minus[4]]
            end
            sum_u[ic, jc] = value
        end
        trace_imag += imag(sum_u[jc, jc])
    end
    trace_imag /= NC

    @inbounds for jc in 1:NC
        for ic in 1:NC
            value = (sum_u[ic, jc] - conj(sum_u[jc, ic])) / 2
            if ic == jc
                value -= im * trace_imag
            end
            Δ[ic, jc, i] = value
        end
    end
    return nothing
end

function get_Δ!(
    Δ::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    U::Array{T,1},
    temps,
    D_fix,
) where {NC,TU,TUv,TS,T<:Gaugefields_4D_accelerator}
    JACC.parallel_for(
        Δ.NV,
        _kernel_accelerator_delta!,
        Δ.U,
        U[1].U,
        U[2].U,
        U[3].U,
        U[4].U,
        Δ.NX,
        Δ.NY,
        Δ.NZ,
        Δ.NT,
        D_fix,
        Val(NC),
    )
    return nothing
end

@inline function _kernel_accelerator_sd!(
    i,
    g,
    Δ,
    u1,
    u2,
    u3,
    u4,
    NX,
    NY,
    NZ,
    NT,
    parity,
    overrelax,
    D_fix,
    ::Val{NC},
) where {NC}
    site_parity, minus, _ = _accelerator_neighbours(i, NX, NY, NZ, NT)
    T = eltype(g)
    G_tmp = MMatrix{NC,NC,T}(undef)

    if site_parity == parity
        Um_tmp = MMatrix{NC,NC,T}(undef)
        Up_tmp = MMatrix{NC,NC,T}(undef)
        Δ_tmp = MMatrix{NC,NC,T}(undef)
        Δ2 = MMatrix{NC,NC,T}(undef)
        @inbounds for jc in 1:NC
            for ic in 1:NC
                minus_value = zero(T)
                plus_value = zero(T)
                if D_fix >= 1
                    minus_value += u1[ic, jc, i] - u1[ic, jc, minus[1]]
                    plus_value += u1[ic, jc, i] + u1[ic, jc, minus[1]]
                end
                if D_fix >= 2
                    minus_value += u2[ic, jc, i] - u2[ic, jc, minus[2]]
                    plus_value += u2[ic, jc, i] + u2[ic, jc, minus[2]]
                end
                if D_fix >= 3
                    minus_value += u3[ic, jc, i] - u3[ic, jc, minus[3]]
                    plus_value += u3[ic, jc, i] + u3[ic, jc, minus[3]]
                end
                if D_fix >= 4
                    minus_value += u4[ic, jc, i] - u4[ic, jc, minus[4]]
                    plus_value += u4[ic, jc, i] + u4[ic, jc, minus[4]]
                end
                Um_tmp[ic, jc] = minus_value
                Up_tmp[ic, jc] = plus_value
                Δ_tmp[ic, jc] = Δ[ic, jc, i]
            end
        end
        _steepest_descent_transform!(
            G_tmp, Um_tmp, Up_tmp, Δ_tmp, Δ2, overrelax, Val(NC),
        )
    else
        _set_identity!(G_tmp, Val(NC))
    end

    @inbounds for jc in 1:NC
        for ic in 1:NC
            g[ic, jc, i] = G_tmp[ic, jc]
        end
    end
    return nothing
end

function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    Δ::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
) where {NC,TU,TUv,TS,T<:Gaugefields_4D_accelerator}
    get_Δ!(Δ, U, temps[1:4], D_fix)
    RT = typeof(real(zero(eltype(g.U))))
    JACC.parallel_for(
        g.NV,
        _kernel_accelerator_sd!,
        g.U,
        Δ.U,
        U[1].U,
        U[2].U,
        U[3].U,
        U[4].U,
        g.NX,
        g.NY,
        g.NZ,
        g.NT,
        parity,
        convert(RT, overrelax),
        D_fix,
        Val(NC),
    )
    normalize_U!(g)
    return nothing
end

@inline function _kernel_accelerator_gauge_transform!(
    i, Uout, Uin, g, NX, NY, NZ, NT, μ, ::Val{NC},
) where {NC}
    _, _, plus = _accelerator_neighbours(i, NX, NY, NZ, NT)
    T = eltype(Uout)
    temp = MMatrix{NC,NC,T}(undef)
    result = MMatrix{NC,NC,T}(undef)

    @inbounds for jc in 1:NC
        for ic in 1:NC
            value = zero(T)
            for kc in 1:NC
                value += g[ic, kc, i] * Uin[kc, jc, i]
            end
            temp[ic, jc] = value
        end
    end
    @inbounds for jc in 1:NC
        for ic in 1:NC
            value = zero(T)
            for kc in 1:NC
                value += temp[ic, kc] * conj(g[jc, kc, plus[μ]])
            end
            result[ic, jc] = value
        end
    end
    @inbounds for jc in 1:NC
        for ic in 1:NC
            Uout[ic, jc, i] = result[ic, jc]
        end
    end
    return nothing
end

function gUgshift!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    temp::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
) where {NC,TU,TUv,TS,T<:Gaugefields_4D_accelerator}
    for μ in 1:4
        JACC.parallel_for(
            g.NV,
            _kernel_accelerator_gauge_transform!,
            U[μ].U,
            U[μ].U,
            g.U,
            g.NX,
            g.NY,
            g.NZ,
            g.NT,
            μ,
            Val(NC),
        )
    end
    return nothing
end

function gUgshift!(
    Uout::Array{T,1},
    Uin::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    temp::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
) where {NC,TU,TUv,TS,T<:Gaugefields_4D_accelerator}
    for μ in 1:4
        JACC.parallel_for(
            g.NV,
            _kernel_accelerator_gauge_transform!,
            Uout[μ].U,
            Uin[μ].U,
            g.U,
            g.NX,
            g.NY,
            g.NZ,
            g.NT,
            μ,
            Val(NC),
        )
    end
    return nothing
end
