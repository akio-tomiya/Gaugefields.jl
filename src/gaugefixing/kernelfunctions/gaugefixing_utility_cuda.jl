# Direct CUDA kernels for the legacy accelerator layout.  The local matrix
# operations are shared with the other gauge-fixing backends so the numerical
# definition does not diverge from LatticeMatrices or portable JACC.

@inline function _cuda_gaugefix_minus_indices(b, r, blockinfo)
    return (
        shiftedindex(b, r, (-1, 0, 0, 0), blockinfo),
        shiftedindex(b, r, (0, -1, 0, 0), blockinfo),
        shiftedindex(b, r, (0, 0, -1, 0), blockinfo),
        shiftedindex(b, r, (0, 0, 0, -1), blockinfo),
    )
end

@inline function _cuda_gaugefix_link_sum(
    u1, u2, u3, u4, ic, jc, b, r, minus, D_fix, daggered,
)
    T = eltype(u1)
    value = zero(T)
    if D_fix >= 1
        bm, rm = minus[1]
        value += daggered ? conj(u1[jc, ic, bm, rm]) : u1[ic, jc, bm, rm]
    end
    if D_fix >= 2
        bm, rm = minus[2]
        value += daggered ? conj(u2[jc, ic, bm, rm]) : u2[ic, jc, bm, rm]
    end
    if D_fix >= 3
        bm, rm = minus[3]
        value += daggered ? conj(u3[jc, ic, bm, rm]) : u3[ic, jc, bm, rm]
    end
    if D_fix >= 4
        bm, rm = minus[4]
        value += daggered ? conj(u4[jc, ic, bm, rm]) : u4[ic, jc, bm, rm]
    end
    return value
end

function cudakernel_gaugefix_transform!(
    u1,
    u2,
    u3,
    u4,
    g,
    parity,
    overrelax,
    coeff2,
    coeff3,
    D_fix,
    ::Val{NC},
    blockinfo,
) where {NC}
    b = Int(CUDA.threadIdx().x)
    r = Int(CUDA.blockIdx().x)
    T = eltype(g)
    G_tmp = MMatrix{NC,NC,T}(undef)
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

    if ((ix + iy + iz + it) & 1) == parity
        minus = _cuda_gaugefix_minus_indices(b, r, blockinfo)
        w_x = MMatrix{NC,NC,T}(undef)
        M_tmp = MMatrix{NC,NC,T}(undef)
        A_tmp = MMatrix{NC,NC,T}(undef)
        @inbounds for jc in 1:NC
            for ic in 1:NC
                onsite = zero(T)
                if D_fix >= 1
                    onsite += u1[ic, jc, b, r]
                end
                if D_fix >= 2
                    onsite += u2[ic, jc, b, r]
                end
                if D_fix >= 3
                    onsite += u3[ic, jc, b, r]
                end
                if D_fix >= 4
                    onsite += u4[ic, jc, b, r]
                end
                w_x[ic, jc] = onsite + _cuda_gaugefix_link_sum(
                    u1, u2, u3, u4, ic, jc, b, r, minus, D_fix, true,
                )
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
            g[ic, jc, b, r] = G_tmp[ic, jc]
        end
    end
    return nothing
end

function cudakernel_gaugefix_steepest_descent!(
    u1,
    u2,
    u3,
    u4,
    Δ,
    g,
    parity,
    overrelax,
    D_fix,
    ::Val{NC},
    blockinfo,
) where {NC}
    b = Int(CUDA.threadIdx().x)
    r = Int(CUDA.blockIdx().x)
    T = eltype(g)
    G_tmp = MMatrix{NC,NC,T}(undef)
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

    if ((ix + iy + iz + it) & 1) == parity
        minus = _cuda_gaugefix_minus_indices(b, r, blockinfo)
        Um_tmp = MMatrix{NC,NC,T}(undef)
        Up_tmp = MMatrix{NC,NC,T}(undef)
        Δ_tmp = MMatrix{NC,NC,T}(undef)
        Δ2 = MMatrix{NC,NC,T}(undef)
        @inbounds for jc in 1:NC
            for ic in 1:NC
                onsite = zero(T)
                if D_fix >= 1
                    onsite += u1[ic, jc, b, r]
                end
                if D_fix >= 2
                    onsite += u2[ic, jc, b, r]
                end
                if D_fix >= 3
                    onsite += u3[ic, jc, b, r]
                end
                if D_fix >= 4
                    onsite += u4[ic, jc, b, r]
                end
                neighbour = _cuda_gaugefix_link_sum(
                    u1, u2, u3, u4, ic, jc, b, r, minus, D_fix, false,
                )
                Um_tmp[ic, jc] = onsite - neighbour
                Up_tmp[ic, jc] = onsite + neighbour
                Δ_tmp[ic, jc] = Δ[ic, jc, b, r]
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
            g[ic, jc, b, r] = G_tmp[ic, jc]
        end
    end
    return nothing
end
