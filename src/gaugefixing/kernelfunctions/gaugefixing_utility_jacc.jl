@inline function _gaugefixing_parity(i, dindexer, coords, local_size)
    local_indices = delinearize(dindexer, i, 0)
    coordinate_sum = 0
    @inbounds for d in eachindex(local_indices)
        coordinate_sum += coords[d] * local_size[d] + local_indices[d]
    end
    return coordinate_sum & 1
end

@inline function jacckernel_SU2_subgroup_hit!(
    i,
    g,
    W,
    dindexer,
    coords,
    local_size,
    parity::Int,
    overrelax,
    coeff2,
    coeff3,
    ::Val{NC},
    ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(g)
    w_x = MMatrix{NC,NC,T}(undef)
    G_tmp = MMatrix{NC,NC,T}(undef)
    M_tmp = MMatrix{NC,NC,T}(undef)
    A_tmp = MMatrix{NC,NC,T}(undef)

    if _gaugefixing_parity(i, dindexer, coords, local_size) == parity
        @inbounds for jc in 1:NC
            for ic in 1:NC
                w_x[ic, jc] = W[ic, jc, indices...]
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
            g[ic, jc, indices...] = G_tmp[ic, jc]
        end
    end
    return nothing
end

@inline function jacckernel_mino_method!(
    i,
    dindexer,
    g,
    Um,
    Up,
    Δ,
    coords,
    local_size,
    parity::Int,
    overrelax,
    ::Val{NC},
    ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(g)
    G_tmp = MMatrix{NC,NC,T}(undef)

    if _gaugefixing_parity(i, dindexer, coords, local_size) == parity
        Um_tmp = MMatrix{NC,NC,T}(undef)
        Up_tmp = MMatrix{NC,NC,T}(undef)
        Δ_tmp = MMatrix{NC,NC,T}(undef)
        Δ2 = MMatrix{NC,NC,T}(undef)
        @inbounds for jc in 1:NC
            for ic in 1:NC
                Um_tmp[ic, jc] = Um[ic, jc, indices...]
                Up_tmp[ic, jc] = Up[ic, jc, indices...]
                Δ_tmp[ic, jc] = Δ[ic, jc, indices...]
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
            g[ic, jc, indices...] = G_tmp[ic, jc]
        end
    end
    return nothing
end
