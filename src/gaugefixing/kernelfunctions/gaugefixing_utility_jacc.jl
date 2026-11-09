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

# ---------------------------------------------------------------------------
# Fused site kernels for the MPILattice gauge fixing. Each repeats, on one site,
# the arithmetic of the LatticeMatrices field operations it replaces (add_matrix!,
# traceless_antihermitian!, mul! with a shifted adjoint), in the same order, so the
# gauge-fixed links are the same; only the passes over the lattice are fewer.

@inline _backward_site(indices, μ) =
    ntuple(d -> indices[d] - ifelse(d == μ, 1, 0), length(indices))

# Um = Σ_μ U_μ(x) - U_μ(x-μ) and Up = Σ_μ U_μ(x) + U_μ(x-μ), accumulated as
# get_Δ! and make_g_steepest_descent! do (u += α * v with integer α).
@inline function _sd_sums!(Um, Up, links, indices, D_fix, ::Val{NC}) where {NC}
    T = eltype(Um)
    @inbounds for jc in 1:NC, ic in 1:NC
        Um[ic, jc] = zero(T)
        Up[ic, jc] = zero(T)
    end
    @inbounds for μ in 1:D_fix
        U = links[μ]
        back = _backward_site(indices, μ)
        for jc in 1:NC, ic in 1:NC
            u = U[ic, jc, indices...]
            ub = U[ic, jc, back...]
            Um[ic, jc] += 1 * u
            Um[ic, jc] += -1 * ub
            Up[ic, jc] += 1 * u
            Up[ic, jc] += 1 * ub
        end
    end
    return nothing
end

# W = Σ_μ U_μ(x) + U_μ(x-μ)† as make_g_los_alamos! accumulates it.
@inline function _la_sum!(W, links, indices, D_fix, ::Val{NC}) where {NC}
    T = eltype(W)
    @inbounds for jc in 1:NC, ic in 1:NC
        W[ic, jc] = zero(T)
    end
    @inbounds for μ in 1:D_fix
        U = links[μ]
        back = _backward_site(indices, μ)
        for jc in 1:NC, ic in 1:NC
            W[ic, jc] += 1 * U[ic, jc, indices...]
        end
        for jc in 1:NC, ic in 1:NC
            W[ic, jc] += 1 * U[jc, ic, back...]'
        end
    end
    return nothing
end

# Traceless antihermitian part, as LatticeMatrices.kernel_traceless_antihermitian!.
@inline function _traceless_antihermitian_local!(out, v, ::Val{3})
    fac13 = 1 / 3
    @inbounds begin
        v11 = v[1, 1]; v21 = v[2, 1]; v31 = v[3, 1]
        v12 = v[1, 2]; v22 = v[2, 2]; v32 = v[3, 2]
        v13 = v[1, 3]; v23 = v[2, 3]; v33 = v[3, 3]
        tri = fac13 * (imag(v11) + imag(v22) + imag(v33))
        y11 = (imag(v11) - tri) * im
        y22 = (imag(v22) - tri) * im
        y33 = (imag(v33) - tri) * im
        x12 = v12 - conj(v21)
        x13 = v13 - conj(v31)
        x23 = v23 - conj(v32)
        x21 = -conj(x12)
        x31 = -conj(x13)
        x32 = -conj(x23)
        out[1, 1] = y11; out[2, 1] = 0.5 * x21; out[3, 1] = 0.5 * x31
        out[1, 2] = 0.5 * x12; out[2, 2] = y22; out[3, 2] = 0.5 * x32
        out[1, 3] = 0.5 * x13; out[2, 3] = 0.5 * x23; out[3, 3] = y33
    end
    return nothing
end

@inline function _traceless_antihermitian_local!(out, v, ::Val{2})
    @inbounds begin
        v11 = v[1, 1]; v22 = v[2, 2]
        tri = (imag(v11) + imag(v22)) * 0.5
        x12 = v[1, 2] - conj(v[2, 1])
        x21 = -conj(x12)
        out[1, 1] = (imag(v11) - tri) * im
        out[1, 2] = 0.5 * x12
        out[2, 1] = 0.5 * x21
        out[2, 2] = (imag(v22) - tri) * im
    end
    return nothing
end

@inline function _traceless_antihermitian_local!(out, v, ::Val{N}) where {N}
    fac1N = 1 / N
    tri = 0.0
    @inbounds for k in 1:N
        tri += imag(v[k, k])
    end
    tri *= fac1N
    @inbounds for k in 1:N
        out[k, k] = (imag(v[k, k]) - tri) * im
    end
    @inbounds for k1 in 1:N, k2 in (k1 + 1):N
        vv = 0.5 * (v[k1, k2] - conj(v[k2, k1]))
        out[k1, k2] = vv
        out[k2, k1] = -conj(vv)
    end
    return nothing
end

@inline function jacckernel_steepest_descent_fused!(
    i, g, links, dindexer, coords, local_size, parity::Int, overrelax, D_fix,
    ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(g)
    G_tmp = MMatrix{NC,NC,T}(undef)
    if _gaugefixing_parity(i, dindexer, coords, local_size) == parity
        Um_tmp = MMatrix{NC,NC,T}(undef)
        Up_tmp = MMatrix{NC,NC,T}(undef)
        Δ_tmp = MMatrix{NC,NC,T}(undef)
        Δ2 = MMatrix{NC,NC,T}(undef)
        _sd_sums!(Um_tmp, Up_tmp, links, indices, D_fix, Val(NC))
        _traceless_antihermitian_local!(Δ_tmp, Um_tmp, Val(NC))
        _steepest_descent_transform!(
            G_tmp, Um_tmp, Up_tmp, Δ_tmp, Δ2, overrelax, Val(NC),
        )
    else
        _set_identity!(G_tmp, Val(NC))
    end
    @inbounds for jc in 1:NC, ic in 1:NC
        g[ic, jc, indices...] = G_tmp[ic, jc]
    end
    return nothing
end

# W(x) of make_g_los_alamos!.
@inline function jacckernel_los_alamos_sum!(
    i, W, links, dindexer, D_fix, ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(W)
    w_x = MMatrix{NC,NC,T}(undef)
    _la_sum!(w_x, links, indices, D_fix, Val(NC))
    @inbounds for jc in 1:NC, ic in 1:NC
        W[ic, jc, indices...] = w_x[ic, jc]
    end
    return nothing
end

# Δ(x) of get_Δ!: the traceless antihermitian part of Σ_μ U_μ(x) - U_μ(x-μ).
@inline function jacckernel_delta_fused!(
    i, Δ, links, dindexer, D_fix, ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    T = eltype(Δ)
    sum_tmp = MMatrix{NC,NC,T}(undef)
    Δ_tmp = MMatrix{NC,NC,T}(undef)
    @inbounds for jc in 1:NC, ic in 1:NC
        sum_tmp[ic, jc] = zero(T)
    end
    @inbounds for μ in 1:D_fix
        U = links[μ]
        back = _backward_site(indices, μ)
        for jc in 1:NC, ic in 1:NC
            sum_tmp[ic, jc] += 1 * U[ic, jc, indices...]
            sum_tmp[ic, jc] += -1 * U[ic, jc, back...]
        end
    end
    _traceless_antihermitian_local!(Δ_tmp, sum_tmp, Val(NC))
    @inbounds for jc in 1:NC, ic in 1:NC
        Δ[ic, jc, indices...] = Δ_tmp[ic, jc]
    end
    return nothing
end

# U_μ(x) ← (g(x) U_μ(x)) g(x+μ)†, the two mul! of gUgshift! on one site.
@inline function jacckernel_gauge_transform_link!(
    i, Uμ, g, dindexer, μ, ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    forward = ntuple(d -> indices[d] + ifelse(d == μ, 1, 0), length(indices))
    T = eltype(Uμ)
    temp = MMatrix{NC,NC,T}(undef)
    _mul_local!(temp, g, Uμ, indices, Val(NC))
    _mul_shiftdag_local!(Uμ, temp, g, indices, forward, Val(NC))
    return nothing
end

# kernel_Dmatrix_mul!: temp = A(x) B(x)
@inline function _mul_local!(C, A, B, indices, ::Val{3})
    @inbounds begin
        a11 = A[1, 1, indices...]; a21 = A[2, 1, indices...]; a31 = A[3, 1, indices...]
        a12 = A[1, 2, indices...]; a22 = A[2, 2, indices...]; a32 = A[3, 2, indices...]
        a13 = A[1, 3, indices...]; a23 = A[2, 3, indices...]; a33 = A[3, 3, indices...]
        b11 = B[1, 1, indices...]; b21 = B[2, 1, indices...]; b31 = B[3, 1, indices...]
        b12 = B[1, 2, indices...]; b22 = B[2, 2, indices...]; b32 = B[3, 2, indices...]
        b13 = B[1, 3, indices...]; b23 = B[2, 3, indices...]; b33 = B[3, 3, indices...]
        C[1, 1] = a11 * b11 + a12 * b21 + a13 * b31
        C[2, 1] = a21 * b11 + a22 * b21 + a23 * b31
        C[3, 1] = a31 * b11 + a32 * b21 + a33 * b31
        C[1, 2] = a11 * b12 + a12 * b22 + a13 * b32
        C[2, 2] = a21 * b12 + a22 * b22 + a23 * b32
        C[3, 2] = a31 * b12 + a32 * b22 + a33 * b32
        C[1, 3] = a11 * b13 + a12 * b23 + a13 * b33
        C[2, 3] = a21 * b13 + a22 * b23 + a23 * b33
        C[3, 3] = a31 * b13 + a32 * b23 + a33 * b33
    end
    return nothing
end

@inline function _mul_local!(C, A, B, indices, ::Val{NC}) where {NC}
    T = eltype(C)
    @inbounds for jc in 1:NC
        for ic in 1:NC
            C[ic, jc] = zero(T)
        end
        for kc in 1:NC
            b = B[kc, jc, indices...]
            for ic in 1:NC
                C[ic, jc] += A[ic, kc, indices...] * b
            end
        end
    end
    return nothing
end

# kernel_Dmatrix_mul_AshiftBdag!: U(x) = temp B(x+μ)†
@inline function _mul_shiftdag_local!(U, A, B, indices, forward, ::Val{3})
    @inbounds begin
        a11 = A[1, 1]; a21 = A[2, 1]; a31 = A[3, 1]
        a12 = A[1, 2]; a22 = A[2, 2]; a32 = A[3, 2]
        a13 = A[1, 3]; a23 = A[2, 3]; a33 = A[3, 3]
        b11 = B[1, 1, forward...]'
        b21 = B[1, 2, forward...]'
        b31 = B[1, 3, forward...]'
        U[1, 1, indices...] = a11 * b11 + a12 * b21 + a13 * b31
        U[2, 1, indices...] = a21 * b11 + a22 * b21 + a23 * b31
        U[3, 1, indices...] = a31 * b11 + a32 * b21 + a33 * b31
        b12 = B[2, 1, forward...]'
        b22 = B[2, 2, forward...]'
        b32 = B[2, 3, forward...]'
        U[1, 2, indices...] = a11 * b12 + a12 * b22 + a13 * b32
        U[2, 2, indices...] = a21 * b12 + a22 * b22 + a23 * b32
        U[3, 2, indices...] = a31 * b12 + a32 * b22 + a33 * b32
        b13 = B[3, 1, forward...]'
        b23 = B[3, 2, forward...]'
        b33 = B[3, 3, forward...]'
        U[1, 3, indices...] = a11 * b13 + a12 * b23 + a13 * b33
        U[2, 3, indices...] = a21 * b13 + a22 * b23 + a23 * b33
        U[3, 3, indices...] = a31 * b13 + a32 * b23 + a33 * b33
    end
    return nothing
end

@inline function _mul_shiftdag_local!(U, A, B, indices, forward, ::Val{NC}) where {NC}
    T = eltype(A)
    @inbounds for jc in 1:NC
        for ic in 1:NC
            U[ic, jc, indices...] = zero(T)
        end
        for kc in 1:NC
            b = conj(B[jc, kc, forward...])
            for ic in 1:NC
                U[ic, jc, indices...] += A[ic, kc] * b
            end
        end
    end
    return nothing
end
