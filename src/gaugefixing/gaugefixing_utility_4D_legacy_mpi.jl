# Legacy MPI storage adapters.  The PR #155 implementations supplied the
# storage-specific iteration pattern; the checkerboard and one-site matrix
# operations here are shared with the LM and serial adapters.
gaugefixing_backend_supported(g::Gaugefields_4D_nowing_mpi) = g.NC in (2, 3)
gaugefixing_backend_supported(g::Gaugefields_4D_wing_mpi) = g.NC in (2, 3)

function unit_U!(g::Gaugefields_4D_wing_mpi{NC}) where {NC}
    @inbounds for nt in 1:g.PN[4]
        for nz in 1:g.PN[3]
            for ny in 1:g.PN[2]
                for nx in 1:g.PN[1]
                    for jc in 1:NC
                        for ic in 1:NC
                            setvalue!(
                                g,
                                ic == jc ? 1.0 + 0.0im : 0.0 + 0.0im,
                                ic,
                                jc,
                                nx,
                                ny,
                                nz,
                                nt,
                            )
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(g)
    return nothing
end

function _normalize_legacy_mpi!(g, ::Val{NC}) where {NC}
    matrix = Matrix{ComplexF64}(undef, NC, NC)
    @inbounds for nt in 1:g.PN[4]
        for nz in 1:g.PN[3]
            for ny in 1:g.PN[2]
                for nx in 1:g.PN[1]
                    for jc in 1:NC
                        for ic in 1:NC
                            matrix[ic, jc] =
                                getvalue(g, ic, jc, nx, ny, nz, nt)
                        end
                    end

                    if NC == 2
                        alpha = matrix[1, 1]
                        beta = matrix[2, 1]
                        column_norm = sqrt(abs2(alpha) + abs2(beta))
                        if column_norm > eps(Float64)
                            matrix[1, 1] = alpha / column_norm
                            matrix[2, 1] = beta / column_norm
                            matrix[1, 2] = -conj(beta) / column_norm
                            matrix[2, 2] = conj(alpha) / column_norm
                        else
                            _set_identity!(matrix, Val(2))
                        end
                    else
                        normalize3!(matrix)
                    end

                    for jc in 1:NC
                        for ic in 1:NC
                            setvalue!(
                                g,
                                matrix[ic, jc],
                                ic,
                                jc,
                                nx,
                                ny,
                                nz,
                                nt,
                            )
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(g)
    return nothing
end

@inline function _legacy_mpi_parity(g, nx, ny, nz, nt)
    rx, ry, rz, rt = g.myrank_xyzt
    lx, ly, lz, lt = g.PN
    return (
        nx + rx * lx + ny + ry * ly + nz + rz * lz + nt + rt * lt
    ) & 1
end

function _make_g_transform_legacy_mpi!(
    U,
    g,
    temp,
    parity,
    overrelax,
    ovr_coeff2,
    ovr_coeff3,
    D_fix,
    ::Val{NC},
) where {NC}
    W = temp
    clear_U!(W)
    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        add_U!(W, U[μ])
        add_U!(W, U_shift')
    end

    w_x = Matrix{ComplexF64}(undef, NC, NC)
    G_tmp = Matrix{ComplexF64}(undef, NC, NC)
    M_tmp = Matrix{ComplexF64}(undef, NC, NC)
    A_tmp = Matrix{ComplexF64}(undef, NC, NC)

    @inbounds for nt in 1:g.PN[4]
        for nz in 1:g.PN[3]
            for ny in 1:g.PN[2]
                for nx in 1:g.PN[1]
                    if _legacy_mpi_parity(g, nx, ny, nz, nt) == parity
                        for jc in 1:NC
                            for ic in 1:NC
                                w_x[ic, jc] =
                                    getvalue(W, ic, jc, nx, ny, nz, nt)
                            end
                        end
                        _su2_subgroup_hit!(
                            G_tmp,
                            w_x,
                            M_tmp,
                            A_tmp,
                            overrelax,
                            ovr_coeff2,
                            ovr_coeff3,
                            Val(NC),
                        )
                    else
                        _set_identity!(G_tmp, Val(NC))
                    end

                    for jc in 1:NC
                        for ic in 1:NC
                            setvalue!(
                                g,
                                G_tmp[ic, jc],
                                ic,
                                jc,
                                nx,
                                ny,
                                nz,
                                nt,
                            )
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(g)
    return nothing
end

function _make_g_steepest_descent_legacy_mpi!(
    U,
    g,
    Δ,
    parity,
    overrelax,
    temps,
    D_fix,
    ::Val{NC},
) where {NC}
    get_Δ!(Δ, U, temps[1:4], D_fix)

    Um = temps[1]
    Up = temps[2]
    clear_U!(Um)
    clear_U!(Up)
    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        add_U!(Um, U[μ])
        add_U!(Um, -1, U_shift)
        add_U!(Up, U[μ])
        add_U!(Up, U_shift)
    end

    G_tmp = Matrix{ComplexF64}(undef, NC, NC)
    Um_tmp = Matrix{ComplexF64}(undef, NC, NC)
    Up_tmp = Matrix{ComplexF64}(undef, NC, NC)
    Δ_tmp = Matrix{ComplexF64}(undef, NC, NC)
    Δ2 = Matrix{ComplexF64}(undef, NC, NC)

    @inbounds for nt in 1:g.PN[4]
        for nz in 1:g.PN[3]
            for ny in 1:g.PN[2]
                for nx in 1:g.PN[1]
                    if _legacy_mpi_parity(g, nx, ny, nz, nt) == parity
                        for jc in 1:NC
                            for ic in 1:NC
                                Um_tmp[ic, jc] =
                                    getvalue(Um, ic, jc, nx, ny, nz, nt)
                                Up_tmp[ic, jc] =
                                    getvalue(Up, ic, jc, nx, ny, nz, nt)
                                Δ_tmp[ic, jc] =
                                    getvalue(Δ, ic, jc, nx, ny, nz, nt)
                            end
                        end
                        _steepest_descent_transform!(
                            G_tmp,
                            Um_tmp,
                            Up_tmp,
                            Δ_tmp,
                            Δ2,
                            overrelax,
                            Val(NC),
                        )
                    else
                        _set_identity!(G_tmp, Val(NC))
                    end

                    for jc in 1:NC
                        for ic in 1:NC
                            setvalue!(
                                g,
                                G_tmp[ic, jc],
                                ic,
                                jc,
                                nx,
                                ny,
                                nz,
                                nt,
                            )
                        end
                    end
                end
            end
        end
    end
    _normalize_legacy_mpi!(g, Val(NC))
    return nothing
end

for MPIField in (Gaugefields_4D_nowing_mpi, Gaugefields_4D_wing_mpi)
    @eval begin
        function make_g_los_alamos!(
            U::Array{T,1},
            g::$MPIField{NC},
            temp::$MPIField{NC},
            parity::Int,
            overrelax::Float64,
            ovr_coeff2::Float64,
            ovr_coeff3::Float64,
            D_fix::Int=4,
        ) where {NC,T<:$MPIField}
            return _make_g_transform_legacy_mpi!(
                U,
                g,
                temp,
                parity,
                overrelax,
                ovr_coeff2,
                ovr_coeff3,
                D_fix,
                Val(NC),
            )
        end

        function make_g_steepest_descent!(
            U::Array{T,1},
            g::$MPIField{NC},
            Δ::$MPIField{NC},
            parity::Int,
            overrelax::Float64,
            temps::Array{T,1},
            D_fix::Int=4,
        ) where {NC,T<:$MPIField}
            return _make_g_steepest_descent_legacy_mpi!(
                U, g, Δ, parity, overrelax, temps, D_fix, Val(NC),
            )
        end
    end
end
