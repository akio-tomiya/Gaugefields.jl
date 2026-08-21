# Serial legacy storage adapter. The algorithm and one-site matrix operations
# are shared with the LatticeMatrices/JACC implementation.
gaugefixing_backend_supported(g::Gaugefields_4D_nowing) = g.NC in (2, 3)

function make_g_los_alamos!(
    U::Array{T,1},
    g::Gaugefields_4D_nowing{NC},
    temp::Gaugefields_4D_nowing{NC},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
) where {NC,T<:Gaugefields_4D_nowing}
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

    @inbounds for nt in 1:g.NT
        for nz in 1:g.NZ
            for ny in 1:g.NY
                for nx in 1:g.NX
                    if ((nx + ny + nz + nt) & 1) == parity
                        for jc in 1:NC
                            for ic in 1:NC
                                w_x[ic, jc] = W[ic, jc, nx, ny, nz, nt]
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
                            g[ic, jc, nx, ny, nz, nt] = G_tmp[ic, jc]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(g)
    return nothing
end

function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_nowing{NC},
    Δ::Gaugefields_4D_nowing{NC},
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
) where {NC,T<:Gaugefields_4D_nowing}
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

    @inbounds for nt in 1:g.NT
        for nz in 1:g.NZ
            for ny in 1:g.NY
                for nx in 1:g.NX
                    if ((nx + ny + nz + nt) & 1) == parity
                        for jc in 1:NC
                            for ic in 1:NC
                                Um_tmp[ic, jc] = Um[ic, jc, nx, ny, nz, nt]
                                Up_tmp[ic, jc] = Up[ic, jc, nx, ny, nz, nt]
                                Δ_tmp[ic, jc] = Δ[ic, jc, nx, ny, nz, nt]
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
                            g[ic, jc, nx, ny, nz, nt] = G_tmp[ic, jc]
                        end
                    end
                end
            end
        end
    end
    normalize_U!(g)
    set_wing_U!(g)
    return nothing
end
