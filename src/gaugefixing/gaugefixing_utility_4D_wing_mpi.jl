# for Gaugefields_4D_wing_mpi

function make_g_transform!(
    U::Array{T,1},
    g::Gaugefields_4D_wing_mpi,
    temp::Gaugefields_4D_wing_mpi,
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
    ) where {T<:Gaugefields_4D_wing_mpi}

    Dim = size(U)[1]
    NC = g.NC
    NT = g.PN[4]
    NX = g.PN[1]
    NY = g.PN[2]
    NZ = g.PN[3]

    w_x = zeros(ComplexF64, NC, NC)
    G_tmp = Matrix{ComplexF64}(I, NC, NC) # the gauge transfromation at a single site
    M_tmp = similar(G_tmp)

    U_shifts = [shift_U(U[1], -1), shift_U(U[2], -2), shift_U(U[3], -3), shift_U(U[4], -4)]

    for nt in 1:NT
        for nx in 1:NX
            for  ny in 1:NY
                for nz in 1:NZ

                    fill!(G_tmp, 0.0+0.0im)
                    for ic in 1:NC
                        G_tmp[ic, ic] = 1.0
                    end
                    
                    # Do only even or odd parity site each time
                    parity_check =  (nt+nx+ny+nz) % 2
                    if parity_check == parity 
                        
                        fill!(w_x, 0.0+0.0im)

                        # Compute w = U_μ(x) + U_μ^†( x - μ)
                        for μ in 1:D_fix
                            for ic in 1:NC
                                @simd for jc in 1:NC
                                    w_x[ic, jc] += getvalue(U[μ], ic, jc, nx, ny, nz, nt) + getvalue(U_shifts[μ]', ic, jc, nx, ny, nz, nt)
                                end
                            end
                        end

                        # Reunitarisation process
                        SU2_group_hit!(G_tmp, w_x, 1, overrelax, M_tmp, ovr_coeff2, ovr_coeff3) # Do we need cooling?
                    end

                    #gramschmidt!(G_ovr)
                    for ic in 1:NC
                        @simd for jc in 1:NC
                            setvalue!(g, G_tmp[ic, jc], ic, jc, nx, ny, nz, nt)
                        end
                    end

                end
            end
        end
    end
    barrier(g)
    set_wing_U!(g)
end


function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_wing_mpi,
    Δ::Gaugefields_4D_wing_mpi,
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
    ) where {T<:Gaugefields_4D_wing_mpi}
    
    get_Δ!(Δ, U, temps[1:4], D_fix)

    α = temps[5]
    compute_α!(α, U, Δ, parity, D_fix)
    one = Matrix{ComplexF64}(I, g.NC, g.NC)

    NT = g.PN[4]
    NX = g.PN[1]
    NY = g.PN[2]
    NZ = g.PN[3]
    NC = g.NC

    for nt in 1:NT
        for nx in 1:NX
            for  ny in 1:NY
                for nz in 1:NZ
                    
                    for ic in 1:NC
                        @simd for jc in 1:NC
                            setvalue!(g, 0, ic, jc, nx, ny, nz, nt)
                        end
                        setvalue!(g, 1, ic, ic, nx, ny, nz, nt)
                    end

                    
                    # Do only even or odd parity site each time
                    parity_check =  (nt+nx+ny+nz) % 2
                    if parity_check == parity
                        alpha = getvalue(α, 1, 1, nx, ny, nz, nt)
                        for ic in 1:NC
                            @simd for jc in 1:NC

                                v =  one[ic, jc] + overrelax * alpha  * getvalue(Δ, ic, jc, nx, ny, nz, nt)
                                setvalue!(g, v, ic, jc, nx, ny, nz, nt)
                            end
                        end
                    end
                end
            end
        end
    end
    Gaugefields.AbstractGaugefields_module.normalize_U!(g)
    set_wing_U!(g)
end


function compute_α!(α::Gaugefields_4D_wing_mpi, U::Array{T,1}, Δ::Gaugefields_4D_wing_mpi, parity, D_fix) where {T<:Gaugefields_4D_wing_mpi}

    U_shifts = [ shift_U(U[1], -1), shift_U(U[2], -2), shift_U(U[3], -3), shift_U(U[4], -4)]

    NC = α.NC
    NT = α.PN[4]
    NX = α.PN[1]
    NY = α.PN[2]
    NZ = α.PN[3]

    Gaugefields.clear_U!(α)

    Δm = Matrix{ComplexF64}(I, NC, NC)
    Um = Matrix{ComplexF64}(I, NC, NC)
    Usm = Matrix{ComplexF64}(I, NC, NC)

    
    for nt in 1:NT
        for nx in 1:NX
            for  ny in 1:NY
                for nz in 1:NZ
                    # Do only even or odd parity site each time
                    parity_check =  (nt+nx+ny+nz) % 2

                    if parity_check == parity
                        
                        fill!(Δm, 0.0+0.0im)
                        fill!(Um, 0.0+0.0im)
                        fill!(Usm, 0.0+0.0im)
                            
                        for ic in 1:NC
                            for jc in 1:NC
                                    
                                Δm[ic, jc] = getvalue(Δ, ic, jc, nx, ny, nz, nt)
                                @simd for μ in 1:D_fix
                                    Um[ic, jc] += getvalue(U[μ], ic, jc, nx, ny, nz, nt)
                                    Usm[ic, jc] += getvalue(U_shifts[μ], ic, jc, nx, ny, nz, nt)
                                end
                            end
                        end

                        v = real( tr( Δm * (Usm - Um) )) / real( tr( Δm * Δm * (Usm + Um) ) )
                        setvalue!(α, v, 1, 1, nx, ny, nz, nt)
                    end
                end
            end
        end
    end
    set_wing_U!(α)
end
