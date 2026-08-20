# for Gaugefields_4D_nowing_mpi

function make_g_transform!(
    U::Array{T,1},
    g::Gaugefields_4D_nowing_mpi{NC},
    temp::Gaugefields_4D_nowing_mpi{NC},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
    ) where {NC, T<:Gaugefields_4D_nowing_mpi}

    
    W = temp
    clear_U!(W)

    #Compute W = U_μ(x) + U_μ^†( x - μ)
    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        add_U!(W, U[μ])
        add_U!(W, U_shift')
    end

    NT = g.PN[4]
    NX = g.PN[1]
    NY = g.PN[2]
    NZ = g.PN[3]

    w_x = @MMatrix zeros(ComplexF64, NC, NC)
    G_tmp = @MMatrix zeros(ComplexF64, NC, NC) # the gauge transfromation at a single site
    M_tmp = @MMatrix zeros(ComplexF64, NC, NC)

    su2_tmp = @MMatrix ones(ComplexF64, 2, 2)
    A_tmp =  @MMatrix ones(ComplexF64, NC, NC)


    @inbounds for nt in 1:NT
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
                        
                            for ic in 1:NC
                                @simd for jc in 1:NC
                                    w_x[ic, jc] = getvalue(W, ic, jc, nx, ny, nz, nt) 
                                end
                            end

                        # Reunitarisation process
                        SU2_group_hit!(G_tmp, w_x, 1, overrelax, M_tmp, ovr_coeff2, ovr_coeff3, su2_tmp, A_tmp, NC)# Do we need cooling?
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
end

@inline function trace_AB(A, B, NC)
    s = zero(eltype(A))
    @inbounds for i in 1:NC
        for k in 1:NC
            s += A[i,k] * B[k,i]
        end
    end
    return s
end


function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_nowing_mpi{NC},
    Δ::Gaugefields_4D_nowing_mpi{NC},
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
    ) where {NC, T<:Gaugefields_4D_nowing_mpi}

    NT = g.PN[4]
    NX = g.PN[1]
    NY = g.PN[2]
    NZ = g.PN[3]

    
    get_Δ!(Δ, U, temps[1:4], D_fix)

    Um = temps[1]
    clear_U!(Um)
    
    Up = temps[2]
    clear_U!(Up)
    
    for μ in 1:D_fix
        U_shift =shift_U(U[μ], -μ)

        #Compute Um = U_μ(x) - U_μ( x - μ)
        add_U!(Um, U[μ])
        add_U!(Um, -1, U_shift)

        #Compute Up = U_μ(x) + U_μ( x - μ)
        add_U!(Up, U[μ])
        add_U!(Up, U_shift)
    end

    Um_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Up_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Δ_tmp  = @MMatrix zeros(ComplexF64, 3, 3)
    Δ2_tmp = @MMatrix zeros(ComplexF64, 3, 3)


   
    alpha = 0
    v = 0
    unit_U!(g)
    @inbounds  for nt in 1:NT
        for nx in 1:NX
            for  ny in 1:NY
                for nz in 1:NZ
                    
                    # Do only even or odd parity site each time
                    parity_check =  (nt+nx+ny+nz) & 1
                    if parity_check == parity

                        for ic in 1:NC
                            @simd for jc in 1:NC
                                Um_tmp[ic, jc] = getvalue(Um, ic, jc, nx, ny, nz, nt)
                                Up_tmp[ic, jc] = getvalue(Up, ic, jc, nx, ny, nz, nt)
                                Δ_tmp[ic, jc] = getvalue(Δ, ic, jc, nx, ny, nz, nt)
                            end
                        end
                        num = real(trace_AB(Δ_tmp, Um_tmp, NC))
                        den = real(trace_AB(Δ_tmp, mul!(Δ2_tmp, Δ_tmp, Up_tmp), NC))
                        alpha = -num / den


                        for ic in 1:NC
                            @simd for jc in 1:NC

                                v =  (ic == jc) + overrelax * alpha  * Δ_tmp[ic, jc]
                                setvalue!(g, v, ic, jc, nx, ny, nz, nt)
                            end
                        end
                    end
                end
            end
        end
    end
    normalize_U!(g)
end

