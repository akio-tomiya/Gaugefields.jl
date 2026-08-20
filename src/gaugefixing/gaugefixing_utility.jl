
function trace_U(U::Array{T,1}; D_fix::Int = 4) where {T<:AbstractGaugefields}

    trace = zero(typeof(tr(U[1])))
    NC = U[1].NC
    NV = U[1].NV

    for μ in 1:D_fix
        trace += tr(U[μ])
    end
    
    return trace / (NC * D_fix * NV)

end

function trace_U_Udag(U::Array{T,1}) where {T<:AbstractGaugefields}

    
    trace = zero(eltype(U[1].U))

    Dim = size(U)[1]

    NC = U[1].NC
    NV = U[1].NV

    for μ in 1:Dim
        Uminus_nu = shift_U(U[μ], -μ)      
        trace += (tr(U[μ]) + tr(Uminus_nu')) * 0.5

    end
    return trace / (NC * Dim *NV)

end

function trace_AAdagger(dA::AbstractGaugefields{NC,Dim}, temp::AbstractGaugefields{NC,Dim}; D_fix::Int = 4) where {NC,Dim}

    trace = zero(eltype(dA.U))
    
    mul!(temp, dA, dA')
    trace += tr(temp) 
    return real(trace / (NC * dA.NV*D_fix))
end


function gUgshift!(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
    ) where {NC,Dim,T<:AbstractGaugefields}
    
    for μ in 1:Dim
        g_shift = shift_U(g, μ) # g_shift(n) = g(n+μ)
        mul!(temp, g, U[μ]) # temp = g(n) * U[μ](n)
        mul!(U[μ], temp, g_shift') # U_transform[μ] = g(n) * U[μ](n) * g(n+μ)†
    end
    set_wing_U!(U)
end


function gUgshift!(
    Uout::Array{T,1},
    Uin::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
    ) where {NC,Dim,T<:AbstractGaugefields}
    
    for μ in 1:Dim
        g_shift = shift_U(g, μ) # g_shift(n) = g(n+μ)
        mul!(temp, g, Uin[μ]) # temp = g(n) * U[μ](n)
        mul!(Uout[μ], temp, g_shift') # U_transform[μ] = g(n) * U[μ](n) * g(n+μ)†
    end
    set_wing_U!(Uout)
end


function g_dagger!(gd::AbstractGaugefields{NC, Dim}, g::AbstractGaugefields{NC, Dim}) where {NC, Dim}

    clear_U!(gd)
    add_U!(gd, 1, g')
    set_wing_U!(gd)
end


function shift_g!(gs::AbstractGaugefields{NC, Dim}, g::AbstractGaugefields{NC, Dim}, μ::Int) where {NC, Dim}
    g_shift = shift_U(g, μ)

    clear_U!(gs)
    add_U!(gs, 1, g_shift)
    
    set_wing_U!(gs)
    return gs
end


function make_g_transform!(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
    ) where {NC,Dim,T<:AbstractGaugefields}

    w_x = zeros(ComplexF64, NC, NC)
    G_tmp = Matrix{ComplexF64}(I, NC, NC) # the gauge transfromation at a single site
    M_tmp = similar(G_tmp)
    
    for nt in 1:g.NT
        for nx in 1:g.NX
            for  ny in 1:g.NY
                for nz in 1:g.NZ
                    # run over single site

                    fill!(G_tmp, 0.0+0.0im)
                    for ic in 1:NC
                        G_tmp[ic, ic] = 1.0
                    end

                    # Do only even or odd parity site each time
                    parity_check =  (nt+nx+ny+nz) % 2
                    if parity_check == parity 
                        
                        fill!(w_x, 0.0+0.0im)

                        # w = U_μ(x) + U_μ^†( x - ̂μ)
                        for μ in 1:D_fix
                            
                            U_shift = shift_U(U[μ], -μ)
                            w_x +=  U[μ][:,:, nx, ny, nz, nt] +  U_shift[:,:, nx, ny, nz, nt]'
                            #w[:,:, nx, ny, nz, nt] +=  U[μ][:,:, nx, ny, nz, nt] +  U_shift[μ][:,:, nx, ny, nz, nt]' # w = U_μ(x) + U_μ^†( x - ̂μ)

                        end

                        # Reunitarisation process
                        cooling = 1
                        G_tmp = SU2_group_hit(w_x, cooling, overrelax, M_tmp) # Do we need cooling?                       
                    end
                    
                    g[:,:, nx, ny, nz, nt] = G_tmp # assigning to the g_transform
                end
            end
        end
    end
    set_wing_U!(g)
end


# Δ = [U - U_shift]_TA
function get_Δ!(Δ::AbstractGaugefields{NC,Dim}, U::Array{T,1}, temps, D_fix) where {NC, Dim, T<:AbstractGaugefields{NC,Dim}}

    temp1 = temps[1]

    clear_U!(Δ)
    clear_U!(temp1)

    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        add_U!(temp1, 1, U[μ])
        add_U!(temp1, -1, U_shift)
    end

    Traceless_antihermitian!(Δ, temp1)
    set_wing_U!(Δ)
end
