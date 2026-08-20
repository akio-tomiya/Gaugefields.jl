function trace_U(U::Array{T,1}; D_fix::Int=4) where {T<:AbstractGaugefields}
    trace = zero(typeof(tr(U[1])))
    NC = U[1].NC
    NV = U[1].NV
    for μ in 1:D_fix
        trace += tr(U[μ])
    end
    return trace / (NC * D_fix * NV)
end

function trace_AAdagger(
    dA::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim};
    D_fix::Int=4,
) where {NC,Dim}
    mul!(temp, dA, dA')
    return real(tr(temp) / (NC * dA.NV * D_fix))
end

function gUgshift!(
    U::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    for μ in 1:Dim
        g_shift = shift_U(g, μ)
        mul!(temp, g, U[μ])
        mul!(U[μ], temp, g_shift')
    end
    set_wing_U!(U)
    return nothing
end

function gUgshift!(
    Uout::Array{T,1},
    Uin::Array{T,1},
    g::AbstractGaugefields{NC,Dim},
    temp::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    for μ in 1:Dim
        g_shift = shift_U(g, μ)
        mul!(temp, g, Uin[μ])
        mul!(Uout[μ], temp, g_shift')
    end
    set_wing_U!(Uout)
    return nothing
end

function g_dagger!(
    gd::AbstractGaugefields{NC,Dim},
    g::AbstractGaugefields{NC,Dim},
) where {NC,Dim}
    clear_U!(gd)
    add_U!(gd, 1, g')
    set_wing_U!(gd)
    return nothing
end

function shift_g!(
    gs::AbstractGaugefields{NC,Dim},
    g::AbstractGaugefields{NC,Dim},
    μ::Int,
) where {NC,Dim}
    g_shift = shift_U(g, μ)
    clear_U!(gs)
    add_U!(gs, 1, g_shift)
    set_wing_U!(gs)
    return gs
end

# Δ = [Σμ (Uμ(x) - Uμ(x-μ))]_TA
function get_Δ!(
    Δ::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    temps,
    D_fix,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
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
    return nothing
end
