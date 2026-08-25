# This adapter is loaded lazily after MPI.jl defines the deprecated winged
# MPI gauge-field type.
using ..AbstractGaugefields_module: Gaugefields_4D_wing_mpi

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

function make_g_los_alamos!(
    U::Array{T,1},
    g::Gaugefields_4D_wing_mpi{NC},
    temp::Gaugefields_4D_wing_mpi{NC},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
) where {NC,T<:Gaugefields_4D_wing_mpi}
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
    g::Gaugefields_4D_wing_mpi{NC},
    Δ::Gaugefields_4D_wing_mpi{NC},
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
) where {NC,T<:Gaugefields_4D_wing_mpi}
    return _make_g_steepest_descent_legacy_mpi!(
        U, g, Δ, parity, overrelax, temps, D_fix, Val(NC),
    )
end
