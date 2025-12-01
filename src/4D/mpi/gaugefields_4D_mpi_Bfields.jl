#const comm = MPI.COMM_WORLD
#=
function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
) where {T1<:Gaugefields_4D_wing_mpi,T2<:Gaugefields_4D_wing_mpi}
    for μ = 1:4
        for ν = 1:4
            if μ == ν
                continue
            end
            substitute_U!(a[μ, ν], b[μ, ν])
        end
    end
end

function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
    iseven,
) where {T1<:Gaugefields_4D_wing_mpi,T2<:Gaugefields_4D_wing_mpi}
    for μ = 1:4
        for ν = 1:4
            if μ == ν
                continue
            end
            substitute_U!(a[μ, ν], b[μ, ν], iseven)
        end
    end
end

function Base.similar(U::Array{T,2}) where {T<:Gaugefields_4D_wing_mpi}
    Uout = Array{T,2}(undef, 4, 4)
    for μ = 1:4
        for ν = 1:4
            if μ == ν
                continue
            end
            Uout[μ, ν] = similar(U[μ, ν])
        end
    end
    return Uout
end
=#

function thooftFlux_4D_B_at_bndry_wing_mpi end
#=
function thooftFlux_4D_B_at_bndry_wing_mpi(
    NC,
    NDW,
    FLUX,
    FLUXNUM,
    NX,
    NY,
    NZ,
    NT,
    PEs;
    overallminus=false,
    mpiinit=true,
    verbose_level=2,
    randomnumber="Random",
    comm=MPI.COMM_WORLD,
)
    dim = 4
    if dim == 4
        if overallminus
            U = minusidentityGaugefields_4D_wing_mpi(
                NC,
                NX,
                NY,
                NZ,
                NT,
                NDW,
                PEs,
                mpiinit=mpiinit,
                verbose_level=verbose_level,
                randomnumber=randomnumber,
            )
        else
            U = identityGaugefields_4D_wing_mpi(
                NC,
                NX,
                NY,
                NZ,
                NT,
                NDW,
                PEs,
                mpiinit=mpiinit,
                verbose_level=verbose_level,
                randomnumber=randomnumber,
            )
        end

        if overallminus
            v = exp(-im * (2pi / NC) * FLUX)
        else
            v = -exp(-im * (2pi / NC) * FLUX)
        end
        if FLUXNUM == 1
            for it = 1:U.PN[4]
                for iz = 1:U.PN[3]
                    #for iy = 1:U.PN[2]
                    #for ix = 1:U.PN[1]
                    @simd for ic = 1:NC
                        setvalue!(U, v, ic, ic, NX, NY, iz, it)
                    end
                    #end
                    #end
                end
            end
        elseif FLUXNUM == 2
            for it = 1:U.PN[4]
                #for iz = 1:U.PN[3]
                for iy = 1:U.PN[2]
                    #for ix = 1:U.PN[1]
                    @simd for ic = 1:NC
                        setvalue!(U, v, ic, ic, NX, iy, NZ, it)
                    end
                    #end
                end
                #end
            end
        elseif FLUXNUM == 3
            #for it = 1:U.PN[4]
            for iz = 1:U.PN[3]
                for iy = 1:U.PN[2]
                    #for ix = 1:U.PN[1]
                    @simd for ic = 1:NC
                        setvalue!(U, v, ic, ic, NX, iy, iz, NT)
                    end
                    #end
                end
            end
            #end
        elseif FLUXNUM == 4
            for it = 1:U.PN[4]
                #for iz = 1:U.PN[3]
                #for iy = 1:U.PN[2]
                for ix = 1:U.PN[1]
                    @simd for ic = 1:NC
                        setvalue!(U, v, ic, ic, ix, NY, NZ, it)
                    end
                end
                #end
                #end
            end
        elseif FLUXNUM == 5
            #for it = 1:U.PN[4]
            for iz = 1:U.PN[3]
                #for iy = 1:U.PN[2]
                for ix = 1:U.PN[1]
                    @simd for ic = 1:NC
                        setvalue!(U, v, ic, ic, ix, NY, iz, NT)
                    end
                end
                #end
            end
            #end
        elseif FLUXNUM == 6
            #for it = 1:U.PN[4]
            #for iz = 1:U.PN[3]
            for iy = 1:U.PN[2]
                for ix = 1:U.PN[1]
                    @simd for ic = 1:NC
                        setvalue!(U, v, ic, ic, ix, iy, NZ, NT)
                    end
                end
            end
            #end
            #end
        end
        set_wing_U!(U)
        return U
    end
end
=#
