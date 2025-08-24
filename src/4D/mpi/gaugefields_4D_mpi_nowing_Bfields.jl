
#=function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
) where {T1<:Gaugefields_4D_nowing_mpi,T2<:Gaugefields_4D_nowing_mpi}
    for μ = 1:4
        for ν = 1:4
            if μ == ν
                continue
            end
            substitute_U!(a[μ, ν], b[μ, ν])
        end
    end
end

=#

#=
function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
    iseven,
) where {T1<:Gaugefields_4D_nowing_mpi,T2<:Gaugefields_4D_nowing_mpi}
    for μ = 1:4
        for ν = 1:4
            if μ == ν
                continue
            end
            substitute_U!(a[μ, ν], b[μ, ν], iseven)
        end
    end
end
=#

#=
function Base.similar(U::Array{T,2}) where {T<:Gaugefields_4D_nowing_mpi}
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

function thooftFlux_4D_B_at_bndry_nowing_mpi(
    NC,
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
            U = minusidentityGaugefields_4D_nowing_mpi(
                NC,
                NX,
                NY,
                NZ,
                NT,
                PEs,
                mpiinit=mpiinit,
                verbose_level=verbose_level,
                randomnumber=randomnumber,
                comm=comm,
            )
        else
            U = identityGaugefields_4D_nowing_mpi(
                NC,
                NX,
                NY,
                NZ,
                NT,
                PEs,
                mpiinit=mpiinit,
                verbose_level=verbose_level,
                randomnumber=randomnumber,
                comm=comm,
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


#=
## coordinates are not correct for MPI
function thooftLoop_4D_B_temporal_nowing_mpi(
    NC,
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
    tloop_pos=[1, 1, 1, 1],
    tloop_dir=[1, 4],
    tloop_dis=1,
)
    dim = 4
    if dim == 4
        if overallminus
            U = minusidentityGaugefields_4D_nowing_mpi(
                NC,
                NX,
                NY,
                NZ,
                NT,
                PEs,
                mpiinit=mpiinit,
                verbose_level=verbose_level,
                randomnumber=randomnumber,
                comm=comm,
            )
        else
            U = identityGaugefields_4D_nowing_mpi(
                NC,
                NX,
                NY,
                NZ,
                NT,
                PEs,
                mpiinit=mpiinit,
                verbose_level=verbose_level,
                randomnumber=randomnumber,
                comm=comm,
            )
        end

        spatial_dir = tloop_dir[1]
        temporal_dir = tloop_dir[2]

        ## This should depend on rank of MPI
        if tloop_dis > 0
            spatial_strpos = tloop_pos[spatial_dir]
            spatial_endpos = spatial_strpos + tloop_dis

            v = exp(-im * (2pi / NC) * FLUX)
        else
            spatial_endpos = tloop_pos[spatial_dir]
            spatial_strpos = spatial_endpos + tloop_dis

            v = exp(im * (2pi / NC) * FLUX)
        end

        if !overallminus
            v *= -1
        end

        if FLUXNUM == 1 && (tloop_dir == [3, 4] || tloop_dir == [4, 3])
            if spatial_dir == 3
                for it = 1:U.PN[4]
                    for iz = spatial_strpos:spatial_endpos
                        #for iy = 1:U.PN[2]
                        #for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, tloop_pos[1], tloop_pos[2], iz, it)
                        end
                        #end
                        #end
                    end
                end
            elseif spatial_dir == 4
                for it = spatial_strpos:spatial_endpos
                    for iz = 1:U.PN[3]
                        #for iy = 1:U.PN[2]
                        #for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, tloop_pos[1], tloop_pos[2], iz, it)
                        end
                        #end
                        #end
                    end
                end
            end
        elseif FLUXNUM == 2 && (tloop_dir == [2, 4] || tloop_dir == [4, 2])
            if spatial_dir == 2
                for it = 1:U.PN[4]
                    #for iz = 1:U.PN[3]
                    for iy = spatial_strpos:spatial_endpos
                        #for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, tloop_pos[1], iy, tloop_pos[3], it)
                        end
                        #end
                    end
                    #end
                end
            elseif spatial_dir == 4
                for it = spatial_strpos:spatial_endpos
                    #for iz = 1:U.PN[3]
                    for iy = 1:U.PN[2]
                        #for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, tloop_pos[1], iy, tloop_pos[3], it)
                        end
                        #end
                    end
                    #end
                end
            end
        elseif FLUXNUM == 3 && (tloop_dir == [2, 3] || tloop_dir == [3, 2])
            if spatial_dir == 2
                #for it = 1:U.PN[4]
                for iz = 1:U.PN[3]
                    for iy = spatial_strpos:spatial_endpos
                        #for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, tloop_pos[1], iy, iz, tloop_pos[4])
                        end
                        #end
                    end
                end
                #end
            elseif spatial_dir == 3
                #for it = 1:U.PN[4]
                for iz = spatial_strpos:spatial_endpos
                    for iy = 1:U.PN[2]
                        #for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, tloop_pos[1], iy, iz, tloop_pos[4])
                        end
                        #end
                    end
                end
                #end
            end
        elseif FLUXNUM == 4 && (tloop_dir == [1, 4] || tloop_dir == [4, 1])
            if spatial_dir == 1
                for it = 1:U.PN[4]
                    #for iz = 1:U.PN[3]
                    #for iy = 1:U.PN[2]
                    for ix = spatial_strpos:spatial_endpos
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, ix, tloop_pos[2], tloop_pos[3], it)
                        end
                    end
                    #end
                    #end
                end
            elseif spatial_dir == 4
                for it = spatial_strpos:spatial_endpos
                    #for iz = 1:U.PN[3]
                    #for iy = 1:U.PN[2]
                    for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, ix, tloop_pos[2], tloop_pos[3], it)
                        end
                    end
                    #end
                    #end
                end
            end
        elseif FLUXNUM == 5 && (tloop_dir == [1, 3] || tloop_dir == [3, 1])
            if spatial_dir == 1
                #for it = 1:U.PN[4]
                for iz = 1:U.PN[3]
                    #for iy = 1:U.PN[2]
                    for ix = spatial_strpos:spatial_endpos
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, ix, tloop_pos[2], iz, tloop_pos[4])
                        end
                    end
                    #end
                end
                #end
            elseif spatial_dir == 3
                #for it = 1:U.PN[4]
                for iz = spatial_strpos:spatial_endpos
                    #for iy = 1:U.PN[2]
                    for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, ix, tloop_pos[2], iz, tloop_pos[4])
                        end
                    end
                    #end
                end
                #end
            end
        elseif FLUXNUM == 6 && (tloop_dir == [1, 2] || tloop_dir == [2, 1])
            if spatial_dir == 1
                #for it = 1:U.PN[4]
                #for iz = 1:U.PN[3]
                for iy = 1:U.PN[2]
                    for ix = spatial_strpos:spatial_endpos
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, ix, iy, tloop_pos[3], tloop_pos[4])
                        end
                    end
                end
                #end
                #end
            elseif spatial_dir == 2
                #for it = 1:U.PN[4]
                #for iz = 1:U.PN[3]
                for iy = spatial_strpos:spatial_endpos
                    for ix = 1:U.PN[1]
                        @simd for ic = 1:NC
                            setvalue!(U, v, ic, ic, ix, iy, tloop_pos[3], tloop_pos[4])
                        end
                    end
                end
                #end
                #end
            end
        end
        set_wing_U!(U)
        return U
    end
end
=#
