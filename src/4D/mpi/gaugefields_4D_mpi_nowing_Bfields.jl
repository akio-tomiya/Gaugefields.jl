

function thooftFlux_4D_B_at_bndry_nowing_mpi end
#=
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
=#
