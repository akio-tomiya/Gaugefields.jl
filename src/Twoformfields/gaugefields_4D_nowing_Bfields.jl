function thooftFlux_4D_B_at_bndry(
    NC,
    FLUX,
    FLUXNUM,
    NN...;
    overallminus=false,
    verbose_level=2,
)
    dim = length(NN)
    if dim == 4
        if overallminus
            U = minusidentityGaugefields_4D_nowing(
                NC,
                NN[1],
                NN[2],
                NN[3],
                NN[4],
                verbose_level=verbose_level,
            )
        else
            U = identityGaugefields_4D_nowing(
                NC,
                NN[1],
                NN[2],
                NN[3],
                NN[4],
                verbose_level=verbose_level,
            )
        end

        v = exp(-im * (2pi / NC) * FLUX)
        if FLUXNUM == 1
            for it = 1:NN[4]
                for iz = 1:NN[3]
                    #for iy = 1:NN[2]
                    #for ix = 1:NN[1]
                    @simd for ic = 1:NC
                        U[ic, ic, NN[1], NN[2], iz, it] *= v
                    end
                    #end
                    #end
                end
            end
        elseif FLUXNUM == 2
            for it = 1:NN[4]
                #for iz = 1:NN[3]
                for iy = 1:NN[2]
                    #for ix = 1:NN[1]
                    @simd for ic = 1:NC
                        U[ic, ic, NN[1], iy, NN[3], it] *= v
                    end
                    #end
                end
                #end
            end
        elseif FLUXNUM == 3
            #for it = 1:NN[4]
            for iz = 1:NN[3]
                for iy = 1:NN[2]
                    #for ix = 1:NN[1]
                    @simd for ic = 1:NC
                        U[ic, ic, NN[1], iy, iz, NN[4]] *= v
                    end
                    #end
                end
            end
            #end
        elseif FLUXNUM == 4
            for it = 1:NN[4]
                #for iz = 1:NN[3]
                #for iy = 1:NN[2]
                for ix = 1:NN[1]
                    @simd for ic = 1:NC
                        U[ic, ic, ix, NN[2], NN[3], it] *= v
                    end
                end
                #end
                #end
            end
        elseif FLUXNUM == 5
            #for it = 1:NN[4]
            for iz = 1:NN[3]
                #for iy = 1:NN[2]
                for ix = 1:NN[1]
                    @simd for ic = 1:NC
                        U[ic, ic, ix, NN[2], iz, NN[4]] *= v
                    end
                end
                #end
            end
            #end
        elseif FLUXNUM == 6
            #for it = 1:NN[4]
            #for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    @simd for ic = 1:NC
                        U[ic, ic, ix, iy, NN[3], NN[4]] *= v
                    end
                end
            end
            #end
            #end
        else
            error("NumofFlux is out")
        end
    end
    set_wing_U!(U)
    return U
end

function thooftLoop_4D_B_temporal(
    NC,
    FLUX,
    FLUXNUM,
    NN...;
    overallminus=false,
    verbose_level=2,
    tloop_pos=[1, 1, 1, 1],
    tloop_dir=[1, 4],
    tloop_dis=1,
)
    dim = length(NN)
    if dim == 4
        if overallminus
            U = minusidentityGaugefields_4D_nowing(
                NC,
                NN[1],
                NN[2],
                NN[3],
                NN[4],
                verbose_level=verbose_level,
            )
        else
            U = identityGaugefields_4D_nowing(
                NC,
                NN[1],
                NN[2],
                NN[3],
                NN[4],
                verbose_level=verbose_level,
            )
        end

        spatial_dir = tloop_dir[1]
        temporal_dir = tloop_dir[2]

        if tloop_dis > 0
            spatial_strpos = tloop_pos[spatial_dir]
            spatial_endpos = spatial_strpos + tloop_dis

            v = exp(-im * (2pi / NC) * FLUX)
        else
            spatial_endpos = tloop_pos[spatial_dir]
            spatial_strpos = spatial_endpos + tloop_dis

            v = exp(im * (2pi / NC) * FLUX)
        end

        if FLUXNUM == 1 && (tloop_dir == [3, 4] || tloop_dir == [4, 3])
            if spatial_dir == 3
                for it = 1:NN[4]
                    for iz = spatial_strpos:spatial_endpos
                        #for iy = 1:NN[2]
                        #for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, tloop_pos[1], tloop_pos[2], iz, it] *= v
                        end
                        #end
                        #end
                    end
                end
            elseif spatial_dir == 4
                for it = spatial_strpos:spatial_endpos
                    for iz = 1:NN[3]
                        #for iy = 1:NN[2]
                        #for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, tloop_pos[1], tloop_pos[2], iz, it] *= v
                        end
                        #end
                        #end
                    end
                end
            end
        elseif FLUXNUM == 2 && (tloop_dir == [2, 4] || tloop_dir == [4, 2])
            if spatial_dir == 2
                for it = 1:NN[4]
                    #for iz = 1:NN[3]
                    for iy = spatial_strpos:spatial_endpos
                        #for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, tloop_pos[1], iy, tloop_pos[3], it] *= v
                        end
                        #end
                    end
                    #end
                end
            elseif spatial_dir == 4
                for it = spatial_strpos:spatial_endpos
                    #for iz = 1:NN[3]
                    for iy = 1:NN[2]
                        #for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, tloop_pos[1], iy, tloop_pos[3], it] *= v
                        end
                        #end
                    end
                    #end
                end
            end
        elseif FLUXNUM == 3 && (tloop_dir == [2, 3] || tloop_dir == [3, 2])
            if spatial_dir == 2
                #for it = 1:NN[4]
                for iz = 1:NN[3]
                    for iy = spatial_strpos:spatial_endpos
                        #for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, tloop_pos[1], iy, iz, tloop_pos[4]] *= v
                        end
                        #end
                    end
                end
                #end
            elseif spatial_dir == 3
                #for it = 1:NN[4]
                for iz = spatial_strpos:spatial_endpos
                    for iy = 1:NN[2]
                        #for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, tloop_pos[1], iy, iz, tloop_pos[4]] *= v
                        end
                        #end
                    end
                end
                #end
            end
        elseif FLUXNUM == 4 && (tloop_dir == [1, 4] || tloop_dir == [4, 1])
            if spatial_dir == 1
                for it = 1:NN[4]
                    #for iz = 1:NN[3]
                    #for iy = 1:NN[2]
                    for ix = spatial_strpos:spatial_endpos
                        @simd for ic = 1:NC
                            U[ic, ic, ix, tloop_pos[2], tloop_pos[3], it] *= v
                        end
                    end
                    #end
                    #end
                end
            elseif spatial_dir == 4
                for it = spatial_strpos:spatial_endpos
                    #for iz = 1:NN[3]
                    #for iy = 1:NN[2]
                    for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, ix, tloop_pos[2], tloop_pos[3], it] *= v
                        end
                    end
                    #end
                    #end
                end
            end
        elseif FLUXNUM == 5 && (tloop_dir == [1, 3] || tloop_dir == [3, 1])
            if spatial_dir == 1
                #for it = 1:NN[4]
                for iz = 1:NN[3]
                    #for iy = 1:NN[2]
                    for ix = spatial_strpos:spatial_endpos
                        @simd for ic = 1:NC
                            U[ic, ic, ix, tloop_pos[2], iz, tloop_pos[4]] *= v
                        end
                    end
                    #end
                end
                #end
            elseif spatial_dir == 3
                #for it = 1:NN[4]
                for iz = spatial_strpos:spatial_endpos
                    #for iy = 1:NN[2]
                    for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, ix, tloop_pos[2], iz, tloop_pos[4]] *= v
                        end
                    end
                    #end
                end
                #end
            end
        elseif FLUXNUM == 6 && (tloop_dir == [1, 2] || tloop_dir == [2, 1])
            if spatial_dir == 1
                #for it = 1:NN[4]
                #for iz = 1:NN[3]
                for iy = 1:NN[2]
                    for ix = spatial_strpos:spatial_endpos
                        @simd for ic = 1:NC
                            U[ic, ic, ix, iy, tloop_pos[3], tloop_pos[4]] *= v
                        end
                    end
                end
                #end
                #end
            elseif spatial_dir == 2
                #for it = 1:NN[4]
                #for iz = 1:NN[3]
                for iy = spatial_strpos:spatial_endpos
                    for ix = 1:NN[1]
                        @simd for ic = 1:NC
                            U[ic, ic, ix, iy, tloop_pos[3], tloop_pos[4]] *= v
                        end
                    end
                end
                #end
                #end
            end
            #else
            #    error("NumofFlux is out")
        end
    end
    set_wing_U!(U)
    return U
end

