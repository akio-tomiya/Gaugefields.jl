
#import Gaugefields.AbstractGaugefields_module:
#    Gaugefields_4D_wing_mpi, Gaugefields_4D_nowing_mpi
#import Gaugefields.AbstractGaugefields_module:
#    identityGaugefields_4D_wing_mpi,
#Gaugefields_4D_wing_mpi,
#    calc_rank_and_indices,
#    barrier,
#    setvalue!,
#    getvalue


function load_binarydata!(
    U::Array{T,1},
    NX,
    NY,
    NZ,
    NT,
    NC,
    filename,
    precision,
) where {T<:Gaugefields_4D_wing_mpi}
    if U[1].myrank == 0
        bi = Binarydata_ILDG(filename, precision)
    end

    data = zeros(ComplexF64, NC, NC, 4, prod(U[1].PN), U[1].nprocs)
    counts = zeros(Int64, U[1].nprocs)
    totalnum = NX * NY * NZ * NT * NC * NC * 2 * 4
    PN = U[1].PN
    barrier(U[1])

    N = NC * NC * 4
    send_mesg1 = Array{ComplexF64}(undef, 1)
    recv_mesg1 = Array{ComplexF64}(undef, 1)

    send_mesg = Array{ComplexF64}(undef, N)
    recv_mesg = Array{ComplexF64}(undef, N)

    #if U[1].myrank == 0
    i = 0
    counttotal = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    rank, ix_local, iy_local, iz_local, it_local =
                        calc_rank_and_indices(U[1], ix, iy, iz, it)
                    #counts[rank+1] += 1
                    counttotal += 1

                    #=
                    if U[1].myrank == 0
                        println("rank = $rank")
                        println("$ix $(ix_local)")
                        println("$iy $(iy_local)")
                        println("$iz $(iz_local)")
                        println("$it $(it_local)")
                    end
                    =#
                    barrier(U[1])
                    if U[1].myrank == 0
                        count = 0
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    count += 1
                                    send_mesg[count] = read!(bi)
                                end
                            end
                        end
                        sreq =
                            MPI.Isend(send_mesg, rank, counttotal, comm)
                    end
                    if U[1].myrank == rank
                        rreq =
                            MPI.Irecv!(recv_mesg, 0, counttotal, comm)
                        MPI.Wait!(rreq)
                        count = 0
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    count += 1
                                    v = recv_mesg[count]
                                    Gaugefields.setvalue!(
                                        U[μ],
                                        v,
                                        ic2,
                                        ic1,
                                        ix_local,
                                        iy_local,
                                        iz_local,
                                        it_local,
                                    )
                                end
                            end
                        end
                    end
                    barrier(U[1])
                end
            end
        end
    end
    #end

    barrier(U[1])
    #=

    N = length(data[:,:,:,:,1])
    send_mesg1 =  Array{ComplexF64}(undef, N)#data[:,:,:,:,1] #Array{ComplexF64}(undef, N)
    recv_mesg1 = Array{ComplexF64}(undef, N)
    #comm = MPI.MPI_COMM_WORLD
    #println(typeof(comm))


    for ip=0:U[1].nprocs-1
        if U[1].myrank == 0
            send_mesg1[:] = reshape(data[:,:,:,:,ip+1],:) #Array{ComplexF64}(undef, N)
            sreq1 = MPI.Isend(send_mesg1, ip, ip+32, comm) 
        end
        if U[1].myrank == ip
            rreq1 = MPI.Irecv!(recv_mesg1, 0, ip+32, comm)
            MPI.Wait!(rreq1)

            count = 0
            for it=1:PN[4]
                for iz=1:PN[3]
                    for iy=1:PN[2]
                        for ix=1:PN[1]
                            for μ=1:4
                                for ic1 = 1:NC
                                    for ic2 = 1:NC
                                        count += 1
                                        v = recv_mesg1[count] 
                                        Gaugefields.setvalue!(U[μ],v,ic2,ic1,ix,iy,iz,it) 
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

    end

    barrier(U[1])
    =#

    update!(U)


    #close(fp)
end

function load_binarydata!(
    U::Array{T,1},
    NX,
    NY,
    NZ,
    NT,
    NC,
    filename,
    precision,
) where {T<:Gaugefields_4D_nowing_mpi}
    if U[1].myrank == 0
        bi = Binarydata_ILDG(filename, precision)
    end

    data = zeros(ComplexF64, NC, NC, 4, prod(U[1].PN), U[1].nprocs)
    counts = zeros(Int64, U[1].nprocs)
    totalnum = NX * NY * NZ * NT * NC * NC * 2 * 4
    PN = U[1].PN
    barrier(U[1])

    N = NC * NC * 4
    send_mesg1 = Array{ComplexF64}(undef, 1)
    recv_mesg1 = Array{ComplexF64}(undef, 1)

    send_mesg = Array{ComplexF64}(undef, N)
    recv_mesg = Array{ComplexF64}(undef, N)

    #if U[1].myrank == 0
    i = 0
    counttotal = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    rank, ix_local, iy_local, iz_local, it_local =
                        calc_rank_and_indices(U[1], ix, iy, iz, it)
                    #counts[rank+1] += 1
                    counttotal += 1

                    #=
                    if U[1].myrank == 0
                        println("rank = $rank")
                        println("$ix $(ix_local)")
                        println("$iy $(iy_local)")
                        println("$iz $(iz_local)")
                        println("$it $(it_local)")
                    end
                    =#
                    barrier(U[1])
                    if U[1].myrank == 0
                        count = 0
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    count += 1
                                    send_mesg[count] = read!(bi)
                                end
                            end
                        end
                        sreq =
                            MPI.Isend(send_mesg, rank, counttotal, comm)
                    end
                    if U[1].myrank == rank
                        rreq =
                            MPI.Irecv!(recv_mesg, 0, counttotal, comm)
                        MPI.Wait!(rreq)
                        count = 0
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    count += 1
                                    v = recv_mesg[count]
                                    setvalue!(
                                        U[μ],
                                        v,
                                        ic2,
                                        ic1,
                                        ix_local,
                                        iy_local,
                                        iz_local,
                                        it_local,
                                    )
                                end
                            end
                        end
                    end
                    barrier(U[1])
                end
            end
        end
    end
    #end

    barrier(U[1])
    update!(U)


    #close(fp)
end

function save_binarydata(
    U::Array{T,1},
    filename; tempfile1="testbin.dat", tempfile2="filelist.dat"
) where {T<:Gaugefields_4D_nowing_mpi}

    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT
    NC = U[1].NC

    barrier(U[1])

    N = NC * NC * 4
    send_mesg1 = Array{ComplexF64}(undef, 1)
    recv_mesg1 = Array{ComplexF64}(undef, 1)

    send_mesg = Array{ComplexF64}(undef, N)
    recv_mesg = Array{ComplexF64}(undef, N)


    #li = LIME_header((NX,NY,NZ,NT),"su3gauge",1,64)
    #print(li.doc)
    #write("test.xml", li.doc)

    if U[1].myrank == 0
        #fp = open("testbin.dat", "w")
        fp = open(tempfile1, "w")
    end
    i = 0
    i = 0

    i = 0
    counttotal = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    rank, ix_local, iy_local, iz_local, it_local =
                        calc_rank_and_indices(U[1], ix, iy, iz, it)
                    #counts[rank+1] += 1
                    counttotal += 1

                    #=
                    if U[1].myrank == 0
                        println("rank = $rank")
                        println("$ix $(ix_local)")
                        println("$iy $(iy_local)")
                        println("$iz $(iz_local)")
                        println("$it $(it_local)")
                    end
                    =#
                    barrier(U[1])
                    if U[1].myrank == rank
                        count = 0
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    count += 1
                                    send_mesg[count] = getvalue(
                                        U[μ],
                                        ic2,
                                        ic1,
                                        ix_local,
                                        iy_local,
                                        iz_local,
                                        it_local,
                                    )
                                    #send_mesg[count] = U[μ][ic2,ic1,ix_local,iy_local,iz_local,it_local]
                                end
                            end
                        end
                        sreq = MPI.Isend(send_mesg, 0, counttotal, U[1].comm) ## HH: corrent sending rank
                    end
                    if U[1].myrank == 0
                        rreq = MPI.Irecv!(recv_mesg, rank, counttotal, U[1].comm) ## HH: corrent receiving rank
                        MPI.Wait!(rreq)
                        count = 0
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    count += 1
                                    v = recv_mesg[count]
                                    write(fp, hton(real(v)))
                                    write(fp, hton(imag(v)))
                                    #Gaugefields.setvalue!(U[μ],v,ic2,ic1,ix_local,iy_local,iz_local,it_local) 
                                end
                            end
                        end
                    end
                    barrier(U[1])
                end
            end
        end
    end

    if U[1].myrank == 0
        close(fp)

        #fp = open("filelist.dat", "w")
        fp = open(tempfile2, "w")
        #println(fp,"test.xml ","ildg-format")
        #println(fp, "testbin.dat ", "ildg-binary-data")
        println(fp, "$tempfile1 ", "ildg-binary-data")
        close(fp)

        lime_pack() do exe
            run(`$exe $tempfile2 $filename`)
            #run(`$exe filelist.dat $filename`)
        end

    end
    barrier(U[1])


    return

end