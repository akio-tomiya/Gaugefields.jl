module Bridge_format
using Requires

import ..AbstractGaugefields_module: set_wing_U!
#=
Bridge++ Text file format
U(x,y,z,t;mu)_{ab} is like 
(outer) [mu][t][z][y][x][a][b][re/im] (inner)
re/im; 
  0: Real
  1: Imaginary
mu: (x,y,z,t)=(0,1,2,3)

Re of U(0,0,0,0;mu=0)_00  # mu=0, site (x,y,z,t)=(0,0,0,0)
Im of U(0,0,0,0;mu=)_00
Re of U(0,0,0,0;mu=0)_01
Im of U(0,0,0,0;mu=0)_01
...
Re of U(1,0,0,0;0)_00     # mu=0, site (x,y,z,t)=(1,0,0,0)
Im of U(1,0,0,0;0)_00
...
Re of U(0,0,0,0;mu=1)_00  # mu=1, site (x,y,z,t)=(0,0,0,0)
Im of U(0,0,0,0;mu=1)_00

=#
import ..AbstractGaugefields_module: AbstractGaugefields, set_wing_U!
update!(U) = set_wing!(U)
update!(U::Array{T,1}) where {T<:AbstractGaugefields} = set_wing_U!(U)

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin
        import ..AbstractGaugefields_module:
            Gaugefields_4D_wing_mpi, Gaugefields_4D_nowing_mpi
        import ..AbstractGaugefields_module:
            identityGaugefields_4D_wing_mpi,
            Gaugefields_4D_wing_mpi,
            calc_rank_and_indices,
            barrier,
            comm,
            setvalue!,
            getvalue

        function load_BridgeText!(initial, U::Array{T,1}, L, NC) where {T<:Gaugefields_4D_nowing_mpi}
            NX = L[1]
            NY = L[2]
            NZ = L[3]
            NT = L[4]
            @assert U[1].NX == NX "NX mismatch"
            @assert U[1].NY == NY "NY mismatch"
            @assert U[1].NZ == NZ "NZ mismatch"
            @assert U[1].NT == NT "NT mismatch"
            @assert U[1].NC == NC "NC mismatch"
            fp = open(initial, "r")
            numdata = countlines(initial)
            @assert numdata == 4 * NX * NY * NT * NZ * NC * NC * 2 "data shape is wrong"

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
                                    for a = 1:NC
                                        for b = 1:NC
                                            count += 1
                                            u = split(readline(fp))
                                            #println(u)
                                            rvalue = parse(Float64, u[1])
                                            u = split(readline(fp))
                                            ivalue = parse(Float64, u[1])
                                            send_mesg[count] = rvalue + im * ivalue
                                            #U[μ][b,a,ix,iy,iz,it] = rvalue + im*ivalue
                                        end
                                    end
                                    #u = U[μ][:,:,ix,iy,iz,it] 
                                    #println(u'*u)
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
                                    for a = 1:NC
                                        for b = 1:NC
                                            count += 1
                                            v = recv_mesg[count]
                                            #U[μ][a, b, ix, iy, iz, it]
                                            setvalue!(
                                                U[μ],
                                                v,
                                                a,
                                                b,
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

        end

        function save_textdata(U::Array{T,1}, filename) where {T<:Gaugefields_4D_nowing_mpi}
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
                fp = open(filename, "w")
            end


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

                            barrier(U[1])
                            if U[1].myrank == rank
                                count = 0
                                for μ = 1:4
                                    for a = 1:NC
                                        for b = 1:NC
                                            count += 1
                                            send_mesg[count] = getvalue(
                                                U[μ],
                                                a,
                                                b,
                                                ix_local,
                                                iy_local,
                                                iz_local,
                                                it_local,
                                            )
                                            #send_mesg[count] = U[μ][ic2,ic1,ix_local,iy_local,iz_local,it_local]
                                        end
                                    end
                                end
                                sreq = MPI.Isend(send_mesg, rank, counttotal, U[1].comm)
                            end
                            if U[1].myrank == 0
                                rreq = MPI.Irecv!(recv_mesg, 0, counttotal, U[1].comm)
                                MPI.Wait!(rreq)
                                count = 0
                                for μ = 1:4
                                    for a = 1:NC
                                        for b = 1:NC
                                            count += 1
                                            v = recv_mesg[count]
                                            rvalue = real(v)
                                            println(fp, rvalue)
                                            ivalue = imag(v)
                                            println(fp, ivalue)

                                            #write(fp, hton(real(v)))
                                            #write(fp, hton(imag(v)))
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
            end
            barrier(U[1])



        end
    end
end

function load_BridgeText!(initial, U, L, NC)
    NX = L[1]
    NY = L[2]
    NZ = L[3]
    NT = L[4]
    @assert U[1].NX == NX "NX mismatch"
    @assert U[1].NY == NY "NY mismatch"
    @assert U[1].NZ == NZ "NZ mismatch"
    @assert U[1].NT == NT "NT mismatch"
    @assert U[1].NC == NC "NC mismatch"
    fp = open(initial, "r")
    numdata = countlines(initial)
    @assert numdata == 4 * NX * NY * NT * NZ * NC * NC * 2 "data shape is wrong"


    #for μ=1:4
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        for a = 1:NC
                            for b = 1:NC
                                u = split(readline(fp))
                                #println(u)
                                rvalue = parse(Float64, u[1])
                                u = split(readline(fp))
                                ivalue = parse(Float64, u[1])
                                U[μ][a, b, ix, iy, iz, it] = rvalue + im * ivalue
                                #U[μ][b,a,ix,iy,iz,it] = rvalue + im*ivalue
                            end
                        end
                        #u = U[μ][:,:,ix,iy,iz,it] 
                        #println(u'*u)
                    end
                end
            end
        end
    end


    set_wing_U!(U)


    close(fp)


end

function save_textdata(U, filename)

    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT
    NC = U[1].NC


    #li = LIME_header((NX,NY,NZ,NT),"su3gauge",1,64)
    #print(li.doc)
    #write("test.xml", li.doc)


    fp = open(filename, "w")
    #fp = open("testbin.dat","w")
    i = 0
    i = 0
    #for μ=1:4
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        for a = 1:NC
                            for b = 1:NC
                                i += 1
                                #rvalue = read(fp, floattype)
                                rvalue = real(U[μ][a, b, ix, iy, iz, it])
                                println(fp, rvalue)
                                ivalue = imag(U[μ][a, b, ix, iy, iz, it])
                                println(fp, ivalue)
                            end
                        end
                    end
                end
            end
        end
    end
    close(fp)




    return

end

end
