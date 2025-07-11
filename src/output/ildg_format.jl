
module ILDG_format
using CLIME_jll
using EzXML
using Requires

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
    end
end
#using MPI


import ..IOmodule: IOFormat
import ..AbstractGaugefields_module: AbstractGaugefields, set_wing_U!
#import ..Gaugefields:GaugeFields,SU3GaugeFields,SU2GaugeFields,set_wing!,AbstractGaugefields,set_wing_U!
#import ..Gaugefields


struct LIME_header
    doc::EzXML.Document
    function LIME_header(L, field, version, precision)
        doc = XMLDocument()
        elm = ElementNode("ildgFormat")
        addelement!(elm, "version", "$version")
        addelement!(elm, "field", "$field")
        addelement!(elm, "precision", "$precision")
        addelement!(elm, "lx", "$(L[1])")
        addelement!(elm, "ly", "$(L[2])")
        addelement!(elm, "lz", "$(L[3])")
        addelement!(elm, "lt", "$(L[4])")
        setroot!(doc, elm)
        return new(doc)
    end
end


struct ILDG <: IOFormat
    header::Array{Dict,1}
    filename::String
    ILDG(filename) = new(read_header(filename), filename)
end

function Base.length(ildg::ILDG)
    return length(ildg.header)
end

function Base.getindex(ildg::ILDG, i)
    return ildg.header[i]
end

function save_binarydata(U, filename; tempfile1="testbin.dat", tempfile2="filelist.dat")

    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT
    NC = U[1].NC


    #li = LIME_header((NX,NY,NZ,NT),"su3gauge",1,64)
    #print(li.doc)
    #write("test.xml", li.doc)


    #fp = open("testbin.dat", "w")
    fp = open(tempfile1, "w")
    i = 0
    i = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        for ic2 = 1:NC
                            for ic1 = 1:NC
                                i += 1
                                #rvalue = read(fp, floattype)
                                rvalue = real(U[μ][ic2, ic1, ix, iy, iz, it])
                                ivalue = imag(U[μ][ic2, ic1, ix, iy, iz, it])
                                write(fp, hton(rvalue))
                                write(fp, hton(ivalue))
                            end
                        end
                    end
                end
            end
        end
    end
    close(fp)

    #fp = open("filelist.dat", "w")
    fp = open(tempfile2, "w")
    #println(fp,"test.xml ","ildg-format")
    #println(fp, "testbin.dat ", "ildg-binary-data")
    println(fp, "$tempfile1 ", "ildg-binary-data")
    close(fp)

    lime_pack() do exe
        #run(`$exe filelist.dat $filename`)
        run(`$exe $tempfile2 $filename`)
    end


    return

end

update!(U) = set_wing!(U)
update!(U::Array{T,1}) where {T<:AbstractGaugefields} = set_wing_U!(U)

mutable struct Binarydata_ILDG
    fp::IOStream
    count::Int64
    floattype::DataType
    function Binarydata_ILDG(filename, precision)
        if precision == 32
            floattype = Float32
        else
            floattype = Float64
        end
        fp = open(filename, "r")
        count = 0

        bi = new(fp, count, floattype)

        finalizer(bi) do bi
            close(bi.fp)
        end

        return bi
    end
end

function read!(x::Binarydata_ILDG)
    x.count += 1
    rvalue = ntoh(read(x.fp, x.floattype))
    ivalue = ntoh(read(x.fp, x.floattype))
    return rvalue + im * ivalue
end

function load_binarydata!(U, NX, NY, NZ, NT, NC, filename, precision)
    bi = Binarydata_ILDG(filename, precision)

    totalnum = NX * NY * NZ * NT * NC * NC * 2 * 4

    i = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        for ic2 = 1:NC
                            for ic1 = 1:NC
                                U[μ][ic2, ic1, ix, iy, iz, it] = read!(bi)
                            end
                        end
                    end
                end
            end
        end
    end

    update!(U)

    #close(fp)
end

function load_binarydata!(U, filename)
    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT
    NC = U[1].NC
    NDW = U[1].NDW
    ildg = ILDG(filename)
    i = 1
    L = [NX, NY, NZ, NT]
    load_gaugefield!(U, i, ildg, L, NC, NDW=NDW)
end

function load_gaugefield!(U, i, ildg::ILDG, L, NC; NDW=0)
    NX = L[1]
    NY = L[2]
    NZ = L[3]
    NT = L[4]
    data = ildg[i]
    filename = ildg.filename

    @assert U[1].NX == NX "NX mismatch"
    @assert U[1].NY == NY "NY mismatch"
    @assert U[1].NZ == NZ "NZ mismatch"
    @assert U[1].NT == NT "NT mismatch U[1].NT=$(U[1].NT) but NT = $NT"
    @assert U[1].NC == NC "NC mismatch"

    message_no = data["message_no"]
    reccord_no = data["reccord_no"]
    if haskey(data, "precision")
        precision = data["precision"]
    else
        precision = 64
    end


    lime_extract_record() do exe
        run(`$exe $filename $message_no $reccord_no tempconf.dat`)
    end

    load_binarydata!(U, NX, NY, NZ, NT, NC, "tempconf.dat", precision)

    return
end

import ..AbstractGaugefields_module: Initialize_Gaugefields

function load_gaugefield(i, ildg::ILDG, L, NC; NDW=0)
    NX = L[1]
    NY = L[2]
    NZ = L[3]
    NT = L[4]
    data = ildg[i]
    filename = ildg.filename

    U = Initialize_Gaugefields(NC, NDW, NX, NY, NZ, NT, condition="cold")
    #=
    if NC == 3
        U = Array{SU3GaugeFields,1}(undef, 4)
    elseif NC == 2
        U = Array{SU2GaugeFields,1}(undef, 4)
    end

    for μ = 1:4
        U[μ] = GaugeFields(NC, NDW, NX, NY, NZ, NT)
    end
    =#

    load_gaugefield!(U, i, ildg::ILDG, L, NC; NDW)
    return U

end

function load_gaugefield(i, ildg::ILDG; NDW=0)
    #@assert length(ildg) != 0 "the header file is not found"
    data = ildg[i]
    filename = ildg.filename
    if haskey(data, "L")
        L = data["L"]
        NX = L[1]
        NY = L[2]
        NZ = L[3]
        NT = L[4]
    else
        error("header file is not found. Please put lattice size")
    end
    if haskey(data, "NC")
        NC = data["NC"]
    else
        error("header file is not found. Please put NC")
    end
    load_gaugefield(i, ildg::ILDG, L, NC; NDW=NDW)



end



function read_header(filename)
    contents_data = ""
    header = nothing
    lime_contents() do exe
        contents_data = read(`$exe $filename`, String)
        #println(contents_data)

        content_dictdata = split_data(contents_data)
        #display(content_dictdata)
        header = extract_info_fromdict(content_dictdata)
        #error("d")
        #display(header)

        #=
        contents_data = split(string(contents_data), "\n")
        println("split")
        println(contents_data)
        header = extract_info(contents_data)
        =#

    end
    #println(header)
    return header
end

function split_data(contents_data)
    #println("split data")
    n = count(==('\n'), contents_data)
    #println(n)
    data = Dict()
    alldata = []
    firstdata = true
    for line in eachline(IOBuffer(contents_data))
        #println("line ", line)
        nl = length(line)
        if nl >= 7 && line[1:7] == "Message"
            if firstdata
            else
                push!(alldata, deepcopy(data))
                data = Dict()
            end
            #println(split(line)[2])
            data["Message"] = parse(Int64, split(line)[2])
            firstdata = false
        elseif nl >= 6 && line[1:6] == "Record"
            #println(split(line)[2])
            data["Record"] = parse(Int64, split(line)[2])
        elseif nl >= 5 && line[1:5] == "Type:"
            #println(split(line)[2])
            data["Type"] = split(line)[2]
        elseif nl >= 12 && line[1:12] == "Data Length:"
            data["Data Length"] = parse(Int64, split(line)[3])
        elseif nl >= 15 && line[1:15] == "Padding Length:"
            data["Padding Length"] = parse(Int64, split(line)[3])
        elseif nl >= 8 && line[1:8] == "MB flag:"
            data["MB flag"] = parse(Int64, split(line)[3])
        elseif nl >= 8 && line[1:8] == "ME flag:"
            data["ME flag"] = parse(Int64, split(line)[3])
        elseif nl >= 5 && line[1:5] == "Data:"
            data["Data"] = line[6:end]
        else
            if haskey(data, "Data")
                data["Data"] *= line
            end
        end
    end
    push!(alldata, deepcopy(data))
    return alldata
end

function extract_info_fromdict(contents_data)
    i = 0
    message_no = 0
    reccord_no = 0
    datatype = ""
    header = Dict[]
    NX = 0
    NY = 0
    NZ = 0
    NT = 0
    NC = 3
    precision = 32
    headerfound = false
    headerdic = Dict()
    for data_dic in contents_data
        message_no = data_dic["Message"]
        reccord_no = data_dic["Record"]
        datatype = data_dic["Type"]
        if datatype == "ildg-format"
            headerdic = Dict()
            data = data_dic["Data"]
            ist = findfirst('\"', data)
            ien = findlast('\"', data)
            doc = parsexml(data[ist+1:ien-1])
            ildgFormat = root(doc)
            for d in eachelement(ildgFormat)
                if d.name == "lx"
                    NX = parse(Int64, d.content)
                elseif d.name == "ly"
                    NY = parse(Int64, d.content)
                elseif d.name == "lz"
                    NZ = parse(Int64, d.content)
                elseif d.name == "lt"
                    NT = parse(Int64, d.content)
                elseif d.name == "field"
                    gauge = d.content
                    if findfirst("su3", gauge) != nothing
                        NC = 3
                    elseif findfirst("su2", gauge) != nothing
                        NC = 2
                    else
                        error("not supported. gauge is ", gauge)
                    end
                elseif d.name == "precision"
                    precision = parse(Int64, d.content)
                end

            end
            headerdic["L"] = (NX, NY, NZ, NT)
            headerdic["NC"] = NC
            headerdic["precision"] = precision
            headerdic["headertype"] = "ildg-format"

            headerfound = false
        elseif datatype == "scidac-private-file-xml"
            headerdic = Dict()
            #println(data[2:end])
            data = data_dic["Data"]
            ist = findfirst('\"', data)
            ien = findlast('\"', data)


            doc = parsexml(data[ist+1:ien-1])

            scidacFile = root(doc)
            #systemdata = elements(ildgFormat)
            #println(systemdata["version"])

            for d in eachelement(scidacFile)
                #println(d)
                if d.name == "dims"
                    L = parse.(Int64, split(d.content))
                    NX = L[1]
                    NY = L[2]
                    NZ = L[3]
                    NT = L[4]
                    #=
                    elseif d.name == "colors"
                    NC = parse(Int64,d.content)
                    println(NC)
                    elseif d.name == "precision"
                    if d.content == "F"
                        precision = 32
                    elseif d.content == "D"
                        precision = 64
                    end
                    =#
                end

            end
            headerdic["L"] = (NX, NY, NZ, NT)
            headerdic["NC"] = NC
            headerdic["headertype"] = "scidac-private-file-xml"
            headerdic["precision"] = precision

            headerfound = true
        end

        if datatype == "ildg-binary-data" #&& headerfound
            #println("message_no = $(message_no)")
            #println("reccord_no = $reccord_no")
            headerdic["message_no"] = message_no
            headerdic["reccord_no"] = reccord_no
            push!(header, headerdic)
            headerfound = false

        end
    end
    return header
end

function extract_info(contents_data)
    println(contents_data)
    i = 0
    message_no = 0
    reccord_no = 0
    datatype = ""
    header = Dict[]
    NX = 0
    NY = 0
    NZ = 0
    NT = 0
    NC = 3
    precision = 32
    headerfound = false
    headerdic = Dict()
    for data in contents_data
        u = split(data)
        println("----")
        println(data)
        println("u = ", u)
        if length(u) ≥ 2
            if u[1] == "Message:"
                message_no = parse(Int64, u[2])
                #println(message_no)
            end
            if u[1] == "Record:"
                reccord_no = parse(Int64, u[2])
            end
            if u[1] == "Type:"
                datatype = u[2]

            end
            if u[1] == "Data:"
                #println("message_no = $(message_no)")
                #println("reccord_no = $reccord_no")
                #println("datatype  = $datatype ")
                @show datatype
                if datatype == "ildg-format"
                    ### HH: There is some bug in this part. I will fix it later. 

                    headerdic = Dict()
                    #println(data[2:end])
                    ist = findfirst('\"', data)
                    ien = findlast('\"', data)

                    #ien =  findlast(""\",data)
                    println("data :::", data, " ien ", ien, "ist ", ist)
                    println(data[ist+1:ien-1])
                    doc = parsexml(data[ist+1:ien-1])
                    #elm_lx = ElementNode("lx")
                    #println("lx = ",elm_lx.content)
                    ildgFormat = root(doc)
                    #systemdata = elements(ildgFormat)
                    #println(systemdata["version"])

                    for d in eachelement(ildgFormat)
                        if d.name == "lx"
                            NX = parse(Int64, d.content)
                        elseif d.name == "ly"
                            NY = parse(Int64, d.content)
                        elseif d.name == "lz"
                            NZ = parse(Int64, d.content)
                        elseif d.name == "lt"
                            NT = parse(Int64, d.content)
                        elseif d.name == "field"
                            gauge = d.content
                            if findfirst("su3", gauge) != nothing
                                NC = 3
                            elseif findfirst("su2", gauge) != nothing
                                NC = 2
                            else
                                error("not supported. gauge is ", gauge)
                            end
                        elseif d.name == "precision"
                            precision = parse(Int64, d.content)
                        end

                    end
                    headerdic["L"] = (NX, NY, NZ, NT)
                    headerdic["NC"] = NC
                    headerdic["precision"] = precision
                    headerdic["headertype"] = "ildg-format"

                    headerfound = false

                elseif datatype == "scidac-private-file-xml"
                    headerdic = Dict()
                    #println(data[2:end])

                    ist = findfirst('\"', data)
                    ien = findlast('\"', data)


                    doc = parsexml(data[ist+1:ien-1])

                    scidacFile = root(doc)
                    #systemdata = elements(ildgFormat)
                    #println(systemdata["version"])

                    for d in eachelement(scidacFile)
                        #println(d)
                        if d.name == "dims"
                            L = parse.(Int64, split(d.content))
                            NX = L[1]
                            NY = L[2]
                            NZ = L[3]
                            NT = L[4]
                            #=
                            elseif d.name == "colors"
                            NC = parse(Int64,d.content)
                            println(NC)
                            elseif d.name == "precision"
                            if d.content == "F"
                                precision = 32
                            elseif d.content == "D"
                                precision = 64
                            end
                            =#
                        end

                    end
                    headerdic["L"] = (NX, NY, NZ, NT)
                    headerdic["NC"] = NC
                    headerdic["headertype"] = "scidac-private-file-xml"
                    headerdic["precision"] = precision

                    headerfound = true

                end

                if datatype == "ildg-binary-data" #&& headerfound
                    #println("message_no = $(message_no)")
                    #println("reccord_no = $reccord_no")
                    headerdic["message_no"] = message_no
                    headerdic["reccord_no"] = reccord_no
                    push!(header, headerdic)
                    headerfound = false

                end

            end

        end



    end

    return header
end

function test()
    contents_data = ""
    header = nothing
    lime_contents() do exe
        contents_data = read(`$exe $(ARGS[1])`, String)
        #println(contents_data)
        contents_data = split(string(contents_data), "\n")
        #println(contents_data)
        header = extract_info(contents_data)

    end
    println(header)

    #println(contents_data)



    #println(text)
    #=
    lime_extract_record() do exe
        run(`$exe $(ARGS[1]) 1 1 out11.dat`)
        run(`$exe $(ARGS[1]) 1 2 out12.dat`)
        run(`$exe $(ARGS[1]) 1 3 out13.dat`)
        run(`$exe $(ARGS[1]) 2 1 out21.dat`)
        run(`$exe $(ARGS[1]) 2 2 out22.dat`)
        run(`$exe $(ARGS[1]) 3 3 out23.dat`)
    end
    =#
end
end