
module ILDG_format
using CLIME_jll
using EzXML
using Requires
import ..LatticeMatricesCompat: mark_lattice_dirty!

const ILDG_NAMESPACE = "http://www.lqcd.org/ildg"
const ILDG_FORMAT_VERSION = "1.2"

@inline function ildg_float_type(precision::Integer)
    precision == 32 && return Float32
    precision == 64 && return Float64
    throw(ArgumentError("ILDG precision must be 32 or 64, got $precision"))
end

@inline real_component_type(::Type{Complex{T}}) where {T<:AbstractFloat} = T
@inline real_component_type(::Type{T}) where {T<:AbstractFloat} = T

function resolve_ildg_precision(precision, field_element_type::Type)
    if precision === :field
        field_precision = 8 * sizeof(real_component_type(field_element_type))
        field_precision in (32, 64) || throw(ArgumentError(
            "ILDG only supports 32- or 64-bit fields, got $field_element_type"))
        return field_precision
    end
    precision isa Integer || throw(ArgumentError(
        "ILDG precision must be :field, 32, or 64, got $(repr(precision))"))
    ildg_float_type(precision)
    return Int(precision)
end

function ildg_format_xml(L, NC, precision)
    ildg_float_type(precision)
    field = "su$(NC)gauge"
    return """<?xml version="1.0" encoding="UTF-8"?>
<ildgFormat xmlns="$ILDG_NAMESPACE">
  <version>$ILDG_FORMAT_VERSION</version>
  <field>$field</field>
  <precision>$precision</precision>
  <lx>$(L[1])</lx>
  <ly>$(L[2])</ly>
  <lz>$(L[3])</lz>
  <lt>$(L[4])</lt>
</ildgFormat>
"""
end

function pack_ildg_file(filename, payload_path, filelist_path, L, NC, precision)
    header_path = payload_path * ".ildg-format.xml"
    try
        open(header_path, "w") do io
            write(io, ildg_format_xml(L, NC, precision))
        end
        open(filelist_path, "w") do io
            println(io, "$header_path ildg-format")
            println(io, "$payload_path ildg-binary-data")
        end
        run(`$(lime_pack()) $filelist_path $filename`)
    finally
        rm(header_path; force=true)
    end
    return nothing
end

function read_ildg_site!(bi, buffer, rawbuffer)
    F = bi.floattype
    expected_bytes = 2 * sizeof(F) * length(buffer)
    length(rawbuffer) == expected_bytes || throw(DimensionMismatch(
        "ILDG site buffer has $(length(rawbuffer)) bytes; expected $expected_bytes"))
    Base.read!(bi.fp, rawbuffer)
    data = reinterpret(F, rawbuffer)
    @inbounds for i in eachindex(buffer)
        realpart = ntoh(data[2i - 1])
        imagpart = ntoh(data[2i])
        buffer[i] = Complex{F}(realpart, imagpart)
    end
    bi.count += length(buffer)
    return buffer
end

ildg_communicator(::Any) = nothing

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

        ildg_communicator(U::Vector{T}) where {T<:Gaugefields_4D_wing_mpi} = comm
        ildg_communicator(U::Vector{T}) where {T<:Gaugefields_4D_nowing_mpi} = U[1].comm

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
            U::Vector{T},
            NX, NY, NZ, NT,
            NC,
            filename,
            precision,
        ) where {T<:Gaugefields_4D_nowing_mpi}

            comm = U[1].comm
            PN = U[1].PN

            Nfields = NC * NC * 4
            bi = Binarydata_ILDG(filename, precision)
            F = bi.floattype
            bytes_per_site = 2 * sizeof(F) * Nfields

            px, py, pz, pt = U[1].myrank_xyzt .* PN
            sitebuf = Vector{Complex{F}}(undef, Nfields)
            rawbuf = Vector{UInt8}(undef, bytes_per_site)
            try
                for it = 1:PN[4], iz = 1:PN[3], iy = 1:PN[2], ix = 1:PN[1]
                    ixg = px + ix
                    iyg = py + iy
                    izg = pz + iz
                    itg = pt + it

                    global_index =
                        (itg - 1) * (NZ * NY * NX) +
                        (izg - 1) * (NY * NX) +
                        (iyg - 1) * NX +
                        (ixg - 1)

                    seek(bi.fp, global_index * bytes_per_site)
                    read_ildg_site!(bi, sitebuf, rawbuf)
                    buf_index = 1
                    for μ = 1:4
                        for ic2 = 1:NC
                            for ic1 = 1:NC
                                setvalue!(
                                    U[μ], sitebuf[buf_index], ic2, ic1,
                                    ix, iy, iz, it)
                                buf_index += 1
                            end
                        end
                    end
                end
            finally
                close(bi)
            end
            update!(U)
            MPI.Barrier(comm)
            return nothing
        end

        function save_binarydata(
            U::Array{T,1},
            filename;
            tempfile1="testbin.dat",
            tempfile2="filelist.dat",
            precision=:field,
        ) where {T<:Gaugefields_4D_nowing_mpi}

            NX = U[1].NX
            NY = U[1].NY
            NZ = U[1].NZ
            NT = U[1].NT
            NC = U[1].NC
            precision = resolve_ildg_precision(precision, ComplexF64)
            F = ildg_float_type(precision)

            barrier(U[1])

            N = NC * NC * 4
            #send_mesg1 = Array{ComplexF64}(undef, 1)
            #recv_mesg1 = Array{ComplexF64}(undef, 1)

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
                                            write(fp, hton(F(real(v))))
                                            write(fp, hton(F(imag(v))))
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

                pack_ildg_file(
                    filename, tempfile1, tempfile2,
                    (NX, NY, NZ, NT), NC, precision)
            end
            barrier(U[1])


            return

        end
    end

    
    @require JACC = "0979c8fe-16a4-4796-9b82-89a9f10403ea" begin
        import ..AbstractGaugefields_module:
            Gaugefields_4D_MPILattice,
            barrier,
            get_myrank,
            set_halo!
        import LatticeMatrices: delinearize, gather_and_bcast_matrix

        ildg_communicator(U::Vector{T}) where {T<:Gaugefields_4D_MPILattice} = U[1].U.comm

        function load_binarydata!(
            U::Array{T,1},
            NX,
            NY,
            NZ,
            NT,
            NC,
            filename,
            precision,) where {T<:Gaugefields_4D_MPILattice}

            bi = Binarydata_ILDG(filename, precision)
            F = bi.floattype
            PN = U[1].U.PN

            Nfields = 4 * NC * NC
            N_localsites = prod(PN)
            total_elems = N_localsites * Nfields

            offset_coords = U[1].U.coords .* PN

            host_data = Vector{Complex{F}}(undef, total_elems)

            bytes_per_site = 2 * sizeof(F) * Nfields
            i = 1
            sitebuf = Vector{Complex{F}}(undef, Nfields)
            rawbuf = Vector{UInt8}(undef, bytes_per_site)
            try
                for it = 1:PN[4], iz = 1:PN[3], iy = 1:PN[2], ix = 1:PN[1]
                    ixg = offset_coords[1] + ix
                    iyg = offset_coords[2] + iy
                    izg = offset_coords[3] + iz
                    itg = offset_coords[4] + it

                    global_index =
                        (itg - 1) * (NZ * NY * NX) +
                        (izg - 1) * (NY * NX) +
                        (iyg - 1) * NX +
                        (ixg - 1)

                    seek(bi.fp, global_index * bytes_per_site)
                    read_ildg_site!(bi, sitebuf, rawbuf)
                    buf_index = 1
                    for μ = 1:4
                        for ic2 = 1:NC
                            for ic1 = 1:NC
                                host_data[i] = sitebuf[buf_index]
                                i += 1
                                buf_index += 1
                            end
                        end
                    end
                end
            finally
                close(bi)
            end

            device_data = JACC.array(host_data)

            for μ = 1:4
                mark_lattice_dirty!(U[μ].U)
                JACC.parallel_for(N_localsites, kernel_assign_configuration!,
                                U[μ].U.A, U[μ].U.indexer, U[μ].U.nw, device_data, NC, μ)
                JACC.synchronize()
                set_halo!(U[μ].U)
            end
            return nothing
        end


        @inline function kernel_assign_configuration!(
            i, u, dindexer, nw, data,
            NC::Int, μ::Int,
            )

            indices = delinearize(dindexer, i, nw)
            ix = indices[1]; iy = indices[2]; iz = indices[3]; it = indices[4]

            # Compute linear offset for this site
            site_stride = 4 * NC * NC

            # local site offset in `data` (i runs 1..N_localsites in the same order
            # the host read loop filled host_data: ix fastest, then iy, iz, it)
            site_offset = (i - 1) * site_stride

            # offset for this μ block
            mu_offset = (μ - 1) * (NC * NC)
            base = site_offset + mu_offset

            @inbounds for ic2 = 1:NC
                for ic1 = 1:NC
                    color_offset = (ic2 - 1) * NC + (ic1 - 1)
                    u[ic2, ic1, ix, iy, iz, it] = data[base + color_offset + 1]
                end
            end
        end

        function save_binarydata(
                    U::Array{T,1},
                    filename;
                    tempfile1="testbin.dat",
                    tempfile2="filelist.dat",
                    precision=:field,
                ) where {T<:Gaugefields_4D_MPILattice}

            # 1. Setup dimensions
            NX, NY, NZ, NT = U[1].NX, U[1].NY, U[1].NZ, U[1].NT
            NC = U[1].NC
            PN = U[1].U.PN
            N_localsites = prod(PN)
            Nfields = 4 * NC * NC
            coords = U[1].U.coords
            precision = resolve_ildg_precision(precision, eltype(U[1]))
            F = ildg_float_type(precision)

            comm = U[1].U.comm
            nprocs = MPI.Comm_size(comm)
            
            # Coordinate offset for this specific MPI rank
            offset_coords =  coords.* PN
            
            # Ensure all ranks are ready
            barrier(U[1])

            # 2. Extract GPU data to Host
            # We do this in parallel across all ranks first
            host_buffer = zeros(Complex{F}, N_localsites * Nfields)
            device_buffer = JACC.array(host_buffer)

            for μ = 1:4
                JACC.parallel_for(N_localsites, kernel_pack_configuration!,
                                U[μ].U.A, U[μ].U.indexer, U[μ].U.nw, device_buffer, NC, μ)
            end
            JACC.synchronize()
            copyto!(host_buffer, device_buffer)

            # 3. Sequential Write (Token Passing)
            # Rank 0 creates the file first to truncate any existing data
            if U[1].U.myrank == 0
                fp = open(tempfile1, "w")
                close(fp)
            end
            barrier(U[1])

            bytes_per_site = 2 * sizeof(F) * Nfields

            # Loop through all ranks; only one rank writes at a time
            for r in 0:(nprocs - 1)
                if U[1].U.myrank == r
                    # Open in read-write mode without truncating ("r+")
                    open(tempfile1, "r+") do fp
                        i = 1
                        for it = 1:PN[4], iz = 1:PN[3], iy = 1:PN[2], ix = 1:PN[1]
                            # Calculate global coordinates for this local site
                            ixg = offset_coords[1] + ix
                            iyg = offset_coords[2] + iy
                            izg = offset_coords[3] + iz
                            itg = offset_coords[4] + it

                            # Calculate global seek position (Matches your load_binarydata logic)
                            global_index = (itg - 1) * (NZ * NY * NX) +
                                        (izg - 1) * (NY * NX) +
                                        (iyg - 1) * NX +
                                        (ixg - 1)
                            
                            seek(fp, global_index * bytes_per_site)

                            # Write the block for this site (μ, then colors)
                            for k = 1:Nfields
                                v = host_buffer[i]
                                # hton converts to Big Endian for ILDG compatibility
                                write(fp, hton(real(v)))
                                write(fp, hton(imag(v)))
                                i += 1
                            end
                        end
                    end
                end
                # Wait for rank 'r' to finish writing and close the file
                barrier(U[1])
            end

            # 4. Finalize LIME packaging on Rank 0
            if U[1].U.myrank == 0
                pack_ildg_file(
                    filename, tempfile1, tempfile2,
                    (NX, NY, NZ, NT), NC, precision)
            end

            barrier(U[1])
            return
        end

        @inline function kernel_pack_configuration!(i, u, dindexer, nw, data, NC, μ)
            indices = delinearize(dindexer, i, nw)
            ix, iy, iz, it = indices[1], indices[2], indices[3], indices[4]

            site_stride = 4 * NC * NC
            site_offset = (i - 1) * site_stride
            mu_offset = (μ - 1) * (NC * NC)
            base = site_offset + mu_offset

            @inbounds for ic2 = 1:NC
                for ic1 = 1:NC
                    color_offset = (ic2 - 1) * NC + (ic1 - 1)
                    data[base + color_offset + 1] = u[ic2, ic1, ix, iy, iz, it]
                end
            end
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
        ildg_float_type(precision)
        doc = parsexml("""<?xml version="1.0" encoding="UTF-8"?>
<ildgFormat xmlns="$ILDG_NAMESPACE">
  <version>$version</version>
  <field>$field</field>
  <precision>$precision</precision>
  <lx>$(L[1])</lx>
  <ly>$(L[2])</ly>
  <lz>$(L[3])</lz>
  <lt>$(L[4])</lt>
</ildgFormat>
""")
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

function save_binarydata(
    U,
    filename;
    tempfile1="testbin.dat",
    tempfile2="filelist.dat",
    precision=:field,
)

    NX = U[1].NX
    NY = U[1].NY
    NZ = U[1].NZ
    NT = U[1].NT
    NC = U[1].NC
    field_element_type = typeof(U[1][1, 1, 1, 1, 1, 1])
    precision = resolve_ildg_precision(precision, field_element_type)
    F = ildg_float_type(precision)


    #li = LIME_header((NX,NY,NZ,NT),"su3gauge",1,64)
    #print(li.doc)
    #write("test.xml", li.doc)


    #fp = open("testbin.dat", "w")
    open(tempfile1, "w") do fp
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for μ = 1:4
                            for ic2 = 1:NC
                                for ic1 = 1:NC
                                    value = U[μ][ic2, ic1, ix, iy, iz, it]
                                    write(fp, hton(F(real(value))))
                                    write(fp, hton(F(imag(value))))
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    pack_ildg_file(
        filename, tempfile1, tempfile2,
        (NX, NY, NZ, NT), NC, precision)

    return nothing
end

update!(U) = set_wing!(U)
update!(U::Array{T,1}) where {T<:AbstractGaugefields} = set_wing_U!(U)

mutable struct Binarydata_ILDG
    fp::IOStream
    count::Int64
    floattype::DataType
    function Binarydata_ILDG(filename, precision)
        floattype = ildg_float_type(precision)
        fp = open(filename, "r")
        count = 0

        bi = new(fp, count, floattype)

        finalizer(bi) do bi
            close(bi)
        end

        return bi
    end
end

function Base.close(bi::Binarydata_ILDG)
    isopen(bi.fp) && close(bi.fp)
    return nothing
end

function read!(x::Binarydata_ILDG)
    x.count += 1
    rvalue = ntoh(read(x.fp, x.floattype))
    ivalue = ntoh(read(x.fp, x.floattype))
    return rvalue + im * ivalue
end

function load_binarydata!(U, NX, NY, NZ, NT, NC, filename, precision)
    bi = Binarydata_ILDG(filename, precision)

    try
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
    finally
        close(bi)
    end

    update!(U)
    return nothing
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

function load_gaugefield!(U, i, ildg::ILDG, L, NC; NDW=0, tmpfilename=nothing)
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


    comm = ildg_communicator(U)
    owns_tmpfile = isnothing(tmpfilename)

    if isnothing(comm)
        payload_path = owns_tmpfile ? tempname() : abspath(tmpfilename)
        try
            run(`$(lime_extract_record()) $filename $message_no $reccord_no $payload_path`)
            load_binarydata!(U, NX, NY, NZ, NT, NC, payload_path, precision)
        finally
            owns_tmpfile && rm(payload_path; force=true)
        end
    else
        rank = MPI.Comm_rank(comm)
        payload_path = if rank == 0
            owns_tmpfile ? tempname(pwd()) : abspath(tmpfilename)
        else
            ""
        end
        payload_path = MPI.bcast(payload_path, 0, comm)

        extraction_error = nothing
        if rank == 0
            try
                run(`$(lime_extract_record()) $filename $message_no $reccord_no $payload_path`)
            catch err
                extraction_error = sprint(showerror, err)
            end
        end
        extraction_error = MPI.bcast(extraction_error, 0, comm)
        extraction_error === nothing || error(
            "failed to extract ILDG binary record: $extraction_error")

        try
            load_binarydata!(U, NX, NY, NZ, NT, NC, payload_path, precision)
        finally
            MPI.Barrier(comm)
            rank == 0 && owns_tmpfile && rm(payload_path; force=true)
            MPI.Barrier(comm)
        end
    end

    return nothing
end

import ..AbstractGaugefields_module: Initialize_Gaugefields

function load_gaugefield(
    i,
    ildg::ILDG,
    L,
    NC;
    NDW=0,
    isMPILattice=false,
    PEs=nothing,
    verbose_level=2,
    singleprecision=false,
    boundarycondition=ones(4),
    elementtype=nothing,
)
    NX = L[1]
    NY = L[2]
    NZ = L[3]
    NT = L[4]
    data = ildg[i]
    filename = ildg.filename

    U = Initialize_Gaugefields(
        NC,
        NDW,
        NX,
        NY,
        NZ,
        NT;
        condition="cold",
        isMPILattice,
        PEs,
        verbose_level,
        singleprecision,
        boundarycondition,
        elementtype,
    )
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

function load_gaugefield(i, ildg::ILDG; kwargs...)
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
    load_gaugefield(i, ildg::ILDG, L, NC; kwargs...)



end



function read_header(filename)
    contents_data = read(`$(lime_contents()) $filename`, String)
    content_dictdata = split_data(contents_data)
    return extract_info_fromdict(content_dictdata)
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
    contents_data = read(`$(lime_contents()) $(ARGS[1])`, String)
    contents_data = split(string(contents_data), "\n")
    header = extract_info(contents_data)
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
