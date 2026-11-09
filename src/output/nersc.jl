import LinearAlgebra: tr

# NERSC SU(3) configuration I/O. Binary payloads are ordered by
# (t,z,y,x,direction,row,column,real/imag), with x the fastest site axis.
# Host staging uses the same collective transport as portable JLD2 I/O.
function _nersc_header(io)
    strip(readline(io)) == "BEGIN_HEADER" || throw(ArgumentError("missing NERSC BEGIN_HEADER"))
    header=Dict{String,String}()
    while !eof(io)
        position(io) <= 65536 || throw(ArgumentError("NERSC header exceeds 64 KiB"))
        line=strip(readline(io))
        line=="END_HEADER" && return header
        isempty(line) && continue
        parts=split(line,'=';limit=2)
        length(parts)==2 || throw(ArgumentError("invalid NERSC header line: $line"))
        key=strip(parts[1]);haskey(header,key) && throw(ArgumentError("duplicate NERSC key $key"))
        header[key]=strip(parts[2])
    end
    throw(ArgumentError("missing NERSC END_HEADER"))
end

function _nersc_metadata(header)
    datatype=get(header,"DATATYPE","")
    rows=datatype=="4D_SU3_GAUGE" ? 2 : datatype=="4D_SU3_GAUGE_3x3" ? 3 :
        throw(ArgumentError("unsupported NERSC DATATYPE $datatype"))
    fp=get(header,"FLOATING_POINT","")
    fp in ("IEEE32","IEEE32BIG","IEEE32LITTLE","IEEE64","IEEE64BIG","IEEE64LITTLE") ||
        throw(ArgumentError("unsupported NERSC FLOATING_POINT $fp"))
    RT=startswith(fp,"IEEE32") ? Float32 : Float64
    dims=ntuple(d->parse(Int,header["DIMENSION_$d"]),4)
    all(>(0),dims) || throw(ArgumentError("NERSC lattice dimensions must be positive"))
    checksum=parse(UInt32,replace(lowercase(header["CHECKSUM"]),"0x"=>"");base=16)
    plaquette=parse(Float64,header["PLAQUETTE"]);linktrace=parse(Float64,header["LINK_TRACE"])
    all(isfinite,(plaquette,linktrace)) || throw(ArgumentError("nonfinite NERSC observables"))
    return (dimension=4,lattice=dims,colors=3,halo=1,boundary=ones(Int,4),
        element_type=Complex{RT},rows,fp,checksum,plaquette,linktrace)
end

function _nersc_reconstruct!(links)
    for a in links, site in CartesianIndices(size(a)[3:end])
        p=Tuple(site)
        a[3,1,p...]=conj(a[1,2,p...]*a[2,3,p...]-a[1,3,p...]*a[2,2,p...])
        a[3,2,p...]=conj(a[1,3,p...]*a[2,1,p...]-a[1,1,p...]*a[2,3,p...])
        a[3,3,p...]=conj(a[1,1,p...]*a[2,2,p...]-a[1,2,p...]*a[2,1,p...])
    end
    return links
end

function _nersc_observables(links)
    dims=size(first(links))[3:end];plaq=0.0;linktrace=0.0
    for site in CartesianIndices(dims)
        p=Tuple(site)
        for mu in 1:4
            u=ComplexF64.(view(links[mu],:,:,p...));linktrace+=real(tr(u))
            for nu in (mu+1):4
                pm=ntuple(d->d==mu ? mod1(p[d]+1,dims[d]) : p[d],4)
                pn=ntuple(d->d==nu ? mod1(p[d]+1,dims[d]) : p[d],4)
                v=ComplexF64.(view(links[nu],:,:,pm...))
                w=ComplexF64.(view(links[mu],:,:,pn...))
                z=ComplexF64.(view(links[nu],:,:,p...))
                plaq+=real(tr(u*v*w'*z'))
            end
        end
    end
    return plaq/(18prod(dims)),linktrace/(12prod(dims))
end

_nersc_checksum_word(x::UInt32)=x
_nersc_checksum_word(x::UInt64)=UInt32(x & 0xffffffff)+UInt32(x>>32)
_nersc_from_file(x,endian)=endian===:big ? ntoh(x) : ltoh(x)
_nersc_to_file(x,endian)=endian===:big ? hton(x) : htol(x)

function _read_nersc_payload(io,metadata,endian)
    RT=real(metadata.element_type);UT=RT==Float32 ? UInt32 : UInt64
    links=[Array{metadata.element_type}(undef,3,3,metadata.lattice...) for _ in 1:4]
    checksum=UInt32(0)
    for site in CartesianIndices(metadata.lattice), mu in 1:4, row in 1:metadata.rows, column in 1:3
        re=_nersc_from_file(read(io,UT),endian);imv=_nersc_from_file(read(io,UT),endian)
        checksum+=_nersc_checksum_word(re)+_nersc_checksum_word(imv)
        links[mu][row,column,Tuple(site)...]=complex(reinterpret(RT,re),reinterpret(RT,imv))
    end
    checksum==metadata.checksum || throw(ArgumentError("NERSC CHECKSUM mismatch"))
    metadata.rows==2 && _nersc_reconstruct!(links)
    all(a->all(isfinite,a),links) || throw(ArgumentError("nonfinite NERSC link values"))
    plaq,trace=_nersc_observables(links)
    # Header observables are often printed to only 8 decimal places.
    tol=RT==Float32 ? 5e-6 : 5e-8
    isapprox(plaq,metadata.plaquette;atol=tol,rtol=tol) || throw(ArgumentError("NERSC PLAQUETTE mismatch: $plaq versus $(metadata.plaquette)"))
    isapprox(trace,metadata.linktrace;atol=tol,rtol=tol) || throw(ArgumentError("NERSC LINK_TRACE mismatch: $trace versus $(metadata.linktrace)"))
    return links
end

function _nersc_checked_metadata(io, filename)
    metadata = _nersc_metadata(_nersc_header(io))
    expected = big(4) * prod(big.(metadata.lattice)) * metadata.rows * 3 * 2 * sizeof(real(metadata.element_type))
    filesize(filename) - position(io) == expected ||
        throw(ArgumentError("NERSC payload size does not match the header"))
    return metadata
end

function _read_nersc(filename)
    open(filename,"r") do io
        metadata=_nersc_checked_metadata(io,filename);offset=position(io)
        # Unsuffixed IEEE32/IEEE64 are used with differing byte orders by
        # existing writers. Accept only an order that passes every check.
        orders=endswith(metadata.fp,"BIG") ? (:big,) : endswith(metadata.fp,"LITTLE") ? (:little,) : (:big,:little)
        last_error=nothing
        for endian in orders
            seek(io,offset)
            try
                return _read_nersc_payload(io,metadata,endian),metadata
            catch err
                err isa ArgumentError || rethrow()
                last_error=err
            end
        end
        throw(last_error)
    end
end

function _load_nersc!(U,filename)
    communicator=gauge_communicator(U)
    result=_portable_jld2_root_operation(communicator) do
        _read_nersc(filename)
    end
    rank=isnothing(communicator) ? 0 : comm_rank(communicator)
    metadata=rank==0 ? result[2] : nothing
    isnothing(communicator) || (metadata=broadcast(metadata,0,communicator))
    _validate_portable_jld2_target(U,metadata)
    links=rank==0 ? result[1] : nothing
    if gauge_backend(U) isa LatticeMatricesBackend
        _load_portable_jld2_lm!(U,links,metadata,communicator)
    else
        _load_portable_jld2_legacy!(U,links,metadata)
    end
    return U
end

function _write_nersc(filename,links;precision::Integer=64,endian::Symbol=:big,compressed::Bool=false)
    precision in (32,64) || throw(ArgumentError("NERSC precision must be 32 or 64"))
    endian in (:big,:little) || throw(ArgumentError("NERSC endian must be :big or :little"))
    RT=precision==32 ? Float32 : Float64;UT=precision==32 ? UInt32 : UInt64
    values=[Complex{RT}.(a) for a in links];dims=size(first(values))[3:end]
    if compressed
        restored=_nersc_reconstruct!(deepcopy(values))
        tol=(precision==32 || any(a -> real(eltype(a)) == Float32, links)) ? 2e-5 : 2e-12
        all(isapprox(a,b;rtol=tol,atol=tol) for (a,b) in zip(values,restored)) ||
            throw(ArgumentError("two-row NERSC output requires SU(3) links"))
        values=restored
    end
    all(a->all(isfinite,a),values) || throw(ArgumentError("cannot write nonfinite NERSC links"))
    rows=compressed ? 2 : 3;checksum=UInt32(0)
    payload=IOBuffer()
    for site in CartesianIndices(dims), mu in 1:4, row in 1:rows, column in 1:3
        z=values[mu][row,column,Tuple(site)...]
        for v in (real(z),imag(z))
            word=reinterpret(UT,v);checksum+=_nersc_checksum_word(word)
            write(payload,_nersc_to_file(word,endian))
        end
    end
    plaq,trace=_nersc_observables(values)
    datatype=compressed ? "4D_SU3_GAUGE" : "4D_SU3_GAUGE_3x3"
    fp="IEEE$(precision)$(endian===:big ? "BIG" : "LITTLE")"
    open(filename,"w") do io
        println(io,"BEGIN_HEADER\nHDR_VERSION = 1.0\nDATATYPE = $datatype")
        for d in 1:4;println(io,"DIMENSION_$d = $(dims[d])");end
        println(io,"FLOATING_POINT = $fp\nCHECKSUM = $(string(checksum;base=16,pad=8))")
        println(io,"PLAQUETTE = $plaq\nLINK_TRACE = $trace\nCREATOR = Gaugefields.jl\nEND_HEADER")
        write(io,take!(payload))
    end
    return nothing
end

function _save_nersc(filename,U;kwargs...)
    length(U)==4 && gauge_num_colors(U)==3 || throw(ArgumentError("NERSC output requires four-dimensional SU(3) fields"))
    links=_portable_jld2_links(U)
    _portable_jld2_root_operation(gauge_communicator(U)) do
        _write_nersc(filename,links;kwargs...)
    end
    return nothing
end
