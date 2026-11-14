# MILC version 5 SU(3) configuration I/O (MILC code `save_serial` and
# `reload_serial` files, magic number 20103).
#
# File layout, as written by MILC generic/io_lat4.c and io_lat_utils.c:
#   magic number          uint32 (20103; its byte order gives the file order)
#   dims                  4 x int32 (nx, ny, nz, nt)
#   time stamp            64 bytes, NUL padded
#   order                 int32 (0: natural site order; otherwise a site list)
#   site list             V x uint32, only when order != 0
#   checksum              uint32 sum29, uint32 sum31
#   links                 for each site: 4 directions (x, y, z, t), each a
#                         3x3 row-major matrix of (real, imaginary) Float32
# Sites appear in natural order with x fastest, or in the order of the site
# list, whose entries encode x + nx*(y + ny*(z + nz*t)).
# The links carry no staggered phases or boundary signs.

const _MILC_MAGIC = UInt32(20103)
const _MILC_HEADER_BYTES = 4 + 16 + 64 + 4
const _MILC_WORDS_PER_SITE = 4 * 3 * 3 * 2

_milc_rotate(v::UInt32, r::Int) = r == 0 ? v : (v << r) | (v >> (32 - r))

# sum29/sum31 checksums: word i of the payload (in file order, after byte
# order correction) is rotated left by i mod 29 (i mod 31) bits and combined
# with exclusive or.
function _milc_checksums(words::AbstractVector{UInt32})
    sum29 = UInt32(0)
    sum31 = UInt32(0)
    r29 = 0
    r31 = 0
    @inbounds for v in words
        sum29 ⊻= _milc_rotate(v, r29)
        sum31 ⊻= _milc_rotate(v, r31)
        r29 += 1
        r29 == 29 && (r29 = 0)
        r31 += 1
        r31 == 31 && (r31 = 0)
    end
    return sum29, sum31
end

_milc_from_file(x, endian) = endian === :big ? ntoh(x) : ltoh(x)
_milc_to_file(x, endian) = endian === :big ? hton(x) : htol(x)

function _milc_metadata(io, filename)
    filesize(filename) >= _MILC_HEADER_BYTES + 8 ||
        throw(ArgumentError("file is too short for a MILC gauge configuration"))
    raw = read(io, UInt32)
    endian = if ltoh(raw) == _MILC_MAGIC
        :little
    elseif ntoh(raw) == _MILC_MAGIC
        :big
    elseif ltoh(raw) == 0x42454749 || ntoh(raw) == 0x42454749
        throw(ArgumentError("this is a NERSC archive file; use format=:nersc"))
    elseif ltoh(raw) == 0x456789ab || ntoh(raw) == 0x456789ab
        throw(ArgumentError("this is a LIME (SciDAC/ILDG) file; use format=:ildg"))
    else
        throw(ArgumentError(
            "unrecognized MILC magic number $(repr(raw)); only version 5 " *
            "files (magic number 20103) are supported",
        ))
    end
    dims = ntuple(_ -> Int(reinterpret(Int32, _milc_from_file(read(io, UInt32), endian))), 4)
    all(>(0), dims) || throw(ArgumentError("MILC lattice dimensions must be positive: $dims"))
    stamp = read(io, 64)
    time_stamp = String(stamp[1:something(findfirst(==(0x00), stamp), 65) - 1])
    order = _milc_from_file(read(io, UInt32), endian)
    volume = prod(big.(dims))
    expected = _MILC_HEADER_BYTES + (order == 0 ? 0 : 4volume) + 8 +
               4 * _MILC_WORDS_PER_SITE * volume
    filesize(filename) == expected || throw(ArgumentError(
        "MILC payload size does not match the header: file has " *
        "$(filesize(filename)) bytes, header implies $expected",
    ))
    return (dimension=4, lattice=dims, colors=3, halo=1, boundary=ones(Int, 4),
        element_type=ComplexF32, endian, order, time_stamp)
end

function _read_milc(filename)
    open(filename, "r") do io
        metadata = _milc_metadata(io, filename)
        dims = metadata.lattice
        volume = prod(dims)
        endian = metadata.endian
        sites = if metadata.order == 0
            nothing
        else
            list = [Int(_milc_from_file(read(io, UInt32), endian)) for _ in 1:volume]
            seen = falses(volume)
            for s in list
                0 <= s < volume || throw(ArgumentError("MILC site list entry $s is out of range"))
                seen[s+1] && throw(ArgumentError("MILC site list repeats site $s"))
                seen[s+1] = true
            end
            list
        end
        sum29 = _milc_from_file(read(io, UInt32), endian)
        sum31 = _milc_from_file(read(io, UInt32), endian)
        words = Vector{UInt32}(undef, _MILC_WORDS_PER_SITE * volume)
        read!(io, words)
        words .= _milc_from_file.(words, endian)
        _milc_checksums(words) == (sum29, sum31) || throw(ArgumentError(
            "MILC checksum mismatch: file has $(string(sum29; base=16)) " *
            "$(string(sum31; base=16)), data give " *
            join(string.(_milc_checksums(words); base=16), " "),
        ))
        values = reinterpret(Float32, words)
        links = [Array{ComplexF32}(undef, 3, 3, dims...) for _ in 1:4]
        linear = LinearIndices(dims)
        cartesian = CartesianIndices(dims)
        k = 0
        for record in 1:volume
            site = isnothing(sites) ? cartesian[record] : cartesian[sites[record]+1]
            for mu in 1:4, row in 1:3, column in 1:3
                links[mu][row, column, Tuple(site)...] =
                    complex(values[k+1], values[k+2])
                k += 2
            end
        end
        all(a -> all(isfinite, a), links) || throw(ArgumentError("nonfinite MILC link values"))
        return links, metadata
    end
end

function _load_milc!(U, filename)
    communicator = gauge_communicator(U)
    result = _portable_jld2_root_operation(communicator) do
        _read_milc(filename)
    end
    rank = isnothing(communicator) ? 0 : comm_rank(communicator)
    metadata = rank == 0 ? result[2] : nothing
    isnothing(communicator) || (metadata = broadcast(metadata, 0, communicator))
    _validate_portable_jld2_target(U, metadata)
    links = rank == 0 ? result[1] : nothing
    if gauge_backend(U) isa LatticeMatricesBackend
        _load_portable_jld2_lm!(U, links, metadata, communicator)
    else
        _load_portable_jld2_legacy!(U, links, metadata)
    end
    return U
end

# Plaquettes in the normalization of the MILC `.info` file and `CHECK PLAQ`
# output: the mean of Re tr over spatial (ss) and space-time (st) plaquettes,
# 3 for unit links. The link trace follows MILC `CHECK NERSC LINKTR`.
function _milc_observables(links)
    dims = size(first(links))[3:end]
    ss = 0.0
    st = 0.0
    linktrace = 0.0
    for site in CartesianIndices(dims)
        p = Tuple(site)
        for mu in 1:4
            u = ComplexF64.(view(links[mu], :, :, p...))
            linktrace += real(tr(u))
            for nu in (mu+1):4
                pm = ntuple(d -> d == mu ? mod1(p[d] + 1, dims[d]) : p[d], 4)
                pn = ntuple(d -> d == nu ? mod1(p[d] + 1, dims[d]) : p[d], 4)
                v = ComplexF64.(view(links[nu], :, :, pm...))
                w = ComplexF64.(view(links[mu], :, :, pn...))
                z = ComplexF64.(view(links[nu], :, :, p...))
                value = real(tr(u * v * w' * z'))
                nu == 4 ? (st += value) : (ss += value)
            end
        end
    end
    volume = prod(dims)
    return (ssplaq=ss / (3volume), stplaq=st / (3volume), linktrace=linktrace / (12volume))
end

# MILC `gauge.nersc_checksum`: the 32-bit sum of the Float32 bit patterns of
# the first two rows of every link.
function _milc_nersc_checksum(links)
    checksum = UInt32(0)
    for a in links, site in CartesianIndices(size(a)[3:end]), row in 1:2, column in 1:3
        z = a[row, column, Tuple(site)...]
        checksum += reinterpret(UInt32, Float32(real(z)))
        checksum += reinterpret(UInt32, Float32(imag(z)))
    end
    return checksum
end

function _write_milc(filename, links; endian::Symbol=:little, info::Bool=true,
    time_stamp::AbstractString=Libc.strftime("%a %b %e %H:%M:%S %Y", time()))
    endian in (:big, :little) || throw(ArgumentError("MILC endian must be :big or :little"))
    ncodeunits(time_stamp) < 64 || throw(ArgumentError("MILC time stamp must be shorter than 64 bytes"))
    values = [ComplexF32.(a) for a in links]
    all(a -> all(isfinite, a), values) || throw(ArgumentError("cannot write nonfinite MILC links"))
    dims = size(first(values))[3:end]
    all(<=(typemax(Int32)), dims) || throw(ArgumentError("MILC lattice dimensions must fit in Int32"))
    words = Vector{UInt32}(undef, _MILC_WORDS_PER_SITE * prod(dims))
    k = 0
    for site in CartesianIndices(dims), mu in 1:4, row in 1:3, column in 1:3
        z = values[mu][row, column, Tuple(site)...]
        words[k+1] = reinterpret(UInt32, real(z))
        words[k+2] = reinterpret(UInt32, imag(z))
        k += 2
    end
    sum29, sum31 = _milc_checksums(words)
    stamp = zeros(UInt8, 64)
    copyto!(stamp, codeunits(time_stamp))
    open(filename, "w") do io
        write(io, _milc_to_file(_MILC_MAGIC, endian))
        for d in dims
            write(io, _milc_to_file(UInt32(d), endian))
        end
        write(io, stamp)
        write(io, _milc_to_file(UInt32(0), endian))
        write(io, _milc_to_file(sum29, endian))
        write(io, _milc_to_file(sum31, endian))
        write(io, _milc_to_file.(words, endian))
    end
    if info
        obs = _milc_observables(values)
        f6(x) = string(round(x; digits=6))
        open(filename * ".info", "w") do io
            println(io, "magic_number = 20103")
            println(io, "time_stamp = \"$time_stamp\"")
            println(io, "checksums = \"$(string(sum29; base=16)) $(string(sum31; base=16))\"")
            for (name, d) in zip(("nx", "ny", "nz", "nt"), dims)
                println(io, "$name = $d")
            end
            println(io, "gauge.ssplaq = $(f6(obs.ssplaq))")
            println(io, "gauge.stplaq = $(f6(obs.stplaq))")
            println(io, "gauge.nersc_linktr = $(f6(obs.linktrace))")
            println(io, "gauge.nersc_checksum = $(_milc_nersc_checksum(values))")
        end
    end
    return nothing
end

function _save_milc(filename, U; kwargs...)
    length(U) == 4 && gauge_num_colors(U) == 3 ||
        throw(ArgumentError("MILC output requires four-dimensional SU(3) fields"))
    links = _portable_jld2_links(U)
    _portable_jld2_root_operation(gauge_communicator(U)) do
        _write_milc(filename, links; kwargs...)
    end
    return nothing
end
