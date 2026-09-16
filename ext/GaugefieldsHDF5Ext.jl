module GaugefieldsHDF5Ext

using Gaugefields
using HDF5

const LCNN = Gaugefields.LCNN

@inline function _c_order_site(site::Int, lattice::NTuple{Dim,Int}) where {Dim}
    zero_based = site - 1
    return ntuple(Val(Dim)) do direction
        stride = prod(lattice[(direction + 1):end]; init=1)
        mod(div(zero_based, stride), lattice[direction]) + 1
    end
end

function _sample_indices(selection, count::Int)
    indices = if selection === :all || selection isa Colon
        collect(1:count)
    elseif selection isa Integer
        [Int(selection)]
    else
        Int.(collect(selection))
    end
    all(index -> 1 <= index <= count, indices) || throw(BoundsError(1:count, indices))
    return indices
end

function _read_selected(dataset, indices, sample_axis::Int)
    sample_count = size(dataset, sample_axis)
    length(indices) == sample_count && indices == collect(1:sample_count) &&
        return read(dataset)
    isempty(indices) && throw(ArgumentError("sample selection must not be empty"))
    if indices == collect(first(indices):last(indices))
        selectors = ntuple(
            axis -> axis == sample_axis ? (first(indices):last(indices)) : Colon(),
            ndims(dataset),
        )
        return dataset[selectors...]
    end
    blocks = map(indices) do index
        selectors = ntuple(
            axis -> axis == sample_axis ? (index:index) : Colon(),
            ndims(dataset),
        )
        dataset[selectors...]
    end
    return length(blocks) == 1 ? only(blocks) : cat(blocks...; dims=sample_axis)
end

"""
    LCNN.read_favoni2022_dataset(path; target="trW_1x2", samples=:all,
                                  link_type=nothing, target_type=nothing)

Load a dataset written by the released `lge-cnn` Python generator. Its HDF5
arrays have Python shapes `(samples, sites, Dim, NC, NC)` for `u` and
`(samples, sites)` for the target. HDF5.jl exposes those axes in reverse
order; this loader additionally expands the C-order flattened site index into
the physical lattice coordinates used by Gaugefields.

The returned `DenseLCNNDataset` stores only `u`, the selected target, `beta`,
and `dims`; the precomputed `w` entry is intentionally ignored because
Gaugefields recomputes plaquettes with its own tested `Plaq` layer.
"""
function LCNN.read_favoni2022_dataset(
    path::AbstractString;
    target::Union{AbstractString,Symbol}="trW_1x2",
    samples=:all,
    link_type=nothing,
    target_type=nothing,
)
    target_name = String(target)
    raw_links, raw_targets, raw_beta, lattice = h5open(path, "r") do file
        haskey(file, "u") || throw(ArgumentError("HDF5 dataset is missing `u`"))
        haskey(file, target_name) || throw(ArgumentError(
            "HDF5 dataset is missing target `$target_name`",
        ))
        haskey(file, "dims") || throw(ArgumentError("HDF5 dataset is missing `dims`"))
        dimensions = Tuple(Int.(vec(read(file["dims"]))))
        link_dataset = file["u"]
        ndims(link_dataset) == 5 || throw(DimensionMismatch(
            "Python `u` must have five axes; HDF5.jl exposed " *
            "$(size(link_dataset))",
        ))
        selected = _sample_indices(samples, size(link_dataset, 5))
        links = _read_selected(link_dataset, selected, 5)
        target_dataset = file[target_name]
        ndims(target_dataset) == 2 || throw(DimensionMismatch(
            "Python target `$target_name` must have two axes; HDF5.jl " *
            "exposed $(size(target_dataset))",
        ))
        targets = _read_selected(target_dataset, selected, 2)
        beta = haskey(file, "beta") ? read(file["beta"])[selected] : nothing
        return links, targets, beta, dimensions
    end

    Dim = length(lattice)
    Dim in (2, 3, 4) || throw(ArgumentError(
        "Favoni dataset dimension must be 2, 3, or 4; got $Dim",
    ))
    ndims(raw_links) == 5 || throw(DimensionMismatch(
        "Python `u` must have shape (samples, sites, Dim, NC, NC); " *
        "HDF5.jl exposed $(size(raw_links))",
    ))
    colors = size(raw_links, 1)
    size(raw_links, 2) == colors || throw(DimensionMismatch(
        "link matrices must be square; got $(size(raw_links, 1)) by " *
        "$(size(raw_links, 2))",
    ))
    size(raw_links, 3) == Dim || throw(DimensionMismatch(
        "dataset contains $(size(raw_links, 3)) link directions but dims has $Dim entries",
    ))
    volume = prod(lattice)
    size(raw_links, 4) == volume || throw(DimensionMismatch(
        "dataset contains $(size(raw_links, 4)) flattened sites but dims=$lattice " *
        "has volume $volume",
    ))
    sample_count = size(raw_links, 5)
    size(raw_targets) == (volume, sample_count) || throw(DimensionMismatch(
        "target `$target_name` has HDF5.jl shape $(size(raw_targets)); " *
        "expected ($volume, $sample_count)",
    ))

    input_type = link_type === nothing ? eltype(raw_links) : link_type
    input_type <: Complex || throw(ArgumentError(
        "link_type must be complex; got $input_type",
    ))
    inferred_target_type = typeof(real(zero(eltype(raw_targets))))
    output_type = target_type === nothing ? inferred_target_type : target_type
    output_type <: Real || throw(ArgumentError(
        "target_type must be real; got $output_type",
    ))

    links = Array{input_type}(undef, colors, colors, lattice..., Dim, sample_count)
    targets = Array{output_type}(undef, lattice..., sample_count)
    for sample in 1:sample_count
        for flattened_site in 1:volume
            site = _c_order_site(flattened_site, lattice)
            for direction in 1:Dim, column in 1:colors, row in 1:colors
                links[row, column, site..., direction, sample] =
                    raw_links[row, column, direction, flattened_site, sample]
            end
            targets[site..., sample] = convert(
                output_type,
                real(raw_targets[flattened_site, sample]),
            )
        end
    end
    beta = raw_beta === nothing ? nothing : collect(raw_beta)
    return LCNN.DenseLCNNDataset(links, targets, lattice; beta)
end

end
