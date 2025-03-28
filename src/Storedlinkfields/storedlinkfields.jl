module Storedlinkfields_module

import Wilsonloop: Wilsonline

mutable struct Storedlinkfields{TG,WL<:Wilsonline}
    _data::Vector{TG}
    _link::Vector{WL}
    _flagusing::Vector{Bool}
    _indices::Vector{Int64}
    Nmax::Int64

    function Storedlinkfields(a::TG, l::WL; num=1, Nmax=1000) where {TG,WL<:Wilsonline}
        _data = Vector{TG}(undef, num)
        _link = Vector{WL}(undef, num)
        _flagusing = zeros(Bool, num)
        _indices = zeros(Int64, num)
        similar_l = Wilsonline([])
        for i = 1:num
            _data[i] = similar(a)
            _link[i] = similar_l
        end
        return new{TG,WL}(_data, _link, _flagusing, _indices, Nmax)
    end

    function Storedlinkfields(_data::Vector{TG}, _link::Vector{WL}, _flagusing, _indices, Nmax) where {TG,WL<:Wilsonline}
        return new{TG,WL}(_data, _link, _flagusing, _indices, Nmax)
    end

end

function Storedlinkfields_fromvector(a::Vector{TG}, l::Vector{WL}; Nmax=1000) where {TG,WL<:Wilsonline}
    num = length(a)
    if num != length(l)
        error("Lengths of TG and WL vectors are mismatched.")
    end
    _flagusing = zeros(Bool, num)
    _indices = zeros(Int64, num)
    return Storedlinkfields(a, l, _flagusing, _indices, Nmax)
end
export Storedlinkfields_fromvector

Base.eltype(::Type{Storedlinkfields{TG,WL}}) where {TG,WL} = TG

Base.length(t::Storedlinkfields{TG,WL}) where {TG,WL} = length(t._data)

Base.size(t::Storedlinkfields{TG,WL}) where {TG,WL} = size(t._data)

function Base.firstindex(t::Storedlinkfields{TG,WL}) where {TG,WL}
    return 1
end

function Base.lastindex(t::Storedlinkfields{TG,WL}) where {TG,WL}
    return length(t._data)
end

function Base.getindex(t::Storedlinkfields{TG,WL}, i::Int) where {TG,WL}
    @assert i <= length(t._data) "The length of the storedlinkfields is shorter than the index $i."
    @assert i <= t.Nmax "The number of the storedlinkfields $i is larger than the maximum number $(Nmax). Change Nmax."
    if t._indices[i] == 0
        index = findfirst(x -> x == 0, t._flagusing)
        t._flagusing[index] = true
        t._indices[i] = index
    end

    return t._data[t._indices[i]], t._link[t._indices[i]]
end

function Base.getindex(t::Storedlinkfields{TG,WL}, I::Vararg{Int,N}) where {TG,WL,N}
    data = TG[]
    link = WL[]
    for i in I
        data_tmp, link_tmp = t[i]
        push!(data, data_tmp)
        push!(link, link_tmp)
    end
    return data, link
end

function Base.getindex(t::Storedlinkfields{TG,WL}, I::AbstractVector{T}) where {TG,WL,T<:Integer}
    data = TG[]
    link = WL[]
    for i in I
        data_tmp, link_tmp = t[i]
        push!(data, data_tmp)
        push!(link, link_tmp)
    end
    return data, link
end

function Base.display(t::Storedlinkfields{TG,WL}) where {TG,WL}
    n = length(t._data)
    println("The strage size of fields: $n")
    numused = sum(t._flagusing)
    println("The total number of fields used: $numused")
    for i = 1:n
        if t._indices[i] != 0
            println("The address $(t._indices[i]) is used as the index $i")
        end
    end
    println("The flags: $(t._flagusing)")
    println("The indices: $(t._indices)")
end

function is_storedlink(t::Storedlinkfields{TG,WL}, l::WL) where {TG,WL}
    if l in t._link
        return true
    else
        return false
    end
end

function store_link!(t::Storedlinkfields{TG,WL}, a::TG, l::WL) where {TG,WL}
    n = length(t._data)
    i = findfirst(x -> x == 0, t._indices)
    if i == nothing
        error("All strage of $n fields are used.")
    end
    index = i
    if !is_storedlink(t,l)
        t._flagusing[index] = true
        t._indices[i] = index
        t._data[index] = deepcopy(a)
        t._link[index] = deepcopy(l)
    end
end

function store_link!(t::Storedlinkfields{TG,WL}, as::Vector{TG}, ls::Vector{WL}) where {TG,WL}
    n = length(as)
    if n != length(ls)
        error("Lengths of TG and WL vectors are mismatched.")
    end
    for i = 1:n
        store_link!(t, as[i], ls[i])
    end
end

function get_storedlink(t::Storedlinkfields{TG,WL}, l::WL) where {TG,WL}
    i = findfirst(x -> x == l, t._link)
    if i == nothing
        error("No such a stored link.")
    end
    index = t._indices[i]
    return t._data[index]
end

export Storedlinkfields, is_storedlink, store_link!, get_storedlink

end
