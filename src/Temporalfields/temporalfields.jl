module Temporalfields_module

mutable struct Temporalfields{TG}
    _data::Vector{TG}
    _flagusing::Vector{Bool}
    _indices::Vector{Int64}
    Nmax::Int64

    function Temporalfields(a::TG; num=1, Nmax=1000) where {TG}
        _data = Vector{TG}(undef, num)
        _flagusing = zeros(Bool, num)
        _indices = zeros(Int64, num)
        for i = 1:num
            _data[i] = similar(a)
        end

        return new{TG}(_data, _flagusing, _indices, Nmax)
    end
end

Base.eltype(::Type{Temporalfields{TG}}) where {TG} = TG

Base.length(t::Temporalfields{TG}) where {TG} = length(t._data)

Base.size(t::Temporalfields{TG}) where {TG} = size(t._data)

function Base.firstindex(t::Temporalfields{TG}) where {TG}
    return t[1]
end

function Base.lastindex(t::Temporalfields{TG}) where {TG}
    return t[length(t)]
end

function Base.getindex(t::Temporalfields{TG}, i::Int) where {TG}
    if i > length(t._data)
        @warn "The length of the temporalfields is shorter than the index $i. New temporal fields are created."
        ndiff = i - length(t._data)
        @assert i <= t.Nmax "The number of the tempralfields $i is larger than the maximum number $(Nmax). Change Nmax."
        for n = 1:ndiff
            push!(t._data, similar(t._data[1]))
            push!(t._flagusing, 0)
            push!(t._indices, 0)
        end
    end
    if t._indices[i] == 0
        index = findfirst(x -> x == 0, t._flagusing)
        t._flagusing[index] = true
        t._indices[i] = index
    else
        @error "This index $i is being using.  You should pay attention"
    end

    return t._data[t._indices[i]]
end

function Base.getindex(t::Temporalfields{TG}, I::Vararg{Int,N}) where {TG,N}
    data = TG[]
    for i in I
        push!(data, t[i])
    end
    return data
end

function Base.getindex(t::Temporalfields{TG}, I::AbstractVector{T}) where {TG,T<:Integer}
    data = TG[]
    for i in I
        push!(data, t[i])
    end
    return data
end

function Base.display(t::Temporalfields{TG}) where {TG}
    n = length(t._data)
    println("The total number of fields: $n")
    numused = sum(t._flagusing)
    println("The total number of fields used: $numused")
    for i = 1:n
        if t._indices[i] != 0
            println("The adress $i is used as the index $(t._indices[i])")
        end
    end
    println("The flags: $(t._flagusing)")
end

function unused!(t::Temporalfields{TG}, i) where {TG}
    if t._indices[i] != 0
        index = t._indices[i]
        t._flagusing[index] = false
        t._indices[i] = 0
    end
end


function unused!(t::Temporalfields{TG}, I::AbstractVector{T}) where {TG,T<:Integer}
    for i in I
        unused!(t, i)
    end
end

function unused!(t::Temporalfields{TG}) where {TG}
    for i = 1:length(t)
        unused!(t, i)
    end
end


export Temporalfields, unused!

end