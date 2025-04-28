module Temporalfields_module
#using PreallocatedArrays
import PreallocatedArrays: PreallocatedArray, get_block, unused!, set_reusemode!



mutable struct Temporalfields{TG}
    _data::PreallocatedArray{TG,Union{Nothing,String},false}
    #PreallocatedArray(a::TG;
    #labeltype=String, haslabel=false, num=1, Nmax=1000, reusemode=false)
    function Temporalfields(a::TG; num=1, Nmax=1000, reusemode=false) where {TG}
        data = PreallocatedArray(a; num, Nmax, reusemode)
        return new{TG}(data)
    end

    #function PreallocatedArray(_data::Vector{TG}, _flagusing, _indices, Nmax, _reusemode) where {TG}
    function Temporalfields(_data::Vector{TG}, _flagusing, _indices, Nmax, _reusemode) where {TG}
        data = PreallocatedArray(_data, _flagusing, _indices, Nmax, _reusemode)
        return new{TG}(data)
    end

    function Temporalfields(data::PreallocatedArray{TG,Nothing,false}) where {TG}
        return new{TG}(data)
    end

end

#function PreallocatedArray(a::AbstractVector{TG};
#Nmax=1000, reusemode=false) where {TG<:AbstractVector}
@inline function Temporalfields_fromvector(a::Vector{TG}; Nmax=1000, reusemode=false) where {TG}
    data = PreallocatedArray(a; Nmax, reusemode)
    return Temporalfields(data)
end
export Temporalfields_fromvector

@inline function set_reusemode!(t::Temporalfields{TG}, reusemode) where {TG}
    set_reusemode!(t._data, reusemode)
end
export set_reusemode!

@inline Base.eltype(t::Type{Temporalfields{TG}}) where {TG} = eltype(t._data)

@inline Base.length(t::Temporalfields{TG}) where {TG} = length(t._data)

@inline Base.size(t::Temporalfields{TG}) where {TG} = size(t._data)

@inline function Base.firstindex(t::Temporalfields{TG}) where {TG}
    return Base.firstindex(t._data)
end

@inline function Base.lastindex(t::Temporalfields{TG}) where {TG}
    return Base.lastindex(t._data)
end

@inline function Base.getindex(t::Temporalfields{TG}, i::Int) where {TG}
    return Base.getindex(t._data, i)
end

@inline function Base.getindex(t::Temporalfields{TG}, I::Vararg{Int,N}) where {TG,N}
    return Base.getindex(t._data, I)
end

@inline function Base.getindex(t::Temporalfields{TG}, I::AbstractVector{T}) where {TG,T<:Integer}
    return Base.getindex(t._data, I)
end

@inline function Base.display(t::Temporalfields{TG}) where {TG}
    Base.display(t._data)
end

@inline function get_temp(t::Temporalfields{TG}) where {TG}
    return get_block(t._data)
end

@inline function get_temp(t::Temporalfields{TG}, num) where {TG}
    return get_block(t._data, num)
end

@inline function unused!(t::Temporalfields{TG}, i) where {TG}
    unused!(t._data, i)
end


@inline function unused!(t::Temporalfields{TG}, I::AbstractVector{T}) where {TG,T<:Integer}
    for i in I
        unused!(t, i)
    end
end

@inline function unused!(t::Temporalfields{TG}) where {TG}
    for i = 1:length(t)
        unused!(t, i)
    end
end


export Temporalfields, unused!, get_temp

end