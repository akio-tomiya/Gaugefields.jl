struct Isingfields_2D <: Gaugefields_2D{1}
    ϕ::Array{Int64,4}
    NX::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print
    ϕshifted::Array{Int64,4}

    function Isingfields_2D(
        NX::T,
        NT::T;
        verbose_level = 2,
    ) where {T<:Integer}
        NC = 1
        NV = NX * NT
        NDW = 0
        ϕ = zeros(Int64, NC, NC, NX + 2NDW, NT + 2NDW)
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        ϕshifted = zero(ϕ)
        #U = Array{Array{ComplexF64,4}}(undef,2)
        #for μ=1:2
        #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        #end
        return new(ϕ, NX, NT, NDW, NV, NC, mpi, verbose_print, ϕshifted)
    end
end

function Base.setindex!(x::Isingfields_2D , v, i1, i2, i3, i6)
    @inbounds x.ϕ[i1, i2, i3, i6] = v
end

@inline function Base.getindex(x::Isingfields_2D , i1, i2, i3, i6)
    @inbounds return x.ϕ[i1, i2, i3, i6]
end

function Base.setindex!(x::Isingfields_2D , v, i3, i6)
    @inbounds x.ϕ[1, 1, i3, i6] = v
end

@inline function Base.getindex(x::Isingfields_2D , i3, i6)
    @inbounds return x.ϕ[1,1,i3, i6]
end

function Base.setindex!(x::Isingfields_2D , v, i1, i2, ii)
    ix, it = get_latticeindex(ii, x.NX, x.NT)
    @inbounds x.ϕ[i1, i2, ix, it] = v
end

@inline function Base.getindex(x::Isingfields_2D , i1, i2, ii)
    ix, it = get_latticeindex(ii, x.NX, x.NT)
    @inbounds return x.ϕ[i1, i2, ix, it]
end

function Base.setindex!(x::Isingfields_2D , v, ii)
    ix, it = get_latticeindex(ii, x.NX, x.NT)
    @inbounds x.ϕ[1, 1, ix, it] = v
end

@inline function Base.getindex(x::Isingfields_2D , ii)
    ix, it = get_latticeindex(ii, x.NX, x.NT)
    @inbounds return x.ϕ[1, 1, ix, it]
end

function substitute_U!(a::Isingfields_2D, b::T2) where {T2<:Abstractfields}
    NT = a.NT
    #NZ = a.NZ
    #NY = a.NY
    NX = a.NX
    for it = 1:NT
        ##for iz=1:NZ
        ##for iy=1:NY
        for ix = 1:NX
            @inbounds a[ix, it] = b[ix, it]
        end
        #end
        #end
    end
end

function substitute_U!(
    a::Isingfields_2D,
    b::T2,
    iseven::Bool,
) where {T2<:Abstractfields}
    NT = a.NT
    #NZ = a.NZ
    #NY = a.NY
    NX = a.NX
    for it = 1:NT
        ##for iz=1:NZ
        #    #for iy=1:NY
        for ix = 1:NX
            evenodd = ifelse((ix + it) % 2 == 0, true, false)
            if evenodd == iseven
                @inbounds a[ix, it] = b[ix, it]
            end
        end
        #    end
        #end
    end

end

function identityIsingfields_2D(NX, NT; verbose_level = 2)
    ϕ =  Isingfields_2D(NX, NT, verbose_level = verbose_level)

    for it = 1:NT
        for ix = 1:NX
            ϕ[ix, it] = 1
        end
    end
    return U
end

