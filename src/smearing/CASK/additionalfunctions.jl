
function trace(U)
    return ComplexField(tr(U))
end

function site_trace(U::AbstractGaugefields{NC,Dim}) where {Dim,NC}
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(U)
    #Nsite = prod(NN)
    #value = IdentityGauges_4D(NC, NN...)
    value = zeros(ComplexF64, NN...)
    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    tr = 0.0im
                    for ic = 1:NC
                        #tr += U[ic, ic, ix, iy, iz, it]
                        value[ix, iy, iz, it] += U[ic, ic, ix, iy, iz, it]
                    end
                    #for ic = 1:NC
                    #    value[ic, ic, ix, iy, iz, it] = tr
                    #end
                end
            end
        end
    end
    return value
end
export site_trace

function site_realtrace!(value, U::AbstractGaugefields{NC,Dim}) where {Dim,NC}
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(U)
    #Nsite = prod(NN)
    #value = IdentityGauges_4D(NC, NN...)
    #value = zeros(ComplexF64, NN...)
    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    tr = 0.0im
                    for ic = 1:NC
                        tr += U[ic, ic, ix, iy, iz, it]
                        #value[ix, iy, iz, it] += U[ic, ic, ix, iy, iz, it]
                    end
                    #for ic = 1:NC
                    value[ix, iy, iz, it] = real(tr)
                    #end
                end
            end
        end
    end
    #return value
end
export site_realtrace!

function site_realtrace!(value, U::AbstractGaugefields{NC,Dim}, a::N) where {Dim,NC,N<:Number}
    #error("site")
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(U)
    #Nsite = prod(NN)
    #value = IdentityGauges_4D(NC, NN...)
    #value = zeros(ComplexF64, NN...)
    #error(NN)

    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    tr = 0.0im
                    for ic = 1:NC
                        tr += U[ic, ic, ix, iy, iz, it]
                        #value[ix, iy, iz, it] += U[ic, ic, ix, iy, iz, it]
                    end
                    #for ic = 1:NC

                    value[ix, iy, iz, it] = real(tr) * a
                    #error("dd")
                    #end
                end
            end
        end
    end
    #return value
end

function site_realtrace_add!(value, U::AbstractGaugefields{NC,Dim}, a::N) where {Dim,NC,N<:Number}
    #error("site")
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(U)
    #Nsite = prod(NN)
    #value = IdentityGauges_4D(NC, NN...)
    #value = zeros(ComplexF64, NN...)
    #error(NN)

    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    tr = 0.0im
                    for ic = 1:NC
                        tr += U[ic, ic, ix, iy, iz, it]
                        #value[ix, iy, iz, it] += U[ic, ic, ix, iy, iz, it]
                    end
                    #for ic = 1:NC

                    value[ix, iy, iz, it] += real(tr) * a
                    #error("dd")
                    #end
                end
            end
        end
    end
    #return value
end

function site_realtrace_add!(value, U::AbstractGaugefields{NC,Dim}) where {Dim,NC}
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(U)
    #Nsite = prod(NN)
    #value = IdentityGauges_4D(NC, NN...)
    #value = zeros(ComplexF64, NN...)
    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    tr = 0.0im
                    for ic = 1:NC
                        tr += U[ic, ic, ix, iy, iz, it]
                        #value[ix, iy, iz, it] += U[ic, ic, ix, iy, iz, it]
                    end
                    #for ic = 1:NC
                    value[ix, iy, iz, it] += real(tr)
                    #end
                end
            end
        end
    end
    #return value
end
export site_realtrace_add!


function site_realtrace_filter_add!(value, U::AbstractGaugefields{NC,Dim}, filtermatrix, filterfunction, a::N) where {Dim,NC,N<:Number}
    #error("site")
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(U)
    #Nsite = prod(NN)
    #value = IdentityGauges_4D(NC, NN...)
    #value = zeros(ComplexF64, NN...)
    #error(NN)

    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    tr = 0.0im
                    for ic = 1:NC
                        tr += U[ic, ic, ix, iy, iz, it]
                        #value[ix, iy, iz, it] += U[ic, ic, ix, iy, iz, it]
                    end
                    #for ic = 1:NC

                    value[ix, iy, iz, it] += real(tr) * a * filterfunction(filtermatrix[ix, iy, iz, it])
                    #error("dd")
                    #end
                end
            end
        end
    end
    #return value
end
export site_realtrace_filter_add!


function Base.exp(Q::T) where {T<:Abstractfields}
    Uout = similar(Q)
    temps = Array{T,1}(undef, 2)
    temps[1] = similar(Q)
    temps[2] = similar(Q)

    Gaugefields.exptU!(Uout, 1, Q, temps)
    return Uout
end

function Base.:*(a::AbstractArray, b::AbstractGaugefields{NC,Dim}) where {NC,Dim}
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(b)
    NNN = size(a)
    @assert NN == NNN "size mismatch! size(a) = $(size(a)) and size(b) = $(size(b))"
    c = similar(b)
    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    for jc = 1:NC
                        for ic = 1:NC
                            c[ic, jc, ix, iy, iz, it] = a[ix, iy, iz, it] * b[ic, jc, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    return c
end

function LinearAlgebra.mul!(c, a::AbstractArray, b::AbstractGaugefields{NC,Dim}) where {NC,Dim}
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(b)
    NNN = size(a)
    @assert NN == NNN "size mismatch! size(a) = $(size(a)) and size(b) = $(size(b))"
    #c = similar(b)
    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    for jc = 1:NC
                        for ic = 1:NC
                            c[ic, jc, ix, iy, iz, it] = a[ix, iy, iz, it] * b[ic, jc, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    #return c
end

function mul_withshift!(c, a::AbstractArray, b::AbstractGaugefields{NC,Dim}, shiftposition) where {NC,Dim}
    @assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    _, _, NN... = size(b)
    NNN = size(a)
    @assert NN == NNN "size mismatch! size(a) = $(size(a)) and size(b) = $(size(b))"
    #c = similar(b)
    for it = 1:NN[4]
        it2 = it + shiftposition[4]
        it2 += ifelse(it2 > NN[4], -NN[4], 0) + ifelse(it2 < 1, NN[4], 0)
        for iz = 1:NN[3]
            iz2 = iz + shiftposition[3]
            iz2 += ifelse(iz2 > NN[3], -NN[3], 0) + ifelse(iz2 < 1, NN[3], 0)
            for iy = 1:NN[2]
                iy2 = iy + shiftposition[2]
                iy2 += ifelse(iy2 > NN[2], -NN[2], 0) + ifelse(iy2 < 1, NN[2], 0)
                for ix = 1:NN[1]
                    ix2 = ix + shiftposition[1]
                    ix2 += ifelse(ix2 > NN[1], -NN[1], 0) + ifelse(ix2 < 1, NN[1], 0)
                    for jc = 1:NC
                        for ic = 1:NC
                            c[ic, jc, ix, iy, iz, it] = a[ix2, iy2, iz2, it2] * b[ic, jc, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    #return c
end

function Base.:*(a::AbstractArray, b::Adjoint_Gaugefields)
    #@assert Dim == 4 "4 dimension is only supported. Now Dim = $Dim"
    NC, _, NN... = size(b)
    NNN = size(a)
    @assert NN == NNN "size mismatch! size(a) = $(size(a)) and size(b) = $(size(b))"
    c = similar(b)
    for it = 1:NN[4]
        for iz = 1:NN[3]
            for iy = 1:NN[2]
                for ix = 1:NN[1]
                    for jc = 1:NC
                        for ic = 1:NC
                            c[ic, jc, ix, iy, iz, it] = a[ix, iy, iz, it] * b[jc, ic, ix, iy, iz, it]'
                        end
                    end
                end
            end
        end
    end
    return c
end


function Base.:*(a::T1, b::T2) where {T1<:Abstractfields,T2<:Number}
    c = similar(a)
    LinearAlgebra.mul!(c, b, a)
    return c
end

function Base.:*(a::T1, b::T2) where {T2<:Abstractfields,T1<:Number}
    c = similar(b)
    LinearAlgebra.mul!(c, a, b)
    return c
end

function Base.:/(a::T1, b::T2) where {T1<:Abstractfields,T2<:Number}
    c = a * (1 / b)
    return c
end



function Base.:+(a::T1, b::T2) where {T1<:Adjoint_Gaugefields,T2<:Abstractfields}
    c = deepcopy(b)
    add_U!(c, a)
    return c
end

function Base.:+(a::T1, b::T2) where {T1<:Adjoint_Gaugefields,T2<:Adjoint_Gaugefields}
    c = deepcopy(b.parent)'
    add_U!(c, a)
    return c
end


function Base.:-(a::T1, b::T2) where {T1<:Abstractfields,T2<:Abstractfields}
    c = a + (-1 * b)
    return c
end

function Base.:-(a::T1, b::T2) where {T1<:Adjoint_Gaugefields,T2<:Abstractfields}
    c = a + (-1 * b)
    return c
end

function Base.:-(a::T1, b::T2) where {T1<:Adjoint_Gaugefields,T2<:Adjoint_Gaugefields}
    c = a + (-1 * b)
    return c
end




function LinearAlgebra.tr(a::Adjoint_Gaugefields{T}) where {T<:AbstractGaugefields}
    return tr(a.parent)'
end

function Base.size(a::Adjoint_Gaugefields{T}) where {T<:AbstractGaugefields}
    return size(a.parent)
end

function Base.similar(a::Adjoint_Gaugefields{T}) where {T<:AbstractGaugefields}
    return similar(a.parent)
end


function shift_U(a::T, shift0) where {T<:Shifted_Gaugefields}
    c = similar(a.parent)
    shift = Tuple(collect(a.shift) .+ collect(shift0))
    return shift_U(c, shift)
end

function shift_U(a::Adjoint_Gaugefields{T}, shift) where {T<:AbstractGaugefields}
    c = deepcopy(a.parent)
    return shift_U(c, shift)'
end



function shift_gaugefield(a::T, shift) where {T<:Abstractfields}
    c = similar(a)
    cshifted = shift_U(a, Tuple(collect(shift)))
    substitute_U!(c, cshifted)
    return c
end

function LinearAlgebra.tr(a::T) where {T<:Shifted_Gaugefields}
    return tr(a.parent)
end

function Base.size(a::T) where {T<:Shifted_Gaugefields}
    return size(a.parent)
end

function make_shift()
    shifts = []
    for μ = 1:4
        shift = zeros(Int64, 4)
        shift[μ] = 1
        push!(shifts, Tuple(shift))
    end
    return shifts
end

function make_shift(ishift)
    shifts = []
    for μ = 1:4
        shift = zeros(Int64, 4)
        shift[μ] = ishift
        push!(shifts, Tuple(shift))
    end
    return shifts
end

function make_shift_ν(ishift, ν)
    shifts = []
    for μ = 1:4
        shift = zeros(Int64, 4)
        shift[μ] = ishift
        shift[ν] = 1
        push!(shifts, Tuple(shift))
    end
    return shifts
end

function make_vec_shift()
    vec_shifts = []
    for ishift = 1:20
        push!(vec_shifts, make_shift(ishift))
    end
    return vec_shifts
end

function make_vec_shift(ν)
    vec_shifts = []
    for ishift = 1:20
        push!(vec_shifts, make_shift_ν(ishift, ν))
    end
    return vec_shifts
end

function make_vec_vec_shift()
    vec_vec_shifts = []
    for ν = 1:4
        push!(vec_vec_shifts, make_vec_shift(ν))
    end
    return vec_vec_shifts
end

const shifts = make_shift()
const vec_shifts = make_vec_shift()
const vec_vec_shifts = make_vec_vec_shift()


const eps_Q = 1e-18




