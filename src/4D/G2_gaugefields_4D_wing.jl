using Random

struct G2Gaugefields_4D_wing <: Gaugefields_4D{7}
    U::Array{ComplexF64,6}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print

    function G2Gaugefields_4D_wing(NDW::T, NX::T, NY::T, NZ::T, NT::T; verbose_level = 2) where {T<:Integer}
        NV = NX * NY * NZ * NT
        U = zeros(ComplexF64, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM,
            NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW)
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        return new(U, NX, NY, NZ, NT, NDW, NV, G2_FUNDAMENTAL_DIM, mpi, verbose_print)
    end
end

function G2Gaugefields_4D_wing(NC::T, NDW::T, NX::T, NY::T, NZ::T, NT::T; verbose_level = 2) where {T<:Integer}
    NC == G2_FUNDAMENTAL_DIM ||
        throw(ArgumentError("G2Gaugefields_4D_wing requires NC == $G2_FUNDAMENTAL_DIM"))
    return G2Gaugefields_4D_wing(NDW, NX, NY, NZ, NT; verbose_level = verbose_level)
end

function Base.setindex!(x::G2Gaugefields_4D_wing, v, i1, i2, i3, i4, i5, i6)
    @inbounds x.U[i1, i2, i3 + x.NDW, i4 + x.NDW, i5 + x.NDW, i6 + x.NDW] = v
end

@inline function Base.getindex(x::G2Gaugefields_4D_wing, i1, i2, i3, i4, i5, i6)
    @inbounds return x.U[i1, i2, i3 + x.NDW, i4 + x.NDW, i5 + x.NDW, i6 + x.NDW]
end

function Base.setindex!(x::G2Gaugefields_4D_wing, v, i1, i2, ii)
    ix, iy, iz, it = get_latticeindex(ii, x.NX, x.NY, x.NZ, x.NT)
    @inbounds x.U[i1, i2, ix + x.NDW, iy + x.NDW, iz + x.NDW, it + x.NDW] = v
end

@inline function Base.getindex(x::G2Gaugefields_4D_wing, i1, i2, ii)
    ix, iy, iz, it = get_latticeindex(ii, x.NX, x.NY, x.NZ, x.NT)
    @inbounds return x.U[i1, i2, ix + x.NDW, iy + x.NDW, iz + x.NDW, it + x.NDW]
end

function Base.similar(U::G2Gaugefields_4D_wing)
    return G2Gaugefields_4D_wing(U.NDW, U.NX, U.NY, U.NZ, U.NT; verbose_level = U.verbose_print.level)
end

function Base.similar(U::Array{T,1}) where {T<:G2Gaugefields_4D_wing}
    Uout = Array{T,1}(undef, 4)
    for μ in 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
end

function substitute_U!(a::G2Gaugefields_4D_wing, b::G2Gaugefields_4D_wing)
    @inbounds for i in eachindex(a.U)
        a.U[i] = b.U[i]
    end
    return nothing
end

function substitute_U!(a::G2Gaugefields_4D_wing, b::T2) where {T2<:Abstractfields}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX
    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            @inbounds a[k1, k2, ix, iy, iz, it] = b[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(a)
    return nothing
end

function substitute_U!(a::Array{T,1}, b::Array{T,1}) where {T<:G2Gaugefields_4D_wing}
    for μ in 1:4
        substitute_U!(a[μ], b[μ])
    end
    return nothing
end

function clear_U!(Uμ::G2Gaugefields_4D_wing)
    fill!(Uμ.U, 0)
    return nothing
end

function add_U!(c::G2Gaugefields_4D_wing, a::T1) where {T1<:Abstractfields}
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            c[k1, k2, ix, iy, iz, it] += a[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
    return nothing
end

function add_U!(c::G2Gaugefields_4D_wing, α::N, a::T1) where {N<:Number,T1<:Abstractfields}
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            c[k1, k2, ix, iy, iz, it] += α * a[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
    return nothing
end

function unit_U!(Uμ::G2Gaugefields_4D_wing)
    fill!(Uμ.U, 0)
    NT = Uμ.NT
    NZ = Uμ.NZ
    NY = Uμ.NY
    NX = Uμ.NX
    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for k in 1:G2_FUNDAMENTAL_DIM
                        @inbounds Uμ[k, k, ix, iy, iz, it] = 1
                    end
                end
            end
        end
    end
    set_wing_U!(Uμ)
    return nothing
end

function identityG2Gaugefields_4D_wing(NX, NY, NZ, NT, NDW; verbose_level = 2)
    U = G2Gaugefields_4D_wing(NDW, NX, NY, NZ, NT; verbose_level = verbose_level)
    unit_U!(U)
    return U
end

function randomG2Gaugefields_4D_wing(NX, NY, NZ, NT, NDW; verbose_level = 2, randomnumber = "Random", scale = 0.2)
    U = G2Gaugefields_4D_wing(NDW, NX, NY, NZ, NT; verbose_level = verbose_level)
    rng = if randomnumber == "Random"
        MersenneTwister()
    elseif randomnumber == "Reproducible"
        StableRNG(123)
    else
        error("randomnumber should be \"Random\" or \"Reproducible\". Now randomnumber = $randomnumber")
    end
    basis = g2_basis()
    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    coeffs = scale .* randn(rng, G2_ALGEBRA_DIM)
                    link = exp(g2_matrix(coeffs; basis = basis))
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            @inbounds U[k1, k2, ix, iy, iz, it] = ComplexF64(link[k1, k2])
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(U)
    return U
end

function set_wing_U!(u::Array{T,1}) where {T<:G2Gaugefields_4D_wing}
    for μ in 1:4
        set_wing_U!(u[μ])
    end
    return nothing
end

function set_wing_U!(u::G2Gaugefields_4D_wing)
    NT = u.NT
    NY = u.NY
    NZ = u.NZ
    NX = u.NX
    NDW = u.NDW

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for id in 1:NDW
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            @inbounds u[k1, k2, -NDW + id, iy, iz, it] =
                                u[k1, k2, NX + (id - NDW), iy, iz, it]
                            @inbounds u[k1, k2, NX + id, iy, iz, it] =
                                u[k1, k2, id, iy, iz, it]
                        end
                    end
                end
            end
        end
    end

    for it in 1:NT
        for iz in 1:NZ
            for ix in -NDW+1:NX+NDW
                for id in 1:NDW
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            @inbounds u[k1, k2, ix, -NDW + id, iz, it] =
                                u[k1, k2, ix, NY + (id - NDW), iz, it]
                            @inbounds u[k1, k2, ix, NY + id, iz, it] =
                                u[k1, k2, ix, id, iz, it]
                        end
                    end
                end
            end
        end
    end

    for id in 1:NDW
        for it in 1:NT
            for iy in -NDW+1:NY+NDW
                for ix in -NDW+1:NX+NDW
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            @inbounds u[k1, k2, ix, iy, id - NDW, it] =
                                u[k1, k2, ix, iy, NZ + (id - NDW), it]
                            @inbounds u[k1, k2, ix, iy, NZ + id, it] =
                                u[k1, k2, ix, iy, id, it]
                        end
                    end
                end
            end
        end
    end

    for id in 1:NDW
        for iz in -NDW+1:NZ+NDW
            for iy in -NDW+1:NY+NDW
                for ix in -NDW+1:NX+NDW
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            @inbounds u[k1, k2, ix, iy, iz, id - NDW] =
                                u[k1, k2, ix, iy, iz, NT + (id - NDW)]
                            @inbounds u[k1, k2, ix, iy, iz, NT + id] =
                                u[k1, k2, ix, iy, iz, id]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

struct Shifted_G2Gaugefields_4D <: Shifted_Gaugefields{7,4}
    parent::G2Gaugefields_4D_wing
    shift::NTuple{4,Int8}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    outside::Bool

    function Shifted_G2Gaugefields_4D(U::G2Gaugefields_4D_wing, shift)
        shift4 = ntuple(i -> Int8(shift[i]), 4)
        outside = check_outside(U.NDW, shift4)
        return new(U, shift4, U.NX, U.NY, U.NZ, U.NT, U.NDW, outside)
    end
end

function shift_U(U::G2Gaugefields_4D_wing, ν::T) where {T<:Integer}
    shift = if ν == 1
        (1, 0, 0, 0)
    elseif ν == 2
        (0, 1, 0, 0)
    elseif ν == 3
        (0, 0, 1, 0)
    elseif ν == 4
        (0, 0, 0, 1)
    elseif ν == -1
        (-1, 0, 0, 0)
    elseif ν == -2
        (0, -1, 0, 0)
    elseif ν == -3
        (0, 0, -1, 0)
    elseif ν == -4
        (0, 0, 0, -1)
    else
        throw(ArgumentError("direction must be one of -4:-1 or 1:4"))
    end
    return Shifted_G2Gaugefields_4D(U, shift)
end

function shift_U(U::G2Gaugefields_4D_wing, shift::NTuple{Dim,T}) where {Dim,T<:Integer}
    Dim == 4 || throw(ArgumentError("G2Gaugefields_4D_wing requires a 4-dimensional shift"))
    return Shifted_G2Gaugefields_4D(U, shift)
end

@inline function Base.getindex(U::Shifted_G2Gaugefields_4D, i1, i2, i3, i4, i5, i6)
    if !U.outside
        @inbounds return U.parent[i1, i2, i3 + U.shift[1], i4 + U.shift[2], i5 + U.shift[3], i6 + U.shift[4]]
    end

    i3_new = i3 + U.shift[1]
    i3_new += ifelse(i3_new > U.NX + U.NDW, -U.NX, 0)
    i3_new += ifelse(i3_new < 1 - U.NDW, U.NX, 0)
    i4_new = i4 + U.shift[2]
    i4_new += ifelse(i4_new > U.NY + U.NDW, -U.NY, 0)
    i4_new += ifelse(i4_new < 1 - U.NDW, U.NY, 0)
    i5_new = i5 + U.shift[3]
    i5_new += ifelse(i5_new > U.NZ + U.NDW, -U.NZ, 0)
    i5_new += ifelse(i5_new < 1 - U.NDW, U.NZ, 0)
    i6_new = i6 + U.shift[4]
    i6_new += ifelse(i6_new > U.NT + U.NDW, -U.NT, 0)
    i6_new += ifelse(i6_new < 1 - U.NDW, U.NT, 0)

    @inbounds return U.parent[i1, i2, i3_new, i4_new, i5_new, i6_new]
end

function LinearAlgebra.tr(a::G2Gaugefields_4D_wing)
    s = zero(ComplexF64)
    @inbounds for it in 1:a.NT
        for iz in 1:a.NZ
            for iy in 1:a.NY
                for ix in 1:a.NX
                    for k in 1:G2_FUNDAMENTAL_DIM
                        s += a[k, k, ix, iy, iz, it]
                    end
                end
            end
        end
    end
    return s
end

function LinearAlgebra.tr(a::G2Gaugefields_4D_wing, b::G2Gaugefields_4D_wing)
    s = zero(ComplexF64)
    @inbounds for it in 1:a.NT
        for iz in 1:a.NZ
            for iy in 1:a.NY
                for ix in 1:a.NX
                    for k1 in 1:G2_FUNDAMENTAL_DIM
                        for k2 in 1:G2_FUNDAMENTAL_DIM
                            s += a[k1, k2, ix, iy, iz, it] * b[k2, k1, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    return s
end

function LinearAlgebra.mul!(c::G2Gaugefields_4D_wing, a::T1, b::T2) where {T1<:Abstractfields,T2<:Abstractfields}
    @inbounds for it in 1:c.NT
        for iz in 1:c.NZ
            for iy in 1:c.NY
                for ix in 1:c.NX
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            value = zero(ComplexF64)
                            for k3 in 1:G2_FUNDAMENTAL_DIM
                                value += a[k1, k3, ix, iy, iz, it] * b[k3, k2, ix, iy, iz, it]
                            end
                            c[k1, k2, ix, iy, iz, it] = value
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
    return c
end

function LinearAlgebra.mul!(c::G2Gaugefields_4D_wing, a::T1, b::T2, α::Ta, β::Tb) where {T1<:Abstractfields,T2<:Abstractfields,Ta<:Number,Tb<:Number}
    product = zeros(ComplexF64, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    @inbounds for it in 1:c.NT
        for iz in 1:c.NZ
            for iy in 1:c.NY
                for ix in 1:c.NX
                    fill!(product, 0)
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            for k3 in 1:G2_FUNDAMENTAL_DIM
                                product[k1, k2] += a[k1, k3, ix, iy, iz, it] * b[k3, k2, ix, iy, iz, it]
                            end
                        end
                    end
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            c[k1, k2, ix, iy, iz, it] =
                                α * product[k1, k2] + β * c[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
    return c
end

function LinearAlgebra.mul!(c::G2Gaugefields_4D_wing, α::N, b::T2) where {N<:Number,T2<:Abstractfields}
    @inbounds for it in 1:c.NT
        for iz in 1:c.NZ
            for iy in 1:c.NY
                for ix in 1:c.NX
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            c[k1, k2, ix, iy, iz, it] = α * b[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
    return c
end
