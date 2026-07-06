struct G2TA_Gaugefields_4D_serial <: TA_Gaugefields_4D{7}
    a::Array{Float64,5}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NC::Int64
    NumofBasis::Int64
    basis::G2Basis{Float64}

    function G2TA_Gaugefields_4D_serial(NX, NY, NZ, NT; basis::G2Basis{Float64} = g2_basis())
        a = zeros(Float64, G2_ALGEBRA_DIM, NX, NY, NZ, NT)
        return new(a, NX, NY, NZ, NT, G2_FUNDAMENTAL_DIM, G2_ALGEBRA_DIM, basis)
    end
end

G2TA_Gaugefields_4D_serial(U::G2Gaugefields_4D_wing) =
    G2TA_Gaugefields_4D_serial(U.NX, U.NY, U.NZ, U.NT)

function initialize_TA_Gaugefields(U::G2Gaugefields_4D_wing)
    return G2TA_Gaugefields_4D_serial(U)
end

function Base.setindex!(x::G2TA_Gaugefields_4D_serial, v, i...)
    @inbounds x.a[i...] = v
end

function Base.getindex(x::G2TA_Gaugefields_4D_serial, i...)
    @inbounds return x.a[i...]
end

function Base.similar(u::G2TA_Gaugefields_4D_serial)
    return G2TA_Gaugefields_4D_serial(u.NX, u.NY, u.NZ, u.NT; basis = u.basis)
end

function clear_U!(Uμ::G2TA_Gaugefields_4D_serial)
    fill!(Uμ.a, 0.0)
    return nothing
end

function gauss_distribution!(p::G2TA_Gaugefields_4D_serial; σ = 1.0)
    d = Normal(0.0, σ)
    NT = p.NT
    NZ = p.NZ
    NY = p.NY
    NX = p.NX
    pwork = rand(d, NX * NY * NZ * NT * G2_ALGEBRA_DIM)
    icount = 0
    @inbounds for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for k in 1:G2_ALGEBRA_DIM
                        icount += 1
                        p[k, ix, iy, iz, it] = pwork[icount]
                    end
                end
            end
        end
    end
    return nothing
end

function substitute_U!(Uμ::G2TA_Gaugefields_4D_serial, pwork)
    length(pwork) == length(Uμ.a) ||
        throw(ArgumentError("work vector length $(length(pwork)) does not match G2 momentum length $(length(Uμ.a))"))
    NT = Uμ.NT
    NZ = Uμ.NZ
    NY = Uμ.NY
    NX = Uμ.NX
    icount = 0
    @inbounds for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for k in 1:G2_ALGEBRA_DIM
                        icount += 1
                        Uμ[k, ix, iy, iz, it] = pwork[icount]
                    end
                end
            end
        end
    end
    return nothing
end

function Base.:*(x::G2TA_Gaugefields_4D_serial, y::G2TA_Gaugefields_4D_serial)
    x.NX == y.NX && x.NY == y.NY && x.NZ == y.NZ && x.NT == y.NT ||
        throw(ArgumentError("G2 momentum fields must have the same lattice size"))
    s = 0.0
    @inbounds for it in 1:x.NT
        for iz in 1:x.NZ
            for iy in 1:x.NY
                for ix in 1:x.NX
                    for k in 1:G2_ALGEBRA_DIM
                        s += x[k, ix, iy, iz, it] * y[k, ix, iy, iz, it]
                    end
                end
            end
        end
    end
    return s
end

function add_U!(c::G2TA_Gaugefields_4D_serial, α::N, a::G2TA_Gaugefields_4D_serial) where {N<:Number}
    c.NX == a.NX && c.NY == a.NY && c.NZ == a.NZ && c.NT == a.NT ||
        throw(ArgumentError("G2 momentum fields must have the same lattice size"))
    @inbounds for i in eachindex(c.a)
        c.a[i] += α * a.a[i]
    end
    return nothing
end

function _g2_site_matrix(U::Abstractfields, ix, iy, iz, it)
    matrix = zeros(ComplexF64, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    @inbounds for k2 in 1:G2_FUNDAMENTAL_DIM
        for k1 in 1:G2_FUNDAMENTAL_DIM
            matrix[k1, k2] = U[k1, k2, ix, iy, iz, it]
        end
    end
    return matrix
end

function Traceless_antihermitian!(c::G2TA_Gaugefields_4D_serial, vin::G2Gaugefields_4D_wing)
    clear_U!(c)
    return Traceless_antihermitian_add!(c, 1.0, vin)
end

function Traceless_antihermitian_add!(c::G2TA_Gaugefields_4D_serial, factor, vin::G2Gaugefields_4D_wing)
    c.NX == vin.NX && c.NY == vin.NY && c.NZ == vin.NZ && c.NT == vin.NT ||
        throw(ArgumentError("G2 momentum and link field must have the same lattice size"))
    basis = c.basis
    @inbounds for it in 1:vin.NT
        for iz in 1:vin.NZ
            for iy in 1:vin.NY
                for ix in 1:vin.NX
                    coeffs = project_to_g2_coefficients(_g2_site_matrix(vin, ix, iy, iz, it); basis = basis)
                    for k in 1:G2_ALGEBRA_DIM
                        c[k, ix, iy, iz, it] += factor * coeffs[k]
                    end
                end
            end
        end
    end
    return nothing
end

function _g2_momentum_coefficients(p::G2TA_Gaugefields_4D_serial, ix, iy, iz, it)
    coeffs = zeros(Float64, G2_ALGEBRA_DIM)
    @inbounds for k in 1:G2_ALGEBRA_DIM
        coeffs[k] = p[k, ix, iy, iz, it]
    end
    return coeffs
end

function exptU!(
    uout::G2Gaugefields_4D_wing,
    t::N,
    p::G2TA_Gaugefields_4D_serial,
    temps::Array{G2Gaugefields_4D_wing,1},
) where {N<:Number}
    uout.NX == p.NX && uout.NY == p.NY && uout.NZ == p.NZ && uout.NT == p.NT ||
        throw(ArgumentError("G2 output link field and momentum field must have the same lattice size"))
    basis = p.basis
    @inbounds for it in 1:p.NT
        for iz in 1:p.NZ
            for iy in 1:p.NY
                for ix in 1:p.NX
                    coeffs = _g2_momentum_coefficients(p, ix, iy, iz, it)
                    update = exp(Float64(t) .* g2_matrix(coeffs; basis = basis))
                    for k2 in 1:G2_FUNDAMENTAL_DIM
                        for k1 in 1:G2_FUNDAMENTAL_DIM
                            uout[k1, k2, ix, iy, iz, it] = ComplexF64(update[k1, k2])
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(uout)
    return nothing
end
