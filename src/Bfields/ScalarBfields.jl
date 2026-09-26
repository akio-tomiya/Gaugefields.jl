# Six independent scalar planes, with lazy, authoritative matrix compatibility
# objects. Once a matrix escapes through B[mu,nu], do not cache derived phases:
# callers may keep that object and mutate it at any later time.
struct ScalarBStorage{T,S,F} <: AbstractMatrix{T}
    phases::Matrix{S}
    matrices::Matrix{Union{Nothing,T}}
    factory::F
    native_accessed::Base.RefValue{Bool}
end

Base.size(storage::ScalarBStorage) = size(storage.phases)
_uses_scalar_storage(::AbstractMatrix) = false
_uses_scalar_storage(storage::ScalarBStorage) = all(isnothing, storage.matrices)

function _expand_Bplane!(out, storage::ScalarBStorage, mu, nu)
    s = storage.phases[min(mu, nu), max(mu, nu)]
    LatticeMatrices.substitute!(out.U, s)
    mu > nu && mul!(out.U, -one(eltype(out.U.A)))
    set_wing_U!(out)
    return out
end

function Base.getindex(storage::ScalarBStorage, mu::Int, nu::Int)
    checkbounds(storage, mu, nu)
    mu == nu && throw(ArgumentError("B has no diagonal plane"))
    existing = storage.matrices[mu, nu]
    if isnothing(existing)
        storage.native_accessed[] && throw(ArgumentError(
            "native scalar references have been exposed; create a matrix-storage B and copy into it for matrix access"))
        existing = _expand_Bplane!(storage.factory(), storage, mu, nu)
        storage.matrices[mu, nu] = existing
    end
    return existing
end

function Base.setindex!(storage::ScalarBStorage{T}, value::T, mu::Int, nu::Int) where {T}
    checkbounds(storage, mu, nu)
    mu == nu && throw(ArgumentError("B has no diagonal plane"))
    storage.native_accessed[] && throw(ArgumentError("cannot replace a plane after exposing native scalar references"))
    storage.matrices[mu, nu] = value
    return value
end

"""
    get_Bplane(B, mu, nu)

Return the live `ScaledIdentityLattice` for an independent B plane
(`mu < nu`). This does not expand B into color matrices. Its `.scalar` is
the 1×1 backing field; mutate it through LatticeMatrices operations (or mark
its halo dirty after direct array writes).
Not available after a matrix has escaped through `B[mu,nu]`; use a fresh
scalar-storage B field in that case. Matrix access and scalar access must not
be mixed when retaining mutable references.
"""
function get_Bplane(B::Bfield, mu, nu)
    _uses_scalar_storage(B.u) || throw(ArgumentError(
        "native B-plane access requires scalar storage with no materialized matrix planes"))
    1 <= mu < nu <= size(B.u, 1) || throw(ArgumentError("require 1 <= mu < nu <= Dim"))
    B.u.native_accessed[] = true
    return B.u.phases[mu, nu]
end

"""Return `get_Bplane(B, mu, nu).scalar`, the live 1×1 coefficient field."""
get_Bphase(B::Bfield, mu, nu) = get_Bplane(B, mu, nu).scalar

_center_Bplane(B::Bfield, mu, nu) =
    _uses_scalar_storage(B.u) ? B.u.phases[mu, nu] : B[mu, nu]

function _initialize_scalar_Bfields(NC, Flux, NDW, NN...;
    condition, PEs, verbose_level, tloop_pos, tloop_dir, tloop_dis,
    singleprecision, boundarycondition, elementtype)
    length(NN) == 4 || throw(ArgumentError("scalar B initialization currently supports 4D"))
    condition in ("tflux", "tloop") || throw(ArgumentError(
        "scalar B initialization supports tflux and tloop, not $condition"))
    length(Flux) == 6 || throw(ArgumentError("Flux must contain the six plane fluxes"))
    # Only a transient prototype is needed for backend/type/communicator setup.
    prototype = Gaugefields_4D_MPILattice(NC, NN...; NDW, PEs,
        verbose_level, singleprecision, boundarycondition, elementtype)
    comm, grid = prototype.U.comm, prototype.U.dims
    dtype = eltype(prototype.U.A)
    boundarycondition = collect(prototype.U.phases)
    factory = () -> Gaugefields_4D_MPILattice(NC, NN...; NDW, PEs=grid,
        verbose_level, boundarycondition, elementtype=dtype, comm)
    scalars = map(enumerate(((1,2), (1,3), (1,4), (2,3), (2,4), (3,4)))) do item
        k, _ = item
        host = if condition == "tflux"
            thooftFlux_4D_B_at_bndry(1, Flux[k], k, NN...; phase_NC=NC, verbose_level=0)
        else
            thooftLoop_4D_B_temporal(1, Flux[k], k, NN...;
                phase_NC=NC, verbose_level=0, tloop_pos, tloop_dir, tloop_dis)
        end
        scalar = LatticeMatrices.LatticeMatrix(dtype.(host.U), 4, grid;
            nw=NDW, phases=boundarycondition, comm0=comm)
        LatticeMatrices.ScaledIdentityLattice(NC, scalar)
    end
    planes = Matrix{typeof(first(scalars))}(undef, 4, 4)
    for (s, (mu, nu)) in zip(scalars, ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4)))
        planes[mu, nu] = s
    end
    T = typeof(prototype)
    matrices = Matrix{Union{Nothing,T}}(nothing, 4, 4)
    storage = ScalarBStorage{T,eltype(planes),typeof(factory)}(planes, matrices, factory, Ref(false))
    return Bfield(storage; center_valued=true, use_path_cache=true)
end

function Base.similar(B::Bfield{T,Dim,S}) where {T,Dim,S<:ScalarBStorage}
    if !_uses_scalar_storage(B.u)
        output = Matrix{T}(undef, Dim, Dim)
        for mu in 1:Dim, nu in 1:Dim
            mu == nu && continue
            output[mu, nu] = similar(B[mu, nu])
        end
        return Bfield(output; center_valued=false, use_path_cache=B.use_path_cache)
    end
    phases = similar(B.u.phases)
    for mu in 1:Dim, nu in (mu+1):Dim
        phases[mu, nu] = similar(B.u.phases[mu, nu])
    end
    matrices = Matrix{Union{Nothing,T}}(nothing, Dim, Dim)
    storage = S(phases, matrices, B.u.factory, Ref(false))
    return Bfield(storage; center_valued=B.center_valued, use_path_cache=B.use_path_cache)
end

function _substitute_Bstorage!(a::AbstractMatrix, b::AbstractMatrix, parity...)
    size(a) == size(b) || throw(DimensionMismatch("B-field matrices must have the same size"))
    a === b && return nothing
    if _uses_scalar_storage(a) && _uses_scalar_storage(b)
        for mu in axes(a, 1), nu in (mu+1):size(a, 2)
            if isempty(parity)
                LatticeMatrices.substitute!(a.phases[mu, nu], b.phases[mu, nu])
            else
                # Match the existing gauge-field parity-copy convention.
                LatticeMatrices.clear_matrix!(a.phases[mu, nu].scalar, only(parity))
                LatticeMatrices.add_matrix_evenodd!(
                    a.phases[mu, nu].scalar, b.phases[mu, nu].scalar, only(parity))
            end
        end
    else
        for mu in axes(a, 1), nu in axes(a, 2)
            mu == nu && continue
            if _uses_scalar_storage(b)
                tmp = _expand_Bplane!(b.factory(), b, mu, nu)
                substitute_U!(a[mu, nu], tmp, parity...)
            else
                substitute_U!(a[mu, nu], b[mu, nu], parity...)
            end
        end
    end
    return nothing
end
