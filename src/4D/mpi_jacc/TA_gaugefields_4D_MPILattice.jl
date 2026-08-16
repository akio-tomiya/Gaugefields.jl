


struct TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis} <: TA_Gaugefields_4D{NC}
    a::LatticeMatrix{4,T,AT,NumofBasis,1}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    NC::Int64
    NumofBasis::Int64
    generators::Union{Nothing,Generator}


    function TA_Gaugefields_4D_MPILattice(u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW},
    ) where {NC,NX,NY,NZ,NT,T,AT,NDW}

        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        elementtype = ifelse(u.singleprecision, Float32, Float64)
        nw = 1
        gsize = (NX, NY, NZ, NT)
        dim = 4
        phases = u.U.phases
        PEs_in = u.U.dims
        comm0 = u.U.comm

        a = LatticeMatrix(NumofBasis, 1, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        Ta = elementtype
        ATa = typeof(a.A)

        return new{NC,NX,NY,NZ,NT,Ta,ATa,NumofBasis}(
            a,
            NX,
            NY,
            NZ,
            NT,
            u.NV,
            NC,
            NumofBasis,
            generators)



    end

    function TA_Gaugefields_4D_MPILattice(
        u::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis},
    ) where {NC,NX,NY,NZ,NT,T,AT,NumofBasis}
        a = similar(u.a)
        ATa = typeof(a.A)

        return new{NC,NX,NY,NZ,NT,T,ATa,NumofBasis}(
            a,
            u.NX,
            u.NY,
            u.NZ,
            u.NT,
            u.NV,
            u.NC,
            u.NumofBasis,
            u.generators,
        )
    end
end

Base.similar(u::TA_Gaugefields_4D_MPILattice) = TA_Gaugefields_4D_MPILattice(u)

@inline function kernel_getindex_TA_4D_MPILattice!(
    _, output, input, basis, ix, iy, iz, it,
)
    @inbounds output[1] = input[basis, 1, ix, iy, iz, it]
    return nothing
end

@inline function kernel_setindex_TA_4D_MPILattice!(
    _, output, value, basis, ix, iy, iz, it,
)
    @inbounds output[basis, 1, ix, iy, iz, it] = value
    return nothing
end

function Base.getindex(
    x::TA_Gaugefields_4D_MPILattice,
    basis,
    ix,
    iy,
    iz,
    it,
)
    indices = (basis, ix + x.a.nw, iy + x.a.nw, iz + x.a.nw, it + x.a.nw)
    if x.a.A isa Array
        @inbounds return x.a.A[basis, 1, indices[2:end]...]
    end
    output = JACC.zeros(eltype(x.a.A), 1)
    JACC.parallel_for(
        1,
        kernel_getindex_TA_4D_MPILattice!,
        output,
        x.a.A,
        indices...,
    )
    return JACC.to_host(output)[1]
end

function Base.setindex!(
    x::TA_Gaugefields_4D_MPILattice,
    value,
    basis,
    ix,
    iy,
    iz,
    it,
)
    indices = (basis, ix + x.a.nw, iy + x.a.nw, iz + x.a.nw, it + x.a.nw)
    if x.a.A isa Array
        @inbounds x.a.A[basis, 1, indices[2:end]...] = value
    else
        JACC.parallel_for(
            1,
            kernel_setindex_TA_4D_MPILattice!,
            x.a.A,
            convert(eltype(x.a.A), value),
            indices...,
        )
    end
    mark_lattice_dirty!(x.a)
    return value
end

get_myrank(U::TA_Gaugefields_4D_MPILattice) = MPI.Comm_rank(U.a.comm)
get_nprocs(U::TA_Gaugefields_4D_MPILattice) = MPI.Comm_size(U.a.comm)

function barrier(U::TA_Gaugefields_4D_MPILattice)
    MPI.Barrier(U.a.comm)
    return nothing
end


function Base.:*(
    x::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis},
    y::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis},
) where {NC,NX,NY,NZ,NT,T,AT,NumofBasis}

    s = dot(x.a, y.a)

    return s
end

function gauss_distribution!(
    p::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis};
    σ=1.0,
    seed=nothing,
    sweep::Integer=0,
    direction::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where {NC,NX,NY,NZ,NT,T,AT,NumofBasis}
    shared_seed = _shared_mpialattice_seed(seed, p.a.comm)
    return _gauss_distribution_mpialattice!(
        p,
        shared_seed;
        σ,
        sweep,
        direction,
        rng_algorithm,
    )
end

function _gauss_distribution_mpialattice!(
    p::TA_Gaugefields_4D_MPILattice,
    shared_seed::UInt64;
    σ=1.0,
    sweep::Integer=0,
    direction::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
)
    key = RNGStreamKey(
        shared_seed,
        sweep,
        direction,
        0,
        _GAUSSIAN_MOMENTUM_STREAM_TAG,
    )
    randomize_gaussian_matrix!(p.a, key; sigma=σ, rng_algorithm)
    return nothing
end


function exptU!(
    uout::Tg,
    t::N,
    v::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis},
    temps::Array{Tg,1},
) where {NC,NX,NY,NZ,NT,T,AT,NumofBasis,Tg<:Gaugefields_4D_MPILattice,N<:Number} #uout = exp(t*u)

    if NC > 3
        Uta = temps[1]
        substitute_U!(Uta, v)
        expt!(uout.U, Uta.U, t)
        set_wing_U!(uout)
    else
        expt!(uout.U, v.a, t)
    end
    #if NC > 3
    #    Uta = temps[1]
    #    substitute_U!(Uta, v)
    #    error("not implemented for NC > 3")
    #else
    #    expt!(uout.U, v.a, t)
    #end
    #set_wing_U!(uout)
end

function substitute_U!(C::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW},
    A::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T1,AT1,NumofBasis}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NumofBasis,AT1,T1}
    @assert NC > 3 "Only NC >3 is supported"
    return _sun_lie2matrix!(C, A, NC)
end

function kernel_lie2matrix!(i, uout, u, NC, NG, dindexer, generators, nw1, nw2)
    indices = delinearize(dindexer, i, nw1)
    indices2 = delinearize(dindexer, i, nw2)

    for jc = 1:NC
        for ic = 1:NC
            value = u[1, 1, indices2...] * generators[1][ic, jc] * (im / 2)
            for ibasis = 2:NG
                value += u[ibasis, 1, indices2...] * generators[ibasis][ic, jc] * (im / 2)
            end
            uout[ic, jc, indices...] = value
        end
    end


    #a = view(u, :, indices2...)
    #u0 = view(uout, :, :, i)
    #lie2matrix_tuple!(u0, a, NG, generators, NC)
end


function lie2matrix_tuple!(matrix, a, NG, generators, NC)
    matrix .= 0
    for i = 1:NG
        for jc = 1:NC
            for ic = 1:NC
                matrix[ic, jc] += a[i] * generators[i][ic, jc] * (im / 2)
            end
        end
    end
    return
end




function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T1,AT1,NumofBasis},
    factor,
    vin::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW},
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NumofBasis,AT1,T1}
    if NC > 3
        _sun_traceless_antihermitian_add!(c, factor, vin, NC)
    else
        traceless_antihermitian_add!(c.a, factor, vin.U)
    end
    return nothing
end

function Traceless_antihermitian!(
    c::TA_Gaugefields_4D_MPILattice,
    vin::Gaugefields_4D_MPILattice,
)
    clear_matrix!(c.a)
    Traceless_antihermitian_add!(c, 1, vin)
    return nothing
end

function clear_U!(c::TA_Gaugefields_4D_MPILattice)
    clear_matrix!(c.a)
end

function add_U!(c::TA_Gaugefields_4D_MPILattice, t::T, a::T1) where {T1<:TA_Gaugefields_4D_MPILattice,T<:Number}
    add_matrix!(c.a, a.a, t)
end
