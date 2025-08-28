


struct TA_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NumofBasis} <: TA_Gaugefields_2D{NC}
    a::LatticeMatrix{2,T,AT,NumofBasis,1}
    NX::Int64
    NY::Int64
    #NZ::Int64
    #NT::Int64
    NV::Int64
    NC::Int64
    NumofBasis::Int64
    generators::Union{Nothing,Generator}


    function TA_Gaugefields_2D_MPILattice(u::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NDW},
    ) where {NC,NX,NY,T,AT,NDW}

        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        elementtype = ifelse(u.singleprecision, Float32, Float64)
        nw = 1
        gsize = (NX, NY)
        dim = 2
        phases = u.U.phases
        PEs_in = u.U.dims
        comm0 = u.U.comm

        a = LatticeMatrix(NumofBasis, 1, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        Ta = elementtype
        ATa = typeof(a.A)

        return new{NC,NX,NY,Ta,ATa,NumofBasis}(
            a,
            NX,
            NY,
            #NZ,
            #NT,
            u.NV,
            NC,
            NumofBasis,
            generators)



    end
end


function Base.:*(
    x::TA_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NumofBasis},
    y::TA_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NumofBasis},
) where {NC,NX,NY,T,AT,NumofBasis}

    s = dot(x.a, y.a)

    return s
end

function gauss_distribution!(
    p::TA_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NumofBasis};
    σ=1.0,
) where {NC,NX,NY,T,AT,NumofBasis}
    d = Normal(0.0, σ)
    pwork = rand(d, NumofBasis, 1, NX, NY)
    PEs = get_PEs(p.a)

    a = LatticeMatrix(pwork, 2, PEs; nw=1, phases=p.a.phases, comm0=p.a.comm)
    substitute!(p.a, a)
end


function exptU!(
    uout::Tg,
    t::N,
    v::TA_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NumofBasis},
    temps::Array{Tg,1},
) where {NC,NX,NY,T,AT,NumofBasis,Tg<:Gaugefields_2D_MPILattice,N<:Number} #uout = exp(t*u)

    if NC > 3
        Uta = temps[1]
        substitute_U!(Uta, v)
        error("not implemented for NC > 3")
    else
        expt!(uout.U, v.a, t)
    end
    set_wing_U!(uout)
end

function substitute_U!(C::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NDW},
    A::TA_Gaugefields_2D_MPILattice{NC,NX,NY,T1,AT1,NumofBasis}) where {NC,NX,NY,T,AT,NDW,NumofBasis,AT1,T1}
    @assert NC > 3 "Only NC >3 is supported"
    generators = Tuple(JACC.array.(A.generators.generator))


    JACC.parallel_for(prod(C.U.PN), kernel_lie2matrix_2D!,
        uout.U, u.a, NC, NG, C.U.PN, generators) #w,u,ww,t
end

function kernel_lie2matrix_2D!(i, uout, u, NC, NG, PN, generators, nw1, nw2)
    ix, iy = get_2Dindex(i, PN)

    a = view(u, :, ix + nw2, iy + nw2)
    u0 = view(uout, :, :, i)
    lie2matrix_tuple!(u0, a, NG, generators, NC)
end





function Traceless_antihermitian_add!(
    c::TA_Gaugefields_2D_MPILattice{NC,NX,NY,T1,AT1,NumofBasis},
    factor,
    vin::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NDW},
) where {NC,NX,NY,T,AT,NDW,NumofBasis,AT1,T1}
    traceless_antihermitian_add!(c.a, factor, vin.U)
end

function clear_U!(c::TA_Gaugefields_2D_MPILattice)
    clear_matrix!(c.a)
end

function add_U!(c::TA_Gaugefields_2D_MPILattice, t::T, a::T1) where {T1<:TA_Gaugefields_2D_MPILattice,T<:Number}
    add_matrix!(c.a, a.a, t)
end