
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
        nw = 0
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
end
