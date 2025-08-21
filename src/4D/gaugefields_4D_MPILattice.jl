import ..MPILattice: LatticeMatrix,
    Shifted_Lattice,
    Adjoint_Lattice,
    TALattice,
    makeidentity_matrix!,
    set_halo!,
    substitute!

abstract type Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT} <: Gaugefields_4D{NC} end

struct Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}
    U::LatticeMatrix{4,T,AT,NC,NC}
    mpi::Bool
    verbose_print::Verbose_print
    singleprecision::Bool
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64


    function Gaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
        NDW=1, singleprecision=false,
        boundarycondition=ones(4),
        PEs=nothing,
        comm=MPI.COMM_WORLD,
        #mpiinit=false,
        verbose_level=2
    )
        if MPI.Initialized() == false
            MPI.Init()
            mpiinit = true
        end

        #if mpiinit == false
        #    MPI.Init()
        #    mpiinit = true
        #end
        comm0 = comm

        gsize = (NX, NY, NZ, NT)
        dim = 4
        nw = NDW
        @assert NDW > 0 "NDW should be larger than 0. We use a halo area."
        elementtype = ifelse(singleprecision, ComplexF32, ComplexF64)
        phases = boundarycondition
        nprocs = MPI.Comm_size(comm)
        if isnothing(PEs)
            PEs_in = (1, 1, 1, nprocs)
        else
            PEs_in = deepcopy(PEs)
        end

        @assert NX > PEs_in[1] "PEs[1] is larger than NX. Now NX = $NX and PEs = $PEs_in"
        @assert NY > PEs_in[2] "PEs[2] is larger than NY. Now NX = $NY and PEs = $PEs_in"
        @assert NZ > PEs_in[3] "PEs[3] is larger than NZ. Now NX = $NZ and PEs = $PEs_in"
        @assert NT > PEs_in[4] "PEs[4] is larger than NT. Now NX = $NT and PEs = $PEs_in"

        @assert NX % PEs_in[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs_in"
        @assert NY % PEs_in[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs_in"
        @assert NZ % PEs_in[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs_in"
        @assert NT % PEs_in[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs_in"

        @assert prod(PEs_in) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        verbose_print = Verbose_print(verbose_level, myid=myrank)


        U = LatticeMatrix(NC, NC, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        T = elementtype
        AT = typeof(U.A)

        mpi = true

        NV = NX * NY * NZ * NT

        return new{NC,NX,NY,NZ,NT,T,AT}(
            U, mpi, verbose_print, singleprecision,
            NX,
            NY,
            NZ,
            NT,
            NDW,
            NV,
            NC)

        #LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)
    end
end





struct TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}
    U::TALattice{4,T,AT,NC}
end



function identityGaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
    NDW=1,
    verbose_level=2,
    singleprecision=false,
    boundarycondition=ones(4),
    PEs=nothing,
    comm=MPI.COMM_WORLD,
    #mpiinit=false
)


    U = Gaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
        NDW,
        singleprecision,
        boundarycondition,
        PEs,
        comm,
        verbose_level
    )

    makeidentity_matrix!(U.U)
    return U

end

function set_wing_U!(u::Array{Gaugefields_4D_MPILattice{NC},1}) where {NC}
    for i = 1:length(u)
        set_halo!(u.U)
    end
    return
end

function set_wing_U!(u::Gaugefields_4D_MPILattice{NC}) where {NC}
    set_halo!(u.U)
    return
end

function substitute_U!(a::Gaugefields_4D_MPILattice, b::Gaugefields_4D_MPILattice)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_MPILattice,T2<:Gaugefields_4D_MPILattice}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_nowing,T2<:Gaugefields_4D_MPILattice}

    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end



function substitute_U!(A::Gaugefields_4D_nowing, B::Gaugefields_4D_MPILattice)
    tempmatrix = gather_and_bcast_matrix(B.U)
    A.U .= tempmatrix
end


function substitute_U!(A::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT},
    B::Gaugefields_4D_nowing{NC}) where {NC,NX,NY,NZ,NT,T,AT}

    dim = 4
    PEs = A.U.dims
    phases = A.U.phases
    nw = A.U.nw
    comm0 = A.U.comm

    tempU = LatticeMatrix(B.U, dim, PEs;
        nw,
        phases,
        comm0)
    substitute!(A.U, tempU)

end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_MPILattice,T2<:Gaugefields_4D_nowing}

    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function ges_PEs(U::Gaugefields_4D_MPILattice)
    return U.U.dims
end

function Base.similar(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}) where {NC,NX,NY,NZ,NT,T,AT}
    NDW = U.U.nw
    boundarycondition = U.U.phases
    PEs = ges_PEs(U)
    comm = U.U.comm

    Uout = Gaugefields_4D_MPILattice(
        NC, NX, NY, NZ, NT;
        NDW,
        U.singleprecision,
        boundarycondition,
        PEs,
        comm,
        verbose_level=U.verbose_print.level
    )
    #identityGaugefields_4D_nowing(U.NC,U.NX,U.NY,U.NZ,U.NT,U.NDW)
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_4D_MPILattice}
    Uout = Array{T,1}(undef, 4)
    for μ = 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
end

struct Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,shift} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}
    U::Shifted_Lattice{LatticeMatrix{4,T,AT,NC,NC},shift}

    function Shifted_Gaugefields_4D_MPILattice(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}, shift) where {NC,NX,NY,NZ,NT,T,AT}
        sU = Shifted_Lattice{typeof(U.U),shift}(U.U)
        return new{NC,NX,NY,NZ,NT,T,AT,shift}(sU)
    end
end

function shift_U(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}, ν::Ts) where {Ts<:Integer,T,NC,NX,NY,NZ,NT,AT}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return shift_U(U, shift)
end


function shift_U(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}, shift::NTuple{4,Ts}) where {NC,NX,NY,NZ,NT,T,AT,Ts<:Int}
    return Shifted_Gaugefields_4D_MPILattice(U, shift)
end



struct Adjoint_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}
    U::Adjoint_Lattice{LatticeMatrix{4,T,AT,NC,NC}}
end


function Base.adjoint(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}) where {NC,NX,NY,NZ,NT,T,AT}
    Adjoint_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}(U.U')
end


struct Adjoint_Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,shift} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}
    U::Adjoint_Lattice{Shifted_Lattice{LatticeMatrix{4,T,AT,NC,NC},shift}}
end

function Base.adjoint(U::Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,shift}) where {NC,NX,NY,NZ,NT,T,AT,shift}
    Adjoint_Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,shift}(U.U')
end

function LinearAlgebra.mul!(
    c::T,
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {T<:Gaugefields_4D_MPILattice,T1<:Fields_4D_MPILattice,T2<:Fields_4D_MPILattice,Ta<:Number,Tb<:Number}
    mul!(c.U, a.U, b.U, α, β)
end

function LinearAlgebra.tr(a::Gaugefields_4D_MPILattice)
    tr(a.U)
end
