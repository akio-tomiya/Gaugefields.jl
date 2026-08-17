abstract type Adjoint_rep_Gaugefields{NC,Dim,NumofBasis} <: AbstractGaugefields{NC,Dim} #Traceless antihermitian matrix
end

include("./4D/deprecated/Adjoint_rep/Adjoint_rep_gaugefields_4D.jl")

function construct_Adjoint_rep_Gaugefields(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
) where {NC,Dim}
    F = Array{Adjoint_rep_Gaugefields{NC,Dim},1}(undef, Dim)
    for μ = 1:Dim
        F[μ] = construct_Adjoint_rep_Gaugefields(U[μ])
    end
    return F
end

function construct_Adjoint_rep_Gaugefields(
    u::AbstractGaugefields{NC,Dim};
    verbose_level=2,
) where {NC,Dim}
    mpi = u.mpi
    if mpi
        error("MPI is not supported")
        if Dim == 4
            return construct_Adjoint_rep_Gaugefields_mpi(u)
        elseif Dim == 2
            error("Dim = $Dim is not supoorted")
        else
            error("Dim = $Dim is not supoorted")
        end

    else
        if Dim == 4
            return construct_Adjoint_rep_Gaugefields_4D_wing(
                u,
                verbose_level=u.verbose_print.level,
            )
        elseif Dim == 2
            error("Dim = $Dim is not supoorted")
        else
            error("Dim = $Dim is not supoorted")
        end


    end
end

function construct_Adjoint_rep_Gaugefields(
    u::Gaugefields_4D_MPILattice{NC};
    verbose_level=u.verbose_print.level,
) where {NC}
    get_nprocs(u) == 1 || error("MPI is not supported")
    legacy = Gaugefields_4D_nowing(
        NC,
        u.NX,
        u.NY,
        u.NZ,
        u.NT;
        verbose_level,
    )
    substitute_U!(legacy, u)
    return construct_Adjoint_rep_Gaugefields_4D_wing(legacy; verbose_level)
end
