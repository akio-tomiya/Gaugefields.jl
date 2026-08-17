import ..SUN_generator: Generator, lie2matrix!, matrix2lie!
import LatticeMatrices: delinearize

abstract type TA_Gaugefields{NC,Dim} <: AbstractGaugefields{NC,Dim} #Traceless antihermitian matrix
end

@inline function _kernel_sun_lie2matrix!(
    i,
    output,
    coefficients,
    NC,
    indexer,
    output_nw,
    coefficient_nw,
)
    output_indices = delinearize(indexer, i, output_nw)
    coefficient_indices = delinearize(indexer, i, coefficient_nw)

    for jc = 1:NC
        for ic = 1:NC
            output[ic, jc, output_indices...] = zero(eltype(output))
        end
    end

    basis = 1
    for ic = 1:(NC-1)
        for jc = (ic+1):NC
            symmetric = coefficients[basis, 1, coefficient_indices...]
            antisymmetric = coefficients[basis+1, 1, coefficient_indices...]
            value = (antisymmetric + im * symmetric) / 2
            output[ic, jc, output_indices...] = value
            output[jc, ic, output_indices...] = -conj(value)
            basis += 2
        end
    end

    for diagonal = 1:(NC-1)
        normalization = sqrt(diagonal * (diagonal + 1) / 2)
        value = im * coefficients[basis, 1, coefficient_indices...] /
                (2 * normalization)
        for ic = 1:diagonal
            output[ic, ic, output_indices...] += value
        end
        output[diagonal+1, diagonal+1, output_indices...] -= diagonal * value
        basis += 1
    end
    return nothing
end

function _sun_lie2matrix!(output, coefficients, NC)
    mark_lattice_dirty!(output.U)
    JACC.parallel_for(
        prod(output.U.PN),
        _kernel_sun_lie2matrix!,
        output.U.A,
        coefficients.a.A,
        NC,
        output.U.indexer,
        output.U.nw,
        coefficients.a.nw,
    )
    return nothing
end

@inline function _kernel_sun_traceless_antihermitian_add!(
    i,
    output,
    input,
    factor,
    NC,
    indexer,
    output_nw,
    input_nw,
)
    output_indices = delinearize(indexer, i, output_nw)
    input_indices = delinearize(indexer, i, input_nw)

    basis = 1
    for ic = 1:(NC-1)
        for jc = (ic+1):NC
            value = (
                input[ic, jc, input_indices...] -
                conj(input[jc, ic, input_indices...])
            ) / 2
            output[basis, 1, output_indices...] += factor * 2 * imag(value)
            output[basis+1, 1, output_indices...] += factor * 2 * real(value)
            basis += 2
        end
    end

    for diagonal = 1:(NC-1)
        diagonal_sum = zero(real(input[1, 1, input_indices...]))
        for ic = 1:diagonal
            diagonal_sum += imag(input[ic, ic, input_indices...])
        end
        diagonal_sum -= diagonal * imag(
            input[diagonal+1, diagonal+1, input_indices...],
        )
        normalization = sqrt(diagonal * (diagonal + 1) / 2)
        output[basis, 1, output_indices...] +=
            factor * diagonal_sum / normalization
        basis += 1
    end
    return nothing
end

function _sun_traceless_antihermitian_add!(output, factor, input, NC)
    mark_lattice_dirty!(output.a)
    JACC.parallel_for(
        prod(input.U.PN),
        _kernel_sun_traceless_antihermitian_add!,
        output.a.A,
        input.U.A,
        factor,
        NC,
        input.U.indexer,
        output.a.nw,
        input.U.nw,
    )
    return nothing
end

include("./4D/TA_gaugefields_4D.jl")
include("./2D/TA_gaugefields_2D.jl")
include("./3D/TA_gaugefields_3D.jl")

function Base.:*(
    x::Array{<:TA_Gaugefields{NC,Dim},1},
    y::Array{<:TA_Gaugefields{NC,Dim},1},
) where {NC,Dim}
    s = 0.0
    for μ = 1:Dim
        s += x[μ] * y[μ]
    end

    return s
end


function initialize_TA_Gaugefields(U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    F1 = initialize_TA_Gaugefields(U[1])
    F = Array{typeof(F1),1}(undef, Dim)
    F[1] = F1
    for μ = 2:Dim
        F[μ] = initialize_TA_Gaugefields(U[μ])
    end
    return F
end


function initialize_TA_Gaugefields(u::AbstractGaugefields{NC,Dim}) where {NC,Dim}
    #println(typeof(u))
    if typeof(u) <: Gaugefields_4D_MPILattice
        if Dim == 4
            return TA_Gaugefields_4D_MPILattice(u)
        else
            error("Dim = $Dim is not supoorted")
        end
    elseif typeof(u) <: Gaugefields_3D_MPILattice
        if Dim == 3
            return TA_Gaugefields_3D_MPILattice(u)
        else
            error("Dim = $Dim is not supoorted")
        end
    elseif typeof(u) <: Gaugefields_2D_MPILattice
        if Dim == 2
            return TA_Gaugefields_2D_MPILattice(u)
        else
            error("Dim = $Dim is not supoorted")
        end
    else
        mpi = u.mpi
        if mpi
            if Dim == 4
                return TA_Gaugefields_4D_mpi(u)
            elseif Dim == 2
                return TA_Gaugefields_2D_mpi(u)
                #error("Dim = $Dim is not supoorted")

            else
                error("Dim = $Dim is not supoorted")
            end
        else
            if Dim == 4
                return TA_Gaugefields(NC, u.NX, u.NY, u.NZ, u.NT)
            elseif Dim == 2
                return TA_Gaugefields(NC, u.NX, u.NT)
            elseif Dim == 3
                return TA_Gaugefields(NC, u.NX, u.NY, u.NT)
            else
                error("Dim = $Dim is not supoorted")
            end
        end
    end
end

#function gauss_distribution!(p::T) where T<: TA_Gaugefields
#    error("gauss_distribution!(p) is not implemented in type $(typeof(p)) ")
#end

function gauss_distribution!(p::T; σ=1.0) where {T<:TA_Gaugefields}
    error("gauss_distribution!(p) is not implemented in type $(typeof(p)) ")
end


function gauss_distribution!(p::Array{<:TA_Gaugefields{NC,Dim},1}; σ=1.0) where {NC,Dim}
    for μ = 1:Dim
        gauss_distribution!(p[μ], σ=σ)
    end
end

"""
    gauss_distribution!(p; σ=1, seed=nothing, sweep=0,
                        rng_algorithm=Philox4x32())

Fill an array of `LatticeMatrices`-backed conjugate momenta with Gaussian
coefficients. One seed is shared over the MPI communicator, while direction,
sweep, and global site select independent streams. Supplying `seed` therefore
makes the global result reproducible across MPI decompositions.
"""
function gauss_distribution!(
    p::Array{T,1};
    σ=1.0,
    seed=nothing,
    sweep::Integer=0,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
) where {
    T<:Union{
        TA_Gaugefields_2D_MPILattice,
        TA_Gaugefields_3D_MPILattice,
        TA_Gaugefields_4D_MPILattice,
    }
}
    isempty(p) && return nothing
    shared_seed = _shared_mpialattice_seed(seed, p[1].a.comm)
    for μ in eachindex(p)
        _gauss_distribution_mpialattice!(
            p[μ],
            shared_seed;
            σ,
            sweep,
            direction=μ,
            rng_algorithm,
        )
    end
    return nothing
end



function Base.setindex!(x::T, v, i...) where {T<:TA_Gaugefields}
    error("setindex! is not implemented in type $(typeof(x)) ")
    x.a[i...] = v
end

function Base.getindex(x::T, i...) where {T<:TA_Gaugefields}
    error("setindex! is not implemented in type $(typeof(x)) ")
    return x.a[i...]
end

function Base.similar(U::T) where {T<:TA_Gaugefields}
    error("similar! is not implemented in type $(typeof(U)) ")
end

function Traceless_antihermitian_add!(U::T, factor, temp1) where {T<:TA_Gaugefields}
    error("Traceless_antihermitian_add! is not implemented in type $(typeof(U)) ")
end

function Traceless_antihermitian!(vout::T, vin::T) where {T<:TA_Gaugefields}
    error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
end
