using LinearAlgebra: I, det, norm, tr

const G2_FUNDAMENTAL_DIM = 7
const G2_ALGEBRA_DIM = 14
const G2_DEFAULT_ATOL = 1.0e-10

"""
    G2Basis{T}

Real seven-dimensional G2 Lie-algebra basis. The generators are real
antisymmetric matrices normalized by `-2tr(A[a] * A[b]) == delta_ab`.
"""
struct G2Basis{T<:AbstractFloat}
    generators::NTuple{G2_ALGEBRA_DIM,Matrix{T}}
end

Base.length(::G2Basis) = G2_ALGEBRA_DIM
Base.getindex(basis::G2Basis, index::Integer) = basis.generators[index]

"""
    G2AlgebraElement{T}

Fourteen real coefficients in the basis returned by `g2_basis(T)`.
"""
struct G2AlgebraElement{T<:AbstractFloat}
    coefficients::Vector{T}

    function G2AlgebraElement{T}(coefficients::AbstractVector{<:Real}) where {T<:AbstractFloat}
        length(coefficients) == G2_ALGEBRA_DIM ||
            throw(ArgumentError("G2 coefficient vector must have length $G2_ALGEBRA_DIM"))
        return new{T}(T.(collect(coefficients)))
    end
end

G2AlgebraElement(coefficients::AbstractVector{<:Real}) =
    G2AlgebraElement{Float64}(coefficients)

coefficients(element::G2AlgebraElement) = element.coefficients
Base.length(::G2AlgebraElement) = G2_ALGEBRA_DIM
Base.getindex(element::G2AlgebraElement, index::Integer) = element.coefficients[index]

const _G2_BASIS_FLOAT64 = Ref{Union{Nothing,G2Basis{Float64}}}(nothing)

function _matrix_unit(i::Int, j::Int)
    matrix = zeros(ComplexF64, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    matrix[i, j] = 1
    return matrix
end

function _complex_g2_generators()
    s2 = sqrt(2.0)
    invsqrt8 = 1.0 / sqrt(8.0)
    invsqrt24 = 1.0 / sqrt(24.0)
    P = _matrix_unit

    t1 = invsqrt8 * (P(1, 2) + P(2, 1) - P(5, 6) - P(6, 5))
    t2 = im * invsqrt8 * (P(2, 1) - P(1, 2) - P(5, 6) + P(6, 5))
    t3 = invsqrt8 * (P(1, 1) - P(2, 2) - P(5, 5) + P(6, 6))
    t4 = invsqrt8 * (P(1, 3) + P(3, 1) - P(5, 7) - P(7, 5))
    t5 = im * invsqrt8 * (P(3, 1) - P(1, 3) - P(5, 7) + P(7, 5))
    t6 = invsqrt8 * (P(2, 3) + P(3, 2) - P(6, 7) - P(7, 6))
    t7 = im * invsqrt8 * (P(3, 2) - P(2, 3) - P(6, 7) + P(7, 6))
    t8 = invsqrt24 * (
        P(1, 1) + P(2, 2) - 2 * P(3, 3) - P(5, 5) - P(6, 6) + 2 * P(7, 7)
    )
    t9 = invsqrt24 * (
        P(1, 6) - P(2, 5) + s2 * P(3, 4) + s2 * P(4, 3) -
        s2 * P(4, 7) - P(5, 2) + P(6, 1) - s2 * P(7, 4)
    )
    t10 = im * invsqrt24 * (
        P(2, 5) - P(1, 6) + s2 * P(3, 4) - s2 * P(4, 3) -
        s2 * P(4, 7) - P(5, 2) + P(6, 1) + s2 * P(7, 4)
    )
    t11 = invsqrt24 * (
        s2 * P(2, 4) - P(1, 7) + P(3, 5) + s2 * P(4, 2) -
        s2 * P(4, 6) + P(5, 3) - s2 * P(6, 4) - P(7, 1)
    )
    t12 = im * invsqrt24 * (
        P(3, 5) - P(1, 7) - s2 * P(2, 4) + s2 * P(4, 2) +
        s2 * P(4, 6) - P(5, 3) - s2 * P(6, 4) + P(7, 1)
    )
    t13 = invsqrt24 * (
        s2 * P(1, 4) + P(2, 7) - P(3, 6) + s2 * P(4, 1) -
        s2 * P(4, 5) - s2 * P(5, 4) - P(6, 3) + P(7, 2)
    )
    t14 = im * invsqrt24 * (
        s2 * P(1, 4) - P(2, 7) + P(3, 6) - s2 * P(4, 1) -
        s2 * P(4, 5) + s2 * P(5, 4) - P(6, 3) + P(7, 2)
    )

    return (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14)
end

function _basis_change_matrix()
    B = zeros(ComplexF64, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    invsqrt2 = 1.0 / sqrt(2.0)
    invsqrt2_over_i = invsqrt2 / im
    for k in 1:3
        B[k, 2 * k - 1] = invsqrt2
        B[k + 4, 2 * k - 1] = invsqrt2
        B[k, 2 * k] = invsqrt2_over_i
        B[k + 4, 2 * k] = -invsqrt2_over_i
    end
    B[4, 7] = 1
    return B
end

function _build_g2_basis_float64()
    B = _basis_change_matrix()
    complex_generators = _complex_g2_generators()
    generators = ntuple(G2_ALGEBRA_DIM) do index
        transformed = B' * complex_generators[index] * B
        Matrix{Float64}(real.(im * transformed))
    end
    return G2Basis{Float64}(generators)
end

"""
    g2_basis([T=Float64])

Return the canonical real seven-dimensional G2 basis.
"""
function g2_basis(::Type{Float64} = Float64)
    cached = _G2_BASIS_FLOAT64[]
    if cached === nothing
        cached = _build_g2_basis_float64()
        _G2_BASIS_FLOAT64[] = cached
    end
    return cached
end

function g2_basis(::Type{T}) where {T<:AbstractFloat}
    basis64 = g2_basis(Float64)
    generators = ntuple(index -> Matrix{T}(basis64[index]), G2_ALGEBRA_DIM)
    return G2Basis{T}(generators)
end

function _g2_coefficients_vector(coefficients::AbstractVector{<:Real}, ::Type{T}) where {T<:AbstractFloat}
    length(coefficients) == G2_ALGEBRA_DIM ||
        throw(ArgumentError("G2 coefficient vector must have length $G2_ALGEBRA_DIM"))
    return T.(collect(coefficients))
end

function _real_g2_matrix(matrix::AbstractMatrix)
    size(matrix) == (G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM) ||
        throw(ArgumentError("G2 matrix must be 7 x 7"))
    return Matrix{Float64}(real.(matrix))
end

function _antisymmetric_part(matrix::AbstractMatrix)
    real_matrix = _real_g2_matrix(matrix)
    return 0.5 .* (real_matrix .- transpose(real_matrix))
end

"""
    g2_matrix(coefficients; basis=g2_basis())

Build the real `7 x 7` algebra matrix `sum_a coefficients[a] * A[a]`.
"""
function g2_matrix(
    coefficients_in::AbstractVector{<:Real};
    basis::G2Basis{T} = g2_basis(),
) where {T<:AbstractFloat}
    coeffs = _g2_coefficients_vector(coefficients_in, T)
    matrix = zeros(T, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    for index in 1:G2_ALGEBRA_DIM
        matrix .+= coeffs[index] .* basis[index]
    end
    return matrix
end

g2_matrix(element::G2AlgebraElement; basis::G2Basis = g2_basis()) =
    g2_matrix(element.coefficients; basis = basis)

"""
    g2_coefficients(matrix; basis=g2_basis(), antisymmetrize=true)

Return the 14 basis coefficients of the G2 projection of `matrix`.
"""
function g2_coefficients(
    matrix::AbstractMatrix;
    basis::G2Basis{T} = g2_basis(),
    antisymmetrize::Bool = true,
) where {T<:AbstractFloat}
    projected_input = antisymmetrize ? _antisymmetric_part(matrix) : _real_g2_matrix(matrix)
    projected = Matrix{T}(projected_input)
    return [T(-2) * tr(basis[index] * projected) for index in 1:G2_ALGEBRA_DIM]
end

project_to_g2_coefficients(matrix::AbstractMatrix; basis::G2Basis = g2_basis()) =
    g2_coefficients(matrix; basis = basis, antisymmetrize = true)

"""
    project_to_g2(matrix; basis=g2_basis())

Project a real or complex `7 x 7` matrix onto the real G2 Lie algebra.
"""
project_to_g2(matrix::AbstractMatrix; basis::G2Basis = g2_basis()) =
    g2_matrix(project_to_g2_coefficients(matrix; basis = basis); basis = basis)

function is_g2_algebra_matrix(
    matrix::AbstractMatrix;
    atol::Real = G2_DEFAULT_ATOL,
    basis::G2Basis = g2_basis(),
)
    real_matrix = _real_g2_matrix(matrix)
    antisymmetric_defect = norm(real_matrix + transpose(real_matrix))
    projection_defect = norm(real_matrix - project_to_g2(real_matrix; basis = basis))
    return antisymmetric_defect <= atol && projection_defect <= atol
end

"""
    g2_link_defects(U; basis=g2_basis())

Return diagnostics for a `7 x 7` matrix as a G2 link candidate.
"""
function g2_link_defects(matrix::AbstractMatrix; basis::G2Basis = g2_basis())
    size(matrix) == (G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM) ||
        throw(ArgumentError("G2 link matrix must be 7 x 7"))
    real_matrix = Matrix{Float64}(real.(matrix))
    identity7 = Matrix{Float64}(I, G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
    imag_defect = maximum(abs.(imag.(matrix)))
    orthogonal_defect = norm(transpose(real_matrix) * real_matrix - identity7)
    determinant_defect = abs(det(real_matrix) - 1.0)

    algebra_defect = 0.0
    for generator in basis.generators
        rotated = real_matrix * generator * transpose(real_matrix)
        algebra_defect = max(algebra_defect, norm(rotated - project_to_g2(rotated; basis = basis)))
    end

    return (
        imaginary = imag_defect,
        orthogonal = orthogonal_defect,
        determinant = determinant_defect,
        algebra = algebra_defect,
    )
end

function is_g2_link(
    matrix::AbstractMatrix;
    atol::Real = G2_DEFAULT_ATOL,
    basis::G2Basis = g2_basis(),
)
    defects = g2_link_defects(matrix; basis = basis)
    return defects.imaginary <= atol &&
           defects.orthogonal <= atol &&
           defects.determinant <= atol &&
           defects.algebra <= atol
end
