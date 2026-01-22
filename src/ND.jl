"""
    Wiltinger_numerical_derivative(f, indices, U; params=(), targets=:all, ϵ=1e-8)

Numerical differentiation for functions of multiple  inputs.

- `U` can be a Vector.
- `indices` is the lattice site indices (e.g. (2,2,2,2)).
- For each target k in `targets`, compute a matrix `grad[k]` of size (NC1,NC2):
      grad[k][jc,ic] = d f / dRe(U[k].A[ic,jc,indices...])
                     + i * d f / dIm(U[k].A[ic,jc,indices...])
  using central differences.

Returns:
- If `U` is a Vector: returns a Vector of gradients, one per element.

Notes:
- Returns the Wirtinger derivative ∂f/∂U = (df/dx - i df/dy)/2.
"""
function LatticeMatrices.Wiltinger_numerical_derivative(f, indices, U::Vector{T};
    params=(),
    targets=:all,
    ϵ::Real=1e-8) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}

    # normalize U to a mutable container for easy copying/replacement
    set_wing_U!(U)
    Uvec = copy(U)


    # which components of U to differentiate
    Ks = targets === :all ? eachindex(Uvec) : targets

    grads = Vector{Any}(undef, length(Uvec))
    for i in eachindex(grads)
        grads[i] = nothing
    end

    # helper to call f with the same container type as the original U
    function callf(Uwork)

        return f(Uwork, params...)   # If your f expects f(U::Vector, ...) keep this

    end

    # Important:
    # - If your f is defined as f(U, temp, ...) (single U container),
    #   then set callf accordingly. Here we support both common styles:
    #
    #   (A) f(U::Vector, params...)
    #   (B) f(U1, U2, ..., params...)
    #
    # If you always use style (A), we can simplify.

    for k in Ks
        Uk = Uvec[k]

        # infer element type and matrix size from Uk.A
        #Aarr = Uk.A
        #T1 = eltype(Aarr)
        T1 = ComplexF64
        NC1, NC2 = NC, NC#size(Aarr, 1), size(Aarr, 2)

        grad = zeros(T1, NC1, NC2)

        for jc = 1:NC2, ic = 1:NC1
            # --- Re part derivative ---
            Up = deepcopy(Uvec)
            Up[k][ic, jc, indices...] += ϵ
            set_wing_U!(Up[k])
            Um = deepcopy(Uvec)
            Um[k][ic, jc, indices...] -= ϵ
            set_wing_U!(Um[k])
            dRe = (callf(Up) - callf(Um)) / (2ϵ)

            # --- Im part derivative ---
            Up = deepcopy(Uvec)
            Up[k][ic, jc, indices...] += im * ϵ
            set_wing_U!(Up[k])
            Um = deepcopy(Uvec)
            Um[k][ic, jc, indices...] -= im * ϵ
            set_wing_U!(Um[k])
            dIm = (callf(Up) - callf(Um)) / (2ϵ)

            # your convention: df/dx + i df/dy
            #grad[ic, jc] = dRe + im * dIm
            # Wiltinger: 
            grad[ic, jc] = (dRe - im * dIm) / 2
        end

        grads[k] = grad
    end

    # return in the same container style
    return grads

end

function LatticeMatrices.Numerical_derivative_Enzyme(f, indices, U::Vector{T};
    params=(),
    targets=:all,
    ϵ::Real=1e-8) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}

    # normalize U to a mutable container for easy copying/replacement
    set_wing_U!(U)
    Uvec = copy(U)


    # which components of U to differentiate
    Ks = targets === :all ? eachindex(Uvec) : targets

    grads = Vector{Any}(undef, length(Uvec))
    for i in eachindex(grads)
        grads[i] = nothing
    end

    # helper to call f with the same container type as the original U
    function callf(Uwork)

        return f(Uwork, params...)   # If your f expects f(U::Vector, ...) keep this

    end

    # Important:
    # - If your f is defined as f(U, temp, ...) (single U container),
    #   then set callf accordingly. Here we support both common styles:
    #
    #   (A) f(U::Vector, params...)
    #   (B) f(U1, U2, ..., params...)
    #
    # If you always use style (A), we can simplify.

    for k in Ks
        Uk = Uvec[k]

        # infer element type and matrix size from Uk.A
        #Aarr = Uk.A
        #T1 = eltype(Aarr)
        T1 = ComplexF64
        NC1, NC2 = NC, NC#size(Aarr, 1), size(Aarr, 2)

        grad = zeros(T1, NC1, NC2)

        for jc = 1:NC2, ic = 1:NC1
            # --- Re part derivative ---
            Up = deepcopy(Uvec)
            Up[k][ic, jc, indices...] += ϵ
            set_wing_U!(Up[k])
            Um = deepcopy(Uvec)
            Um[k][ic, jc, indices...] -= ϵ
            set_wing_U!(Um[k])
            dRe = (callf(Up) - callf(Um)) / (2ϵ)

            # --- Im part derivative ---
            Up = deepcopy(Uvec)
            Up[k][ic, jc, indices...] += im * ϵ
            set_wing_U!(Up[k])
            Um = deepcopy(Uvec)
            Um[k][ic, jc, indices...] -= im * ϵ
            set_wing_U!(Um[k])
            dIm = (callf(Up) - callf(Um)) / (2ϵ)

            # your convention: df/dx + i df/dy
            grad[ic, jc] = dRe + im * dIm
            # Wiltinger: 
            #grad[ic, jc] = (dRe - im * dIm) / 2
        end

        grads[k] = grad
    end

    # return in the same container style
    return grads

end

function LatticeMatrices.Numerical_derivative_Enzyme(f, indices, U1::T, U2::T, U3::T, U4::T;
    params=(),
    targets=:all,
    ϵ::Real=1e-8) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    U = [U1, U2, U3, U4]
    # normalize U to a mutable container for easy copying/replacement
    set_wing_U!(U)
    Uvec = copy(U)


    # which components of U to differentiate
    Ks = targets === :all ? eachindex(Uvec) : targets

    grads = Vector{Any}(undef, length(Uvec))
    for i in eachindex(grads)
        grads[i] = nothing
    end

    # helper to call f with the same container type as the original U
    function callf(Uwork...)

        return f(Uwork..., params...)   # If your f expects f(U::Vector, ...) keep this

    end

    # Important:
    # - If your f is defined as f(U, temp, ...) (single U container),
    #   then set callf accordingly. Here we support both common styles:
    #
    #   (A) f(U::Vector, params...)
    #   (B) f(U1, U2, ..., params...)
    #
    # If you always use style (A), we can simplify.

    for k in Ks
        Uk = Uvec[k]

        # infer element type and matrix size from Uk.A
        #Aarr = Uk.A
        #T1 = eltype(Aarr)
        T1 = ComplexF64
        NC1, NC2 = NC, NC#size(Aarr, 1), size(Aarr, 2)

        grad = zeros(T1, NC1, NC2)

        for jc = 1:NC2, ic = 1:NC1
            # --- Re part derivative ---
            Up = deepcopy(Uvec)
            Up[k][ic, jc, indices...] += ϵ
            set_wing_U!(Up[k])
            Um = deepcopy(Uvec)
            Um[k][ic, jc, indices...] -= ϵ
            set_wing_U!(Um[k])
            dRe = (callf(Up...) - callf(Um...)) / (2ϵ)

            # --- Im part derivative ---
            Up = deepcopy(Uvec)
            Up[k][ic, jc, indices...] += im * ϵ
            set_wing_U!(Up[k])
            Um = deepcopy(Uvec)
            Um[k][ic, jc, indices...] -= im * ϵ
            set_wing_U!(Um[k])
            dIm = (callf(Up...) - callf(Um...)) / (2ϵ)

            # your convention: df/dx + i df/dy
            grad[ic, jc] = dRe + im * dIm
            # Wiltinger: 
            #grad[ic, jc] = (dRe - im * dIm) / 2
        end

        grads[k] = grad
    end

    # return in the same container style
    return grads

end


