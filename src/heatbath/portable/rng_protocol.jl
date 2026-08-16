"""
    heatbath_uniform(rng, ::Type{T}) -> updated_rng, value

Return a uniform draw in `[0, 1)` together with the updated RNG state.
Hosts provide methods for their RNG types.  Mutable RNG adapters return the
same object as `updated_rng`.
"""
function heatbath_uniform end

"""
    heatbath_log_uniform(rng, ::Type{T}) -> updated_rng, value

Return the backend's uniform draw for a logarithm input.  Device RNG adapters
normally return a value in `(0, 1)`; the legacy Julia adapter deliberately
retains the historical `rand()` endpoint behavior.
"""
function heatbath_log_uniform end

"""
    heatbath_normalize3!(u) -> success::Bool

Reunitarize a site-local 3x3 matrix.  This host hook keeps normalization,
failure handling, and storage-specific dispatch outside the portable physics
kernel.
"""
function heatbath_normalize3! end

# The clean portable and Web paths have no discarded prefix draws.  A host
# may specialize this hook when compatibility with an older stream position
# is required.
@inline heatbath_prepare_su2(rng, ::Type{T}) where T = rng

# The portable path preserves the caller's beta type.  Device adapters may
# specialize this to keep all local arithmetic in the matrix real type.
@inline heatbath_beta(rng, beta, ::Type{T}) where T = beta
