module LatticeMatricesCompat

import LatticeMatrices

const HAS_SHIFT_RELEASE = isdefined(LatticeMatrices, :release!)
const HAS_HALO_EPOCHS = isdefined(LatticeMatrices, :mark_halo_dirty!)

if HAS_SHIFT_RELEASE
    @inline release_lattice!(shifted) = LatticeMatrices.release!(shifted)
    @inline lattice_isopen(shifted) = isopen(shifted)
else
    # LatticeMatrices 0.3 shifted lattices are non-owning and need no release.
    @inline release_lattice!(shifted) = nothing
    @inline lattice_isopen(shifted) = true
end

if HAS_HALO_EPOCHS
    @inline mark_lattice_dirty!(lattice) =
        LatticeMatrices.mark_halo_dirty!(lattice)
else
    # LatticeMatrices 0.3 does not track core/halo epochs.
    @inline mark_lattice_dirty!(lattice) = nothing
end

end
