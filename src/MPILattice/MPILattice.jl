module MPILattice
import LatticeMatrices: LatticeMatrix,
    Shifted_Lattice,
    Adjoint_Lattice,
    TALattice,
    makeidentity_matrix!,
    set_halo!,
    substitute!,
    partial_trace,
    get_PEs,
    clear_matrix!,
    add_matrix!,
    expt!,
    #get_2Dindex,
    traceless_antihermitian_add!,
    normalize_matrix!,
    randomize_matrix!,
    #get_2Dindex,
    get_shift,
    gather_and_bcast_matrix,
    traceless_antihermitian!

export LatticeMatrix




# Generic, fast path using divrem (works for any sizes)
@inline function get_2Dindex(i::I, dims::NTuple{2,I}) where {I<:Integer}
    # Decode linear index i (1-based) into four 1-based coordinates.
    # Use divrem to compute quotient and remainder in one shot, reducing idiv count.
    @inbounds begin
        Nx, Ny = dims
        o = i - one(I)                  # zero-based offset
        o, rx = divrem(o, Nx)
        ix = rx + one(I)
        iy = o + one(I)               # remaining quotient
        return ix, iy
    end
end

# Generic, fast path using divrem (works for any sizes)
@inline function get_4Dindex(i::I, dims::NTuple{4,I}) where {I<:Integer}
    # Decode linear index i (1-based) into four 1-based coordinates.
    # Use divrem to compute quotient and remainder in one shot, reducing idiv count.
    @inbounds begin
        Nx, Ny, Nz, Nt = dims
        o = i - one(I)                  # zero-based offset
        o, rx = divrem(o, Nx)
        ix = rx + one(I)
        o, ry = divrem(o, Ny)
        iy = ry + one(I)
        o, rz = divrem(o, Nz)
        iz = rz + one(I)
        it = o + one(I)                 # remaining quotient
        return ix, iy, iz, it
    end
end




end