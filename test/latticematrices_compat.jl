import LatticeMatrices

const LMCompat = Gaugefields.LatticeMatricesCompat

@test LMCompat.HAS_SHIFT_RELEASE == isdefined(LatticeMatrices, :release!)
@test LMCompat.HAS_HALO_EPOCHS ==
      isdefined(LatticeMatrices, :mark_halo_dirty!)

U = Initialize_Gaugefields(
    3,
    1,
    4,
    4,
    4,
    4;
    condition="cold",
    isMPILattice=true,
    PEs=(1, 1, 1, 1),
    verbose_level=0,
)

if LMCompat.HAS_HALO_EPOCHS
    @test !LatticeMatrices.halo_is_dirty(U[1].U)
end

U[1][1, 1, 1, 1, 1, 1] = 2 + 0im

if LMCompat.HAS_HALO_EPOCHS
    @test LatticeMatrices.halo_is_dirty(U[1].U)
end

# This is wider than NDW=1. LatticeMatrices 1.x borrows storage for the
# materialized shift; 0.3 shifted lattices are non-owning.
shifted = shift_U(U[1], (2, 0, 0, 0))
@test isopen(shifted)
close(shifted)
@test isopen(shifted) == !LMCompat.HAS_SHIFT_RELEASE

adjoint_shifted = shift_U(U[1], (2, 0, 0, 0))'
@test isopen(adjoint_shifted)
close(adjoint_shifted)
@test isopen(adjoint_shifted) == !LMCompat.HAS_SHIFT_RELEASE

temp1 = similar(U[1])
temp2 = similar(U[1])
plaquette = calculate_Plaquette(U, temp1, temp2)
polyakov = calculate_Polyakov_loop(U, temp1, temp2)
@test isfinite(real(plaquette))
@test isfinite(real(polyakov))
