# ---------------- Device-safe tiny PRNG (PCG32, stateless per element) ----------------
# All functions are @inline and pure; no dynamic dispatch, no allocations.

# ---------------- PCG32 core (safe on GPU) ----------------
# All integer ops; rotation count as Int to avoid implicit conversions.
# PCG32 step: safe for Julia/CPU & CUDA
@inline function pcg32_step(state::UInt64, inc::UInt64)
    oldstate = state
    # 64-bit LCG update (wraps mod 2^64 automatically for UInt64)
    state = oldstate * 0x5851F42D4C957F2D + (inc | 0x01)

    # Mix and squeeze to 32 bits
    # IMPORTANT: mask to 32 bits before converting to UInt32 to avoid InexactError.
    x = ((oldstate >> 18) ⊻ oldstate) >> 27         # still up to 37 bits here
    xorshifted = UInt32(x & 0xFFFF_FFFF)            # explicit truncation to 32 bits

    # Rotate by high bits of oldstate
    rot = Int(oldstate >> 59)                       # shift count must be Int
    out = (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))

    return state, out
end

# Map UInt32 -> uniform in [0,1) as Float32
@inline u01_f32(x::UInt32) = Float32(x) * Float32(2.3283064365386963f-10)  # 1/2^32

# Map two UInt32 -> uniform in [0,1) with ~53 bits as Float64
@inline function u01_f64(x::UInt32, y::UInt32)
    # (x<<21 ^ y>>11) gives 53 bits; scale by 2^-53
    mant = (UInt64(x) << 21) ⊻ (UInt64(y) >> 11)
    return Float64(mant) * 1.1102230246251565e-16  # 2^-53
end

# Deterministic per-site stream: combine indices into (state,inc)
@inline function mix_seed(ix,iy,iz,it,ic,jc,seed0::UInt64)
    h = seed0 ⊻ (UInt64(ix) * 0x9E3779B97F4A7C15) ⊻ (UInt64(iy) * 0xBF58476D1CE4E5B9) ⊻
        (UInt64(iz) * 0x94D049BB133111EB) ⊻ (UInt64(it) * 0xD6E8FEB86659FD93) ⊻
        (UInt64(ic) * 0xA24BAED4963EE407) ⊻ (UInt64(jc) * 0x9FB21C651E98DF25)
    state = h ⊻ 0xDA942042E4DD58B5
    inc   = (h >> 1) | 0x1                 # must be odd
    return state, inc
end
