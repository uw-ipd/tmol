#pragma once

// Philox4x32-10, the counter-based generator the CUDA annealer draws from.
//
// Counter-based rather than state-based: the n-th draw of a stream is a pure
// function of (key, counter), so a trajectory's random numbers depend only on
// its own index and need no per-trajectory state to carry or seed forward.
// That is what lets thousands of independent trajectories run concurrently
// without a state table, and it is pure 32-bit integer arithmetic, so it
// compiles for every backend including Metal.
//
// Reference: Salmon, Moraes, Dror & Shaw, "Parallel Random Numbers: As Easy
// as 1, 2, 3", SC'11.

#include "qualifiers.hh"

#if !TMOL_PORTABLE_METAL
#include <cstdint>
#endif

namespace tmol {
namespace portable {

class Philox4x32_10 {
 public:
  // Weyl increments applied to the key between rounds, and the two
  // multipliers, all from the reference implementation.
  enum : uint32_t {
    kWeyl0 = 0x9E3779B9u,
    kWeyl1 = 0xBB67AE85u,
    kMul0 = 0xD2511F53u,
    kMul1 = 0xCD9E8D57u
  };

  struct uint4 {
    uint32_t x, y, z, w;
  };

  TMOL_HD Philox4x32_10()
      : key0_(0), key1_(0), c0_(0), c1_(0), c2_(0), c3_(0) {}

  // One stream per (seed, stream_id). counter_start lets a caller skip ahead.
  TMOL_HD Philox4x32_10(uint64_t seed, uint64_t stream_id, uint64_t counter = 0)
      : key0_(uint32_t(seed & 0xFFFFFFFFu)),
        key1_(uint32_t(seed >> 32)),
        c0_(uint32_t(counter & 0xFFFFFFFFu)),
        c1_(uint32_t(counter >> 32)),
        c2_(uint32_t(stream_id & 0xFFFFFFFFu)),
        c3_(uint32_t(stream_id >> 32)) {}

  // Four fresh 32-bit words, then advance the counter.
  TMOL_HD uint4 next4() {
    uint32_t k0 = key0_, k1 = key1_;
    uint32_t c0 = c0_, c1 = c1_, c2 = c2_, c3 = c3_;
    for (int round = 0; round < 10; ++round) {
      uint32_t hi0, lo0, hi1, lo1;
      mulhilo(kMul0, c0, hi0, lo0);
      mulhilo(kMul1, c2, hi1, lo1);
      uint32_t const n0 = hi1 ^ c1 ^ k0;
      uint32_t const n1 = lo1;
      uint32_t const n2 = hi0 ^ c3 ^ k1;
      uint32_t const n3 = lo0;
      c0 = n0;
      c1 = n1;
      c2 = n2;
      c3 = n3;
      if (round != 9) {
        k0 += uint32_t(kWeyl0);
        k1 += uint32_t(kWeyl1);
      }
    }
    bump();
    uint4 out;
    out.x = c0;
    out.y = c1;
    out.z = c2;
    out.w = c3;
    return out;
  }

  // A single word, buffering the other three.
  TMOL_HD uint32_t next() {
    if (have_ == 0) {
      buf_ = next4();
      have_ = 4;
    }
    --have_;
    if (have_ == 3) return buf_.x;
    if (have_ == 2) return buf_.y;
    if (have_ == 1) return buf_.z;
    return buf_.w;
  }

  // In (0, 1], matching curand_uniform's conversion so the CUDA (cuRAND) and
  // Metal (Philox) annealers can be compared draw for draw if that is ever
  // wanted.
  TMOL_HD float next_uniform() {
    return float(next()) * 2.3283064365386963e-10f + 1.1641532182693481e-10f;
  }

  // Uniform on [0, n), via the modulo the CUDA annealer uses (random_below).
  TMOL_HD int next_below(int n) { return int(next() % uint32_t(n)); }

 private:
  // The high half of a 32x32 product, without a 64-bit multiply: Apple GPUs
  // emulate 64-bit integer arithmetic, and this runs once per round per
  // draw. Each backend has a single instruction for it.
  TMOL_HD static uint32_t mul_high(uint32_t a, uint32_t b) {
#if TMOL_PORTABLE_METAL
    return ::metal::mulhi(a, b);
#elif defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return uint32_t((uint64_t(a) * uint64_t(b)) >> 32);
#endif
  }

  TMOL_HD static void mulhilo(
      uint32_t a,
      uint32_t b,
      TMOL_THREAD uint32_t& hi,
      TMOL_THREAD uint32_t& lo) {
    hi = mul_high(a, b);
    lo = a * b;  // wraps mod 2^32, which is what the algorithm wants
  }

  TMOL_HD void bump() {
    if (++c0_ == 0)
      if (++c1_ == 0)
        if (++c2_ == 0) ++c3_;
  }

  uint32_t key0_, key1_;
  uint32_t c0_, c1_, c2_, c3_;
  uint4 buf_ = {0, 0, 0, 0};
  int have_ = 0;
};

}  // namespace portable
}  // namespace tmol
