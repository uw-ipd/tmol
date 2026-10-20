#pragma once

#include "qualifiers.hh"

#if TMOL_PORTABLE_METAL
#include <metal_math>
#else
#include <cmath>
#endif

namespace tmol {
namespace portable {
namespace math {

// Scalar math shim. Host and CUDA resolve to the standard library; Metal has
// no <cmath> and supplies its own overloads in namespace metal. Call sites use
// tmol::portable::math::* so a single source file serves all three.

#if TMOL_PORTABLE_METAL
#define TMOL_MATH_NS ::metal
#else
#define TMOL_MATH_NS ::std
#endif

#define TMOL_MATH_UNARY(name)     \
  template <typename T>           \
  TMOL_HD T name(T x) {           \
    return TMOL_MATH_NS::name(x); \
  }

#define TMOL_MATH_BINARY(name)       \
  template <typename T>              \
  TMOL_HD T name(T x, T y) {         \
    return TMOL_MATH_NS::name(x, y); \
  }

#define TMOL_MATH_TERNARY(name)         \
  template <typename T>                 \
  TMOL_HD T name(T x, T y, T z) {       \
    return TMOL_MATH_NS::name(x, y, z); \
  }

TMOL_MATH_UNARY(sqrt)

// Relaxed-precision exp and log, for call sites that have been measured.
//
// Metal compiles with MTLMathModeSafe and the shader API gives us no
// MTLCompileOptions to change it, so ::metal::exp is
// the correctly-rounded version. metal::fast:: trades the last couple of ulps
// for a shorter hardware sequence.
//
// Deliberately NOT the default for exp/log: it is a win only where a kernel
// evaluates them per atom pair. Measured at 30 poses -- lk-ball 1150 -> 903 ms
// with the relative error unchanged (5.5e-07 -> 5.0e-07), ljlk 1073 -> 953 ms
// bit-for-bit identical, but elec 782 -> 804 ms with its error rising
// 9.2e-07 -> 1.3e-06. So lk-ball and lk-isotropic opt in and elec's
// sigmoidal dielectric stays on the precise version.
//
// Host and CUDA resolve these to ::std and stay bit-identical.
#if TMOL_PORTABLE_METAL
#define TMOL_MATH_FAST_NS ::metal::fast
#else
#define TMOL_MATH_FAST_NS TMOL_MATH_NS
#endif

#define TMOL_MATH_FAST_UNARY(exported, name) \
  template <typename T>                      \
  TMOL_HD T exported(T x) {                  \
    return TMOL_MATH_FAST_NS::name(x);       \
  }

TMOL_MATH_FAST_UNARY(fast_exp, exp)
TMOL_MATH_FAST_UNARY(fast_log, log)

TMOL_MATH_UNARY(exp)
TMOL_MATH_UNARY(log)
TMOL_MATH_UNARY(sin)
TMOL_MATH_UNARY(cos)
TMOL_MATH_UNARY(tan)
TMOL_MATH_UNARY(asin)
TMOL_MATH_UNARY(acos)
TMOL_MATH_UNARY(atan)
TMOL_MATH_UNARY(floor)
TMOL_MATH_UNARY(ceil)
TMOL_MATH_UNARY(rint)

TMOL_MATH_BINARY(pow)
TMOL_MATH_BINARY(atan2)
TMOL_MATH_BINARY(fmin)
TMOL_MATH_BINARY(fmax)
TMOL_MATH_BINARY(fmod)

// A single rounding for x * y + z. The error-free transformations in
// double_float.hh need it: `fma(a, b, -(a * b))` is the exact rounding error
// of the product, which no sequence of separately-rounded operations gives.
TMOL_MATH_TERNARY(fma)

#undef TMOL_MATH_UNARY
#undef TMOL_MATH_BINARY
#undef TMOL_MATH_TERNARY

// Replaces EIGEN_PI. Metal has no double, so the literal degrades to float
// there, which is all that backend can represent anyway.
template <typename T>
TMOL_HD T pi() {
  return T(3.14159265358979323846);
}

// abs() is spelled fabs() for floating point in <cmath>; Metal names it abs().
template <typename T>
TMOL_HD T abs(T x) {
#if TMOL_PORTABLE_METAL
  return ::metal::fabs(x);
#else
  return ::std::fabs(x);
#endif
}

// isnan, by bit pattern on Metal.
//
// Not `::metal::isnan` and not `x != x`: the Metal compiler is free to assume
// finite operands, and both spellings can be folded to false. lk_ball marks
// absent waters with NaN and relies on the test, so it is done on the bits,
// which no float-contraction rule can elide. Host and CUDA keep <cmath>.
#if TMOL_PORTABLE_METAL
TMOL_HD bool isnan(float x) {
  uint const bits = as_type<uint>(x);
  return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}
#else
template <typename T>
TMOL_HD bool isnan(T x) {
  return ::std::isnan(x);
}
#endif

#undef TMOL_MATH_NS

}  // namespace math
}  // namespace portable
}  // namespace tmol
