#pragma once

#include <tmol/utility/portable/qualifiers.hh>

#if !TMOL_PORTABLE_METAL
// Metal supplies int64_t itself, and libc++ does not compile as MSL.
#include <cstdint>
#endif

// Carrying a float through the int64 metadata buffer.
//
// The metadata buffer is int64 because that is what sizes and strides are.
// The few real-valued launch parameters -- a cutoff distance, say -- ride
// along in the same buffer as the bit pattern of a float, rather than as a
// second buffer binding for one number.

namespace tmol {
namespace score {
namespace common {

#if TMOL_PORTABLE_METAL

inline float tmol_unpack_float(int64_t bits) {
  // as_type is a keyword-like builtin in MSL, not a member of namespace
  // metal, so it is spelled unqualified.
  return as_type<float>(uint32_t(bits));
}

#else

inline int64_t tmol_pack_float(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return int64_t(bits);
}

inline float tmol_unpack_float(int64_t bits) {
  uint32_t const narrowed = uint32_t(bits);
  float value;
  __builtin_memcpy(&value, &narrowed, sizeof(value));
  return value;
}

#endif

}  // namespace common
}  // namespace score
}  // namespace tmol
