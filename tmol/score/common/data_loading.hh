#pragma once

#include <tmol/utility/portable/metal_view.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace common {

// The vector type is a parameter so a caller can ask for a specific
// spelling; it defaults to the portable vector.
template <typename Real, typename V3 = tmol::portable::Vec<Real, 3>>
TMOL_HD V3 coord_from_shared(TMOL_SHARED_PTR(Real) coord_array, int atom_ind) {
  V3 local_coord;
  for (int i = 0; i < 3; ++i) {
    local_coord[i] = coord_array[3 * atom_ind + i];
  }
  return local_coord;
}

#if TMOL_PORTABLE_METAL
// Same read, for a tile that was never staged into threadgroup memory and is
// read straight out of the source tensor. On CPU and CUDA the two pointer
// types are the same type and this overload would be a redefinition; on Metal
// the address space is part of the type, so both spellings are needed.
template <typename Real, typename V3 = tmol::portable::Vec<Real, 3>>
TMOL_HD V3 coord_from_shared(TMOL_DEVICE_PTR(Real) coord_array, int atom_ind) {
  V3 local_coord;
  for (int i = 0; i < 3; ++i) {
    local_coord[i] = coord_array[3 * atom_ind + i];
  }
  return local_coord;
}
#endif

}  // namespace common
}  // namespace score
}  // namespace tmol
