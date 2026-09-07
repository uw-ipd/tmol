#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmol {
namespace score {
namespace common {

inline int checked_dispatch_size(int64_t count, char const* operation) {
  if (count < 0 || count > std::numeric_limits<int>::max()) {
    int64_t const minimum_bytes =
        count <= 0 ? 0
                   : (count > std::numeric_limits<int64_t>::max() / 16
                          ? std::numeric_limits<int64_t>::max()
                          : count * int64_t(16));
    throw std::overflow_error(
        std::string(operation) + " requires " + std::to_string(count)
        + " interactions, exceeding the signed 32-bit native dispatch limit "
          "(2147483647). The minimum three-int32-index + float-value storage "
          "alone is "
        + std::to_string(minimum_bytes)
        + " bytes. Reduce the dispatched workload; for packing, lower "
          "TMOL_PACK_MAX_POSES_PER_CHUNK.");
  }
  return static_cast<int>(count);
}

inline int64_t checked_count_product(
    int64_t lhs, int64_t rhs, char const* operation) {
  if (lhs < 0 || rhs < 0
      || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
    throw std::overflow_error(
        std::string(operation) + " overflows a signed 64-bit count");
  }
  return lhs * rhs;
}

inline int checked_dispatch_product(
    int64_t lhs, int64_t rhs, char const* operation) {
  return checked_dispatch_size(
      checked_count_product(lhs, rhs, operation), operation);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
