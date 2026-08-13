#pragma once

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tuple>

namespace tmol {
namespace tests {
namespace score {
namespace common {
namespace device_operations {

template <tmol::Device D>
struct DevOpsTests {
  // dst[i] = src[i] + i
  static auto test_forall(TView<int32_t, 1, D> src) -> TPack<int32_t, 1, D>;

  // dst[stack][i] = src[stack][i] * 2
  static auto test_forall_stacks(TView<int32_t, 2, D> src)
      -> TPack<int32_t, 2, D>;

  // dst[i][j][k] = src[i][j][k] + 1
  static auto test_foreach_combination_triple(TView<int32_t, 3, D> src)
      -> TPack<int32_t, 3, D>;

  // dst[wg] = src[wg] + wg
  static auto test_foreach_workgroup(TView<int32_t, 1, D> src)
      -> TPack<int32_t, 1, D>;

  static auto test_scan_inclusive(TView<int32_t, 1, D> src)
      -> TPack<int32_t, 1, D>;

  // dst[0] is pre-initialized to 0 (identity) before calling scan.
  static auto test_scan_exclusive(TView<int32_t, 1, D> src)
      -> TPack<int32_t, 1, D>;

  static auto test_scan_and_return_total_inclusive(TView<int32_t, 1, D> src)
      -> std::tuple<TPack<int32_t, 1, D>, int32_t>;

  // dst[0] is pre-initialized to 0 (identity) before calling scan.
  static auto test_scan_and_return_total_exclusive(TView<int32_t, 1, D> src)
      -> std::tuple<TPack<int32_t, 1, D>, int32_t>;

  static auto test_reduce(TView<int32_t, 1, D> src) -> int32_t;

  static auto test_load_balancing_search(
      TView<int32_t, 1, D> exc_scan_offsets, int n_work_units_total)
      -> TPack<int32_t, 1, D>;

  // padded_seg_starts has n_segs+1 entries: the real segment start indices
  // followed by a sentinel equal to n (length of src), which guards against
  // the out-of-bounds read that occurs when count_seg reaches n_segs inside
  // segmented_scan.
  static auto test_segmented_scan_inclusive(
      TView<int32_t, 1, D> src,
      TView<int32_t, 1, D> padded_seg_starts,
      int n_segs) -> TPack<int32_t, 1, D>;

  static auto test_segmented_scan_exclusive(
      TView<int32_t, 1, D> src,
      TView<int32_t, 1, D> padded_seg_starts,
      int n_segs) -> TPack<int32_t, 1, D>;
};

}  // namespace device_operations
}  // namespace common
}  // namespace score
}  // namespace tests
}  // namespace tmol
