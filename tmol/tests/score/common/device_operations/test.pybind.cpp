#include <tmol/utility/tensor/pybind.h>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/tests/score/common/device_operations/test.hh>
#include <moderngpu/meta.hxx>

namespace tmol {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  using namespace tmol::tests::score::common::device_operations;

  using CPU = DevOpsTests<Device::CPU>;

  m.def(
      "test_checked_dispatch_size",
      [](int64_t count) {
        return tmol::score::common::checked_dispatch_size(
            count, "test dispatch");
      },
      "count"_a);
  m.def(
      "test_checked_dispatch_product",
      [](int64_t lhs, int64_t rhs) {
        return tmol::score::common::checked_dispatch_product(
            lhs, rhs, "test dispatch product");
      },
      "lhs"_a,
      "rhs"_a);
  m.def(
      "test_checked_triangular_size",
      [](int64_t count, bool include_diagonal) {
        return tmol::score::common::checked_triangular_size(
            count, include_diagonal, "test triangular dispatch");
      },
      "count"_a,
      "include_diagonal"_a);
  m.def(
      "test_upper_triangle_indices",
      [](int linear_index, int dimension) {
        auto result =
            tmol::score::common::upper_triangle_inds_from_linear_index(
                linear_index, dimension);
        return std::make_pair(
            tmol::score::common::get<0>(result),
            tmol::score::common::get<1>(result));
      },
      "linear_index"_a,
      "dimension"_a);
  m.def(
      "test_safe_div_up",
      [](int value, int divisor) { return mgpu::div_up(value, divisor); },
      "value"_a,
      "divisor"_a);

  m.def("test_forall", &CPU::test_forall, "src"_a);
  m.def("test_forall_independent", &CPU::test_forall_independent, "src"_a);
  m.def("test_forall_grouped", &CPU::test_forall_grouped, "src"_a);
  m.def(
      "test_foreach_combination_triple",
      &CPU::test_foreach_combination_triple,
      "src"_a);
  m.def("test_foreach_workgroup", &CPU::test_foreach_workgroup, "src"_a);
  m.def(
      "test_foreach_independent_workgroup",
      &CPU::test_foreach_independent_workgroup,
      "src"_a);
  m.def(
      "test_foreach_pose_workgroup",
      &CPU::test_foreach_pose_workgroup,
      "src"_a);
  m.def("test_scan_inclusive", &CPU::test_scan_inclusive, "src"_a);
  m.def("test_scan_exclusive", &CPU::test_scan_exclusive, "src"_a);
  m.def(
      "test_scan_and_return_total_inclusive",
      &CPU::test_scan_and_return_total_inclusive,
      "src"_a);
  m.def(
      "test_scan_and_return_total_exclusive",
      &CPU::test_scan_and_return_total_exclusive,
      "src"_a);
  m.def("test_reduce", &CPU::test_reduce, "src"_a);
  m.def(
      "test_load_balancing_search",
      &CPU::test_load_balancing_search,
      "exc_scan_offsets"_a,
      "n_work_units_total"_a);
  m.def(
      "test_segmented_scan_inclusive",
      &CPU::test_segmented_scan_inclusive,
      "src"_a,
      "seg_starts"_a);
  m.def(
      "test_segmented_scan_exclusive",
      &CPU::test_segmented_scan_exclusive,
      "src"_a,
      "seg_starts"_a);

#ifdef WITH_CUDA
  using CUDA = DevOpsTests<Device::CUDA>;

  // All functions take TView arguments; pybind resolves CPU vs CUDA overloads
  // by inspecting the device of the input tensors.
  m.def("test_forall", &CUDA::test_forall, "src"_a);
  m.def("test_forall_independent", &CUDA::test_forall_independent, "src"_a);
  m.def("test_forall_grouped", &CUDA::test_forall_grouped, "src"_a);
  m.def(
      "test_foreach_combination_triple",
      &CUDA::test_foreach_combination_triple,
      "src"_a);
  m.def("test_foreach_workgroup", &CUDA::test_foreach_workgroup, "src"_a);
  m.def(
      "test_foreach_independent_workgroup",
      &CUDA::test_foreach_independent_workgroup,
      "src"_a);
  m.def(
      "test_foreach_pose_workgroup",
      &CUDA::test_foreach_pose_workgroup,
      "src"_a);
  m.def("test_scan_inclusive", &CUDA::test_scan_inclusive, "src"_a);
  m.def("test_scan_exclusive", &CUDA::test_scan_exclusive, "src"_a);
  m.def(
      "test_scan_and_return_total_inclusive",
      &CUDA::test_scan_and_return_total_inclusive,
      "src"_a);
  m.def(
      "test_scan_and_return_total_exclusive",
      &CUDA::test_scan_and_return_total_exclusive,
      "src"_a);
  m.def("test_reduce", &CUDA::test_reduce, "src"_a);
  m.def(
      "test_load_balancing_search",
      &CUDA::test_load_balancing_search,
      "exc_scan_offsets"_a,
      "n_work_units_total"_a);
  m.def(
      "test_segmented_scan_inclusive",
      &CUDA::test_segmented_scan_inclusive,
      "src"_a,
      "seg_starts"_a);
  m.def(
      "test_segmented_scan_exclusive",
      &CUDA::test_segmented_scan_exclusive,
      "src"_a,
      "seg_starts"_a);
#endif
}
}  // namespace tmol
