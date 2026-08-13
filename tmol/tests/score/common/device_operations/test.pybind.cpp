#include <tmol/utility/tensor/pybind.h>
#include <tmol/tests/score/common/device_operations/test.hh>

namespace tmol {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  using namespace tmol::tests::score::common::device_operations;

  using CPU = DevOpsTests<Device::CPU>;

  m.def("test_forall", &CPU::test_forall, "src"_a);
  m.def("test_forall_stacks", &CPU::test_forall_stacks, "src"_a);
  m.def(
      "test_foreach_combination_triple",
      &CPU::test_foreach_combination_triple,
      "src"_a);
  m.def("test_foreach_workgroup", &CPU::test_foreach_workgroup, "src"_a);
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
      "padded_seg_starts"_a,
      "n_segs"_a);
  m.def(
      "test_segmented_scan_exclusive",
      &CPU::test_segmented_scan_exclusive,
      "src"_a,
      "padded_seg_starts"_a,
      "n_segs"_a);

#ifdef WITH_CUDA
  using CUDA = DevOpsTests<Device::CUDA>;

  // All functions take TView arguments; pybind resolves CPU vs CUDA overloads
  // by inspecting the device of the input tensors.
  m.def("test_forall", &CUDA::test_forall, "src"_a);
  m.def("test_forall_stacks", &CUDA::test_forall_stacks, "src"_a);
  m.def(
      "test_foreach_combination_triple",
      &CUDA::test_foreach_combination_triple,
      "src"_a);
  m.def("test_foreach_workgroup", &CUDA::test_foreach_workgroup, "src"_a);
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
      "padded_seg_starts"_a,
      "n_segs"_a);
  m.def(
      "test_segmented_scan_exclusive",
      &CUDA::test_segmented_scan_exclusive,
      "src"_a,
      "padded_seg_starts"_a,
      "n_segs"_a);
#endif
}
}  // namespace tmol
