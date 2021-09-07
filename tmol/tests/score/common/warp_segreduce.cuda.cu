#include <tmol/tests/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_segreduce.hh>

#include <moderngpu/transform.hxx>

#include <cooperative_groups.h>

tmol::TPack<float, 1, tmol::Device::CUDA> warp_segreduce_gpu(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags) {
  int const n_vals = values.size(0);
  assert(n_vals == flags.size(0));
  int const n_cta = n_vals / 32;
  assert(n_cta * 32 == n_vals);

  using namespace mgpu;
  typedef launch_box_t<
      arch_20_cta<32, 1>,
      arch_35_cta<32, 1>,
      arch_52_cta<32, 1>>
      launch_t;

  auto output_t = tmol::TPack<float, 1, tmol::Device::CUDA>::zeros({n_vals});
  auto output = output_t.view;

  auto run_warp_segreduce([=] MGPU_DEVICE(int tid, int cta) {
    auto g = cooperative_groups::coalesced_threads();
    float value = values[cta * 32 + tid];
    int flag = flags[cta * 32 + tid];
    float reduced_value = tmol::score::common::warp_segreduce_shfl(
        g, value, flag, mgpu::plus_t<float>());
    // printf("%d %d original %f reduced %f\n", cta, tid, value, reduced_value);
    if (flag) {
      output[cta * 32 + tid] = reduced_value;
    }
  });

  mgpu::standard_context_t context;
  mgpu::cta_launch<launch_t>(run_warp_segreduce, n_cta, context);

  return output_t;
}

// This time, not all 32 threads are active in the warp
tmol::TPack<float, 1, tmol::Device::CUDA> warp_segreduce_gpu2(
    tmol::TView<float, 1, tmol::Device::CUDA> values,
    tmol::TView<int, 1, tmol::Device::CUDA> flags) {
  int const n_vals = values.size(0);
  assert(n_vals == flags.size(0));
  int const n_cta = n_vals / 32;
  assert(n_cta * 32 == n_vals);

  using namespace mgpu;
  typedef launch_box_t<
      arch_20_cta<32, 1>,
      arch_35_cta<32, 1>,
      arch_52_cta<32, 1>>
      launch_t;

  auto output_t = tmol::TPack<float, 1, tmol::Device::CUDA>::zeros({n_vals});
  auto output = output_t.view;

  auto run_warp_segreduce([=] MGPU_DEVICE(int tid, int cta) {
    if (tid < 30) {
      auto g = cooperative_groups::coalesced_threads();
      float value = values[cta * 32 + tid];
      int flag = flags[cta * 32 + tid];
      float reduced_value = tmol::score::common::warp_segreduce_shfl(
          g, value, flag, mgpu::plus_t<float>());
      // printf("%d %d original %f reduced %f\n", cta, tid, value,
      // reduced_value);
      if (flag) {
        output[cta * 32 + tid] = reduced_value;
      }
    }
  });

  mgpu::standard_context_t context;
  mgpu::cta_launch<launch_t>(run_warp_segreduce, n_cta, context);

  return output_t;
}
