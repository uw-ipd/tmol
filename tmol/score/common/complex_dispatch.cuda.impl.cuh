#pragma once

#include <Eigen/Core>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_reduce.hxx>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/kinematics/compiled/kernel_segscan.cuh>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ComplexDispatch {
  template <typename Int, typename Func>
  static void forall(Int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform(f, N, context);
  }

  template <typename T, typename Func>
  static T reduce(TView<T, 1, D> vals, Func op) {
    auto v_t = tmol::TPack<T, 1, D>::zeros({1});
    auto v = v_t.view;
    mgpu::standard_context_t context;

    mgpu::transform_reduce(
      [=] MGPU_DEVICE(int i) {return vals[i];},
      vals.size(0),
      &v[0],
      op,
      context);

    T val_cpu;
    cudaMemcpy(&val_cpu, &v[0], sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "Reduce " << val_cpu << std::endl;
    return val_cpu;
  }

  template <typename T, typename Func>
  static void exclusive_scan(TView<T, 1, D> vals, TView<T, 1, D> out, Func op) {
    mgpu::standard_context_t context;
    mgpu::scan_event<mgpu::scan_type_exc>(
      &vals[0], vals.size(0), &out[0], op, mgpu::discard_iterator_t<T>(),
      context, 0);
  }

  template <typename T, typename Func>
  static void inclusive_scan(TView<T, 1, D> vals, TView<T, 1, D> out, Func op) {
    mgpu::standard_context_t context;
    mgpu::scan_event<mgpu::scan_type_inc>(
      &vals[0], vals.size(0), &out[0], op, mgpu::discard_iterator_t<T>(),
      context, 0);
  }

  template <typename T, typename Func>
  static T exclusive_scan_w_final_val(
      TView<T, 1, D> vals, TView<T, 1, D> out, Func op) {
    auto final_val_t = TPack<T, 1, D>::empty({1});
    auto final_val = final_val_t.view;
    mgpu::standard_context_t context;
    mgpu::scan_event<mgpu::scan_type_exc>(
      &vals[0], vals.size(0), &out[0], op, &final_val[0],
      context, 0);
    T final_val_cpu;
    cudaMemcpy(&final_val_cpu, final_val.data(), sizeof(T), cudaMemcpyDeviceToHost);
    return final_val_cpu;
  }

  template <typename T, typename B, typename Func>
  static void exclusive_segmented_scan(
      TView<T, 1, D> vals,
      TView<B, 1, D> seg_start,
      TView<T, 1, D> out,
      Func op) {
      
    assert(vals.size(0) == out.size(0));
    mgpu::standard_context_t context;

    typedef typename mgpu::launch_params_t<128, 2> launch_t;
    constexpr int nt = launch_t::nt, vt = launch_t::vt;

    int const n_segments = seg_start.size(0);
    int scanBuffer = n_segments + vals.size(0);
    float scanleft = std::ceil(((float) scanBuffer) / (nt*vt));
    int lbsBuffer = (int) scanleft + 1;
    int carryoutBuffer = (int) scanleft;
    while (scanleft > 1) {
      scanleft = std::ceil(scanleft / nt);
      carryoutBuffer += (int)scanleft;
    }
    
    auto scanCarryout_tpack = TPack<T, 1, D>::empty({carryoutBuffer});
    auto scanCodes_tpack = TPack<int, 1, D>::empty({carryoutBuffer});
    auto LBS_tpack = TPack<int, 1, D>::empty({lbsBuffer});
    auto scan_tpack = TPack<T, 1, D>::empty({scanBuffer});

    auto scanCarryout = scanCarryout_tpack.view;
    auto scanCodes = scanCodes_tpack.view;
    auto LBS = LBS_tpack.view;
    auto scan = scan_tpack.view;

    auto straight_indexing = [=] MGPU_DEVICE(int index, int seg, int rank) {
      return vals[index];
    };
    tmol::kinematics::kernel_segscan<launch_t, mgpu::scan_type_exc>(
      straight_indexing, vals.size(0), seg_start.data(), n_segments,
      out.data(), scanCarryout.data(), scanCodes.data(), LBS.data(),
      op, T(0), context
    );
  }

  static void synchronize()
  {
    cudaDeviceSynchronize();
  }


};

}
}
}
