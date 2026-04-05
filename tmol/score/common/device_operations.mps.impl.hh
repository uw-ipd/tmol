#pragma once

// device_operations.mps.impl.hh
//
// DeviceOperations<tmol::Device::MPS> specialization.
//
// Strategy — Phase 1 (CPU-loop via unified memory):
//   Apple Silicon uses a unified memory architecture: the physical memory
//   backing an MPS tensor is the same as CPU memory.  PyTorch's MPS backend
//   allocates buffers in shared GPU/CPU memory, so data_ptr() on an MPS tensor
//   is directly readable/writable from the CPU after calling
//   torch::mps::synchronize() or by keeping the command queue flushed.
//
//   DeviceOperations<MPS> therefore executes all lambdas on the CPU, using
//   exactly the same serial loop implementation as the CPU specialization.
//   This gives correct results with no data copies on Apple Silicon.
//
// Strategy — Phase 2 (Metal GPU dispatch):
//   The metal_context.hh / metal_context.mm bridge provides a
//   MtlContext singleton that owns the MTLCommandQueue and pre-compiled
//   MTLComputePipelineState objects loaded from tmol_primitives.metallib.
//   Per-score-term .mps.mm files call into those pipelines via the C++ API
//   declared in metal_context.hh to achieve true GPU parallelism.
//
// This file must NOT be included from CUDA translation units (.cu / .cuh).

#ifdef __NVCC__
error_this_file_must_not_be_compiled_by_nvcc();
#endif

#include "device_operations.hh"

#ifdef WITH_MPS
#include <torch/mps.h>   // torch::mps::synchronize()
#endif

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::MPS> {
  // -------------------------------------------------------------------------
  // forall — iterate over [0, N), calling f(i) for each i.
  // launch_t is ignored on MPS (it is a CUDA-specific launch-box type).
  // -------------------------------------------------------------------------
  template <typename launch_t, typename Func>
  static void forall(int N, Func f) {
    // Flush any pending Metal command buffers so CPU data_ptr() reads see
    // GPU-written data (unified memory coherency for Phase 1).
    synchronize_device();
    for (int i = 0; i < N; ++i) {
      f(i);
    }
  }

  // -------------------------------------------------------------------------
  // forall_stacks — iterate over [0, Nstacks) x [0, N)
  // -------------------------------------------------------------------------
  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f) {
    synchronize_device();
    for (Int stack = 0; stack < Nstacks; ++stack) {
      for (Int i = 0; i < N; ++i) {
        f(stack, i);
      }
    }
  }

  // -------------------------------------------------------------------------
  // foreach_combination_triple — iterate over dim1 x dim2 x dim3
  // -------------------------------------------------------------------------
  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f) {
    synchronize_device();
    for (Int i = 0; i < dim1; ++i) {
      for (Int j = 0; j < dim2; ++j) {
        for (Int k = 0; k < dim3; ++k) {
          f(i, j, k);
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // foreach_workgroup — iterate over [0, n_workgroups), calling f(cta_id).
  // On MPS the concept of a "workgroup" maps to a Metal threadgroup; here we
  // serialise them on the CPU for Phase 1 correctness.
  // -------------------------------------------------------------------------
  template <typename launch_t, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    synchronize_device();
    for (int i = 0; i < n_workgroups; ++i) {
      f(i);
    }
  }

  // -------------------------------------------------------------------------
  // scan — prefix scan (inclusive or exclusive) with operator op.
  // -------------------------------------------------------------------------
  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static void scan(T* src, T* dst, int n, OP op) {
    if (n <= 0) return;
    T last_val = src[0];
    if (scan_type == mgpu::scan_type_inc) {
      dst[0] = last_val;
    }
    for (int i = 1; i < n; ++i) {
      T i_val   = src[i];
      T next_val = op(last_val, i_val);
      dst[i]    = (scan_type == mgpu::scan_type_exc) ? last_val : next_val;
      last_val  = next_val;
    }
  }

  // -------------------------------------------------------------------------
  // scan_and_return_total
  // -------------------------------------------------------------------------
  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static T scan_and_return_total(T* src, T* dst, int n, OP op) {
    if (n == 0) return T(0);
    T last_val = src[0];
    if (scan_type == mgpu::scan_type_inc) {
      dst[0] = last_val;
    }
    for (int i = 1; i < n; ++i) {
      T i_val   = src[i];
      T next_val = op(last_val, i_val);
      dst[i]    = (scan_type == mgpu::scan_type_exc) ? last_val : next_val;
      last_val  = next_val;
    }
    return last_val;
  }

  // -------------------------------------------------------------------------
  // load_balancing_search
  // -------------------------------------------------------------------------
  template <typename launch_t, typename Int>
  static TPack<Int, 1, tmol::Device::MPS> load_balancing_search(
      int n_work_units_total,
      Int* exc_scan_offsets,
      int n_generators) {
    // Allocate on MPS device; data is CPU-accessible via unified memory.
    auto gen_for_work_item_t =
        TPack<Int, 1, tmol::Device::MPS>::zeros({n_work_units_total});
    auto gen_for_work_item = gen_for_work_item_t.view;

    for (int i = 0; i < n_generators; ++i) {
      int i_offset      = exc_scan_offsets[i];
      int i_n_work_units =
          (i + 1 == n_generators ? n_work_units_total
                                 : exc_scan_offsets[i + 1])
          - i_offset;
      for (int j = 0; j < i_n_work_units; ++j) {
        gen_for_work_item[i_offset + j] = i;
      }
    }
    return gen_for_work_item_t;
  }

  // -------------------------------------------------------------------------
  // reduce — reduce array of n elements with op, return result to CPU.
  // n must be > 0.
  // -------------------------------------------------------------------------
  template <typename T, typename OP>
  static T reduce(T* src, int n, OP op) {
    assert(n > 0);
    T val = src[0];
    for (int i = 1; i < n; ++i) {
      val = op(val, src[i]);
    }
    return val;
  }

  // -------------------------------------------------------------------------
  // segmented_scan
  // -------------------------------------------------------------------------
  template <
      mgpu::scan_type_t scan_type,
      typename launch_t,
      typename T,
      typename Int,
      typename OP>
  static auto segmented_scan(
      T* src, Int* seg_start_inds, int n, int n_segs, OP op, T identity)
      -> TPack<T, 1, tmol::Device::MPS> {
    auto dst_t = TPack<T, 1, tmol::Device::MPS>::empty({n});
    auto dst   = dst_t.view;

    T   last_val  = identity;
    int count_seg = 0;

    for (int i = 0; i < n; ++i) {
      T i_val = src[i];
      if (i == seg_start_inds[count_seg]) {
        last_val = identity;
        count_seg++;
      }
      T next_val = op(last_val, i_val);
      dst[i]     = (scan_type == mgpu::scan_type_exc) ? last_val : next_val;
      last_val   = next_val;
    }
    return dst_t;
  }

  // -------------------------------------------------------------------------
  // Intra-workgroup helpers — on MPS (Phase 1) these execute on the CPU in a
  // single logical "workgroup" of width N_T, so shared-memory and shuffle
  // semantics reduce trivially.
  // -------------------------------------------------------------------------

  template <int N_T, int WIDTH, typename T>
  static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    for (int i = 0; i < n; ++i) {
      dst[i] = src[i];
    }
  }

  template <int N_T, int WIDTH, typename TD, typename TS>
  static void copy_contiguous_data_and_cast(
      TD* __restrict__ dst, TS* __restrict__ src, int n) {
    for (int i = 0; i < n; ++i) {
      dst[i] = static_cast<TD>(src[i]);
    }
  }

  template <int N_T, typename Func>
  static void for_each_in_workgroup(Func f) {
    for (int i = 0; i < N_T; ++i) {
      f(i);
    }
  }

  // No parallel reduction on CPU; just return the value from "thread 0".
  template <int N_T, typename T, typename S, typename OP>
  static T reduce_in_workgroup(T val, S, OP) {
    return val;
  }

  template <int N_T, typename T, typename OP>
  static T shuffle_reduce_in_workgroup(T val, OP) {
    return val;
  }

  template <int N_T, typename T, typename OP>
  static T shuffle_reduce_and_broadcast_in_workgroup(T val, OP) {
    return val;
  }

  // No-op: MPS command completion is managed at the PyTorch level via
  // torch::mps::synchronize() or stream completion.
  static void synchronize_workgroup() {}

  static void synchronize_device() {
#ifdef WITH_MPS
    // Flush any pending Metal commands so CPU reads see GPU writes.
    // torch::mps::synchronize() is the public PyTorch API for this.
    // Declared in <torch/csrc/api/include/torch/mps.h> (PyTorch >= 2.0).
    torch::mps::synchronize();
#endif
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
