#pragma once

#include <moderngpu/transform.hxx>
#include <moderngpu/loadstore.hxx>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/cta_scan.hxx>
#include <moderngpu/cta_segscan.hxx>

#include "device_operations.hh"

#include <tmol/score/common/accumulate.hh>

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::CUDA> {
  template <typename launch_t, typename Func>
  static void forall(int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform<launch_t>(f, N, context);
  }

  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int stack = index / N;
          int i = index % N;
          f(stack, i);
        },
        N * Nstacks,
        context);
  }

  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / (dim2 * dim3);
          index = index % (dim2 * dim3);
          int j = index / dim3;
          int k = index % dim3;
          f(i, j, k);
        },
        dim1 * dim2 * dim3,
        context);
  }

  template <typename launch_t, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    auto wrapper = ([=] __device__(int tid, int cta) { f(cta); });
    mgpu::standard_context_t context;
    mgpu::cta_launch<launch_t>(wrapper, n_workgroups, context);
  }

  template <int N_T, int WIDTH, typename T>
  __device__ static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    mgpu::mem_to_shared<N_T, WIDTH>(src, threadIdx.x, n, dst, false);
  }

  template <int N_T, int WIDTH, typename TD, typename TS>
  __device__ static void copy_contiguous_data_and_cast(
      TD* __restrict__ dst, TS* __restrict__ src, int n) {
    // taken from mgpu::mem_to_shared w/ static cast to TD inserted

    mgpu::array_t<TS, WIDTH> x =
        mgpu::mem_to_reg_strided<N_T, WIDTH, WIDTH>(src, threadIdx.x, n);
    mgpu::strided_iterate<N_T, WIDTH, WIDTH>(
        [&](int i, int j) { dst[j] = static_cast<TD>(x[i]); }, threadIdx.x, n);
  }

  template <int N_T, typename Func>
  __device__ static void for_each_in_workgroup(Func f) {
    f(threadIdx.x);
  }

  template <int N_T, typename T, typename S, typename OP>
  __device__ static T reduce_in_workgroup(T val, S& shared, OP op) {
    typedef mgpu::cta_reduce_t<N_T, T> reduce_t;
    return reduce_t().reduce(threadIdx.x, val, shared.reduce, N_T, op);
  }

  // For this to work, NT must be <= 32; this operation relies on intra-warp
  // communication using the nvidia shfl primatives and is more efficient
  // than reduction operations that rely on shared memory, but requires
  // that the whole work group reside in a single warp's width. It can work
  // with NT < 32, but I don't believe it is possible to launch a kernel with
  // NT < 32, so, NT == 32 is the most reasonable use case.
  template <int N_T, typename T, typename OP>
  __device__ static T shuffle_reduce_in_workgroup(T val, OP op) {
    assert(N_T <= 32);
    auto g = cooperative_groups::coalesced_threads();
    return reduce<tmol::Device::CUDA, T>::reduce_to_head(g, val, op);
  }

  // See comments for shuffle_reduce_in_workgroup above
  template <int N_T, typename T, typename OP>
  __device__ static T shuffle_reduce_and_broadcast_in_workgroup(T val, OP op) {
    assert(N_T <= 32);
    auto g = cooperative_groups::coalesced_threads();
    return reduce<tmol::Device::CUDA, T>::reduce_to_all(g, val, op);
  }

  __device__ static void synchronize_workgroup() { __syncthreads(); }

  // Perform an in-place exclusive scan within the workgroup
  // over an array of WIDTH elements. If
  // broadcast_and_return_total is true, then all members
  // of the workgroup will receive the total of the scan
  template <int N_T, int WIDTH, typename T, typename OP>
  __device__ static T exclusive_scan_in_workgroup(
      T* data, T identity, OP op, bool broadcast_and_return_total) {
    // auto* storage =
    //     reinterpret_cast<typename mgpu::cta_scan_t<N_T,
    //     T>::storage_t*>(data);
    // T start_val = data[threadIdx.x];
    // auto result = mgpu::cta_scan_t<N_T, T>().scan(
    //     threadIdx.x,
    //     start_val,
    //     *storage,
    //     N_T,
    //     op,
    //     identity,
    //     mgpu::scan_type_exc);
    //
    // // printf("exc scan %d start %d final %d total %d\n", threadIdx.x,
    // // start_val, result.scan, result.reduction);
    // data[threadIdx.x] = result.scan;
    //
    // return result.reduction;

    // Let's do this with shuffle
    assert(N_T == 32 && WIDTH == 32);
    unsigned int m = 0xffffffff;
    T val = data[threadIdx.x];
    for (int i = 1; i < WIDTH; i *= 2) {
      T prev = __shfl_up_sync(m, val, i);
      if (threadIdx.x >= i) {
        val = op(prev, val);
      }
    }
    T total = identity;
    if (broadcast_and_return_total) {
      total = __shfl_sync(m, val, N_T - 1);
    }
    val = __shfl_up_sync(m, val, 1);
    if (threadIdx.x == 0) {
      val = identity;
    }
    data[threadIdx.x] = val;
    return total;
  }

  // Perform an in-place inclusive scan within the workgroup
  // over an array of WIDTH elements. If
  // broadcast_and_return_total is true, then all members
  // of the workgroup will receive the total of the scan
  template <int N_T, int WIDTH, typename T, typename OP>
  __device__ static T inclusive_scan_in_workgroup(
      T* data, T identity, OP op, bool broadcast_and_return_total) {
    // auto* storage =
    //     reinterpret_cast<typename mgpu::cta_scan_t<N_T,
    //     T>::storage_t*>(data);
    // auto result = mgpu::cta_scan_t<N_T, T>().scan(
    //     threadIdx.x,
    //     data[threadIdx.x],
    //     *storage,
    //     N_T,
    //     op,
    //     identity,
    //     mgpu::scan_type_inc);
    // data[threadIdx.x] = result.scan;
    // return result.reduction;
    assert(N_T == 32 && WIDTH == 32);
    unsigned int m = 0xffffffff;
    T val = data[threadIdx.x];
    for (int i = 1; i < WIDTH; i *= 2) {
      T prev = __shfl_up_sync(m, val, i);
      if (threadIdx.x >= i) {
        val = op(prev, val);
      }
    }
    T total = identity;
    if (broadcast_and_return_total) {
      total = __shfl_sync(m, val, N_T - 1);
    }
    data[threadIdx.x] = val;
    return total;
  }

  // Perform an in-place inclusive segmented scan within the
  // workgroup over an array of WIDTH elements with the
  // seg_begin array informing where each segment begins.
  template <int N_T, int WIDTH, typename T, typename S, typename OP>
  __device__ static void inclusive_seg_scan_in_workgroup(
      T* data, T identity, S* seg_begin, OP op) {
    // auto * storage = reinterpret_cast<typename mgpu::cta_segscan_t<N_T,
    // T>::storage_t *>(data); T start_val = data[threadIdx.x]; int start_head =
    // seg_begin[threadIdx.x]; auto result = mgpu::cta_segscan_t<N_T,
    // T>().segscan(
    //   threadIdx.x, bool(seg_begin[threadIdx.x]), true, data[threadIdx.x],
    //   *storage, identity, op);
    // printf("inc seg scan %d head %d start %f final %f total %f\n",
    // threadIdx.x, start_head, start_val, result.scan, result.reduction);
    // data[threadIdx.x] = result.scan;

    // assert(N_T == 32);

    // T val = data[threadIdx.x];
    // S look_back = seg_begin[threadIdx.x];
    // for (int i = 1; i < N_T; i *= 2) {
    //   __syncthreads();
    //
    //   T val_prev = identity;
    //   S look_back_prev = false;
    //   if (threadIdx.x - i >= 0) {
    //     val_prev = data[threadIdx.x - i];
    //     look_back_prev = seg_begin[threadIdx.x - i];
    //   }
    //
    //   __syncthreads();
    //   if (threadIdx.x - i >= 0 && !look_back) {
    //     val = op(val_prev, val);
    //     data[threadIdx.x] = val;
    //     look_back = look_back || look_back_prev;
    //     seg_begin[threadIdx.x] = look_back;
    //   }
    // }
    assert(N_T == 32 && WIDTH == 32);
    unsigned int m = 0xffffffff;
    T val = data[threadIdx.x];
    bool flag = seg_begin[threadIdx.x];
    for (int i = 1; i < WIDTH; i *= 2) {
      T prev = __shfl_up_sync(m, val, i);
      bool flag_prev = __shfl_up_sync(m, flag, i);
      if (threadIdx.x >= i) {
        if (!flag) {
          val = op(prev, val);
        }
        flag = flag_prev || flag;
      }
    }
    data[threadIdx.x] = val;
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
