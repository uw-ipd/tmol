#pragma once

#ifndef __NVCC__
error_this_should_not_be_compiled();  // gcc should not include this file
#endif

#include <moderngpu/transform.hxx>
#include <moderngpu/loadstore.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/cta_reduce.hxx>

#include "device_operations.hh"

#include <tmol/score/common/accumulate.hh>
#include <tmol/kinematics/compiled/kernel_segscan.cuh>

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::CUDA> {
  static void* get_current_context() {
    // TO DO: ask torch for the current stream / current context
    // and create the context on that stream
    mgpu::standard_context_t* context = new mgpu::standard_context_t();
    return reinterpret_cast<void*>(context);
  }

  static void release_context(void* context) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    delete context_ptr;
  }

  template <typename launch_t, typename Func>
  static void forall(int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform<launch_t>(f, N, context);
  }

  template <typename launch_t, typename Func>
  static void forall(void* context, int N, Func f) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::transform<launch_t>(f, N, *context_ptr);
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
  static void forall_stacks(void* context, Int Nstacks, Int N, Func f) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int stack = index / N;
          int i = index % N;
          f(stack, i);
        },
        N * Nstacks,
        *context_ptr);
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

  template <typename Int, typename Func>
  static void foreach_combination_triple(
      void* context, Int dim1, Int dim2, Int dim3, Func f) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / (dim2 * dim3);
          index = index % (dim2 * dim3);
          int j = index / dim3;
          int k = index % dim3;
          f(i, j, k);
        },
        dim1 * dim2 * dim3,
        *context_ptr);
  }

  template <typename launch_t, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    auto wrapper = ([=] __device__(int tid, int cta) { f(cta); });
    mgpu::standard_context_t context;
    mgpu::cta_launch<launch_t>(wrapper, n_workgroups, context);
  }

  template <typename launch_t, typename Func>
  static void foreach_workgroup(void* context, int n_workgroups, Func f) {
    auto wrapper = ([=] __device__(int tid, int cta) { f(cta); });
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::cta_launch<launch_t>(wrapper, n_workgroups, *context_ptr);
  }

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static void scan(T* src, T* dst, int n, OP op) {
    mgpu::standard_context_t context;
    mgpu::scan<scan_type>(
        src, n, dst, op, mgpu::discard_iterator_t<T>(), context);
  }

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static void scan(void* context, T* src, T* dst, int n, OP op) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::scan<scan_type>(
        src, n, dst, op, mgpu::discard_iterator_t<T>(), *context_ptr);
  }

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static T scan_and_return_total(T* src, T* dst, int n, OP op) {
    mgpu::standard_context_t context;
    mgpu::mem_t<T> total(1, context, mgpu::memory_space_host);
    mgpu::scan<scan_type>(src, n, dst, op, total.data(), context);
    cudaStreamSynchronize(0);
    return total.data()[0];
  }

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static T scan_and_return_total(void* context, T* src, T* dst, int n, OP op) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::mem_t<T> total(1, *context_ptr, mgpu::memory_space_host);
    mgpu::scan<scan_type>(src, n, dst, op, total.data(), *context_ptr);

    // TO DO: synchronize with the context's stream rather than stream 0
    cudaStreamSynchronize(0);
    return total.data()[0];
  }

  template <typename T>
  static void* allocate_scan_total_storage(void* context) {
    // When we need the total computed from scan, we should allocate
    // page-locked host memory to hold it, and we can read from
    // this memory after the first part of the scan operation completes:
    // the "upswing"; we do not have to block until the entirety of the
    // scan operation completes before the CPU can do its work of
    // submitting the work that relies on the total computed during scan.
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::mem_t<T>* total =
        new mgpu::mem_t<T>(1, *context_ptr, mgpu::memory_space_host);
    return reinterpret_cast<void*>(total);
  }

  template <typename T>
  static void deallocate_scan_total_storage(void* context, void* total) {
    // This will synchronize with the device, so we only do this when we're done
    // done.
    // printf("CUDA deallocate scan total storage %p\n", total);
    delete reinterpret_cast<mgpu::mem_t<T>*>(total);
  }

  static void* allocate_synchronization_event() {
    // Allocate a cudaEvent_t object, create the event, and return the pointer
    cudaEvent_t* cuda_event = new cudaEvent_t();
    cudaEventCreate(cuda_event);
    // printf("allocated cuda event %p\n", cuda_event);
    return reinterpret_cast<void*>(cuda_event);
  }

  static void deallocate_synchronization_event(void* event) {
    // Destroy the event _and_ the cudaEvent_t object
    cudaEvent_t* cuda_event = reinterpret_cast<cudaEvent_t*>(event);
    cudaEventDestroy(*cuda_event);
    delete cuda_event;
  }

  static void synchronize_on_event(void* event) {
    // Destroy the event _and_ the cudaEvent_t object
    // printf("CUDA synchronize on event\n");
    cudaEvent_t* cuda_event = reinterpret_cast<cudaEvent_t*>(event);
    // printf("cuda_event %p\n", event);
    cudaEventSynchronize(*cuda_event);
  }

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static void submit_scan_w_event(
      void* context, T* src, T* dst, int n, void* event, void* total, OP op) {
    // event should be a pointer to a cudaEvent_t
    // total should be a pointer to an integer allocated in page-locked host
    // memory
    //   where the scan total will be written when the downswing has completed
    //   (but perhaps) before the entirety of scan has completed
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);

    cudaEvent_t* cuda_event = reinterpret_cast<cudaEvent_t*>(event);
    mgpu::mem_t<T>* total_mem = reinterpret_cast<mgpu::mem_t<T>*>(total);

    // This call returns before the entire scan is complete; the CPU may read
    // from the "total" variable before the entire scan is complete, but only
    // after the cuda_event has been synchronized on.
    mgpu::scan_event<scan_type>(
        src, n, dst, op, total_mem->data(), *context_ptr, *cuda_event);
  }

  template <typename T>
  static T read_scan_total(void* total) {
    // For the asyncronous scan-launch; wait for result combo:
    // in "submit_scan_w_event". Make sure that you have called
    // syncrhonize_on_event before calling this function.
    mgpu::mem_t<T>* total_mem = reinterpret_cast<mgpu::mem_t<T>*>(total);
    return total_mem->data()[0];
  }

  template <typename T>
  static void set_zero(void* context, T* dst, int n) {
    // TO DO: submit this to the context's stream rather than stream 0
    cudaMemset(dst, 0, n * sizeof(T));
  }

  // Construct load-balanced-search mapping of work items to their generator
  // index; see https://moderngpu.github.io/loadbalance.html
  // Arguments:
  //   - n_work_units_total: the sum of the number of work units
  //
  //   - exc_scan_offsets: the result of running exclusive scan on the
  //     the number of work units that each generator produces
  //.  - n_generators: the number of generators / length of exc_scan_offset
  template <typename launch_t, typename Int>
  static TPack<Int, 1, tmol::Device::CUDA> load_balancing_search(
      int n_work_units_total,  // The count of the total number of work units
      Int* exc_scan_offsets,
      int n_generators) {
    mgpu::standard_context_t context;

    auto gen_for_work_item_t =
        TPack<Int, 1, tmol::Device::CUDA>::zeros({n_work_units_total});
    auto gen_for_work_item = gen_for_work_item_t.view;

    load_balance_search(
        n_work_units_total,
        exc_scan_offsets,
        n_generators,
        gen_for_work_item.data(),
        context);
    return gen_for_work_item_t;
  }

  template <typename launch_t, typename Int>
  static TPack<Int, 1, tmol::Device::CUDA> load_balancing_search(
      void* context,
      int n_work_units_total,  // The count of the total number of work units
      Int* exc_scan_offsets,
      int n_generators) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);

    auto gen_for_work_item_t =
        TPack<Int, 1, tmol::Device::CUDA>::zeros({n_work_units_total});
    auto gen_for_work_item = gen_for_work_item_t.view;

    load_balance_search(
        n_work_units_total,
        exc_scan_offsets,
        n_generators,
        gen_for_work_item.data(),
        *context_ptr);
    return gen_for_work_item_t;
  }

  template <typename T, typename OP>
  static T reduce(T* src, int n, OP op) {
    mgpu::standard_context_t context;
    mgpu::mem_t<T> total(1, context, mgpu::memory_space_host);
    mgpu::reduce(src, n, total.data(), op, context);
    cudaStreamSynchronize(0);
    return total.data()[0];
  }

  template <typename T, typename OP>
  static T reduce(void* context, T* src, int n, OP op) {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);
    mgpu::mem_t<T> total(1, *context_ptr, mgpu::memory_space_host);
    mgpu::reduce(src, n, total.data(), op, *context_ptr);
    cudaStreamSynchronize(0);
    return total.data()[0];
  }

  // Segmented scan expects the indices for the beginning of each segment rather
  // than, e.g., a boolean tensor indicating the start of each segment.
  // The identity value (e.g. 0) must be given because pre-initialization is not
  // always possible. seg_starts_inds must be sorted in ascending order.
  template <
      mgpu::scan_type_t scan_type,
      typename launch_t,
      typename T,
      typename Int,
      typename OP>
  static auto segmented_scan(
      T* src, Int* seg_start_inds, int n, int n_segs, OP op, T identity)
      -> TPack<T, 1, tmol::Device::CUDA> {
    mgpu::standard_context_t context;

    int const nt = launch_t::nt;
    int const vt = launch_t::vt;

    auto src_indexing = [=] MGPU_DEVICE(int i) { return src[i]; };

    // Copying Frank's code from kinematics/compiled/compiled.cuda.cuh
    int const scanBuffer = n + n_segs;
    float scanleft = std::ceil(((float)scanBuffer) / (nt * vt));
    Int lbsBuffer = (Int)scanleft + 1;
    Int carryoutBuffer = (Int)scanleft;
    while (scanleft > 1) {
      scanleft = std::ceil(scanleft / nt);
      carryoutBuffer += (Int)scanleft;
    }

    auto scanCarryout_t =
        TPack<T, 1, tmol::Device::CUDA>::empty({carryoutBuffer});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t =
        TPack<Int, 1, tmol::Device::CUDA>::empty({carryoutBuffer});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<Int, 1, tmol::Device::CUDA>::empty({lbsBuffer});
    auto LBS = LBS_t.view;

    // The return tensor
    auto dst_scan_t = TPack<T, 1, tmol::Device::CUDA>::empty({scanBuffer});
    auto dst_scan = dst_scan_t.view;

    tmol::kinematics::kernel_segscan<launch_t>(
        src_indexing,
        n,
        &seg_start_inds[0],
        n_segs,
        &dst_scan.data()[0],
        &scanCarryout.data()[0],
        &scanCodes.data()[0],
        &LBS.data()[0],
        op,
        identity,
        context);
    return dst_scan_t;
  }

  // Segmented scan expects the indices for the beginning of each segment rather
  // than, e.g., a boolean tensor indicating the start of each segment.
  // The identity value (e.g. 0) must be given because pre-initialization is not
  // always possible. seg_starts_inds must be sorted in ascending order.
  template <
      mgpu::scan_type_t scan_type,
      typename launch_t,
      typename T,
      typename Int,
      typename OP>
  static auto segmented_scan(
      void* context,
      T* src,
      Int* seg_start_inds,
      int n,
      int n_segs,
      OP op,
      T identity) -> TPack<T, 1, tmol::Device::CUDA> {
    auto context_ptr = reinterpret_cast<mgpu::standard_context_t*>(context);

    int const nt = launch_t::nt;
    int const vt = launch_t::vt;

    auto src_indexing = [=] MGPU_DEVICE(int i) { return src[i]; };

    // Copying Frank's code from kinematics/compiled/compiled.cuda.cuh
    int const scanBuffer = n + n_segs;
    float scanleft = std::ceil(((float)scanBuffer) / (nt * vt));
    Int lbsBuffer = (Int)scanleft + 1;
    Int carryoutBuffer = (Int)scanleft;
    while (scanleft > 1) {
      scanleft = std::ceil(scanleft / nt);
      carryoutBuffer += (Int)scanleft;
    }

    auto scanCarryout_t =
        TPack<T, 1, tmol::Device::CUDA>::empty({carryoutBuffer});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t =
        TPack<Int, 1, tmol::Device::CUDA>::empty({carryoutBuffer});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<Int, 1, tmol::Device::CUDA>::empty({lbsBuffer});
    auto LBS = LBS_t.view;

    // The return tensor
    auto dst_scan_t = TPack<T, 1, tmol::Device::CUDA>::empty({scanBuffer});
    auto dst_scan = dst_scan_t.view;

    tmol::kinematics::kernel_segscan<launch_t>(
        src_indexing,
        n,
        &seg_start_inds[0],
        n_segs,
        &dst_scan.data()[0],
        &scanCarryout.data()[0],
        &scanCodes.data()[0],
        &LBS.data()[0],
        op,
        identity,
        *context_ptr);
    return dst_scan_t;
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
    return tmol::score::common::reduce<tmol::Device::CUDA, T>::reduce_to_head(
        g, val, op);
  }

  // See comments for shuffle_reduce_in_workgroup above
  template <int N_T, typename T, typename OP>
  __device__ static T shuffle_reduce_and_broadcast_in_workgroup(T val, OP op) {
    assert(N_T <= 32);
    auto g = cooperative_groups::coalesced_threads();
    return tmol::score::common::reduce<tmol::Device::CUDA, T>::reduce_to_all(
        g, val, op);
  }

  __device__ static void synchronize_workgroup() { __syncthreads(); }

  // No op on 1-core CPU
  static void synchronize_device() { cudaStreamSynchronize(0); }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
