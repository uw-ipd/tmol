#pragma once

// Workgroup primitives for the Apple Metal backend.
//
// Included only by a `.metal` translation unit, where TMOL_PORTABLE_METAL is
// set. The CPU specialization runs a workgroup's lanes serially and the CUDA
// one maps them onto a CTA; here a workgroup is a Metal threadgroup and a lane
// is a thread within it, so the per-lane loops collapse to the thread's own
// index and reductions use simdgroup instructions.
//
// Only the primitives tmol's kernels actually reach are defined. The host-side
// launch helpers on DeviceOperations (load_balancing_search, scan) return
// TPack and belong to the dispatch layer, which never compiles for Metal.

#include <tmol/score/common/device_operations_fwd.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/reduce_ops.hh>

#if !TMOL_PORTABLE_METAL
#error "device_operations.mps.impl.hh is for Metal translation units only"
#endif

namespace tmol {
namespace score {
namespace common {

typedef WorkgroupLane<tmol::Device::MPS> MetalLane;

// Map tmol's reduction operators onto Metal's simdgroup intrinsics. The
// operators are empty tag types, so overload resolution picks the intrinsic;
// there is no generic simd_reduce to pass a functor to.
template <typename T>
inline T tmol_simd_reduce(T val, tmol::portable::plus_t<T>) {
  return metal::simd_sum(val);
}

template <typename T>
inline T tmol_simd_reduce(T val, tmol::portable::maximum_t<T>) {
  return metal::simd_max(val);
}

template <typename T>
inline T tmol_simd_reduce(T val, tmol::portable::minimum_t<T>) {
  return metal::simd_min(val);
}

// Vectors reduce component-wise.
template <typename T, int N, typename OP>
inline tmol::portable::Vec<T, N> tmol_simd_reduce(
    tmol::portable::Vec<T, N> val, OP op) {
  tmol::portable::Vec<T, N> out;
  for (int i = 0; i < N; ++i) {
    out[i] = tmol_simd_reduce(val[i], op);
  }
  return out;
}

// Component access, so a scratch-based reduction can serve scalars and
// vectors alike. The threadgroup scratch is a float array -- it cannot be
// typed on T, because the kernel entry declares it -- so a wider-than-
// simdgroup reduction moves one component at a time.
inline int tmol_n_components(thread float const&) { return 1; }

template <typename T, int N>
inline int tmol_n_components(thread tmol::portable::Vec<T, N> const&) {
  return N;
}

inline thread float& tmol_component(thread float& v, int) { return v; }

template <typename T, int N>
inline thread T& tmol_component(thread tmol::portable::Vec<T, N>& v, int i) {
  return v[i];
}

template <>
struct DeviceOperations<tmol::Device::MPS> {
  // On CPU this loops over the workgroup's lanes; on Metal each thread *is* a
  // lane, so the body runs once with this thread's index.
  template <int N_T, typename Func>
  static void for_each_in_workgroup(Func f, MetalLane lane) {
    f(int(lane.lane));
  }

  // Sum across the threadgroup. simd_sum reduces within a 32-wide simdgroup;
  // the per-simdgroup partials are combined through threadgroup scratch, which
  // is what the CUDA path uses mgpu::cta_reduce_t's storage for.
  template <int N_T, typename T, typename S, typename OP>
  static T reduce_in_workgroup(
      T val, threadgroup S& shared, OP op, MetalLane lane) {
    uint const simd_lane = lane.lane % 32u;
    uint const simd_id = lane.lane / 32u;
    uint const n_simds = (lane.lanes_per_group + 31u) / 32u;

    T partial = metal::simd_sum(val);
    if (simd_lane == 0u) {
      lane.reduce_scratch[simd_id] = partial;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    T total = T(0);
    for (uint i = 0; i < n_simds; ++i) {
      total = op(total, T(lane.reduce_scratch[i]));
    }
    return total;
  }

  // Reduce across the threadgroup without threadgroup memory. The simdgroup
  // functions already broadcast their result to every lane, so the two
  // spellings differ only in which lanes are expected to use the answer --
  // but a threadgroup wider than one simdgroup needs a scratch pass to bring
  // the per-simdgroup partials together, which the implementation below does
  // when the launch is wider than 32.
  template <int N_T, typename T, typename OP, typename Lane = MetalLane>
  static T shuffle_reduce_in_workgroup(T val, OP op, MetalLane lane) {
    return shuffle_reduce_and_broadcast_in_workgroup<N_T>(val, op, lane);
  }

  template <int N_T, typename T, typename OP, typename Lane = MetalLane>
  static T shuffle_reduce_and_broadcast_in_workgroup(
      T val, OP op, MetalLane lane) {
    T partial = tmol_simd_reduce(val, op);
    if (N_T <= 32) {
      // One simdgroup: the simdgroup function already broadcasts.
      return partial;
    }
    // Wider than a simdgroup, so the per-simdgroup partials have to meet in
    // threadgroup scratch. Every lane folds all of them, which is what makes
    // this the broadcasting form -- the answer ends up in every lane rather
    // than only in lane zero. One component at a time, because the scratch is
    // a float array declared by the kernel entry and cannot be typed on T.
    uint const simd_lane = lane.lane % 32u;
    uint const simd_id = lane.lane / 32u;
    uint const n_simds = (lane.lanes_per_group + 31u) / 32u;
    T total = partial;
    int const n_comp = tmol_n_components(partial);
    for (int c = 0; c < n_comp; ++c) {
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (simd_lane == 0u) {
        lane.reduce_scratch[simd_id] = tmol_component(partial, c);
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      float acc = lane.reduce_scratch[0];
      for (uint i = 1; i < n_simds; ++i) {
        acc = op(acc, lane.reduce_scratch[i]);
      }
      tmol_component(total, c) = acc;
    }
    return total;
  }

  static void synchronize_workgroup(MetalLane) {
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }

  // Cooperative copy: the workgroup's threads stride through the range.
  template <int N_T, int WIDTH, typename T>
  static void copy_contiguous_data(
      threadgroup T* dst, device T* src, int n, MetalLane lane) {
    for (int i = int(lane.lane); i < n; i += int(lane.lanes_per_group)) {
      dst[i] = src[i];
    }
  }

  template <int N_T, int WIDTH, typename TD, typename TS>
  static void copy_contiguous_data_and_cast(
      threadgroup TD* dst, device TS* src, int n, MetalLane lane) {
    for (int i = int(lane.lane); i < n; i += int(lane.lanes_per_group)) {
      dst[i] = TD(src[i]);
    }
  }

  // Atomic OR into a device word, as the CPU and CUDA backends provide. MSL
  // has no atomic reference type, so the word is reinterpreted as atomic
  // storage; see accumulate.hh for why that is sound for a tensor element.
  static void bitwise_or(device int32_t& target, int32_t value) {
    metal::atomic_fetch_or_explicit(
        reinterpret_cast<device metal::atomic_int*>(&target),
        value,
        metal::memory_order_relaxed);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
