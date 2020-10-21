#pragma once

#include <Eigen/Core>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <moderngpu/operators.hxx>
#include <tmol/numeric/log2.hh>
#endif

namespace tmol {
namespace score {
namespace common {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

#ifdef __CUDA_ARCH__

// Perform reduction within the active threads.
// F is a functor that needs to be associative and
// commutative
// This function could/should be moved elsewhere.
template <typename T, typename F>
__device__ __inline__ T reduce_tile_shfl(
    cooperative_groups::coalesced_group g, T val, F f) {
  // Adapted from https://devblogs.nvidia.com/cooperative-groups/

  // First: have the lower threads shuffle from the
  // threads that hang off the end of the largest-power-of-2
  // less than or equal to the number of active threads.
  unsigned int const gsize = g.size();
  unsigned int const biggest_pow2_base = numeric::most_sig_bit(gsize);

  unsigned int const overhang = gsize - biggest_pow2_base;
  if (overhang > 0) {
    T const overhang_val = g.shfl_down(val, biggest_pow2_base);
    if (g.thread_rank() < overhang) {
      val = f(val, overhang_val);
    }
  }

  // Second: perform a canonical reduction with the group of
  // active threads; the number of iterations would otherwise
  // have missed the "overhang" set if the first shfl_down
  // above had not been performed.
  for (int i = biggest_pow2_base / 2; i > 0; i /= 2) {
    T const shfl_val = g.shfl_down(val, i);
    val = f(val, shfl_val);
  }

  return val;  // note: only thread 0 will return full sum
}
#endif

template <tmol::Device D, typename T, class Enable = void>
struct accumulate {};

template <typename T>
struct accumulate<
    tmol::Device::CPU,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->void { target += val; }

  // This is safe to use when all threads are going to write to the same address
  template <class A>
  static def add_one_dst(A& target, int ind, const T& val)->void {
    target[ind] += val;
  }
};

template <tmol::Device D, int N, typename T>
struct accumulate<
    D,
    Eigen::Matrix<T, N, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  typedef Eigen::Matrix<T, N, 1> V;

  static def add(V& target, const V& val)->void {
#pragma unroll
    for (int i = 0; i < N; i++) {
      accumulate<D, T>::add(target[i], val[i]);
    }
  }

  static def add_one_dst(V& target, const V& val)->void {
#pragma unroll
    for (int i = 0; i < N; i++) {
      accumulate<D, T>::add_one_dst(target[i], val[i]);
    }
  }

};  // namespace potentials

#ifdef __CUDACC__

template <typename T>
struct accumulate<
    tmol::Device::CUDA,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->void { atomicAdd(&target, val); }

  // Use this function to accummulate into an array, target, at a position,
  // ind, when most threads in a warp are going to write to the same
  // address to reduce the number of calls to atomicAdd and the
  // serialization this function creates.
  //
  // This function will iterate across the range of indices in the target
  // that will be written to (first figuring out this range with a min-
  // and max- reduction and two shuffles), reduce the values being summed
  // into the iterated dest within the coallesced group that wants to
  // write to that index, and then have thread 0 of the group perform
  // the single atomicAdd.
  //
  // A is an array-like class that will be indexed by [ind].
  // "ind" is the index that this thread should write to.
  template <typename A>
  static def add_one_dst(A& target, int ind, const T& val)->void {
#ifdef __CUDA_ARCH__

    auto g = cooperative_groups::coalesced_threads();

    int min_ind = reduce_tile_shfl(g, ind, mgpu::minimum_t<int>());
    min_ind = g.shfl(min_ind, 0);
    int max_ind = reduce_tile_shfl(g, ind, mgpu::maximum_t<int>());
    max_ind = g.shfl(max_ind, 0);

    for (int iter_ind = min_ind; iter_ind <= max_ind; ++iter_ind) {
      if (iter_ind == ind) {
        auto g2 = cooperative_groups::coalesced_threads();
        T warp_sum = reduce_tile_shfl(g2, val, mgpu::plus_t<T>());
        if (g2.thread_rank() == 0 && warp_sum != 0) {
          atomicAdd(&target[ind], warp_sum);
        }
      }
      g.sync();
    }
#endif
  }
};

#endif

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
