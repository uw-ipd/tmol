#pragma once

#include <Eigen/Core>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <tmol/numeric/log2.hh>
#endif

namespace tmol {
namespace score {
namespace common {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <tmol::Device D, typename T, class Enable = void>
struct accumulate {};

template <tmol::Device D, typename T, class Enable = void>
struct accumulate_kahan {};

template <typename T>
struct accumulate<
    tmol::Device::CPU,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->void { target += val; }
};

template <typename T>
struct accumulate_kahan<
    tmol::Device::CPU,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T* target, const T& val)->void {
    // from wikipedia
    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    T y = val - target[1];
    T t = target[0] + y;
    target[1] = (t - target[0]) - y;
    target[0] = t;
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
};  // namespace potentials

#ifdef __CUDACC__

template <typename T>
struct accumulate<
    tmol::Device::CUDA,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->void { atomicAdd(&target, val); }
};

union float2UllUnion {
  float2 f;
  unsigned long long int ull;
};

// Kahan summation on the GPU:
// 1. Perform a reduction on the active threads (the coalesced group)
// 2. Thread 0 then performs the Kahan summation with the sum
//    by spinning on calls to atomicCAS until the summation goes
//    trough.
// Note that the input desination pointer must be to an array of
// floats of size 2; contiguity is necessary for the atomicCAS
// to work.
template <>
struct accumulate_kahan<
    tmol::Device::CUDA,
    float,
    typename std::enable_if<std::is_arithmetic<float>::value>::type> {
#ifdef __CUDA_ARCH__

  // Perform reduction (sum) within the active threads.
  // This function could/should be moved elsewhere.
  static __device__ __inline__ float reduce_sum_tile_shfl(
      cooperative_groups::coalesced_group g, float val) {
    // Adapted from https://devblogs.nvidia.com/cooperative-groups/

    // First: have the lower threads shuffle from the
    // threads that hang off the end of the largest-power-of-2
    // less than or equal to the number of active threads.
    unsigned int const gsize = g.size();
    unsigned int const biggest_pow2_base = numeric::most_sig_bit128(gsize);

    unsigned int const overhang = gsize - biggest_pow2_base;
    if (overhang > 0) {
      float const overhang_val = g.shfl_down(val, biggest_pow2_base);
      if (g.thread_rank() < overhang) {
        val += overhang_val;
      }
    }

    // Second: perform a canonical reduction with the group of
    // active threads; the number of iterations would otherwise
    // have missed the "overhang" set if the first shfl_down
    // above had not been performed.
    for (int i = biggest_pow2_base / 2; i > 0; i /= 2) {
      val += g.shfl_down(val, i);
    }

    return val;  // note: only thread 0 will return full sum
  }
#endif

  static def add(float* address, const float& val)->void {
#ifdef __CUDA_ARCH__
    // Neither atomicCAS nor reduce_sum_tile_shfl can be
    // called from a host/device function -- only from a
    // device function. This function isn't truly
    // host/device function as it will never be invoked
    // from a host. Indeed, the whole lambda model for
    // our functions

    auto g = cooperative_groups::coalesced_threads();

    float warp_sum = reduce_sum_tile_shfl(g, val);

    if (g.thread_rank() == 0) {
      unsigned long long int* address_as_ull = (unsigned long long int*)address;
      float2UllUnion old, assumed, tmp;
      old.ull = *address_as_ull;
      do {
        assumed = old;
        tmp = assumed;
        // kahan summation
        const float y = warp_sum - tmp.f.y;
        const float t = tmp.f.x + y;
        tmp.f.y = (t - tmp.f.x) - y;
        tmp.f.x = t;

        old.ull = atomicCAS(address_as_ull, assumed.ull, tmp.ull);

      } while (assumed.ull != old.ull);
    }
#endif
  }
};

template <>
struct accumulate_kahan<
    tmol::Device::CUDA,
    double,
    typename std::enable_if<std::is_arithmetic<double>::value>::type> {
  static def add(double* address, const double& val)->void {
    // Kahan summation cannot be performed at double precision
    // so just perform standard double-precision atomic addition.
    // TO DO: figure out Alex's magic to avoid the error about
    // invoking atomic operations from a __host__ __device__ function.
#ifdef __CUDA_ARCH__
    atomicAdd(address, val);
#endif
  }
};

#endif

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
