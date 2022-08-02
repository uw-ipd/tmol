#pragma once

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <tmol/numeric/log2.hh>
#include <type_traits>
#include <Eigen/Core>

namespace tmol {
namespace score {
namespace common {

template <typename T, class Enable = void>
struct WarpStrideReduceShfl {};

template <typename T>
struct WarpStrideReduceShfl<
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <typename F>
  static __device__ __inline__ T stride_reduce(
      cooperative_groups::coalesced_group g, T val, int stride, F f) {
    unsigned int const grank = g.thread_rank();
    unsigned int const gsize = g.size();

    for (int i = 1; i * stride < 32; i *= 2) {
      T val_i = g.shfl_down(val, i * stride);
      if (i * stride + grank < gsize) {
        val = f(val, val_i);
      }
    }
    return val;
  }
};

template <typename T, int N>
struct WarpStrideReduceShfl<
    Eigen::Matrix<T, N, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <typename F>
  static __device__ __inline__ Eigen::Matrix<T, N, 1> stride_reduce(
      cooperative_groups::coalesced_group g,
      Eigen::Matrix<T, N, 1> val,
      int stride,
      F f) {
    unsigned int const grank = g.thread_rank();
    unsigned int const gsize = g.size();

    for (int i = 1; i * stride < 32; i *= 2) {
      Eigen::Matrix<T, N, 1> val_i;
      for (int j = 0; j < N; ++j) {
        val_i[j] = g.shfl_down(val[j], i * stride);
      }

      if (i * stride + grank < gsize) {
        for (int j = 0; j < N; ++j) {
          val[j] = f(val[j], val_i[j]);
        }
      }
    }
    return val;
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol

#endif
