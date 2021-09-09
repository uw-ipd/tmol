#pragma once

#ifdef __CUDACC__

#include <cooperative_groups.h>
#include <tmol/numeric/log2.hh>
#include <moderngpu/operators.hxx>
#include <moderngpu/meta.hxx>
#include <Eigen/Core>

namespace tmol {
namespace score {
namespace common {

// Perform a reduction over a set of ranges. The beginning of each
// range is marked with a "flag" set to 1. At the end, the threads
// which had a flag value of 1 will have the values of the complete
// reduction; the values returned by other threads will be incomplete.
// The reduction requires that the input operation be associative,
// but, unlike most reductions, does not require commutativity; this
// reduction implementation is more like segmented scan in that sense.
//
// The algorithm works by first shuffling the flag bit down 1 so that
// its meaning changes from "the first value in a range" to "the last
// value in a range."
template <typename T, class Enable = void>
struct WarpSegReduceShfl {};

template <typename T>
struct WarpSegReduceShfl<
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <typename F>
  static __device__ __inline__ T segreduce(
      cooperative_groups::coalesced_group g, T val, bool flag, F f) {
    unsigned int const grank = g.thread_rank();
    unsigned int const gsize = g.size();

    flag = g.shfl_down(flag, 1);

    for (int i = 1; i < 32; i *= 2) {
      T val_i = g.shfl_down(val, i);
      bool flag_i = g.shfl_down(flag, i);
      if (!flag && i + grank < gsize) {
        val = f(val, val_i);
      }
      flag |= flag_i;
    }
    return val;
  }
};

template <typename T, int N>
struct WarpSegReduceShfl<
    Eigen::Matrix<T, N, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <typename F>
  static __device__ __inline__ Eigen::Matrix<T, N, 1> segreduce(
      cooperative_groups::coalesced_group g,
      Eigen::Matrix<T, N, 1> val,
      bool flag,
      F f) {
    unsigned int const grank = g.thread_rank();
    unsigned int const gsize = g.size();
    flag = g.shfl_down(flag, 1);

    for (int i = 1; i < 32; i *= 2) {
      Eigen::Matrix<T, N, 1> val_i;
      // mgpu::iterate<N>([&](int j) {val_i[j] = g.shfl_down(val[j], i);});
      for (int j = 0; j < N; ++j) {
        val_i[j] = g.shfl_down(val[j], i);
      }
      bool flag_i = g.shfl_down(flag, i);
      if (!flag && i + grank < gsize) {
        // mgpu::iterate<N>([&](int j) {val_i[j] = f(val[j], val_i[j]);});
        for (int j = 0; j < N; ++j) {
          val[j] = f(val[j], val_i[j]);
        }
      }
      flag |= flag_i;
    }
    return val;
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol

#endif
