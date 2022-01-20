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
      unsigned int active, T val, bool flag, F f) {
    // unsigned int active = __activemask();
    // int active = __ballot_sync(0xFFFFFFFF, 1);
    unsigned int limit = __popc(active);
    unsigned int mask_higher = 0xFFFFFFFF << (threadIdx.x + 1);
    unsigned int rank = __popc(active & ~mask_higher);

    flag = __shfl_down_sync(active, flag, 1);

    for (int i = 1; i < 32; i *= 2) {
      T val_i = __shfl_down_sync(active, val, i);
      bool flag_i = __shfl_down_sync(active, flag, i);
      if (!flag && i + rank <= limit) {
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
      // cooperative_groups::coalesced_group g,
      unsigned int active,
      Eigen::Matrix<T, N, 1> val,
      bool flag,
      F f) {
    // unsigned int const grank = g.thread_rank();
    // unsigned int const gsize = g.size();
    // int const limit = g.size() - g.thread_rank();

    // compute a mask which will be 1 for every thread with higher rank than me
    unsigned int mask_higher = 0xFFFFFFFF << (threadIdx.x + 1);
    unsigned int limit = __popc(active);

    // find my rank by negating the mask-higher, bitwise-anding it with the set
    // of active threads, and then using the bit counting function, __popc
    unsigned int rank = __popc(active & ~mask_higher);

    // flag = g.shfl_down(flag, 1);
    flag = __shfl_down_sync(active, flag, 1);

    for (int i = 1; i < 32; i *= 2) {
      Eigen::Matrix<T, N, 1> val_i;
      // mgpu::iterate<N>([&](int j) {val_i[j] = g.shfl_down(val[j], i);});
      for (int j = 0; j < N; ++j) {
        // val_i[j] = g.shfl_down(val[j], i);
        val_i[j] = __shfl_down_sync(active, val[j], i);
      }
      // bool flag_i = g.shfl_down(flag, i);
      bool flag_i = __shfl_down_sync(active, flag, i);
      if (!flag && i + rank <= limit) {
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

template <typename T>
struct WarpSegReduceShfl<
    Eigen::Matrix<T, 3, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <typename F>
  static __device__ __inline__ Eigen::Matrix<T, 3, 1> segreduce(
      unsigned int active, Eigen::Matrix<T, 3, 1> val, bool flag, F f) {
    unsigned int limit = __popc(active);
    unsigned int mask_higher = 0xFFFFFFFF << (threadIdx.x + 1);
    unsigned int rank = __popc(active & ~mask_higher);

    flag = __shfl_down_sync(active, flag, 1);

    Eigen::Matrix<T, 3, 1> val_i;
    bool flag_i;

    // 1
    val_i[0] = __shfl_down_sync(active, val[0], 1);
    val_i[1] = __shfl_down_sync(active, val[1], 1);
    val_i[2] = __shfl_down_sync(active, val[2], 1);
    flag_i = __shfl_down_sync(active, flag, 1);
    if (!flag && 1 + rank <= limit) {
      val[0] = f(val[0], val_i[0]);
      val[1] = f(val[1], val_i[1]);
      val[2] = f(val[2], val_i[2]);
    }
    flag = flag || flag_i;

    // 2
    val_i[0] = __shfl_down_sync(active, val[0], 2);
    val_i[1] = __shfl_down_sync(active, val[1], 2);
    val_i[2] = __shfl_down_sync(active, val[2], 2);
    flag_i = __shfl_down_sync(active, flag, 2);
    if (!flag && 2 + rank <= limit) {
      val[0] = f(val[0], val_i[0]);
      val[1] = f(val[1], val_i[1]);
      val[2] = f(val[2], val_i[2]);
    }
    flag = flag || flag_i;

    // 4
    val_i[0] = __shfl_down_sync(active, val[0], 4);
    val_i[1] = __shfl_down_sync(active, val[1], 4);
    val_i[2] = __shfl_down_sync(active, val[2], 4);
    flag_i = __shfl_down_sync(active, flag, 4);
    if (!flag && 4 + rank <= limit) {
      val[0] = f(val[0], val_i[0]);
      val[1] = f(val[1], val_i[1]);
      val[2] = f(val[2], val_i[2]);
    }
    flag = flag || flag_i;

    // 8
    val_i[0] = __shfl_down_sync(active, val[0], 8);
    val_i[1] = __shfl_down_sync(active, val[1], 8);
    val_i[2] = __shfl_down_sync(active, val[2], 8);
    flag_i = __shfl_down_sync(active, flag, 8);
    if (!flag && 8 + rank <= limit) {
      val[0] = f(val[0], val_i[0]);
      val[1] = f(val[1], val_i[1]);
      val[2] = f(val[2], val_i[2]);
    }
    flag = flag || flag_i;

    // 16
    val_i[0] = __shfl_down_sync(active, val[0], 16);
    val_i[1] = __shfl_down_sync(active, val[1], 16);
    val_i[2] = __shfl_down_sync(active, val[2], 16);
    flag_i = __shfl_down_sync(active, flag, 16);
    if (!flag && 16 + rank <= limit) {
      val[0] = f(val[0], val_i[0]);
      val[1] = f(val[1], val_i[1]);
      val[2] = f(val[2], val_i[2]);
    }
    flag = flag || flag_i;

    return val;
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol

#endif
