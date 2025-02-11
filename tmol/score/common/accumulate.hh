#pragma once

#include <Eigen/Core>
#include <tmol/score/common/shuffle_reduce.hh>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <moderngpu/operators.hxx>
#include <tmol/numeric/log2.hh>
#endif

namespace tmol {
namespace score {
namespace common {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <tmol::Device D, typename T, class Enable = void>
struct accumulate {};

template <typename T>
struct accumulate<
    tmol::Device::CPU,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->T {
    //  // Try the atomic-add solution from stack overflow:
    //  //
    //  https://stackoverflow.com/questions/48746540/are-there-any-more-efficient-ways-for-atomically-adding-two-floats
    //  int *ip_x= reinterpret_cast<int*>( &target ); //1
    //  int expected= __atomic_load_n( ip_x, __ATOMIC_SEQ_CST ); //2
    //  int desired;
    //  do  {
    //    float sum= *reinterpret_cast<T*>( &expected ) + val; //3
    //    desired=   *reinterpret_cast<int*>( &sum );
    //  } while( ! __atomic_compare_exchange_n( ip_x, &expected, desired, //4
    //                                          /* weak = */ true,
    //                                          __ATOMIC_SEQ_CST,
    //                                          __ATOMIC_SEQ_CST ) );
    //
    T old_target = target;
    target += val;
    return old_target;
  }

  // This is safe to use when all threads are going to write to the same address
  template <class A>
  static def add_one_dst(A& target, int ind, const T& val)->void {
    target[ind] += val;
  }

  // All threads must write to the same ind1; threads may write to different
  // ind0s. The CPU version is safe as long as there's only one thread.
  template <class A>
  static def add_two_dim_one_dst(A& target, int ind0, int ind1, const T& val)
      ->void {
    target[ind0][ind1] += val;
  }
};

// template partial specialization for Eigen matrices
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

  template <class A>
  static def add_one_dst(A& target, int ind, const V& val)->void {
#pragma unroll
    for (int i = 0; i < N; i++) {
      accumulate<D, T>::add_two_dim_one_dst(target, ind, i, val[i]);
    }
  }

  template <class A>
  static def add_two_dim_one_dst(A& target, int ind0, int ind1, const V& val)
      ->void {
    // ???
  }
};

#ifndef __CUDACC__
// Compile this reduction class only for CPU

template <tmol::Device D, typename T, class Enable = void>
struct reduce {};

template <typename T>
struct reduce<
    tmol::Device::CPU,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <class G, class OP>
  static def reduce_to_head(G&, const T& val, OP)->T {
    T retval = val;
    return retval;
  }

  template <class G, class OP>
  static def reduce_to_all(G&, const T& val, OP)->T {
    T retval = val;
    return retval;
  }
};

template <tmol::Device D, int N, typename T>
struct reduce<
    D,
    Eigen::Matrix<T, N, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  typedef Eigen::Matrix<T, N, 1> V;
  template <class G, class OP>
  static def reduce_to_head(G&, const V& val, OP)->T {
    V retval = val;
    return retval;
  }

  template <class G, class OP>
  static def reduce_to_all(G&, const V& val, OP)->T {
    V retval = val;
    return retval;
  }
};

#endif

#ifdef __CUDACC__

template <typename T>
struct accumulate<
    tmol::Device::CUDA,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  static def add(T& target, const T& val)->T { return atomicAdd(&target, val); }

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

  // All threads must write to the same ind1; threads may write to different
  // ind0s. The CPU version is safe as long as there's only one thread.
  template <class A>
  static def add_two_dim_one_dst(A& target, int ind0, int ind1, const T& val)
      ->void {
    // basically
    // target[ind0][ind1] += val;
    // where all threads have the same ind1 and may have different ind0s
#ifdef __CUDA_ARCH__

    auto g = cooperative_groups::coalesced_threads();

    int min_ind0 = reduce_tile_shfl(g, ind0, mgpu::minimum_t<int>());
    min_ind0 = g.shfl(min_ind0, 0);
    int max_ind0 = reduce_tile_shfl(g, ind0, mgpu::maximum_t<int>());
    max_ind0 = g.shfl(max_ind0, 0);

    for (int iter_ind = min_ind0; iter_ind <= max_ind0; ++iter_ind) {
      if (iter_ind == ind0) {
        auto g2 = cooperative_groups::coalesced_threads();
        T warp_sum = reduce_tile_shfl(g2, val, mgpu::plus_t<T>());
        if (g2.thread_rank() == 0 && warp_sum != 0) {
          atomicAdd(&target[ind0][ind1], warp_sum);
        }
      }
      g.sync();
    }
#endif
  }
};

template <tmol::Device D, typename T, class Enable = void>
struct reduce {};

template <typename T>
struct reduce<
    tmol::Device::CUDA,
    T,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  template <class G, class OP>
  static def reduce_to_head(G& g, const T& val, OP op)->T {
    T retval = reduce_tile_shfl(g, val, op);
    return retval;
  }

  template <class G, class OP>
  static def reduce_to_all(G& g, const T& val, OP op)->T {
    T retval = reduce_tile_shfl(g, val, op);
    return retval = g.shfl(retval, 0);
  }
};

template <int N, typename T>
struct reduce<
    tmol::Device::CUDA,
    Eigen::Matrix<T, N, 1>,
    typename std::enable_if<std::is_arithmetic<T>::value>::type> {
  typedef Eigen::Matrix<T, N, 1> V;
  template <class G, class OP>
  static def reduce_to_head(G& g, const V& val, OP op)->V {
    V retval;
    for (int i = 0; i < N; ++i) {
      retval[i] = reduce_tile_shfl(g, val[i], op);
    }
    return retval;
  }

  template <class G, class OP>
  static def reduce_to_all(G& g, const V& val, OP op)->V {
    V retval = val;
    for (int i = 0; i < N; ++i) {
      retval[i] = reduce_tile_shfl(g, val[i], op);
      retval[i] = g.shfl(retval[i], 0);
    }
    return retval;
  }
};

#endif

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
