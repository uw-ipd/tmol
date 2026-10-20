#pragma once

#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/traits.hh>
#include <tmol/score/common/shuffle_reduce.hh>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <moderngpu/operators.hxx>
#include <tmol/numeric/log2.hh>
#endif

namespace tmol {
namespace score {
namespace common {

#define def auto TMOL_HD

template <tmol::Device D, typename T, class Enable = void>
struct accumulate {};

template <typename T>
struct accumulate<
    tmol::Device::CPU,
    T,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  // CPU accumulation is intentionally non-atomic. Parallel callers must write
  // to disjoint output elements, as pose-parallel scoring does.
  static def add(TMOL_THREAD T& target, TMOL_THREAD const T& val) -> T {
    T old_target = target;
    target += val;
    return old_target;
  }

  template <class A>
  static def add_one_dst(
      TMOL_THREAD A& target, int ind, TMOL_THREAD const T& val) -> void {
    target[ind] += val;
  }

  template <class A>
  static def add_two_dim_one_dst(
      TMOL_THREAD A& target, int ind0, int ind1, TMOL_THREAD const T& val)
      -> void {
    target[ind0][ind1] += val;
  }
};

// template partial specialization for Eigen matrices
template <tmol::Device D, int N, typename T>
struct accumulate<
    D,
    tmol::portable::Vec<T, N>,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  typedef tmol::portable::Vec<T, N> V;

  static def add(TMOL_THREAD V& target, TMOL_THREAD const V& val) -> void {
#pragma unroll
    for (int i = 0; i < N; i++) {
      accumulate<D, T>::add(target[i], val[i]);
    }
  }

  template <class A>
  static def add_one_dst(
      TMOL_THREAD A& target, int ind, TMOL_THREAD const V& val) -> void {
#pragma unroll
    for (int i = 0; i < N; i++) {
      accumulate<D, T>::add_two_dim_one_dst(target, ind, i, val[i]);
    }
  }
};

#if !TMOL_PORTABLE_METAL

// The host also instantiates kernels under the MPS tag -- see
// device_operations.mps.host.impl.hh for why -- and those instantiations are
// dead code that still has to compile. Non-atomic accumulation is the right
// stand-in, so this borrows the CPU behavior. The real Metal accumulation,
// which is atomic, is the specialization below.
template <typename T>
struct accumulate<
    tmol::Device::MPS,
    T,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type>
    : accumulate<tmol::Device::CPU, T> {};

#endif

// Accumulation of derivative contributions, as distinct from energies.
//
// Identical to `accumulate` on CPU and CUDA -- the alias exists so the Metal
// backend can treat the two differently, and so a reader of a kernel can see
// which is which at the call site. See the Metal specialization for why
// derivatives need the distinction.
#if TMOL_PORTABLE_METAL
// Declared, not defined: Metal has no derived classes, so the generic
// "same as accumulate" form below cannot be spelled here, and a Metal
// translation unit only ever instantiates the MPS specialization anyway.
template <tmol::Device D, typename T, typename Enable = void>
struct accumulate_grad;
#else
template <tmol::Device D, typename T, typename Enable = void>
struct accumulate_grad : accumulate<D, T> {};
#endif

#if TMOL_PORTABLE_METAL

// Metal accumulation is atomic, as CUDA's is: several threadgroups reach the
// same whole-pose energy slot and several threads reach the same derivative
// slot.
//
// MSL has no atomic reference type, so the target is reinterpreted as a
// pointer to atomic storage. This is well defined here because the targets are
// elements of device buffers backed by torch tensors, which are suitably
// aligned, and because nothing reads them non-atomically while a kernel runs.
//
// The value is taken by value, not by const reference. Metal folds the
// address space into a type deduced from a reference parameter, so a
// `const T&` value argument deduces T as `thread float` and then conflicts
// with the T deduced from the device-space target. Deducing from a by-value
// parameter yields the plain scalar type.
//
// There is no coalesced-group equivalent of the CUDA add_one_dst
// optimization, which pre-reduces within a warp before issuing one atomicAdd.
// Metal's simdgroup functions cannot do the ballot-driven variant, so these
// issue one atomic per lane.
template <typename T>
struct accumulate<
    tmol::Device::MPS,
    T,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  static def add(device T& target, T val) -> void {
    metal::atomic_fetch_add_explicit(
        reinterpret_cast<device metal::atomic<T>*>(&target),
        val,
        metal::memory_order_relaxed);
  }

  static def add(threadgroup T& target, T val) -> void {
    metal::atomic_fetch_add_explicit(
        reinterpret_cast<threadgroup metal::atomic<T>*>(&target),
        val,
        metal::memory_order_relaxed);
  }

  template <class A>
  static def add_one_dst(TMOL_THREAD A& target, int ind, T val) -> void {
    add(target[ind], val);
  }

  template <class A>
  static def add_two_dim_one_dst(
      TMOL_THREAD A& target, int ind0, int ind1, T val) -> void {
    add(target[ind0][ind1], val);
  }
};

// Vectors accumulate component-wise, as elsewhere, but the target lives in a
// device buffer so the address space has to be named. This is more
// specialized than the backend-agnostic Vec partial specialization above and
// so wins for Metal.
template <int N, typename T>
struct accumulate<
    tmol::Device::MPS,
    tmol::portable::Vec<T, N>,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  typedef tmol::portable::Vec<T, N> V;

  static def add(device V& target, V val) -> void {
    for (int i = 0; i < N; i++) {
      accumulate<tmol::Device::MPS, T>::add(target[i], val[i]);
    }
  }

  template <class A>
  static def add_one_dst(TMOL_THREAD A& target, int ind, V val) -> void {
    for (int i = 0; i < N; i++) {
      accumulate<tmol::Device::MPS, T>::add_two_dim_one_dst(
          target, ind, i, val[i]);
    }
  }
};

// Derivative accumulation on Metal, snapped to a fixed grid.
//
// Several threadgroups reach the same atom's derivative slot, and the order
// they arrive in varies from launch to launch. Float addition is not
// associative, so the accumulated gradient is neither reproducible nor a
// smooth function of the coordinates -- and a line search comparing an
// analytic derivative against a finite difference needs both. Measured on a
// 100-pose torsion relax, the resulting thrash cost about sixty per cent more
// energy evaluations than the host, which ate the entire speedup the Metal
// kernels had won. Substituting the host's gradient while keeping the Metal
// energy restored the host's evaluation count exactly, which is what
// identified this.
//
// Rounding every contribution to a multiple of 2^-13 makes the sum exact and
// therefore order-independent: a float32 holds an exact multiple of 2^-13 up
// to a magnitude of 2^23 * 2^-13 = 1024, so every partial sum below that is
// represented without rounding and the adds commute. Derivative components in
// practice reach tens, not hundreds. Past 1024 exactness lapses and that
// component is approximate again -- a graceful degradation, not a wrong
// answer.
//
// The grid costs accuracy: each contribution moves by up to 6.1e-05. That was
// measured against the host's exact gradient in a relax and is well inside
// what the minimizer tolerates -- a gradient deliberately coarsened to 1e-03
// changed the evaluation count by two per cent.
//
// CPU and CUDA accumulate exactly as before; this is a Metal-only trade.
template <typename T>
struct accumulate_grad<
    tmol::Device::MPS,
    T,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  // 2^13. The reciprocal is a power of two, so scaling back is exact.
  enum { grid = 8192 };

  static def snap(T val) -> T {
    return tmol::portable::math::rint(val * T(grid)) * (T(1) / T(grid));
  }

  static def add(device T& target, T val) -> void {
    accumulate<tmol::Device::MPS, T>::add(target, snap(val));
  }

  static def add(threadgroup T& target, T val) -> void {
    accumulate<tmol::Device::MPS, T>::add(target, snap(val));
  }

  template <class A>
  static def add_one_dst(TMOL_THREAD A& target, int ind, T val) -> void {
    add(target[ind], val);
  }

  template <class A>
  static def add_two_dim_one_dst(
      TMOL_THREAD A& target, int ind0, int ind1, T val) -> void {
    add(target[ind0][ind1], val);
  }
};

#endif

#ifdef __CUDACC__

template <typename T>
struct accumulate<
    tmol::Device::CUDA,
    T,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  static def add(TMOL_THREAD T& target, TMOL_THREAD const T& val) -> T {
    return atomicAdd(&target, val);
  }

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
  static def add_one_dst(
      TMOL_THREAD A& target, int ind, TMOL_THREAD const T& val) -> void {
#ifdef __CUDA_ARCH__

    auto g = cooperative_groups::coalesced_threads();

    int min_ind = reduce_tile_shfl(g, ind, tmol::portable::minimum_t<int>());
    min_ind = g.shfl(min_ind, 0);
    int max_ind = reduce_tile_shfl(g, ind, tmol::portable::maximum_t<int>());
    max_ind = g.shfl(max_ind, 0);

    for (int iter_ind = min_ind; iter_ind <= max_ind; ++iter_ind) {
      if (iter_ind == ind) {
        auto g2 = cooperative_groups::coalesced_threads();
        T warp_sum = reduce_tile_shfl(g2, val, tmol::portable::plus_t<T>());
        if (g2.thread_rank() == 0 && warp_sum != 0) {
          atomicAdd(&target[ind], warp_sum);
        }
      }
      g.sync();
    }
#endif
  }

  // All threads must write to the same ind1; threads may write to different
  // ind0s.
  template <class A>
  static def add_two_dim_one_dst(
      TMOL_THREAD A& target, int ind0, int ind1, TMOL_THREAD const T& val)
      -> void {
    // basically
    // target[ind0][ind1] += val;
    // where all threads have the same ind1 and may have different ind0s
#ifdef __CUDA_ARCH__

    auto g = cooperative_groups::coalesced_threads();

    int min_ind0 = reduce_tile_shfl(g, ind0, tmol::portable::minimum_t<int>());
    min_ind0 = g.shfl(min_ind0, 0);
    int max_ind0 = reduce_tile_shfl(g, ind0, tmol::portable::maximum_t<int>());
    max_ind0 = g.shfl(max_ind0, 0);

    for (int iter_ind = min_ind0; iter_ind <= max_ind0; ++iter_ind) {
      if (iter_ind == ind0) {
        auto g2 = cooperative_groups::coalesced_threads();
        T warp_sum = reduce_tile_shfl(g2, val, tmol::portable::plus_t<T>());
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
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  template <class G, class OP>
  static def reduce_to_head(G& g, const T& val, OP op) -> T {
    T retval = reduce_tile_shfl(g, val, op);
    return retval;
  }

  template <class G, class OP>
  static def reduce_to_all(G& g, const T& val, OP op) -> T {
    T retval = reduce_tile_shfl(g, val, op);
    return retval = g.shfl(retval, 0);
  }
};

template <int N, typename T>
struct reduce<
    tmol::Device::CUDA,
    tmol::portable::Vec<T, N>,
    typename tmol::portable::enable_if<
        tmol::portable::is_arithmetic<T>::value>::type> {
  typedef tmol::portable::Vec<T, N> V;
  template <class G, class OP>
  static def reduce_to_head(G& g, const V& val, OP op) -> V {
    V retval;
    for (int i = 0; i < N; ++i) {
      retval[i] = reduce_tile_shfl(g, val[i], op);
    }
    return retval;
  }

  template <class G, class OP>
  static def reduce_to_all(G& g, const V& val, OP op) -> V {
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
