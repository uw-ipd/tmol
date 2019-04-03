#pragma once

#include <Eigen/Core>

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
  static def add(T& target, const T& val)->void { target += val; }
};  // namespace potentials

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

#endif

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
