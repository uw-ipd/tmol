#pragma once

#include <Eigen/Core>

namespace tmol {
namespace score {
namespace common {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <tmol::Device D, typename T, class Enable = void>
struct accumulate {};

template <tmol::Device D, typename T>
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
    T
  > {
  static def add(T * target, const T & val)->void {
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
  // static def add_kahan(T * target, const T& val)->void {
  //   // from https://devtalk.nvidia.com/default/topic/817899/atomicadd-kahan-summation/
  //   T oldacc = atomicAdd(target,val);
  //   T newacc = oldacc + val;
  //   T r = val - (newacc - oldacc);
  //   atomicAdd(&target[1], r);
  // }

};


union float2UllUnion {
  float2 f;
  unsigned long long int ull;
};

template <typename T>
struct accumulate_kahan<
    tmol::Device::CUDA, T> {

  static
  def
  add(T * address, const T & val)->void {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    float2UllUnion old, assumed, tmp;
    old.ull = *address_as_ull;
    do {
        assumed = old;
        tmp = assumed;
        // kahan summation
        const T y = val - tmp.f.y;
        const T t = tmp.f.x + y;
        tmp.f.y = (t - tmp.f.x) - y;
        tmp.f.x = t;

#ifdef __CUDA_ARCH__
        old.ull = atomicCAS(address_as_ull, assumed.ull, tmp.ull);
#endif

    } while (assumed.ull != old.ull);
  }
};


#endif

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
