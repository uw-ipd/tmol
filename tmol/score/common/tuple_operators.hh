#pragma once

#include "tuple.hh"

#ifdef __CUDACC__
#define def auto __host__ __device__ __inline__
#else
#define def auto
#endif

namespace tmol {
namespace score {
namespace common {

// iadd (+=) tuple operator
namespace internal {

template <typename T, typename T2, size_t... Is>
def iadd(T& t1, const T2& t2, integer_sequence<size_t, Is...>) -> void {
  auto l = {(get<Is>(t1) += get<Is>(t2), 0)...};
  (void)l;
}

}  // namespace internal

template <typename... T, typename... T2>
def iadd(tuple<T&...> lhs, const tuple<T2...>& rhs) -> void {
  internal::iadd(lhs, rhs, index_sequence_for<T...>{});
}

// assignment (=) tuple operator
namespace internal {

template <typename T, typename T2, size_t... Is>
def assign(T& t1, const T2& t2, integer_sequence<size_t, Is...>) -> void {
  auto l = {(get<Is>(t1) = get<Is>(t2), 0)...};
  (void)l;
}

}  // namespace internal

template <typename... T, typename... T2>
def assign(tuple<T&...> lhs, const tuple<T2...>& rhs) -> void {
  internal::assign(lhs, rhs, index_sequence_for<T...>{});
}

namespace internal {

template <size_t I, typename... T, typename... T2>
def add_i(const tuple<T...>& a, const tuple<T2...>& b)
    -> decltype(get<I>(a) + get<I>(b)) {
  return get<I>(a) + get<I>(b);
}

template <typename... T, typename... T2, size_t... I>
def add(
    const tuple<T...>& a,
    const tuple<T2...>& b,
    integer_sequence<size_t, I...>) {
  return make_tuple(add_i<I>(a, b)...);
}
}  // namespace internal

template <typename... T, typename... T2>
def add(const tuple<T...>& a, const tuple<T2...>& b) {
  return internal::add(a, b, index_sequence_for<T...>{});
}

template <typename... T, typename... T2>
def operator+(const tuple<T...>& a, const tuple<T2...>& b) {
  return internal::add(a, b, index_sequence_for<T...>{});
}

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
