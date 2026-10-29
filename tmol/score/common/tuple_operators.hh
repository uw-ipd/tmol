#pragma once

#include "tuple.hh"

#include <tmol/utility/portable/qualifiers.hh>

#define def auto TMOL_HD

namespace tmol {
namespace score {
namespace common {

// iadd (+=) tuple operator
namespace internal {

template <typename T, typename T2, int... Is>
def iadd(
    TMOL_THREAD T const& t1, TMOL_THREAD const T2& t2, index_sequence<Is...>)
    -> void {
  auto l = {(get<Is>(t1) += get<Is>(t2), 0)...};
  (void)l;
}

}  // namespace internal

// lhs is a tie() bundle: a pack of pointers to the variables being updated.
template <typename... Ps, typename... T2>
def iadd(
    tmol::portable::ref_tuple<Ps...> lhs, TMOL_THREAD const tuple<T2...>& rhs)
    -> void {
  internal::iadd(lhs, rhs, index_sequence_for<Ps...>{});
}

// assignment (=) tuple operator
namespace internal {

template <typename T, typename T2, int... Is>
def assign(
    TMOL_THREAD T const& t1, TMOL_THREAD const T2& t2, index_sequence<Is...>)
    -> void {
  auto l = {(get<Is>(t1) = get<Is>(t2), 0)...};
  (void)l;
}

}  // namespace internal

template <typename... Ps, typename... T2>
def assign(
    tmol::portable::ref_tuple<Ps...> lhs, TMOL_THREAD const tuple<T2...>& rhs)
    -> void {
  internal::assign(lhs, rhs, index_sequence_for<Ps...>{});
}

namespace internal {

template <int I, typename... T, typename... T2>
def add_i(TMOL_THREAD const tuple<T...>& a, TMOL_THREAD const tuple<T2...>& b)
    -> decltype(get<I>(a) + get<I>(b)) {
  return get<I>(a) + get<I>(b);
}

template <typename... T, typename... T2, int... I>
def add(
    TMOL_THREAD const tuple<T...>& a,
    TMOL_THREAD const tuple<T2...>& b,
    index_sequence<I...>) {
  return make_tuple(add_i<I>(a, b)...);
}
}  // namespace internal

template <typename... T, typename... T2>
def add(TMOL_THREAD const tuple<T...>& a, TMOL_THREAD const tuple<T2...>& b) {
  return internal::add(a, b, index_sequence_for<T...>{});
}

template <typename... T, typename... T2>
def operator+(
    TMOL_THREAD const tuple<T...>& a, TMOL_THREAD const tuple<T2...>& b) {
  return internal::add(a, b, index_sequence_for<T...>{});
}

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
