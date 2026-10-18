#pragma once

#include "qualifiers.hh"
#include "traits.hh"

namespace tmol {
namespace portable {

// Minimal fixed-arity tuple.
//
// tmol's potentials return several values at once (a value and its partial
// derivatives) and unpack them with tie(). std::tuple is unavailable in Metal
// and mgpu::tuple is CUDA-only, so this supplies one spelling for all three
// toolchains. Only the operations tmol uses are provided: construction,
// get<I>, make_tuple, and tie.

// Compile-time index pack, standing in for std::index_sequence.
template <int... Is>
struct index_sequence {};

namespace detail {

// Written with a member typedef rather than inheritance: Metal does not
// support derived classes.
template <int N, int... Is>
struct make_index_seq {
  typedef typename make_index_seq<N - 1, N - 1, Is...>::type type;
};

template <int... Is>
struct make_index_seq<0, Is...> {
  typedef index_sequence<Is...> type;
};

}  // namespace detail

template <int N>
using make_index_sequence = typename detail::make_index_seq<N>::type;

template <typename... Ts>
using index_sequence_for = make_index_sequence<int(sizeof...(Ts))>;

template <typename... Ts>
struct tuple;

template <>
struct tuple<> {
  TMOL_HD tuple() {}
};

template <typename T, typename... Ts>
struct tuple<T, Ts...> {
  T head;
  tuple<Ts...> tail;

  TMOL_HD tuple() : head(), tail() {}

  // Element-wise converting constructor. Potentials routinely build a tuple
  // from expressions of a wider type (a double-precision polynomial feeding a
  // float result), which aggregate initialization would reject as narrowing;
  // this matches std::tuple's behavior of converting instead.
  template <
      typename U,
      typename... Us,
      typename =
          typename enable_if<sizeof...(Us) == sizeof...(Ts)>::type>
  TMOL_HD tuple(TMOL_THREAD U const& h, TMOL_THREAD Us const&... t)
      : head(T(h)), tail(t...) {}

  // Cross-type assignment is what makes `tie(a, b) = f()` unpack: the left
  // side is a tuple of references and the right side a tuple of values.
  template <typename... Us>
  TMOL_HD TMOL_THREAD tuple& operator=(TMOL_THREAD tuple<Us...> const& o) {
    head = o.head;
    tail = o.tail;
    return *this;
  }
};

namespace detail {

template <int I, typename Tup>
struct tuple_at;

template <typename T, typename... Ts>
struct tuple_at<0, tuple<T, Ts...>> {
  typedef T type;
  TMOL_HD static TMOL_THREAD T& get(TMOL_THREAD tuple<T, Ts...>& t) {
    return t.head;
  }
  TMOL_HD static TMOL_THREAD T const& get(
      TMOL_THREAD tuple<T, Ts...> const& t) {
    return t.head;
  }
};

template <int I, typename T, typename... Ts>
struct tuple_at<I, tuple<T, Ts...>> {
  typedef typename tuple_at<I - 1, tuple<Ts...>>::type type;
  TMOL_HD static TMOL_THREAD type& get(TMOL_THREAD tuple<T, Ts...>& t) {
    return tuple_at<I - 1, tuple<Ts...>>::get(t.tail);
  }
  TMOL_HD static TMOL_THREAD type const& get(
      TMOL_THREAD tuple<T, Ts...> const& t) {
    return tuple_at<I - 1, tuple<Ts...>>::get(t.tail);
  }
};

}  // namespace detail

template <int I, typename... Ts>
TMOL_HD TMOL_THREAD typename detail::tuple_at<I, tuple<Ts...>>::type& get(
    TMOL_THREAD tuple<Ts...>& t) {
  return detail::tuple_at<I, tuple<Ts...>>::get(t);
}

template <int I, typename... Ts>
TMOL_HD TMOL_THREAD typename detail::tuple_at<I, tuple<Ts...>>::type const&
get(TMOL_THREAD tuple<Ts...> const& t) {
  return detail::tuple_at<I, tuple<Ts...>>::get(t);
}

// Number of elements, for generic code such as the pybind caster.
template <typename Tup>
struct tuple_size;

template <typename... Ts>
struct tuple_size<tuple<Ts...>> {
  enum { value = sizeof...(Ts) };
};

TMOL_HD inline tuple<> make_tuple() { return tuple<>(); }

// Taken by value rather than by const reference: Metal folds the address space
// into a type deduced from a reference parameter, so `T const&` binding a
// thread-space int deduces T as `thread int`, and a struct field may not be
// address-space qualified. Deducing from a by-value parameter yields the plain
// type. make_tuple copies its arguments either way.
template <typename T, typename... Ts>
TMOL_HD tuple<T, Ts...> make_tuple(T v, Ts... vs) {
  return tuple<T, Ts...>(v, vs...);
}

// tie() yields a bundle of pointers to the tied variables so that
// `tie(a, b) = f()` unpacks a returned tuple.
//
// Pointers rather than references: in Metal the address space is part of a
// reference's type, and a struct field may not be address-space qualified, so
// `tuple<T&, ...>` is not a type that can exist there. The pointer types are
// picked up from decltype(&v), which carries whichever address space the
// caller's variables live in.
template <typename... Ps>
struct ref_tuple;

template <>
struct ref_tuple<> {
  template <typename Tup>
  TMOL_HD void assign(TMOL_THREAD Tup const&) const {}
};

template <typename P, typename... Ps>
struct ref_tuple<P, Ps...> {
  P head;
  ref_tuple<Ps...> tail;

  template <typename U, typename... Us>
  TMOL_HD void assign(TMOL_THREAD tuple<U, Us...> const& t) const {
    *head = t.head;
    tail.assign(t.tail);
  }

  template <typename Tup>
  TMOL_HD void operator=(TMOL_THREAD Tup const& t) const {
    assign(t);
  }
};

namespace detail {

template <int I, typename RT>
struct ref_tuple_at;

template <typename P, typename... Ps>
struct ref_tuple_at<0, ref_tuple<P, Ps...>> {
  TMOL_HD static P ptr(TMOL_THREAD ref_tuple<P, Ps...> const& t) {
    return t.head;
  }
};

template <int I, typename P, typename... Ps>
struct ref_tuple_at<I, ref_tuple<P, Ps...>> {
  TMOL_HD static auto ptr(TMOL_THREAD ref_tuple<P, Ps...> const& t)
      -> decltype(ref_tuple_at<I - 1, ref_tuple<Ps...>>::ptr(t.tail)) {
    return ref_tuple_at<I - 1, ref_tuple<Ps...>>::ptr(t.tail);
  }
};

}  // namespace detail

// get<I> on a tie() bundle yields the tied variable itself, so that generic
// code can treat it the same way it treats a tuple element.
template <int I, typename... Ps>
TMOL_HD auto get(TMOL_THREAD ref_tuple<Ps...> const& t)
    -> decltype(*detail::ref_tuple_at<I, ref_tuple<Ps...>>::ptr(t)) {
  return *detail::ref_tuple_at<I, ref_tuple<Ps...>>::ptr(t);
}

template <typename... Ps>
struct tuple_size<ref_tuple<Ps...>> {
  enum { value = sizeof...(Ps) };
};

TMOL_HD inline ref_tuple<> tie() { return ref_tuple<>(); }

template <typename T, typename... Ts>
TMOL_HD auto tie(TMOL_THREAD T& v, TMOL_THREAD Ts&... vs)
    -> ref_tuple<decltype(&v), decltype(&vs)...> {
  return {&v, {&vs...}};
}

}  // namespace portable
}  // namespace tmol
