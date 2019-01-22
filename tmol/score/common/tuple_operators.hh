#pragma once

#include <tuple>
#include <utility>

namespace tmol {
namespace score {
namespace common {

// iadd (+=) tuple operator
namespace internal {

template <typename T, typename T2, size_t... Is>
void iadd(T& t1, const T2& t2, std::integer_sequence<size_t, Is...>) {
  auto l = {(std::get<Is>(t1) += std::get<Is>(t2), 0)...};
  (void)l;
}

}  // namespace internal

template <typename... T, typename... T2>
void iadd(std::tuple<T&...> lhs, const std::tuple<T2...>& rhs) {
  internal::iadd(lhs, rhs, std::index_sequence_for<T...>{});
}

namespace internal {

template <size_t I, typename... T, typename... T2>
auto add_i(const std::tuple<T...>& a, const std::tuple<T2...>& b) -> decltype(std::get<I>(a) + std::get<I>(b)) {
  return std::get<I>(a) + std::get<I>(b);
}

template <typename... T, typename... T2, size_t... I>
auto add(
    const std::tuple<T...>& a,
    const std::tuple<T2...>& b,
    std::integer_sequence<size_t, I...>) {
  return std::make_tuple(add_i<I>(a, b)...);
}
}  // namespace internal

template <typename... T, typename... T2>
auto add(const std::tuple<T...>& a, const std::tuple<T2...>& b) {
  return internal::add(a, b, std::index_sequence_for<T...>{});
}

template <typename... T, typename... T2>
auto operator+(
    const std::tuple<T...>& a, const std::tuple<T2...>& b) {
  return internal::add(a, b, std::index_sequence_for<T...>{});
}

}  // namespace common
}  // namespace score
}  // namespace tmol
