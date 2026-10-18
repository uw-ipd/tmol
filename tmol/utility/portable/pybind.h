#pragma once

// pybind11 casters for the portable math types.
//
// These live beside the types rather than in tmol/utility/tensor/pybind.h so
// that a module binding a portable vector or tuple cannot silently miss them
// and fail at runtime with "Unregistered type".

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>

#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/tuple.hh>

namespace pybind11 {
namespace detail {

// tmol::portable::tuple -> Python tuple. The potentials return these from
// astuple() and the test bindings hand them straight to Python. Output only:
// nothing passes a Python tuple back into one of these.
template <typename... Ts>
struct type_caster<tmol::portable::tuple<Ts...>> {
  using Type = tmol::portable::tuple<Ts...>;
  static constexpr size_t size = sizeof...(Ts);

  PYBIND11_TYPE_CASTER(
      Type, _("Tuple[") + concat(make_caster<Ts>::name...) + _("]"));

  bool load(handle, bool) { return false; }

  static handle cast(
      Type const& src, return_value_policy policy, handle parent) {
    return cast_impl(src, policy, parent, make_index_sequence<size>{});
  }

  template <size_t... Is>
  static handle cast_impl(
      Type const& src,
      return_value_policy policy,
      handle parent,
      index_sequence<Is...>) {
    std::array<object, size> entries{
        {reinterpret_steal<object>(make_caster<Ts>::cast(
            tmol::portable::get<Is>(src), policy, parent))...}};
    for (auto const& entry : entries) {
      if (!entry) {
        return handle();
      }
    }
    pybind11::tuple result(size);
    size_t index = 0;
    for (auto& entry : entries) {
      PyTuple_SET_ITEM(result.ptr(), index++, entry.release().ptr());
    }
    return result.release();
  }
};

// tmol::portable::array -> Python tuple. The multi-term potentials return one
// score per term in one of these; like the tuple caster above it is output
// only.
template <typename T, int N>
struct type_caster<tmol::portable::array<T, N>> {
  using Type = tmol::portable::array<T, N>;

  PYBIND11_TYPE_CASTER(
      Type, _("Tuple[") + make_caster<T>::name + _(", ...]"));

  bool load(handle, bool) { return false; }

  static handle cast(
      Type const& src, return_value_policy policy, handle parent) {
    pybind11::tuple result(N);
    for (int i = 0; i < N; ++i) {
      object entry = reinterpret_steal<object>(
          make_caster<T>::cast(src[i], policy, parent));
      if (!entry) {
        return handle();
      }
      PyTuple_SET_ITEM(result.ptr(), i, entry.release().ptr());
    }
    return result.release();
  }
};

// numpy <-> tmol::portable::Mat. pybind11/eigen.h supplies this for Eigen
// types; the portable matrix needs its own. numpy buffers are row-major while
// Mat stores column-major, so the element mapping is explicit.
template <typename T, int R, int C>
struct type_caster<tmol::portable::Mat<T, R, C>> {
  using Type = tmol::portable::Mat<T, R, C>;
  PYBIND11_TYPE_CASTER(
      Type,
      _("numpy.ndarray[") + npy_format_descriptor_name<T>::name + _("[")
          + _<R>() + _(", ") + _<C>() + _("]]"));

  bool load(handle src, bool convert) {
    // Honor pybind's two-phase overload resolution: in the non-converting
    // pass only an exact dtype match may bind, otherwise a float overload
    // would greedily forcecast double arguments and silently lose precision.
    if (!convert && !isinstance<array_t<T>>(src)) {
      return false;
    }
    auto buf = array_t<T, array::c_style | array::forcecast>::ensure(src);
    if (!buf || buf.size() != static_cast<ssize_t>(R) * C) {
      return false;
    }
    T const* p = buf.data();
    for (int i = 0; i < R; ++i) {
      for (int j = 0; j < C; ++j) {
        value(i, j) = p[i * C + j];
      }
    }
    return true;
  }

  static handle cast(Type const& src, return_value_policy, handle) {
    std::vector<ssize_t> shape;
    if (C == 1) {
      shape = {R};
    } else {
      shape = {R, C};
    }
    array_t<T> out(shape);
    T* p = out.mutable_data();
    for (int i = 0; i < R; ++i) {
      for (int j = 0; j < C; ++j) {
        p[i * C + j] = src(i, j);
      }
    }
    return out.release();
  }
};

template <typename T, int M, int N>
struct npy_format_descriptor_name<tmol::portable::Mat<T, M, N>> {
  static constexpr auto name = npy_format_descriptor_name<T>::name + _("[")
                               + _<M>() + _(", ") + _<N>() + _("]");
};

}  // namespace detail
}  // namespace pybind11
