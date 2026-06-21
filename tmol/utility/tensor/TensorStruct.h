#pragma once

#include <string>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
namespace tmol {

template <
    typename T,
    int N,
    Device D,
    PtrTag P = PtrTag::Restricted,
    typename TMap,
    typename TKey,
    typename std::enable_if<enable_tensor_view<T>::enabled>::type* = nullptr>
auto view_tensor_item(TMap& input_map, TKey key) -> tmol::TView<T, N, D, P> {
  auto key_t = input_map.find(key);

  AT_ASSERTM(
      key_t != input_map.end(),
      "Map does not contain key '" + (std::string)key + "'");

  try {
    return view_tensor<T, N, D, P>(key_t->second, key);
  } catch (at::Error err) {
    AT_ERROR(
        "Error viewing tensor map key '" + (std::string)key + "': \n"
        + err.what_without_backtrace());
  }
}

}  // namespace tmol
