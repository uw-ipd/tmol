#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {

// Helper class for tensor casts.
//
// Wraps a tensor with implicit conversion operators to TView, TPack, and
// Tensor data types. Supports type-inferred casts from torch::Tensor to
// TView/TPack function parameters, with helpful error messages on conversion
// failures.
//
// Use the "TCAST" macro to capture a variable with it's name from the local
// scope.
//
// Prefer this helper type to adding implicit casts to TView to (a) preserve
// zero-dependency definition of TView and (b) preserve type-checking for
// Tensor->TView conversions if casts are explicitly required.
struct TCast {
  at::Tensor tensor;
  const std::string& name;

  TCast(at::Tensor tensor, const std::string& name)
      : tensor(tensor), name(name) {}

  template <typename T, size_t N, Device D, PtrTag P>
  operator TView<T, N, D, P>() {
    return view_tensor<T, N, D, P>(tensor, name);
  };

  template <typename T, size_t N, Device D, PtrTag P>
  operator TPack<T, N, D, P>() {
    return TPack<T, N, D, P>(tensor, view_tensor<T, N, D, P>(tensor, name));
  };

  operator at::Tensor() { return tensor; };
};

#define TCAST(NAME) TCast(NAME, #NAME)

}  // namespace tmol
