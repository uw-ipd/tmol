#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

template <typename Real, tmol::Device D>
struct sumx {
  static at::Tensor f(tmol::TView<Real, 1, D> t);
};
