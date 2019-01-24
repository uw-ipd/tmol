#include <cppitertools/range.hpp>

#include "hybrid.hh"

template <typename Real>
struct sum<Real, tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  static at::Tensor f(tmol::TView<Real, 1, D> t) {
    at::Tensor v_t;
    tmol::TView<Real, 1, D> v;
    std::tie(v_t, v) = tmol::new_tensor<Real, 1, D>({1});

    v[0] = 0;
    for (int i = 0; i < t.size(0); i++) {
      v[0] += t[i];
    }

    return v_t;
  }
};

template struct sum<float, tmol::Device::CPU>;
template struct sum<double, tmol::Device::CPU>;
