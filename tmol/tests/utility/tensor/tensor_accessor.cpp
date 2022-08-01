#include <torch/extension.h>
#include <Eigen/Core>
#include <cmath>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

at::Tensor vector_magnitude_aten(at::Tensor input) {
  return (input * input).sum(-1).sqrt();
}

at::Tensor vector_magnitude_accessor(at::Tensor input_t) {
  static const tmol::Device D = tmol::Device::CPU;
  AT_ASSERT(input_t.device().is_cpu());

  auto input = input_t.accessor<float, 2>();
  auto output_t = tmol::TPack<float, 1, D>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    auto v = input[i];
    output[i] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  return output_t.tensor;
}

auto vector_magnitude_accessor_arg(
    tmol::TView<float, 2, tmol::Device::CPU> input)
    -> tmol::TPack<float, 1, tmol::Device::CPU> {
  auto output_t =
      tmol::TPack<float, 1, tmol::Device::CPU>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    auto v = input[i];
    output[i] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  return output_t;
}

at::Tensor vector_magnitude_eigen(at::Tensor input_t) {
  static const tmol::Device D = tmol::Device::CPU;
  AT_ASSERT(input_t.device().is_cpu());

  auto input = tmol::view_tensor<Eigen::Vector3f, 1, D>(input_t);
  auto output_t = tmol::TPack<float, 1, D>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    output[i] = input[i].norm();
  }

  return output_t.tensor;
}

auto vector_magnitude_eigen_arg(
    tmol::TView<Eigen::Vector3f, 1, tmol::Device::CPU> input)
    -> tmol::TPack<float, 1, tmol::Device::CPU> {
  auto output_t =
      tmol::TPack<float, 1, tmol::Device::CPU>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    output[i] = input[i].norm();
  }

  return output_t;
}

auto matrix_sum_eigen_arg(
    tmol::TView<Eigen::Matrix3f, 1, tmol::Device::CPU> input)
    -> tmol::TPack<float, 1, tmol::Device::CPU> {
  auto output_t =
      tmol::TPack<float, 1, tmol::Device::CPU>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    output[i] = input[i].sum();
  }

  return output_t;
}

auto tensor_pack_construct() {
  typedef tmol::TPack<Eigen::Vector3f, 2, tmol::Device::CPU> T;

  return std::make_tuple(
      T::empty({2, 5}),
      T::ones({2, 5}),
      T::zeros({2, 5}),
      T::full({2, 5}, NAN));
}

auto tensor_pack_construct_like_aten(at::Tensor t) {
  typedef tmol::TPack<Eigen::Vector3f, 2, tmol::Device::CPU> T;

  return std::make_tuple(
      T::empty_like(t),
      T::ones_like(t),
      T::zeros_like(t),
      T::full_like(t, NAN));
}

auto tensor_pack_construct_like_tview(
    tmol::TView<float, 2, tmol::Device::CPU> t) {
  typedef tmol::TPack<Eigen::Vector3f, 2, tmol::Device::CPU> T;

  return std::make_tuple(
      T::empty_like(t),
      T::ones_like(t),
      T::zeros_like(t),
      T::full_like(t, NAN));
}

auto tensor_pack_construct_like_tpack(
    tmol::TPack<float, 2, tmol::Device::CPU> t) {
  typedef tmol::TPack<Eigen::Vector3f, 2, tmol::Device::CPU> T;

  return std::make_tuple(
      T::empty_like(t),
      T::ones_like(t),
      T::zeros_like(t),
      T::full_like(t, NAN));
}

auto tensor_pack_construct_eigen_matrix() {
  typedef tmol::TPack<Eigen::Matrix3f, 2, tmol::Device::CPU> T;

  return std::make_tuple(
      T::empty({2, 5}),
      T::ones({2, 5}),
      T::zeros({2, 5}),
      T::full({2, 5}, NAN));
}

auto tensor_view_take_slice_one() {
  auto Vp = tmol::TPack<int, 2, tmol::Device::CPU>::empty({4, 10});
  auto V = Vp.view;

  int count = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 10; ++j) {
      V[i][j] = count;
      ++count;
    }
  }

  auto Outp = tmol::TPack<int, 1, tmol::Device::CPU>::zeros({4});
  auto Out = Outp.view;

  auto Vslice = V.slice_one(1, 5);
  for (int i = 0; i < 4; ++i) {
    Out[i] = Vslice[i];
  }
  return Outp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "aten", &vector_magnitude_aten, "ATen-based vector_magnitude function.");
  m.def(
      "accessor",
      &vector_magnitude_accessor,
      "TensorAccessor-based vector_magnitude function.");
  m.def(
      "accessor_arg",
      &vector_magnitude_accessor_arg,
      "TensorAccessor-based vector_magnitude function with pybind11 argument "
      "conversion.");
  m.def(
      "eigen",
      &vector_magnitude_eigen,
      "Eigen-based vector_magnitude function, implicit squeeze of dim-1 minor "
      "dimension.");
  m.def(
      "eigen_vector_arg",
      &vector_magnitude_eigen_arg,
      "Tensoreigen-based vector_magnitude function with pybind11 argument "
      "conversion, implictly squeezing minor dimension.");
  m.def(
      "eigen_matrix_arg",
      &matrix_sum_eigen_arg,
      "Tensoreigen-based matrix_sum function with pybind11 argument "
      "conversion, implictly squeezing minor dimensions.");
  m.def(
      "tensor_pack_construct",
      &tensor_pack_construct,
      "Construct {2, 5, 3} tensors via TensorPack constructors.");
  m.def(
      "tensor_pack_construct_eigen_matrix",
      &tensor_pack_construct_eigen_matrix,
      "Construct {2, 5, 3, 3} tensors via TensorPack constructors.");
  m.def(
      "tensor_pack_construct_like_aten",
      &tensor_pack_construct_like_aten,
      "Construct tensors via TensorPack constructors.");

  m.def(
      "tensor_pack_construct_like_tview",
      &tensor_pack_construct_like_tview,
      "Construct tensors via TensorPack constructors.");

  m.def(
      "tensor_pack_construct_like_tpack",
      &tensor_pack_construct_like_tpack,
      "Construct tensors via TensorPack constructors.");

  m.def(
      "tensor_view_take_slice_one",
      &tensor_view_take_slice_one,
      "Construct 2D tensor and take a 1D slice of it");
}
