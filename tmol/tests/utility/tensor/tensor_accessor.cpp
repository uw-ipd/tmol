#include <torch/torch.h>
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
    *output[i] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
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
    *output[i] = std::sqrt(*v[0] * *v[0] + *v[1] * *v[1] + *v[2] * *v[2]);
  }

  return output_t;
}

at::Tensor vector_magnitude_eigen(at::Tensor input_t) {
  static const tmol::Device D = tmol::Device::CPU;
  AT_ASSERT(input_t.device().is_cpu());

  auto output_t = at::empty(input_t.size(0), input_t.options());

  auto input = tmol::view_tensor<Eigen::Vector3f, 2, D>(input_t);
  auto output = output_t.accessor<float, 1>();

  for (int64_t i = 0; i < input.size(0); ++i) {
    output[i] = (*input[i][0]).norm();
  }

  return output_t;
}

at::Tensor vector_magnitude_eigen_squeeze(at::Tensor input_t) {
  static const tmol::Device D = tmol::Device::CPU;
  AT_ASSERT(input_t.device().is_cpu());

  auto input = tmol::view_tensor<Eigen::Vector3f, 1, D>(input_t);
  auto output_t = tmol::TPack<float, 1, D>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    *output[i] = (*input[i]).norm();
  }

  return output_t.tensor;
}

auto vector_magnitude_eigen_arg(
    tmol::TView<Eigen::Vector3f, 2, tmol::Device::CPU> input)
    -> tmol::TPack<float, 1, tmol::Device::CPU> {
  auto output_t =
      tmol::TPack<float, 1, tmol::Device::CPU>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    *output[i] = (*input[i][0]).norm();
  }

  return output_t;
}

auto vector_magnitude_eigen_arg_squeeze(
    tmol::TView<Eigen::Vector3f, 1, tmol::Device::CPU> input)
    -> tmol::TPack<float, 1, tmol::Device::CPU> {
  auto output_t =
      tmol::TPack<float, 1, tmol::Device::CPU>::empty(input.size(0));
  auto output = output_t.view;

  for (int64_t i = 0; i < input.size(0); ++i) {
    *output[i] = (*input[i]).norm();
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
      "Eigen-based vector_magnitude function.");
  m.def(
      "eigen_squeeze",
      &vector_magnitude_eigen_squeeze,
      "Eigen-based vector_magnitude function, implicit squeeze of dim-1 minor "
      "dimension.");
  m.def(
      "eigen_arg",
      &vector_magnitude_eigen_arg,
      "Tensoreigen-based vector_magnitude function with pybind11 argument "
      "conversion, implictly squeezing minor dimension.");
  m.def(
      "eigen_arg_squeeze",
      &vector_magnitude_eigen_arg_squeeze,
      "Tensoreigen-based vector_magnitude function with pybind11 argument "
      "conversion, implictly squeezing minor dimension.");

  m.def(
      "tensor_pack_construct",
      &tensor_pack_construct,
      "Construct {2, 5, 3} tensors via TensorPack constructors.");

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
}
