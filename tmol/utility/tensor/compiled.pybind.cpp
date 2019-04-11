#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

//#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/function_dispatch/pybind.hh>

#include <torch/csrc/Device.h>
#include <torch/csrc/Size.h>

namespace tmol {
namespace utility {
namespace tensor {

// Basically coerce an input list of Tensors into a TensorCollection
// and then return that coerced variable. Lucky for us, coercion happens for free!
template <tmol::Device D, typename Real, size_t N>
struct TensorCollectionCreator {
  static auto f(
    std::vector<at::Tensor> tensors
  )
  {
    return std::make_unique< TCollection<Real, N, D> >(tensors);
  }
};


pybind11::handle
py_device( at::Device const & device )
{
  // ATen provided wrapper for an at::Device
  PyObject * ptr = THPDevice_New(device);
  pybind11::handle h(ptr);
  return h;
}

pybind11::handle
py_shape_from_tensor( at::Tensor const & tensor )
{
  PyObject * ptr = THPSize_NewFromSizes(tensor.dim(), tensor.sizes().data());
  return pybind11::handle(ptr);
}

template <typename T, size_t N, Device D, PtrTag P>
pybind11::handle
py_shape( tmol::TCollection<T, N, D, P> const & tc, size_t entry )
{
  AT_ASSERTM(
    entry < tc.tensors.size(),
    "requested tensor is out of range for TCollection");
  return py_shape_from_tensor(tc.tensors[entry].tensor);
}


template <tmol::Device Dev, typename Real>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "create_tensor_collection1",
      &TensorCollectionCreator<Dev, Real, 1>::f,
      "collection"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "create_tensor_collection2",
      &TensorCollectionCreator<Dev, Real, 2>::f,
      "collection"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "create_tensor_collection3",
      &TensorCollectionCreator<Dev, Real, 3>::f,
      "collection"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "create_tensor_collection4",
      &TensorCollectionCreator<Dev, Real, 4>::f,
      "collection"_a);
};

//#define TCOLL_FUNCS(classname, dev) .def( "device", #dev &#classname::device(), "return tensors' device" )
//template <typename T, size_t N, Device D, PtrTag P = PtrTag::Restricted>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  py::class_<TCollection<float, 1, tmol::Device::CPU> >(m, "TCollection_f_1_cpu")
    .def_property_readonly( "device", [](TCollection<float, 1, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 1, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 1, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<float, 2, tmol::Device::CPU> >(m, "TCollection_f_2_cpu")
    .def_property_readonly( "device", [](TCollection<float, 2, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 2, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 2, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<float, 3, tmol::Device::CPU> >(m, "TCollection_f_3_cpu")
    .def_property_readonly( "device", [](TCollection<float, 3, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 3, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 3, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<float, 4, tmol::Device::CPU> >(m, "TCollection_f_4_cpu")
    .def_property_readonly( "device", [](TCollection<float, 4, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 4, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 4, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );

  py::class_<TCollection<double, 1, tmol::Device::CPU> >(m, "TCollection_double_1_cpu")
    .def_property_readonly( "device", [](TCollection<double, 1, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<double, 1, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<double, 1, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<double, 2, tmol::Device::CPU> >(m, "TCollection_double_2_cpu")
    .def_property_readonly( "device", [](TCollection<double, 2, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<double, 2, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<double, 2, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<double, 3, tmol::Device::CPU> >(m, "TCollection_double_3_cpu")
    .def_property_readonly( "device", [](TCollection<double, 3, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<double, 3, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<double, 3, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<double, 4, tmol::Device::CPU> >(m, "TCollection_double_4_cpu")
    .def_property_readonly( "device", [](TCollection<double, 4, tmol::Device::CPU> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<double, 4, tmol::Device::CPU>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<double, 4, tmol::Device::CPU> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );

  bind_dispatch<tmol::Device::CPU, float>(m);
  bind_dispatch<tmol::Device::CPU, double>(m);

  py::class_<TCollection<float, 1, tmol::Device::CUDA> >(m, "TCollection_f_1_cuda")
    .def_property_readonly( "device", [](TCollection<float, 1, tmol::Device::CUDA> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 1, tmol::Device::CUDA>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 1, tmol::Device::CUDA> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<float, 2, tmol::Device::CUDA> >(m, "TCollection_f_2_cuda")
    .def_property_readonly( "device", [](TCollection<float, 2, tmol::Device::CUDA> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 2, tmol::Device::CUDA>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 2, tmol::Device::CUDA> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<float, 3, tmol::Device::CUDA> >(m, "TCollection_f_3_cuda")
    .def_property_readonly( "device", [](TCollection<float, 3, tmol::Device::CUDA> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 3, tmol::Device::CUDA>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 3, tmol::Device::CUDA> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );
  py::class_<TCollection<float, 4, tmol::Device::CUDA> >(m, "TCollection_f_4_cuda")
    .def_property_readonly( "device", [](TCollection<float, 4, tmol::Device::CUDA> const & tc) { return py_device(tc.device());}, "device" )
    .def( "__len__", &TCollection<float, 4, tmol::Device::CUDA>::size, "the number of Tensors in the collection" )
    .def( "shape", [](TCollection<float, 4, tmol::Device::CUDA> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" );

  bind_dispatch<tmol::Device::CUDA, float>(m);

}


}  // namespace tensor
}  // namespace utility
}  // namespace tmol
