#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/function_dispatch/pybind.hh>

#include <torch/csrc/Device.h>
#include <torch/csrc/Size.h>

namespace tmol {
namespace utility {
namespace tensor {

// Coerce an input list of Tensors into a TensorCollection
// and then return that coerced variable. Lucky for us, coercion
// happens for free!
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


#define TCOLL_DEF( precision, ndim, dev ) \
  py::class_<TCollection<precision, ndim, dev> >(m, (std::string("TCollection_") + \
    (#precision == "float" ? "f" : "d") + "_" +	\
    #ndim + "_" + \
      (#dev == "tmol::Device::CPU" ? "cpu" : "cuda")).c_str()) \
    .def_property_readonly( "device", [](TCollection<precision, ndim, dev> const & tc) { return py_device(tc.device());}, "the device for the tensors in the TCollection" ) \
    .def( "__len__", &TCollection<precision, ndim, dev>::size, "the number of Tensors in the collection" ) \
    .def( "shape", [](TCollection<precision, ndim, dev> const & tc, size_t entry) {return py_shape(tc, entry);}, "the shape for a particular tensor in the collection" )


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  TCOLL_DEF(float, 1, tmol::Device::CPU);
  TCOLL_DEF(float, 2, tmol::Device::CPU);
  TCOLL_DEF(float, 3, tmol::Device::CPU);
  TCOLL_DEF(float, 4, tmol::Device::CPU);
  
  TCOLL_DEF(double, 1, tmol::Device::CPU);
  TCOLL_DEF(double, 2, tmol::Device::CPU);
  TCOLL_DEF(double, 3, tmol::Device::CPU);
  TCOLL_DEF(double, 4, tmol::Device::CPU);

  bind_dispatch<tmol::Device::CPU, float>(m);
  bind_dispatch<tmol::Device::CPU, double>(m);

  TCOLL_DEF(float, 1, tmol::Device::CUDA);
  TCOLL_DEF(float, 2, tmol::Device::CUDA);
  TCOLL_DEF(float, 3, tmol::Device::CUDA);
  TCOLL_DEF(float, 4, tmol::Device::CUDA);
  
  TCOLL_DEF(double, 1, tmol::Device::CUDA);
  TCOLL_DEF(double, 2, tmol::Device::CUDA);
  TCOLL_DEF(double, 3, tmol::Device::CUDA);
  TCOLL_DEF(double, 4, tmol::Device::CUDA);
  
  bind_dispatch<tmol::Device::CUDA, float>(m);
  bind_dispatch<tmol::Device::CUDA, double>(m);

}


}  // namespace tensor
}  // namespace utility
}  // namespace tmol
