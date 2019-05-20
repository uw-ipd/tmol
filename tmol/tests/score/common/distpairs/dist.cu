#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

// #include <tmol/score/common/geom.hh>
// #include <tmol/score/common/tuple.hh>
// #include <tmol/score/common/tuple_operators.hh>

#include <tmol/score/common/dist_pairs.cuda.impl.cuh>

template <typename Real, tmol::Device D>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol;
  using namespace tmol::score::common;
  typedef Eigen::Matrix<Real, 3, 1> Real3;

  m.def(
      "triu_distpairs",
      [](TView<Real3, 1, D> coords, Real cutoff) {
        int natoms = coords.size(0);
        int npairs = natoms * (natoms - 1) / 2;
        auto nearby_t = tmol::TPack<int, 1, D>::empty({npairs});
        auto nearby_scan_t = tmol::TPack<int, 1, D>::empty({npairs});
        auto nearby_i_t = tmol::TPack<int, 1, D>::empty({npairs});
        auto nearby_j_t = tmol::TPack<int, 1, D>::empty({npairs});

        auto nearby = nearby_t.view;
        auto nearby_scan = nearby_scan_t.view;
        auto nearby_i = nearby_i_t.view;
        auto nearby_j = nearby_j_t.view;

        TriuDistanceCutoff<Real, int, 8>::f(
            coords, nearby, nearby_scan, nearby_i, nearby_j, cutoff);

        return std::make_tuple(
            nearby_t.tensor,
            nearby_scan_t.tensor,
            nearby_i_t.tensor,
            nearby_j_t.tensor);
      },
      "coords"_a,
      "cutoff"_a);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind<double, tmol::Device::CUDA>(m);
  bind<float, tmol::Device::CUDA>(m);
}
