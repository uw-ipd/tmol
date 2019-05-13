#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/geom.hh>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
struct CartBondedLengthDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 2>, 1, D> atompair_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Real, 1, D> K,
      TView<Real, 1, D> x0)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atompair_indices[i][0];
      Int atj = atompair_indices[i][1];
      Int pari = parameter_indices[i];

      auto cblength = cblength_V_dV(
          coords[ati], coords[atj], K[pari], x0[pari]);

      V[0] += std::get<0>(cblength);
      dV_dx[ati] += std::get<1>(cblength);
      dV_dx[atj] += std::get<2>(cblength);
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {V_t, dV_dx_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedAngleDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 3>, 1, D> atomtriple_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Real, 1, D> K,
      TView<Real, 1, D> x0)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atomtriple_indices[i][0];
      Int atj = atomtriple_indices[i][1];
      Int atk = atomtriple_indices[i][2];
      Int pari = parameter_indices[i];
      auto cbangle = cbangle_V_dV(
          coords[ati], coords[atj], coords[atk], K[pari], x0[pari]);

      V[0] += std::get<0>(cbangle);
      dV_dx[ati] += std::get<1>(cbangle);
      dV_dx[atj] += std::get<2>(cbangle); 
      dV_dx[atk] += std::get<3>(cbangle);

    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {V_t, dV_dx_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 4>, 1, D> atomquad_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Real, 1, D> K,
      TView<Real, 1, D> x0,
      TView<Int, 1, D> period)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atomquad_indices[i][0];
      Int atj = atomquad_indices[i][1];
      Int atk = atomquad_indices[i][2];
      Int atl = atomquad_indices[i][3];
      Int pari = parameter_indices[i];
      auto cbtorsion = cbtorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          K[pari],
          x0[pari],
          period[pari]);

      V[0] += std::get<0>(cbtorsion);
      dV_dx[ati] += std::get<1>(cbtorsion);
      dV_dx[atj] += std::get<2>(cbtorsion); 
      dV_dx[atk] += std::get<3>(cbtorsion);
      dV_dx[atl] += std::get<4>(cbtorsion);
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {V_t, dV_dx_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedHxlTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 4>, 1, D> atomquad_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Real, 1, D> K1,
      TView<Real, 1, D> K2,
      TView<Real, 1, D> K3,
      TView<Real, 1, D> phi1,
      TView<Real, 1, D> phi2,
      TView<Real, 1, D> phi3)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atomquad_indices[i][0];
      Int atj = atomquad_indices[i][1];
      Int atk = atomquad_indices[i][2];
      Int atl = atomquad_indices[i][3];
      Int pari = parameter_indices[i];
      auto cbhxltorsion =
          cbhxltorsion_V_dV(
              coords[ati],
              coords[atj],
              coords[atk],
              coords[atl],
              K1[pari],
              K2[pari],
              K3[pari],
              phi1[pari],
              phi2[pari],
              phi3[pari]);

      V[0] += std::get<0>(cbhxltorsion);
      dV_dx[ati] += std::get<1>(cbhxltorsion);
      dV_dx[atj] += std::get<2>(cbhxltorsion); 
      dV_dx[atk] += std::get<3>(cbhxltorsion);
      dV_dx[atl] += std::get<4>(cbhxltorsion);

    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {V_t, dV_dx_t};
  }
};

template struct CartBondedLengthDispatch<tmol::Device::CPU, float, int32_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU, double, int32_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU, float, int64_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU, double, int64_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU, float, int32_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU, double, int32_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU, float, int64_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU, double, int64_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU, float, int32_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU, double, int32_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU, float, int64_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU, double, int64_t>;
template struct CartBondedHxlTorsionDispatch<tmol::Device::CPU, float, int32_t>;
template struct CartBondedHxlTorsionDispatch<
    tmol::Device::CPU,
    double,
    int32_t>;
template struct CartBondedHxlTorsionDispatch<tmol::Device::CPU, float, int64_t>;
template struct CartBondedHxlTorsionDispatch<
    tmol::Device::CPU,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
