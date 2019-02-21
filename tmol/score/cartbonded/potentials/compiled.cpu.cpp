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


template <
	tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedLengthDispatch {
  static auto f(
    TView<Vec<Int, 2>, 1, D> atompair_indices,
    TView<Int, 1, D> parameter_indices,
    TView<Vec<Real, 3>, 1, D> coords,
    TView<Real, 1, D> K,
    TView<Real, 1, D> x0
  )
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atompair_indices[i][0];
      Int atj = atompair_indices[i][1];
      Int pari = parameter_indices[i];
      tie(Vs[i],dV_dIs[i],dV_dJs[i]) = cblength_V_dV(
        coords[ati], coords[atj], K[ pari ], x0[ pari ] );
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {Vs_t, dV_dIs_t, dV_dJs_t};
  }
};

template <
	tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedAngleDispatch {
  static auto f(
    TView<Vec<Int, 3>, 1, D> atomtriple_indices,
    TView<Int, 1, D> parameter_indices,
    TView<Vec<Real, 3>, 1, D> coords,
    TView<Real, 1, D> K,
    TView<Real, 1, D> x0
  )
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dKs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;
    auto dV_dKs = dV_dKs_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atomtriple_indices[i][0];
      Int atj = atomtriple_indices[i][1];
      Int atk = atomtriple_indices[i][2];
      Int pari = parameter_indices[i];
      tie(Vs[i],dV_dIs[i],dV_dJs[i],dV_dKs[i]) = cbangle_V_dV(
        coords[ati], coords[atj], coords[atk], K[ pari ], x0[ pari ] );
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {Vs_t, dV_dIs_t, dV_dJs_t, dV_dKs_t};
  }
};

template <
	tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedTorsionDispatch {
  static auto f(
    TView<Vec<Int, 4>, 1, D> atomquad_indices,
    TView<Int, 1, D> parameter_indices,
    TView<Vec<Real, 3>, 1, D> coords,
    TView<Real, 1, D> K,
    TView<Real, 1, D> x0,
    TView<Int, 1, D> period
  )
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dKs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dLs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;
    auto dV_dKs = dV_dKs_t.view;
    auto dV_dLs = dV_dLs_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atomquad_indices[i][0];
      Int atj = atomquad_indices[i][1];
      Int atk = atomquad_indices[i][2];
      Int atl = atomquad_indices[i][3];
      Int pari = parameter_indices[i];
      tie(Vs[i],dV_dIs[i],dV_dJs[i],dV_dKs[i],dV_dLs[i]) = cbtorsion_V_dV(
        coords[ati], coords[atj], coords[atk], coords[atl], K[ pari ], x0[ pari ], period[ pari ] );
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {Vs_t, dV_dIs_t, dV_dJs_t, dV_dKs_t, dV_dLs_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedHxlTorsionDispatch {
  static auto f(
      TView<Vec<Int, 4>, 1, D> atomquad_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> K1,
      TView<Real, 1, D> K2,
      TView<Real, 1, D> K3,
      TView<Real, 1, D> phi1,
      TView<Real, 1, D> phi2,
      TView<Real, 1, D> phi3)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> > {
    auto num_Vs = parameter_indices.size(0);

    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto dV_dIs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dJs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dKs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);
    auto dV_dLs_t = TPack<Vec<Real, 3>, 1, D>::empty(num_Vs);

    auto Vs = Vs_t.view;
    auto dV_dIs = dV_dIs_t.view;
    auto dV_dJs = dV_dJs_t.view;
    auto dV_dKs = dV_dKs_t.view;
    auto dV_dLs = dV_dLs_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atomquad_indices[i][0];
      Int atj = atomquad_indices[i][1];
      Int atk = atomquad_indices[i][2];
      Int atl = atomquad_indices[i][3];
      Int pari = parameter_indices[i];
      tie(Vs[i],dV_dIs[i],dV_dJs[i],dV_dKs[i],dV_dLs[i]) = cbhxltorsion_V_dV(
        coords[ati], coords[atj], coords[atk], coords[atl], 
		K1[ pari ], K2[ pari ], K3[ pari ],
		phi1[ pari ], phi2[ pari ], phi3[ pari ] );
    });

    for (int i = 0; i < num_Vs; i++) {
      f_i(i);
    }

    return {Vs_t, dV_dIs_t, dV_dJs_t, dV_dKs_t, dV_dLs_t};
  }
};


template struct CartBondedLengthDispatch<tmol::Device::CPU,float,int32_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU,double,int32_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU,float,int64_t>;
template struct CartBondedLengthDispatch<tmol::Device::CPU,double,int64_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU,float,int32_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU,double,int32_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU,float,int64_t>;
template struct CartBondedAngleDispatch<tmol::Device::CPU,double,int64_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU,float,int32_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU,double,int32_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU,float,int64_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CPU,double,int64_t>;
template struct CartBondedHxlTorsionDispatch<tmol::Device::CPU,float,int32_t>;
template struct CartBondedHxlTorsionDispatch<tmol::Device::CPU,double,int32_t>;
template struct CartBondedHxlTorsionDispatch<tmol::Device::CPU,float,int64_t>;
template struct CartBondedHxlTorsionDispatch<tmol::Device::CPU,double,int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
