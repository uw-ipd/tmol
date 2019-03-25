#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCollection.h>

#include "common.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define QuatPlusTranslation Eigen::Matrix<Real, 7, 1>

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<Vec<Real, 9>, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TCollection<Int, 1, D> nodes,
      TCollection<Int, 1, D> scans)
      -> TPack<HomogeneousTransform, 1, D> {
    auto num_atoms = dofs.size(0);

    auto nodeview = nodes.view;
    auto scansview = scans.view;

    auto QTs_t = TPack<QuatPlusTranslation, 1, D>::empty({num_atoms});
    auto QTs = QTs_t.view;
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;

    // dofs -> HTs
    auto k_dof2qt = ([=] EIGEN_DEVICE_FUNC(int i) {
      DOFtype doftype = (DOFtype)doftypes[i];
      HomogeneousTransform HT;
      if (doftype == ROOT) {
        HT = HomogeneousTransform::Identity();
      } else if (doftype == JUMP) {
        HT = common<D,Real,Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HT = common<D,Real,Int>::bondTransform(dofs[i]);
      }
      QTs[i] = common<D,Real,Int>::ht2quat_trans(HT);
    });

    for (int i = 0; i < num_atoms; i++) {
      k_dof2qt(i);
    }

    // scan and accumulate HTs down atom tree
    auto k_compose = ([=] EIGEN_DEVICE_FUNC(int p, int i) {
        QTs[i] = common<D,Real,Int>::quat_trans_compose(QTs[i], QTs[p]);
    });

    auto ngens = nodeview.size(0);
    for (int i = 0; i < ngens; i++) { // loop over generations
        // one could trivially parallelize this loop over j!
        auto nscans = scansview[i].size(0);
        for (int j = 0; j < nscans; j++) { // loop over scans
            auto scanstart = scansview[i][j];
            auto scanstop = (j==nscans-1) ? nodeview[i].size(0) : scansview[i][j+1];
            for (int k = scanstart; k < scanstop-1; k++) { // loop over path
                k_compose(nodeview[i][k], nodeview[i][k+1]);
            }
        }
    }

    for (int i = 0; i < num_atoms; i++) {
        HTs[i] = common<D,Real,Int>::quat_trans2ht(QTs[i]);
    }

    return HTs_t;
  }
};


template <tmol::Device D, typename Real, typename Int>
struct DOFTransformsDispatch {
  static auto f(
      TView<Vec<Real, 9>, 1, D> dofs,
      TView<Int, 1, D> doftypes)
      -> TPack<HomogeneousTransform, 1, D> {
    auto num_atoms = dofs.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;

    // dofs -> HTs
    auto k_dof2ht = ([=] EIGEN_DEVICE_FUNC(int i) {
      DOFtype doftype = (DOFtype)doftypes[i];
      HomogeneousTransform HT;
      if (doftype == ROOT) {
        HTs[i] = HomogeneousTransform::Identity();
      } else if (doftype == JUMP) {
        HTs[i] = common<D,Real,Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HTs[i] = common<D,Real,Int>::bondTransform(dofs[i]);
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_dof2ht(i);
    }

    return HTs_t;
  }
};

template <tmol::Device D, typename Real, typename Int>
struct BackwardKinDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Int, 1, D> doftypes,
      TView<Int, 1, D> parents,
      TView<Int, 1, D> frame_x,
      TView<Int, 1, D> frame_y,
      TView<Int, 1, D> frame_z,
	  TView<Vec<Real, 9>, 1, D> dofs
  ) -> TPack<HomogeneousTransform, 1, D> {
    auto num_atoms = coords.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;

    auto k_coords2hts = ([=] EIGEN_DEVICE_FUNC(int i) {
		if (i==0) {
			HTs[i] = HomogeneousTransform::Identity();
		} else {
			HTs[i] = common<D,Real,Int>::hts_from_frames( 
				coords[i], coords[frame_x[i]], coords[frame_y[i]], coords[frame_z[i]] );
		}
	});

    for (int i = 0; i < num_atoms; i++) {
      k_coords2hts(i);
    }	

    auto k_hts2dofs = ([=] EIGEN_DEVICE_FUNC(int i) {
	  HomogeneousTransform lclHT;
	  if (doftypes[i] != ROOT) {
		lclHT = HTs[i] * common<D,Real,Int>::ht_inv( HTs[parents[i]] );

        if (doftypes[i] == JUMP) {
          dofs[i]= common<D,Real,Int>::invJumpTransform(lclHT);
        } else if (doftypes[i] == BOND) {
          dofs[i]= common<D,Real,Int>::invBondTransform(lclHT);
        }
	  }
	});

    for (int i = 0; i < num_atoms; i++) {
      k_hts2dofs(i);
    }

	return HTs_t;
  }
};


template struct ForwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CPU, float, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CPU, double, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CPU, double, int32_t>;

#undef HomogeneousTransform
#undef QuatPlusTranslation

}  // namespace kinematics
}  // namespace tmol
