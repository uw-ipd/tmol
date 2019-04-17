#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>

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
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<Vec<Int, 2>, 1, tmol::Device::CPU> gens)
      -> TPack<HomogeneousTransform, 1, D> {
    auto num_atoms = dofs.size(0);

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

    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) { // loop over generations
      int scanstart = gens[gen][1];
      int scanstop = gens[gen+1][1];
      for (int j = scanstart; j < scanstop; j++) { // loop over scans
        int nodestart = gens[gen][0] + scans[j];
        int nodestop = (j == scanstop-1) ? gens[gen+1][0] : (gens[gen][0] + scans[j+1]);
        for (int k = nodestart; k < nodestop-1; k++) { // loop over path
            k_compose(nodes[k], nodes[k+1]);
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

template <tmol::Device D, typename Real, typename Int>
struct f1f2ToDerivsDispatch {
  static auto f(
      TView<HomogeneousTransform, 1, D> hts,
      TView<Vec<Real, 9>, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TView<Int, 1, D> parents,
      TView<Vec<Real, 6>, 1, D> f1f2s) -> TPack<Vec<Real, 9>, 1, D> {
    auto num_atoms = dofs.size(0);
    auto dsc_ddofs_t = TPack<Vec<Real, 9>, 1, D>::empty({num_atoms});
    auto dsc_ddofs = dsc_ddofs_t.view;

    auto k_f1f2s2derivs = ([=] EIGEN_DEVICE_FUNC(int i) {
      Vec<Real, 3> f1 = f1f2s[i].topRows(3);
      Vec<Real, 3> f2 = f1f2s[i].bottomRows(3);
      if (doftypes[i] == ROOT) {
        dsc_ddofs[i] = Vec<Real, 9>::Constant(0);
      } else if (doftypes[i] == JUMP) {
        dsc_ddofs[i]= common<D,Real,Int>::jumpDerivatives(
          dofs[i], hts[i], hts[parents[i]], f1, f2 );
      } else if (doftypes[i] == BOND) {
        dsc_ddofs[i]= common<D,Real,Int>::bondDerivatives(
          dofs[i], hts[i], hts[parents[i]], f1, f2 );
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_f1f2s2derivs(i);
    }

    return dsc_ddofs_t;
  }
};

template <tmol::Device D, typename Real, typename Int>
struct SegscanF1f2sDispatch {
  static auto f(
      TView<Vec<Real, 6>, 1, D> f1f2s,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<Vec<Int, 2>, 1, tmol::Device::CPU> gens) -> void {
    auto num_atoms = f1f2s.size(0);

    // scan and accumulate f1s/f2s up atom tree
    auto k_compose = ([=] EIGEN_DEVICE_FUNC(int p, int i) {
        f1f2s[i] = f1f2s[i] + f1f2s[p];
    });

    // note: if this is parallelized (over j/k) 
    //   then k_compose needs to be atomic
    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) { // loop over generations
      int scanstart = gens[gen][1];
      int scanstop = gens[gen+1][1];
      for (int j = scanstart; j < scanstop; j++) { // loop over scans
        int nodestart = gens[gen][0] + scans[j];
        int nodestop = (j == scanstop-1) ? gens[gen+1][0] : (gens[gen][0] + scans[j+1]);
        for (int k = nodestart; k < nodestop-1; k++) { // loop over path
            k_compose(nodes[k], nodes[k+1]);
        }
      }
    }

    return;
  }
};

template struct ForwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CPU, float, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CPU, double, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct f1f2ToDerivsDispatch<tmol::Device::CPU, float, int32_t>;
template struct f1f2ToDerivsDispatch<tmol::Device::CPU, double, int32_t>;
template struct SegscanF1f2sDispatch<tmol::Device::CPU, float, int32_t>;
template struct SegscanF1f2sDispatch<tmol::Device::CPU, double, int32_t>;

#undef HomogeneousTransform
#undef QuatPlusTranslation

}  // namespace kinematics
}  // namespace tmol
