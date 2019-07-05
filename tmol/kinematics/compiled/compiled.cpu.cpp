#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>

#include "common.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Coord Eigen::Matrix<Real, 3, 1>
#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<DofTypes<Real>, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinTreeGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinTreeParams<Int>, 1, D> kintree
  ) -> std::tuple< TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D> > {
    auto num_atoms = dofs.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto xs_t = TPack<Coord, 1, D>::empty({num_atoms});
    auto xs = xs_t.view;

    // dofs -> HTs
    auto k_dof2ht = ([=] EIGEN_DEVICE_FUNC(int i) {
      DOFtype doftype = (DOFtype)kintree[i].doftype;
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

    // scan and accumulate HTs down atom tree
    auto k_compose = ([=] EIGEN_DEVICE_FUNC(int p, int i) {
        HTs[i] = HTs[i]*HTs[p];
    });

    int ngens = scangens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) { // loop over generations
      int scanstart = gens[gen].scanstart;
      int scanstop = gens[gen+1].scanstart;
      for (int j = scanstart; j < scanstop; j++) { // loop over scans
        int nodestart = gens[gen].nodestart + scans[j];
        int nodestop = (j == scanstop-1) ? gens[gen+1].nodestart : (gens[gen].nodestart + scans[j+1]);
        for (int k = nodestart; k < nodestop-1; k++) { // loop over path
            k_compose(nodes[k], nodes[k+1]);
        }
      }
    }

    // copy atom positions
    auto k_getcoords = ([=] EIGEN_DEVICE_FUNC(int i) {
        xs[i] = HTs[i].block(3,0,1,3);
    });

    for (int i = 0; i < num_atoms; i++) {
      k_getcoords(i);
    }

    return {xs_t, HTs_t};
  }
};


template <tmol::Device D, typename Real, typename Int>
struct InverseKinDispatch {
  static auto f(
      TView<Coord, 1, D> coord,
      TView<KinTreeParams<Int>, 1, D> kintree
  ) -> TPack<DofTypes<Real>, 1, D> {
    auto num_atoms = coords.size(0);

    //fd: we could eliminate HT allocation and calculate on the fly
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto dofs_t = TPack<DofTypes<Real>, 1, D>::empty({num_atoms});
    auto dofs = dofs_t.view;

    auto k_coords2hts = ([=] EIGEN_DEVICE_FUNC(int i) {
		if (i==0) {
			HTs[i] = HomogeneousTransform::Identity();
		} else {
			HTs[i] = common<D,Real,Int>::hts_from_frames( 
				coords[i], 
                coords[kintree[i].frame_x], coords[kintree[i].frame_y], coords[kintree[i].frame_z]
            );
		}
	});

    for (int i = 0; i < num_atoms; i++) {
      k_coords2hts(i);
    }	

    auto k_hts2dofs = ([=] EIGEN_DEVICE_FUNC(int i) {
	  HomogeneousTransform lclHT;
	  if (kintree[i].doftype != ROOT) {
		lclHT = HTs[i] * common<D,Real,Int>::ht_inv( HTs[parents[i]] );

        if (kintree[i].doftype == JUMP) {
          dofs[i]= common<D,Real,Int>::invJumpTransform(lclHT);
        } else if (kintree[i].doftype == BOND) {
          dofs[i]= common<D,Real,Int>::invBondTransform(lclHT);
        }
	  }
	});

    for (int i = 0; i < num_atoms; i++) {
      k_hts2dofs(i);
    }

	return dofs_t;
  }
};

template <tmol::Device D, typename Real, typename Int>
struct KinDerivDispatch {
  static auto f(
      TView<Coord, 1, D> dVdx,
      TView<HT, 1, tmol::Device::CPU> hts,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinTreeGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinTreeParams<Int>, 1, D> kintree
  ) -> TPack<DofTypes<Real>, 1, D> {
    auto num_atoms = dVdx.size(0);

    auto f1f2s_t = TPack<Vec<Real,6>, 1, D>::empty({num_atoms});
    auto f1f2s = f1f2s_t.view;
    auto dsc_ddofs_t = TPack<DofTypes<Real>, 1, D>::empty({num_atoms});
    auto dsc_ddofs = dsc_ddofs_t.view;

    // calculate f1s and f2s from dVdx and HT
    auto k_f1f2s = ([=] EIGEN_DEVICE_FUNC(int i) {
        f1f2s[i].topRows(3) = 
            hts[i].block(3,0,1,3).cross( hts[i].block(3,0,1,3) - dVdx[i]);
        f1f2s[i].bottomRows(3) = dVdx[i];
    });

    for (int i = 0; i < num_atoms; i++) {
      k_f1f2s(i);
    }

    // scan and accumulate f1s/f2s up atom tree
    auto k_compose = ([=] EIGEN_DEVICE_FUNC(int p, int i) {
        f1f2s[i] = f1f2s[i] + f1f2s[p];
    });

    // note: if this is parallelized (over j/k) 
    //   then k_compose needs to be atomic
    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) { // loop over generations
      int scanstart = gens[gen].scanstart;
      int scanstop = gens[gen+1].scanstart;
      for (int j = scanstart; j < scanstop; j++) { // loop over scans
        int nodestart = gens[gen].nodestart + scans[j];
        int nodestop = (j == scanstop-1) ? gens[gen+1].nodestart : (gens[gen].nodestart + scans[j+1]);
        for (int k = nodestart; k < nodestop-1; k++) { // loop over path
            k_compose(nodes[k], nodes[k+1]);
        }
      }
    }

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

template struct ForwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct InverseKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::CPU, float, int32_t>;
template struct KinDerivDispatch<tmol::Device::CPU, double, int32_t>;

#undef HomogeneousTransform
#undef Coord

}  // namespace kinematics
}  // namespace tmol
