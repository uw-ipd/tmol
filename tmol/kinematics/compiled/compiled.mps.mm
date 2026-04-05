// compiled.mps.mm — MPS instantiation of kinematic dispatch structs.
//
// The ForwardKinDispatch / InverseKinDispatch / KinDerivDispatch template
// structs are pure C++ (no CUDA intrinsics) and work for any tmol::Device D
// via explicit loops.  We copy their definition here (parameterised over D)
// and instantiate for Device::MPS so that compiled.cpp/.ops.cpp can resolve
// MPS calls through TORCH_LIBRARY dispatch.

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/device_operations.mps.impl.hh>

#include "common.hh"
#include "params.hh"
#include "compiled.impl.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define KintreeDof Eigen::Matrix<Real, 9, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D> > {
    auto num_atoms = dofs.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto xs_t = TPack<Coord, 1, D>::empty({num_atoms});
    auto xs = xs_t.view;

    auto k_dof2ht = ([=](int i) {
      DOFtype doftype = (DOFtype)kintree[i].doftype;
      if (doftype == ROOT) {
        HTs[i] = HomogeneousTransform::Identity();
      } else if (doftype == JUMP) {
        HTs[i] = common<D, Real, Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HTs[i] = common<D, Real, Int>::bondTransform(dofs[i]);
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_dof2ht(i);
    }

    auto k_compose =
        ([=](int p, int i) { HTs[i] = HTs[i] * HTs[p]; });

    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) {
      int scanstart = gens[gen].scan_start;
      int scanstop = gens[gen + 1].scan_start;
      for (int j = scanstart; j < scanstop; j++) {
        int nodestart = gens[gen].node_start + scans[j];
        int nodestop = (j == scanstop - 1)
                           ? gens[gen + 1].node_start
                           : (gens[gen].node_start + scans[j + 1]);
        for (int k = nodestart; k < nodestop - 1; k++) {
          k_compose(nodes[k], nodes[k + 1]);
        }
      }
    }

    auto k_getcoords = ([=](int i) {
      xs[i] = HTs[i].block(3, 0, 1, 3).transpose();
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
      TView<Coord, 1, D> coords,
      TView<Int, 1, D> parent,
      TView<Int, 1, D> frame_x,
      TView<Int, 1, D> frame_y,
      TView<Int, 1, D> frame_z,
      TView<Int, 1, D> doftype) -> TPack<KintreeDof, 1, D> {
    auto num_atoms = coords.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto dofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dofs = dofs_t.view;

    auto k_coords2hts = ([=](int i) {
      if (i == 0) {
        HTs[i] = HomogeneousTransform::Identity();
      } else {
        HTs[i] = common<D, Real, Int>::hts_from_frames(
            coords[i],
            coords[frame_x[i]],
            coords[frame_y[i]],
            coords[frame_z[i]]);
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_coords2hts(i);
    }

    auto k_hts2dofs = ([=](int i) {
      HomogeneousTransform lclHT;
      if (doftype[i] == ROOT) {
        dofs[i] = KintreeDof::Constant(0);
      } else {
        lclHT = HTs[i] * common<D, Real, Int>::ht_inv(HTs[parent[i]]);
        if (doftype[i] == JUMP) {
          dofs[i] = common<D, Real, Int>::invJumpTransform(lclHT);
        } else if (doftype[i] == BOND) {
          dofs[i] = common<D, Real, Int>::invBondTransform(lclHT);
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
      TView<HomogeneousTransform, 1, D> hts,
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree) -> TPack<KintreeDof, 1, D> {
    auto num_atoms = dVdx.size(0);

    auto f1f2s_t = TPack<Vec<Real, 6>, 1, D>::empty({num_atoms});
    auto f1f2s = f1f2s_t.view;
    auto dsc_ddofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dsc_ddofs = dsc_ddofs_t.view;

    auto k_f1f2s = ([=](int i) {
      Coord trans = hts[i].block(3, 0, 1, 3).transpose();
      Coord f1 = dVdx[i].isZero(0) ? dVdx[i]
                                    : trans.cross(trans - dVdx[i]).transpose();
      f1f2s[i].topRows(3) = f1;
      f1f2s[i].bottomRows(3) = dVdx[i];
    });

    for (int i = 0; i < num_atoms; i++) {
      k_f1f2s(i);
    }

    auto k_compose = ([=](int p, int i) {
      f1f2s[i] = f1f2s[i] + f1f2s[p];
    });

    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) {
      int scanstart = gens[gen].scan_start;
      int scanstop = gens[gen + 1].scan_start;
      for (int j = scanstart; j < scanstop; j++) {
        int nodestart = gens[gen].node_start + scans[j];
        int nodestop = (j == scanstop - 1)
                           ? gens[gen + 1].node_start
                           : (gens[gen].node_start + scans[j + 1]);
        for (int k = nodestart; k < nodestop - 1; k++) {
          k_compose(nodes[k], nodes[k + 1]);
        }
      }
    }

    auto k_f1f2s2derivs = ([=](int i) {
      Vec<Real, 3> f1 = f1f2s[i].topRows(3);
      Vec<Real, 3> f2 = f1f2s[i].bottomRows(3);
      if (kintree[i].doftype == ROOT) {
        dsc_ddofs[i] = Vec<Real, 9>::Constant(0);
      } else if (kintree[i].doftype == JUMP) {
        dsc_ddofs[i] = common<D, Real, Int>::jumpDerivatives(
            dofs[i], hts[i], hts[kintree[i].parent], f1, f2);
      } else if (kintree[i].doftype == BOND) {
        dsc_ddofs[i] = common<D, Real, Int>::bondDerivatives(
            dofs[i], hts[i], hts[kintree[i].parent], f1, f2);
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_f1f2s2derivs(i);
    }

    return dsc_ddofs_t;
  }
};

template struct ForwardKinDispatch<tmol::Device::MPS, float,  int32_t>;
template struct ForwardKinDispatch<tmol::Device::MPS, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::MPS, float,  int32_t>;
template struct InverseKinDispatch<tmol::Device::MPS, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::MPS,   float,  int32_t>;
template struct KinDerivDispatch<tmol::Device::MPS,   double, int32_t>;

template struct KinForestFromStencil<
    tmol::score::common::DeviceOperations,
    tmol::Device::MPS,
    int32_t>;

#undef HomogeneousTransform
#undef KintreeDof
#undef Coord

}  // namespace kinematics
}  // namespace tmol
