#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/device_operations.cpu.impl.hh>

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
    // printf("dofs.size(0): %d\n", num_atoms);
    // printf("nodes.size(0): %d\n", nodes.size(0));
    printf("ForwardKinDispatch\n");

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
        HTs[i] = common<D, Real, Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HTs[i] = common<D, Real, Int>::bondTransform(dofs[i]);
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_dof2ht(i);
    }

    // scan and accumulate HTs down atom tree
    auto k_compose = ([=] EIGEN_DEVICE_FUNC(int p, int i) {
      HTs[i] = HTs[i] * HTs[p];
      // if (i == 58) {printf("setting 58! %6.3f %6.3f %6.3f\n", HTs[i](3, 0),
      // HTs[i](3, 1), HTs[i](3, 2));}
    });

    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) {  // loop over generations
      int scanstart = gens[gen].scan_start;
      int scanstop = gens[gen + 1].scan_start;
      for (int j = scanstart; j < scanstop; j++) {  // loop over scans
        // printf("scan %d %d star %d stop %d\n", gen, j, scanstart, scanstop);
        int nodestart = gens[gen].node_start + scans[j];
        int nodestop = (j == scanstop - 1)
                           ? gens[gen + 1].node_start
                           : (gens[gen].node_start + scans[j + 1]);
        // printf("node start %d node stop %d\n", nodestart, nodestop);
        for (int k = nodestart; k < nodestop - 1; k++) {  // loop over path

          // printf("k: %d %d %d\n", gen, j, k);
          //     print_three_frames(2, 74, 73, 59)
          // int kn = nodes[k];
          // int kp1n = nodes[k + 1];
          // bool any = kn == 58 || kn == 59;
          // if (any) {
          //   printf("gen %d j %d scanstart %d scanstop %d nodestart %d
          //   nodestop %d k %d kn %d kp1n %d\n",
          //     gen, j, scanstart, scanstop, nodestart, nodestop, k, kn, kp1n);
          // }
          //   printf(
          //       "b HT %3d: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f
          //       "
          //       "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n "
          //       "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //       kn,
          //       HTs[kn](0, 0),
          //       HTs[kn](0, 1),
          //       HTs[kn](0, 2),
          //       HTs[kn](0, 3),
          //       HTs[kn](1, 0),
          //       HTs[kn](1, 1),
          //       HTs[kn](1, 2),
          //       HTs[kn](1, 3),
          //       HTs[kn](2, 0),
          //       HTs[kn](2, 1),
          //       HTs[kn](2, 2),
          //       HTs[kn](2, 3),
          //       HTs[kn](3, 0),
          //       HTs[kn](3, 1),
          //       HTs[kn](3, 2),
          //       HTs[kn](3, 3));
          // }
          // if (any) {
          //   printf(
          //       "b HT %3d: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f
          //       "
          //       "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n "
          //       "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //       kp1n,
          //       HTs[kp1n](0, 0),
          //       HTs[kp1n](0, 1),
          //       HTs[kp1n](0, 2),
          //       HTs[kp1n](0, 3),
          //       HTs[kp1n](1, 0),
          //       HTs[kp1n](1, 1),
          //       HTs[kp1n](1, 2),
          //       HTs[kp1n](1, 3),
          //       HTs[kp1n](2, 0),
          //       HTs[kp1n](2, 1),
          //       HTs[kp1n](2, 2),
          //       HTs[kp1n](2, 3),
          //       HTs[kp1n](3, 0),
          //       HTs[kp1n](3, 1),
          //       HTs[kp1n](3, 2),
          //       HTs[kp1n](3, 3));
          // }
          k_compose(nodes[k], nodes[k + 1]);
          // if (any) {
          //   printf(
          //       "a HT %3d: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f
          //       "
          //       "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n "
          //       "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //       kn,
          //       HTs[kn](0, 0),
          //       HTs[kn](0, 1),
          //       HTs[kn](0, 2),
          //       HTs[kn](0, 3),
          //       HTs[kn](1, 0),
          //       HTs[kn](1, 1),
          //       HTs[kn](1, 2),
          //       HTs[kn](1, 3),
          //       HTs[kn](2, 0),
          //       HTs[kn](2, 1),
          //       HTs[kn](2, 2),
          //       HTs[kn](2, 3),
          //       HTs[kn](3, 0),
          //       HTs[kn](3, 1),
          //       HTs[kn](3, 2),
          //       HTs[kn](3, 3));
          // }
          // if (any) {
          //   printf(
          //       "a HT %3d: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f
          //       "
          //       "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n "
          //       "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //       kp1n,
          //       HTs[kp1n](0, 0),
          //       HTs[kp1n](0, 1),
          //       HTs[kp1n](0, 2),
          //       HTs[kp1n](0, 3),
          //       HTs[kp1n](1, 0),
          //       HTs[kp1n](1, 1),
          //       HTs[kp1n](1, 2),
          //       HTs[kp1n](1, 3),
          //       HTs[kp1n](2, 0),
          //       HTs[kp1n](2, 1),
          //       HTs[kp1n](2, 2),
          //       HTs[kp1n](2, 3),
          //       HTs[kp1n](3, 0),
          //       HTs[kp1n](3, 1),
          //       HTs[kp1n](3, 2),
          //       HTs[kp1n](3, 3));
          // }
        }
      }
    }

    // copy atom positions
    auto k_getcoords = ([=] EIGEN_DEVICE_FUNC(int i) {
      xs[i] = HTs[i].block(3, 0, 1, 3).transpose();
    });

    for (int i = 0; i < num_atoms; i++) {
      k_getcoords(i);
    }

    printf("ForwardKinDispatch ... done\n");
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
    printf("InverseKinDispatch\n");
    auto num_atoms = coords.size(0);
    // auto num_atoms = parent.size(0);
    auto num_nodes = parent.size(0);

    // fd: we could eliminate HT allocation and calculate on the fly
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto dofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dofs = dofs_t.view;

    auto k_coords2hts = ([=] EIGEN_DEVICE_FUNC(int i) {
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

    auto k_hts2dofs = ([=] EIGEN_DEVICE_FUNC(int i) {
      HomogeneousTransform lclHT;
      if (doftype[i] == ROOT) {
        dofs[i] = KintreeDof::Constant(0);  // for num deriv check
      } else {
        lclHT = HTs[i] * common<D, Real, Int>::ht_inv(HTs[parent[i]]);

        if (doftype[i] == JUMP) {
          dofs[i] = common<D, Real, Int>::invJumpTransform(lclHT);
          // printf("Jump HT: %d w/ parent %d\n", i, parent[i]);
          // printf(
          //     "%4d HT: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f "
          //     "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n          "
          //     "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //     i,
          //     HTs[i](0, 0),
          //     HTs[i](0, 1),
          //     HTs[i](0, 2),
          //     HTs[i](0, 3),
          //     HTs[i](1, 0),
          //     HTs[i](1, 1),
          //     HTs[i](1, 2),
          //     HTs[i](1, 3),
          //     HTs[i](2, 0),
          //     HTs[i](2, 1),
          //     HTs[i](2, 2),
          //     HTs[i](2, 3),
          //     HTs[i](3, 0),
          //     HTs[i](3, 1),
          //     HTs[i](3, 2),
          //     HTs[i](3, 3));
          // printf(
          //     "%4d HT: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f "
          //     "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n          "
          //     "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //     parent[i],
          //     HTs[parent[i]](0, 0),
          //     HTs[parent[i]](0, 1),
          //     HTs[i](0, 2),
          //     HTs[parent[i]](0, 3),
          //     HTs[parent[i]](1, 0),
          //     HTs[parent[i]](1, 1),
          //     HTs[i](1, 2),
          //     HTs[parent[i]](1, 3),
          //     HTs[parent[i]](2, 0),
          //     HTs[parent[i]](2, 1),
          //     HTs[i](2, 2),
          //     HTs[parent[i]](2, 3),
          //     HTs[parent[i]](3, 0),
          //     HTs[parent[i]](3, 1),
          //     HTs[i](3, 2),
          //     HTs[parent[i]](3, 3));
          // printf(
          //     "jump HT: [[%8.3f %8.3f %8.3f %8.3f]\n          [%8.3f %8.3f "
          //     "%8.3f %8.3f]\n          [%8.3f %8.3f %8.3f %8.3f]\n          "
          //     "[%8.3f %8.3f %8.3f %8.3f]]\n",
          //     lclHT(0, 0),
          //     lclHT(0, 1),
          //     lclHT(0, 2),
          //     lclHT(0, 3),
          //     lclHT(1, 0),
          //     lclHT(1, 1),
          //     lclHT(1, 2),
          //     lclHT(1, 3),
          //     lclHT(2, 0),
          //     lclHT(2, 1),
          //     lclHT(2, 2),
          //     lclHT(2, 3),
          //     lclHT(3, 0),
          //     lclHT(3, 1),
          //     lclHT(3, 2),
          //     lclHT(3, 3));

          // printf(
          //     "jump DOFs %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f "
          //     "%8.3f\n",
          //     dofs[i][0],
          //     dofs[i][1],
          //     dofs[i][2],
          //     dofs[i][3],
          //     dofs[i][4],
          //     dofs[i][5],
          //     dofs[i][6],
          //     dofs[i][7],
          //     dofs[i][8]);
        } else if (doftype[i] == BOND) {
          dofs[i] = common<D, Real, Int>::invBondTransform(lclHT);
        }
      }
    });

    for (int i = 0; i < num_atoms; i++) {
      k_hts2dofs(i);
    }

    printf("InverseKinDispatch... Done!\n");
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

    // calculate f1s and f2s from dVdx and HT
    auto k_f1f2s = ([=] EIGEN_DEVICE_FUNC(int i) {
      Coord trans = hts[i].block(3, 0, 1, 3).transpose();
      Coord f1 = trans.cross(trans - dVdx[i]).transpose();
      f1f2s[i].topRows(3) = f1;
      f1f2s[i].bottomRows(3) = dVdx[i];
    });

    for (int i = 0; i < num_atoms; i++) {
      k_f1f2s(i);
    }

    // scan and accumulate f1s/f2s up atom tree
    auto k_compose = ([=] EIGEN_DEVICE_FUNC(int p, int i) {
      f1f2s[i] = f1f2s[i] + f1f2s[p];
      // if (i == 20) {
      //   printf("k_compose p %d i %d val: %f\n", p, i, f1f2s[i][3]);
      // }
    });

    // note: if this is parallelized (over j/k)
    //   then k_compose needs to be atomic
    int ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; gen++) {  // loop over generations
      int scanstart = gens[gen].scan_start;
      int scanstop = gens[gen + 1].scan_start;
      for (int j = scanstart; j < scanstop; j++) {  // loop over scans
        int nodestart = gens[gen].node_start + scans[j];
        int nodestop = (j == scanstop - 1)
                           ? gens[gen + 1].node_start
                           : (gens[gen].node_start + scans[j + 1]);
        for (int k = nodestart; k < nodestop - 1; k++) {  // loop over path
          k_compose(nodes[k], nodes[k + 1]);
        }
      }
    }

    // auto k_print = [=] EIGEN_DEVICE_FUNC(int index) {
    //   printf(
    //       "f1f2s[%d]: %f %f %f %f %f %f\n",
    //       index,
    //       f1f2s[index][0],
    //       f1f2s[index][1],
    //       f1f2s[index][2],
    //       f1f2s[index][3],
    //       f1f2s[index][4],
    //       f1f2s[index][5]);
    // };

    // for (int i = 0; i < num_atoms; ++i) {
    //   k_print(i);
    // }

    auto k_f1f2s2derivs = ([=] EIGEN_DEVICE_FUNC(int i) {
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

template struct ForwardKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::CPU, float, int32_t>;
template struct InverseKinDispatch<tmol::Device::CPU, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::CPU, float, int32_t>;
template struct KinDerivDispatch<tmol::Device::CPU, double, int32_t>;

template struct KinForestFromStencil<
    tmol::score::common::DeviceOperations,
    tmol::Device::CPU,
    int32_t>;
// template struct KinForestFromStencil<
//     tmol::score::common::DeviceOperations,
//     tmol::Device::CPU,
//     int64_t>;

#undef HomogeneousTransform
#undef KintreeDof
#undef Coord

}  // namespace kinematics
}  // namespace tmol
