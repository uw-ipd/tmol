#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/kinematics/compiled/kernel_segscan.cuh>

#include <moderngpu/transform.hxx>

#include "common.hh"
#include "params.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define KintreeDof Eigen::Matrix<Real, 9, 1>
#define f1f2Vectors Eigen::Matrix<Real, 6, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

template <typename Real>
struct HTRawBuffer {
  Real data[16];
};

template <typename Real>
struct f1f2VecsRawBuffer {
  Real data[6];
};

// the composite operation for the forward pass: apply a transform
//   qt1/qt2 -> HT1/HT2 -> HT1*HT2 -> qt12' -> norm(qt12')
template <tmol::Device D, typename Real, typename Int>
struct htcompose_t : public std::binary_function<
                         HTRawBuffer<Real>,
                         HTRawBuffer<Real>,
                         HTRawBuffer<Real>> {
  MGPU_HOST_DEVICE HTRawBuffer<Real> operator()(
      HTRawBuffer<Real> p, HTRawBuffer<Real> i) const {
    HomogeneousTransform ab = Eigen::Map<HomogeneousTransform>(i.data)
                              * Eigen::Map<HomogeneousTransform>(p.data);

    return *((HTRawBuffer<Real>*)ab.data());
  }
};

// the composite operation for the backward pass: sum f1s & f2s
template <tmol::Device D, typename Real, typename Int>
struct f1f2compose_t : public std::binary_function<
                           f1f2VecsRawBuffer<Real>,
                           f1f2VecsRawBuffer<Real>,
                           f1f2VecsRawBuffer<Real>> {
  MGPU_HOST_DEVICE f1f2VecsRawBuffer<Real> operator()(
      f1f2VecsRawBuffer<Real> p, f1f2VecsRawBuffer<Real> i) const {
    f1f2Vectors ab =
        Eigen::Map<f1f2Vectors>(i.data) + Eigen::Map<f1f2Vectors>(p.data);

    return *((f1f2VecsRawBuffer<Real>*)ab.data());
  }
};

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinTreeGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinTreeParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D>> {
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
        HTs[i] = common<D, Real, Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HTs[i] = common<D, Real, Int>::bondTransform(dofs[i]);
      }
    });

    mgpu::standard_context_t context;
    mgpu::transform(k_dof2ht, num_atoms, context);

    // memory for scan (longest scan possible is 2 times # atoms)
    auto HTscan_t = TPack<HomogeneousTransform, 1, D>::empty({2 * num_atoms});
    auto HTscan = HTscan_t.view;
    HTRawBuffer<Real> init = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};  // identity xform

    auto ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; ++gen) {
      int nodestart = gens[gen].node_start, scanstart = gens[gen].scan_start;
      int nnodes = gens[gen + 1].node_start - nodestart;
      int nscans = gens[gen + 1].scan_start - scanstart;

      // reindexing function
      auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
        return *((HTRawBuffer<Real>*)HTs[nodes[nodestart + index]].data());
      };

      // mgpu does not play nicely with eigen types
      // instead, we wrap the raw data buffer as QuatTransRawBuffer
      //      and use eigen:map to reconstruct on device
      tmol::kinematics::kernel_segscan<mgpu::launch_params_t<128, 2>>(
          k_reindex,
          nnodes,
          &scans.data()[scanstart],
          nscans,
          (HTRawBuffer<Real>*)(HTscan.data()->data()),
          htcompose_t<D, Real, Int>(),
          init,
          context);

      // unindex for gen i
      // this would be nice to incorporate into kernel_segscan (as the indexing
      // is)
      auto k_unindex = [=] MGPU_DEVICE(int index) {
        HTs[nodes[nodestart + index]] = HTscan[index];
      };

      mgpu::transform(k_unindex, nnodes, context);
    }

    // copy atom positions
    auto k_getcoords = ([=] EIGEN_DEVICE_FUNC(int i) {
      xs[i] = HTs[i].block(3, 0, 1, 3).transpose();
    });

    mgpu::transform(k_getcoords, num_atoms, context);

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

    mgpu::standard_context_t context;
    mgpu::transform(k_coords2hts, num_atoms, context);

    auto k_hts2dofs = ([=] EIGEN_DEVICE_FUNC(int i) {
      HomogeneousTransform lclHT;
      if (doftype[i] == ROOT) {
        dofs[i] = KintreeDof::Constant(0);  // for num deriv check
      } else {
        lclHT = HTs[i] * common<D, Real, Int>::ht_inv(HTs[parent[i]]);

        if (doftype[i] == JUMP) {
          dofs[i] = common<D, Real, Int>::invJumpTransform(lclHT);
        } else if (doftype[i] == BOND) {
          dofs[i] = common<D, Real, Int>::invBondTransform(lclHT);
        }
      }
    });

    mgpu::transform(k_hts2dofs, num_atoms, context);

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
      TView<KinTreeGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinTreeParams<Int>, 1, D> kintree) -> TPack<KintreeDof, 1, D> {
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

    mgpu::standard_context_t context;
    mgpu::transform(k_f1f2s, num_atoms, context);

    // temp memory for scan (longest scan possible is 2 times # atoms)
    auto f1f2scan_t = TPack<f1f2Vectors, 1, D>::empty({2 * num_atoms});
    auto f1f2scan = f1f2scan_t.view;
    f1f2VecsRawBuffer<Real> init = {0, 0, 0, 0, 0, 0};  // identity

    auto ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; ++gen) {
      int nodestart = gens[gen].node_start, scanstart = gens[gen].scan_start;
      int nnodes = gens[gen + 1].node_start - nodestart;
      int nscans = gens[gen + 1].scan_start - scanstart;

      // reindexing function
      auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
        return *(
            (f1f2VecsRawBuffer<Real>*)f1f2s[nodes[nodestart + index]].data());
      };

      // mgpu does not play nicely with eigen types
      // instead, we wrap the raw data buffer
      //      and use eigen:map to reconstruct on device
      tmol::kinematics::
          kernel_segscan<mgpu::launch_params_t<256, 3>, scan_type_exc>(
              k_reindex,
              nnodes,
              &scans.data()[scanstart],
              nscans,
              (f1f2VecsRawBuffer<Real>*)(f1f2scan.data()->data()),
              f1f2compose_t<D, Real, Int>(),
              init,
              context);

      // unindex for gen i.  ENSURE ATOMIC
      auto k_unindex = [=] MGPU_DEVICE(int index) {
        for (int kk = 0; kk < 6; ++kk) {
          atomicAdd(
              &(f1f2s[nodes[nodestart + index]][kk]), f1f2scan[index][kk]);
        }
      };

      mgpu::transform(k_unindex, nnodes, context);
    }

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

    mgpu::transform(k_f1f2s2derivs, num_atoms, context);

    return dsc_ddofs_t;
  }
};

template struct ForwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct InverseKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::CUDA, float, int32_t>;
template struct KinDerivDispatch<tmol::Device::CUDA, double, int32_t>;

#undef HomogeneousTransform
#undef KintreeDof
#undef f1f2Vectors
#undef Coord

}  // namespace kinematics
}  // namespace tmol
