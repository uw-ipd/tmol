#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/kinematics/compiled/kernel_segscan.cuh>

#include <moderngpu/transform.hxx>

#include "common.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define QuatTranslation Eigen::Matrix<Real, 7, 1>
#define QuatTransRawBuffer std::array<Real, 7>
#define f1f2Vectors Eigen::Matrix<Real, 6, 1>
#define f1f2VecsRawBuffer std::array<Real, 6>

// the composite operation for the forward pass: apply a transform
//   qt1/qt2 -> HT1/HT2 -> HT1*HT2 -> qt12' -> norm(qt12')
template <tmol::Device D, typename Real, typename Int>
struct qcompose_t : public std::binary_function<
                        QuatTransRawBuffer,
                        QuatTransRawBuffer,
                        QuatTransRawBuffer> {
  MGPU_HOST_DEVICE QuatTransRawBuffer
  operator()(QuatTransRawBuffer p, QuatTransRawBuffer i) const {
    QuatTranslation ab = common<D, Real, Int>::quat_trans_compose(
        Eigen::Map<QuatTranslation>(i.data()),
        Eigen::Map<QuatTranslation>(p.data()));

    return *((QuatTransRawBuffer*)ab.data());
  }
};

// the composite operation for the backward pass: sum f1s & f2s
template <tmol::Device D, typename Real, typename Int>
struct f1f2compose_t : public std::binary_function<
                           f1f2VecsRawBuffer,
                           f1f2VecsRawBuffer,
                           f1f2VecsRawBuffer> {
  MGPU_HOST_DEVICE f1f2VecsRawBuffer
  operator()(f1f2VecsRawBuffer p, f1f2VecsRawBuffer i) const {
    f1f2Vectors ab =
        Eigen::Map<f1f2Vectors>(i.data()) + Eigen::Map<f1f2Vectors>(p.data());

    return *((f1f2VecsRawBuffer*)ab.data());
  }
};

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<Vec<Real, 9>, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TCollection<Int, 1, D> nodes,
      TCollection<Int, 1, D> scans) -> TPack<HomogeneousTransform, 1, D> {
    auto nodeview = nodes.view;

    auto num_atoms = dofs.size(0);

    auto QTs_t = TPack<QuatTranslation, 1, D>::empty({num_atoms});
    auto QTs = QTs_t.view;

    // dofs -> quaterion_xform function
    auto k_dof2qt = ([=] EIGEN_DEVICE_FUNC(int i) {
      DOFtype doftype = (DOFtype)doftypes[i];
      HomogeneousTransform HT;
      if (doftype == ROOT) {
        HT = HomogeneousTransform::Identity();
      } else if (doftype == JUMP) {
        HT = common<D, Real, Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HT = common<D, Real, Int>::bondTransform(dofs[i]);
      }
      QTs[i] = common<D, Real, Int>::ht2quat_trans(HT);
    });

    mgpu::standard_context_t context;
    mgpu::transform(
        [=] MGPU_DEVICE(int idx) { k_dof2qt(idx); }, num_atoms, context);

    // memory for scan (longest scan possible is 2 times # atoms)
    auto QTscan_t = TPack<QuatTranslation, 1, D>::empty({2 * num_atoms});
    auto QTscan = QTscan_t.view;
    QuatTransRawBuffer init = {0, 0, 0, 1, 0, 0, 0};  // identity xform

    auto ngens = nodeview.size(0);
    for (int i = 0; i < ngens; ++i) {
      int nnodes = nodes.tensors[i].size(0);
      int nscans = scans.tensors[i].size(0);

      // reindexing function
      auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
        return *((QuatTransRawBuffer*)QTs[nodeview[i][index]].data());
      };

      // mgpu does not play nicely with eigen types
      // instead, we wrap the raw data buffer as QuatTransRawBuffer
      //      and use eigen:map to reconstruct on device
      tmol::kinematics::kernel_segscan<mgpu::launch_params_t<256, 3>>(
          k_reindex,
          nnodes,
          scans.tensors[i].view.data(),  // get a single tensor view on CPU
          nscans,
          (QuatTransRawBuffer*)(QTscan.data()->data()),
          qcompose_t<D, Real, Int>(),
          init,
          context);

      // unindex for gen i
      // this would be nice to incorporate into kernel_segscan (as the indexing
      // is)
      auto k_unindex = [=] MGPU_DEVICE(int index) {
        QTs[nodeview[i][index]] = QTscan[index];
      };

      mgpu::transform(k_unindex, nnodes, context);
    }

    // quats -> HTs
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;

    auto k_qt2dof = ([=] EIGEN_DEVICE_FUNC(int i) {
      HTs[i] = common<D, Real, Int>::quat_trans2ht(QTs[i]);
    });

    mgpu::transform(
        [=] MGPU_DEVICE(int idx) { k_qt2dof(idx); }, num_atoms, context);

    return HTs_t;
  }
};

template <tmol::Device D, typename Real, typename Int>
struct DOFTransformsDispatch {
  static auto f(TView<Vec<Real, 9>, 1, D> dofs, TView<Int, 1, D> doftypes)
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
        HTs[i] = common<D, Real, Int>::jumpTransform(dofs[i]);
      } else if (doftype == BOND) {
        HTs[i] = common<D, Real, Int>::bondTransform(dofs[i]);
      }
    });

    mgpu::standard_context_t context;
    mgpu::transform(
        [=] MGPU_DEVICE(int idx) { k_dof2ht(idx); }, num_atoms, context);

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
      TView<Vec<Real, 9>, 1, D> dofs) -> TPack<HomogeneousTransform, 1, D> {
    auto num_atoms = coords.size(0);

    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;

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
    mgpu::transform(
        [=] MGPU_DEVICE(int idx) { k_coords2hts(idx); }, num_atoms, context);

    auto k_hts2dofs = ([=] EIGEN_DEVICE_FUNC(int i) {
      HomogeneousTransform lclHT;
      if (doftypes[i] != ROOT) {
        lclHT = HTs[i] * common<D, Real, Int>::ht_inv(HTs[parents[i]]);

        if (doftypes[i] == JUMP) {
          dofs[i] = common<D, Real, Int>::invJumpTransform(lclHT);
        } else if (doftypes[i] == BOND) {
          dofs[i] = common<D, Real, Int>::invBondTransform(lclHT);
        }
      }
    });

    mgpu::transform(
        [=] MGPU_DEVICE(int idx) { k_hts2dofs(idx); }, num_atoms, context);

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
        dsc_ddofs[i] = common<D, Real, Int>::jumpDerivatives(
            dofs[i], hts[i], hts[parents[i]], f1, f2);
      } else if (doftypes[i] == BOND) {
        dsc_ddofs[i] = common<D, Real, Int>::bondDerivatives(
            dofs[i], hts[i], hts[parents[i]], f1, f2);
      }
    });

    mgpu::standard_context_t context;
    mgpu::transform(
        [=] MGPU_DEVICE(int idx) { k_f1f2s2derivs(idx); }, num_atoms, context);

    return dsc_ddofs_t;
  }
};

template <tmol::Device D, typename Real, typename Int>
struct SegscanF1f2sDispatch {
  static void f(
      TView<Vec<Real, 6>, 1, D> f1f2s,
      TCollection<Int, 1, D> nodes,
      TCollection<Int, 1, D> scans) {
    auto num_atoms = f1f2s.size(0);

    auto nodeview = nodes.view;

    mgpu::standard_context_t context;

    // temp memory for scan (longest scan possible is 2 times # atoms)
    auto f1f2scan_t = TPack<f1f2Vectors, 1, D>::empty({2 * num_atoms});
    auto f1f2scan = f1f2scan_t.view;
    f1f2VecsRawBuffer init = {0, 0, 0, 0, 0, 0};  // identity

    auto ngens = nodeview.size(0);
    for (int i = 0; i < ngens; ++i) {
      int nnodes = nodes.tensors[i].size(0);
      int nscans = scans.tensors[i].size(0);

      // reindexing function
      auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
        return *((f1f2VecsRawBuffer*)f1f2s[nodeview[i][index]].data());
      };

      // mgpu does not play nicely with eigen types
      // instead, we wrap the raw data buffer
      //      and use eigen:map to reconstruct on device
      tmol::kinematics::
          kernel_segscan<mgpu::launch_params_t<256, 1>, scan_type_exc>(
              k_reindex,
              nnodes,
              scans.tensors[i].view.data(),  // get a single tensor view on CPU
              nscans,
              (f1f2VecsRawBuffer*)(f1f2scan.data()->data()),
              f1f2compose_t<D, Real, Int>(),
              init,
              context);

      // unindex for gen i.  ENSURE ATOMIC
      auto k_unindex = [=] MGPU_DEVICE(int index) {
            index, f1f2scan[index][0],
            nodeview[i][index], f1f2s[nodeview[i][index]][0],
            f1f2scan[index][0] + f1f2s[nodeview[i][index]][0]
        );
            for (int kk = 0; kk < 6; ++kk) {
              atomicAdd(&(f1f2s[nodeview[i][index]][kk]), f1f2scan[index][kk]);
            }
      };

      mgpu::transform(k_unindex, nnodes, context);
    }

    return;
  }
};

template struct ForwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CUDA, float, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CUDA, double, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct f1f2ToDerivsDispatch<tmol::Device::CUDA, float, int32_t>;
template struct f1f2ToDerivsDispatch<tmol::Device::CUDA, double, int32_t>;
template struct SegscanF1f2sDispatch<tmol::Device::CUDA, float, int32_t>;
template struct SegscanF1f2sDispatch<tmol::Device::CUDA, double, int32_t>;

#undef HomogeneousTransform
#undef QuatTranslation
#undef QuatTransRawBuffer
#undef f1f2Vectors
#undef f1f2VecsRawBuffer

}  // namespace kinematics
}  // namespace tmol
