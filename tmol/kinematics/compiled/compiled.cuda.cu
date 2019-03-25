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

// the composite operation: apply a transform
//   qt1/qt2 -> HT1/HT2 -> HT1*HT2 -> qt12' -> norm(qt12')
template <tmol::Device D, typename Real, typename Int>
struct qcompose_t : public std::binary_function<
                        std::array<Real, 7>,
                        std::array<Real, 7>,
                        std::array<Real, 7>> {
  MGPU_HOST_DEVICE std::array<Real, 7> operator()(
      std::array<Real, 7> p, std::array<Real, 7> i) const {
    QuatTranslation ab = common<D, Real, Int>::quat_trans_compose(
        Eigen::Map<QuatTranslation>(i.data()),
        Eigen::Map<QuatTranslation>(p.data()));

    return *((QuatTransRawBuffer*)ab.data());
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

template struct ForwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CUDA, float, int32_t>;
template struct DOFTransformsDispatch<tmol::Device::CUDA, double, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct BackwardKinDispatch<tmol::Device::CUDA, double, int32_t>;

#undef HomogeneousTransform
#undef QuatTranslation
#undef QuatTransRawBuffer

}  // namespace kinematics
}  // namespace tmol
