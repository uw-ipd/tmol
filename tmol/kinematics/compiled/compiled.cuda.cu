#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/kinematics/compiled/kernel_segscan.cuh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/utility/nvtx.hh>

#include <moderngpu/transform.hxx>

#include "common.hh"
#include "params.hh"
#include "compiled.impl.hh"

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

// This function gets the memory needed for the output and temporary buffers
//   in the segmented scan.
//
// It first loops over all generations and finds the generation with longest
//   scan_length = #elements + #segments
//
// Then using the kernel launch parameters 'nt' and 'vt':
//      vt: # of threads per CTA
//      nt: # of elements processed per thread
// it computes three quantities:
//   scanSize - "scan_length" above, used for the output buffer
//   lbsSize - a temporary buffer used for the load-balancing scan parameters
//     = ceil (scanSize / (vt*nt)) + 1
//   carryoutSize - a temporary buffer used for managing "carryout" from
//                  each of the subscans.
//      = ceil( (count+num_segments) / (vt*nt) )
//        + ceil( (count+num_segments) / (vt*nt*nt) )
//        + ceil( (count+num_segments) / (vt*nt*nt*nt) )
//        + ...
//      [stopping the summation when the arg to ceil becomes < 1].
//
// These are used to preallocate the memory used in each generation of the scan.
template <typename Int>
auto getScanBufferSize(
    TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens, Int nt, Int vt)
    -> mgpu::tuple<Int, Int, Int> {
  auto ngens = gens.size(0) - 1;
  Int scanSize = 0;
  for (int gen = 0; gen < ngens; ++gen) {
    Int nnodes = gens[gen + 1].node_start - gens[gen].node_start;
    Int nsegs = gens[gen + 1].scan_start - gens[gen].scan_start;
    scanSize = std::max(nnodes + nsegs, scanSize);
  }

  float scanleft = std::ceil(((float)scanSize) / (nt * vt));
  Int lbsSize = (Int)scanleft + 1;
  Int carryoutSize = (Int)scanleft;
  while (scanleft > 1) {
    scanleft = std::ceil(scanleft / nt);
    carryoutSize += (Int)scanleft;
  }

  return {scanSize, carryoutSize, lbsSize};
}

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

// Dispatch class for cuda-based generational segmented scan operations.
//
// # Scan Overview
//
// The scan operation processes linear scan paths with an associative binary
// operator, where the scan paths many have any number of additional off-path
// node inputs added *before* the scan path. For example, consider the
// operation composed of path values (P), off path values (OP) joined by an
// operator (+)::
//
//     OP_0            OP_1
//       +               +
//       |               |
//       v               v
//     P_0+--->P_1+--->P_2+--->P_3+--->P_4
//                       ^
//                       |
//                       +
//                     OP_2
//
// Represents the complete operation::
//   (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3 + P_4
//
// As this is a scan, rather than reduction, this results in::
//
//     R_0-----R_1-----R_2-----R_3-----R_4
//     R_0 = OP_0 + P_0
//     R_1 = (OP_0 + P_0) + P_1
//     R_2 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2)
//     R_3 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3
//     R_4 = (OP_0 + P_0) + P_1 + (OP_1 + OP_2 + P_2) + P_3 + P_4
//
// The off-path inputs of a scan are taken from the result values of previous
// scan operations. Scans are processed by "generation", arranged such that
// the off-path inputs of any segment are draw *exclusively* from earlier
// generations. Scans within a generation are processed via a parallel
// segmented scan.
//
// The input arguments 'nodes', 'scans' give the individual scans, with
// 'nodes' providing the scan ordering and 'scans' giving scan boundaries.
//
// The input argument 'gens' breaks these down into individual generations,
// providing both 'node' and 'scan' boundaries for each generation.
//
template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D>> {
    NVTXRange _function(__FUNCTION__);
    using tmol::score::common::tie;
    typedef typename mgpu::launch_params_t<128, 2> launch_t;
    constexpr int nt = launch_t::nt, vt = launch_t::vt;
    auto num_atoms = dofs.size(0);

    nvtx_range_push("dispatch::alloc");
    auto HTs_t = TPack<HomogeneousTransform, 1, D>::empty({num_atoms});
    auto HTs = HTs_t.view;
    auto xs_t = TPack<Coord, 1, D>::empty({num_atoms});
    auto xs = xs_t.view;
    nvtx_range_pop();

    // temp memory for scan
    nvtx_range_push("dispatch::alloc_temp");
    int carryoutBuffer, scanBuffer, lbsBuffer;
    tie(scanBuffer, carryoutBuffer, lbsBuffer) =
        getScanBufferSize(gens, nt, vt);
    auto scanCarryout_t =
        TPack<HomogeneousTransform, 1, D>::empty({carryoutBuffer});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t = TPack<int, 1, D>::empty({carryoutBuffer});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<Int, 1, D>::empty({lbsBuffer});
    auto LBS = LBS_t.view;
    auto HTscan_t = TPack<HomogeneousTransform, 1, D>::empty({scanBuffer});
    auto HTscan = HTscan_t.view;
    HTRawBuffer<Real> init = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // identity xform
    nvtx_range_pop();

    // dofs -> HTs
    nvtx_range_push("dispatch::dof2ht");
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
    nvtx_range_pop();

    auto ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; ++gen) {
      int nodestart = gens[gen].node_start, scanstart = gens[gen].scan_start;

      int nnodes = gens[gen + 1].node_start - nodestart;
      int nscans = gens[gen + 1].scan_start - scanstart;
      if (nnodes == 0 && nscans == 0) {
        continue;
      }

      // reindexing function
      nvtx_range_push("dispatch::segscan");
      auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
        if (nodestart + index >= nodes.size(0) || nodestart + index < 0) {
          printf(
              "oops! nodestart %d + index %d vs nodes.size(0) %d\n",
              nodestart,
              index,
              nodes.size(0));
          return *((HTRawBuffer<Real>*)HTs[0].data());
        }
        if (nodes[nodestart + index] >= HTs.size(0)
            || nodes[nodestart + index] < 0) {
          printf(
              "oops2! nodestart %d + index %d gives nodes[%d] = %d and "
              "HTs.size() = %d\n",
              nodestart,
              index,
              nodestart + index,
              nodes[nodestart + index],
              HTs.size(0));
          return *((HTRawBuffer<Real>*)HTs[0].data());
        }

        assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
        assert(
            nodes[nodestart + index] < HTs.size(0)
            && nodes[nodestart + index] >= 0);
        return *((HTRawBuffer<Real>*)HTs[nodes[nodestart + index]].data());
      };

      // mgpu does not play nicely with eigen types
      // instead, we wrap the raw data buffer as QuatTransRawBuffer
      //      and use eigen:map to reconstruct on device
      tmol::kinematics::kernel_segscan<launch_t>(
          k_reindex,
          nnodes,
          &scans.data()[scanstart],
          nscans,
          (HTRawBuffer<Real>*)(HTscan.data()->data()),
          (HTRawBuffer<Real>*)(scanCarryout.data()->data()),
          &scanCodes.data()[0],
          &LBS.data()[0],
          htcompose_t<D, Real, Int>(),
          init,
          context);
      nvtx_range_pop();
      // gpuErrPeek;
      // gpuErrSync;

      // unindex for gen i
      // this would be nice to incorporate into kernel_segscan (as the indexing
      // is)
      nvtx_range_push("dispatch::unindex");
      auto k_unindex = [=] MGPU_DEVICE(int index) {
        if (nodestart + index >= nodes.size(0) || nodestart + index < 0) {
          printf(
              "oops3! nodestart %d + index %d vs nodes.size(0) %d\n",
              nodestart,
              index,
              nodes.size(0));
          return;  // *((HTRawBuffer<Real>*)HTs[0].data());
        }
        assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
        if (nodes[nodestart + index] >= HTs.size(0)
            || nodes[nodestart + index] < 0) {
          printf(
              "oops4! nodestart %d + index %d gives nodes[%d] = %d and "
              "HTs.size() = %d\n",
              nodestart,
              index,
              nodestart + index,
              nodes[nodestart + index],
              HTs.size(0));
          return;  // *((HTRawBuffer<Real>*)HTs[0].data());
        }

        assert(
            nodes[nodestart + index] < HTs.size(0)
            && nodes[nodestart + index] >= 0);
        HTs[nodes[nodestart + index]] = HTscan[index];
      };

      mgpu::transform(k_unindex, nnodes, context);
      nvtx_range_pop();
      // gpuErrPeek;
      // gpuErrSync;
      nvtx_range_pop();
    }

    // copy atom positions
    auto k_getcoords = ([=] EIGEN_DEVICE_FUNC(int i) {
      assert(i < HTs.size(0) && i >= 0);
      xs[i] = HTs[i].block(3, 0, 1, 3).transpose();
    });

    mgpu::transform(k_getcoords, num_atoms, context);
    // gpuErrPeek;
    // gpuErrSync;

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
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree) -> TPack<KintreeDof, 1, D> {
    NVTXRange _function(__FUNCTION__);
    using tmol::score::common::tie;
    typedef typename mgpu::launch_params_t<256, 3> launch_t;
    constexpr int nt = launch_t::nt, vt = launch_t::vt;

    auto num_atoms = dVdx.size(0);

    nvtx_range_push("dispatch::dalloc");
    auto f1f2s_t = TPack<f1f2Vectors, 1, D>::empty({num_atoms});
    auto f1f2s = f1f2s_t.view;
    auto dsc_ddofs_t = TPack<KintreeDof, 1, D>::empty({num_atoms});
    auto dsc_ddofs = dsc_ddofs_t.view;
    nvtx_range_pop();

    // temp memory for scan
    nvtx_range_push("dispatch::dalloc_temp");
    int carryoutBuffer, scanBuffer, lbsBuffer;
    tie(scanBuffer, carryoutBuffer, lbsBuffer) =
        getScanBufferSize(gens, nt, vt);
    auto scanCarryout_t = TPack<f1f2Vectors, 1, D>::empty({carryoutBuffer});
    auto scanCarryout = scanCarryout_t.view;
    auto scanCodes_t = TPack<int, 1, D>::empty({carryoutBuffer});
    auto scanCodes = scanCodes_t.view;
    auto LBS_t = TPack<Int, 1, D>::empty({lbsBuffer});
    auto LBS = LBS_t.view;
    auto f1f2scan_t = TPack<f1f2Vectors, 1, D>::empty({scanBuffer});
    auto f1f2scan = f1f2scan_t.view;
    f1f2VecsRawBuffer<Real> init = {0, 0, 0, 0, 0, 0};  // identity
    nvtx_range_pop();

    // calculate f1s and f2s from dVdx and HT
    nvtx_range_push("dispatch::ddof2ht");
    auto k_f1f2s = ([=] EIGEN_DEVICE_FUNC(int i) {
      Coord trans = hts[i].block(3, 0, 1, 3).transpose();
      Coord f1 = dVdx[i].isZero(0) ? dVdx[i]
                                   : trans.cross(trans - dVdx[i]).transpose();
      f1f2s[i].topRows(3) = f1;
      f1f2s[i].bottomRows(3) = dVdx[i];
    });

    mgpu::standard_context_t context;
    mgpu::transform(k_f1f2s, num_atoms, context);
    nvtx_range_pop();

    auto ngens = gens.size(0) - 1;
    for (int gen = 0; gen < ngens; ++gen) {
      int nodestart = gens[gen].node_start, scanstart = gens[gen].scan_start;
      int nnodes = gens[gen + 1].node_start - nodestart;
      int nscans = gens[gen + 1].scan_start - scanstart;

      if (nnodes == 0 && nscans == 0) {
        continue;
      }

      // reindexing function
      nvtx_range_push("dispatch::dsegscan");
      auto k_reindex = [=] MGPU_DEVICE(int index, int seg, int rank) {
        assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
        assert(
            nodes[nodestart + index] < f1f2s.size(0)
            && nodes[nodestart + index] >= 0);
        return *(
            (f1f2VecsRawBuffer<Real>*)f1f2s[nodes[nodestart + index]].data());
      };

      // mgpu does not play nicely with eigen types
      // instead, we wrap the raw data buffer
      //      and use eigen:map to reconstruct on device
      tmol::kinematics::kernel_segscan<launch_t, scan_type_exc>(
          k_reindex,
          nnodes,
          &scans.data()[scanstart],
          nscans,
          (f1f2VecsRawBuffer<Real>*)(f1f2scan.data()->data()),
          (f1f2VecsRawBuffer<Real>*)(scanCarryout.data()->data()),
          &scanCodes.data()[0],
          &LBS.data()[0],
          f1f2compose_t<D, Real, Int>(),
          init,
          context);
      nvtx_range_pop();

      // unindex for gen i.  ENSURE ATOMIC
      nvtx_range_push("dispatch::dunindex");
      auto k_unindex = [=] MGPU_DEVICE(int index) {
        assert(nodestart + index < nodes.size(0) && nodestart + index >= 0);
        assert(
            nodes[nodestart + index] < f1f2s.size(0)
            && nodes[nodestart + index] >= 0);
        for (int kk = 0; kk < 6; ++kk) {
          atomicAdd(
              &(f1f2s[nodes[nodestart + index]][kk]), f1f2scan[index][kk]);
        }
      };

      mgpu::transform(k_unindex, nnodes, context);
      nvtx_range_pop();
    }

    nvtx_range_push("dispatch::f1f2_to_deriv");
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
    nvtx_range_pop();

    return dsc_ddofs_t;
  }
};

template struct ForwardKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct ForwardKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct InverseKinDispatch<tmol::Device::CUDA, float, int32_t>;
template struct InverseKinDispatch<tmol::Device::CUDA, double, int32_t>;
template struct KinDerivDispatch<tmol::Device::CUDA, float, int32_t>;
template struct KinDerivDispatch<tmol::Device::CUDA, double, int32_t>;

template struct KinForestFromStencil<
    tmol::score::common::DeviceOperations,
    tmol::Device::CUDA,
    int32_t>;
// NOTE: Intetionally not enabling int64_t as there are atomic_incremement
// operations that are needed for several steps in the kin-forest construction
// algorithm and atomic_increment is not implemented in CUDA for int64_t.
// template struct KinForestFromStencil<
//     tmol::score::common::DeviceOperations,
//     tmol::Device::CUDA,
//     int64_t>;

#undef HomogeneousTransform
#undef KintreeDof
#undef f1f2Vectors
#undef Coord

}  // namespace kinematics
}  // namespace tmol
