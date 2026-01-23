#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/uaid_util.hh>

#include <tmol/io/details/compiled/his_taut_params.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct ResolveHisTaut {
  static auto f(
      TView<Vec<Real, 3>, 3, Dev> coords,
      TView<Int, 2, Dev> res_types,
      TView<Int, 2, Dev> res_type_variants,
      TView<int64_t, 1, Dev> his_pose_ind,
      TView<int64_t, 1, Dev> his_res_ind,
      TView<bool, 3, Dev> atom_is_present,
      TView<HisAtomIndsInCanonicalOrdering<Int>, 1, Dev> his_atom_inds,
      TView<int64_t, 3, Dev> his_remapping_dst_index

      ) -> TPack<int64_t, 2, Dev> {
    int const n_poses = coords.size(0);
    int const max_n_blocks = coords.size(1);
    int const max_n_block_atoms = coords.size(2);
    int const n_his = his_pose_ind.size(0);

    assert(res_types.size(0) == n_poses);
    assert(res_types.size(1) == max_n_blocks);
    assert(res_type_variants.size(0) == n_poses);
    assert(res_type_variants.size(1) == max_n_blocks);
    assert(his_res_ind.size(0) == n_his);
    assert(his_atom_inds.size(0) == 1);
    assert(his_remapping_dst_index.size(0) == n_poses);
    assert(his_remapping_dst_index.size(1) == max_n_blocks);
    assert(his_remapping_dst_index.size(2) == max_n_block_atoms);

    auto his_taut_t = TPack<int64_t, 2, Dev>::zeros({n_poses, max_n_blocks});
    auto his_taut = his_taut_t.view;

    LAUNCH_BOX_32;

    auto f_his_resolver = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const ip = his_pose_ind[ind];
      int const ir = his_res_ind[ind];
      HisAtomIndsInCanonicalOrdering<Int> atom_inds = his_atom_inds[0];
      bool const ND1_present = atom_is_present[ip][ir][atom_inds.his_ND1_in_co];
      bool const NE2_present = atom_is_present[ip][ir][atom_inds.his_NE2_in_co];
      bool const HD1_present = atom_is_present[ip][ir][atom_inds.his_HD1_in_co];
      bool const HE2_present = atom_is_present[ip][ir][atom_inds.his_HE2_in_co];
      bool const HN_present = atom_is_present[ip][ir][atom_inds.his_HN_in_co];
      bool const NH_present = atom_is_present[ip][ir][atom_inds.his_NH_in_co];
      bool const NN_present = atom_is_present[ip][ir][atom_inds.his_NN_in_co];
      bool const CG_present = atom_is_present[ip][ir][atom_inds.his_CG_in_co];

      int state = his_taut_unresolved;

      if (HD1_present && !HE2_present) {
        state = his_taut_HD1;
        res_type_variants[ip][ir] = his_taut_variant_ND1_protonated;
      } else if (HE2_present && !HD1_present) {
        state = his_taut_HE2;
        res_type_variants[ip][ir] = his_taut_variant_NE2_protonated;
      } else if (
          HN_present && !HD1_present && !HE2_present && ND1_present
          && NE2_present) {
        Vec<Real, 3> his_ND1_coord = coords[ip][ir][atom_inds.his_ND1_in_co];
        Vec<Real, 3> his_NE2_coord = coords[ip][ir][atom_inds.his_NE2_in_co];
        Vec<Real, 3> his_HN_coord = coords[ip][ir][atom_inds.his_HN_in_co];
        Real dis2_ND1 = (his_ND1_coord - his_HN_coord).squaredNorm();
        Real dis2_NE2 = (his_NE2_coord - his_HN_coord).squaredNorm();

        if (dis2_ND1 < dis2_NE2) {
          state = his_taut_HD1;
          res_type_variants[ip][ir] = his_taut_variant_ND1_protonated;
          his_remapping_dst_index[ip][ir][atom_inds.his_HD1_in_co] =
              atom_inds.his_HN_in_co;
        } else {
          state = his_taut_HE2;
          res_type_variants[ip][ir] = his_taut_variant_NE2_protonated;
          his_remapping_dst_index[ip][ir][atom_inds.his_HE2_in_co] =
              atom_inds.his_HN_in_co;
        }
      } else if (NH_present && NN_present && HN_present && CG_present) {
        Vec<Real, 3> his_NH_coord = coords[ip][ir][atom_inds.his_NH_in_co];
        Vec<Real, 3> his_NN_coord = coords[ip][ir][atom_inds.his_NN_in_co];
        Vec<Real, 3> his_CG_coord = coords[ip][ir][atom_inds.his_CG_in_co];
        Real dis2_NH = (his_NH_coord - his_CG_coord).squaredNorm();
        Real dis2_NN = (his_NN_coord - his_CG_coord).squaredNorm();
        if (dis2_NH < dis2_NN) {
          state = his_taut_NH_is_ND1;
          his_remapping_dst_index[ip][ir][atom_inds.his_ND1_in_co] =
              atom_inds.his_NH_in_co;
          his_remapping_dst_index[ip][ir][atom_inds.his_HD1_in_co] =
              atom_inds.his_HN_in_co;
          his_remapping_dst_index[ip][ir][atom_inds.his_NE2_in_co] =
              atom_inds.his_NN_in_co;
          res_type_variants[ip][ir] = his_taut_variant_ND1_protonated;
        } else {
          state = his_taut_NN_is_ND1;
          his_remapping_dst_index[ip][ir][atom_inds.his_ND1_in_co] =
              atom_inds.his_NN_in_co;
          his_remapping_dst_index[ip][ir][atom_inds.his_HE2_in_co] =
              atom_inds.his_HN_in_co;
          his_remapping_dst_index[ip][ir][atom_inds.his_NE2_in_co] =
              atom_inds.his_NH_in_co;
          res_type_variants[ip][ir] = his_taut_variant_NE2_protonated;
        }

      } else if (!HD1_present && !HE2_present && !HN_present) {
        // arbitrary choice: go with his_taut_HE2
        state = his_taut_HE2;
      }
      his_taut[ip][ir] = state;
    });

    DeviceOps<Dev>::template forall<launch_t>(n_his, f_his_resolver);
    return his_taut_t;
  };
};

#undef def

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
