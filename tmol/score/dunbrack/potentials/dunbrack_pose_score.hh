#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/unresolved_atom.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct DunbrackPoseScoreDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 3, D> block_type_atom_downstream_of_conn,

      // TView<DunbrackGlobalParams<Real>, 1, D> global_params,
      TView<Real, 3, D> rotameric_neglnprob_tables,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
      TView<Real, 3, D> rotameric_mean_tables,
      TView<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

      TView<Int, 1, D> rotameric_rotind2tableind,
      TView<Int, 1, D> semirotameric_rotind2tableind,

      TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
      TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
      TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

      TView<Int, 1, D> res_n_dihedrals,
      TView<UnresolvedAtomID<Int>, 2, D> res_phi_uaids,
      TView<UnresolvedAtomID<Int>, 2, D> res_psi_uaids,
      TView<UnresolvedAtomID<Int>, 3, D> res_chi_uaids,
      TView<UnresolvedAtomID<Int>, 3, D> res_dih_uaids,
      TView<Int, 1, D> res_rotamer_tablet_set,
      TView<Int, 1, D> res_rotameric_index,
      TView<Int, 1, D> res_semirotameric_index,
      TView<Int, 1, D> res_n_chi,
      TView<Int, 1, D> res_n_rotameric_chi,
      TView<Int, 1, D> res_probability_table_offset,
      TView<Int, 1, D> res_mean_table_offset,
      TView<Int, 1, D> res_rotamer_index_to_table_index,
      TView<Int, 1, D> block_semirotameric_tableset_offset,

      bool compute_derivs

      ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>>;
};

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
