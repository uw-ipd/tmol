// Instantiations of the fused LJ/LK + elec dispatch for the Metal backend.
//
// Only the two members packing reaches are instantiated: the rotamer-pair
// dispatch builder and the weighted rotamer scorer, whose Metal path is
// ljlk_elec_kernels.hh. The whole-pose members accumulate in double, which
// Metal does not have, and are never reached by an MPS tensor. Only float:
// the dispatch macro rejects a float64 tensor before it gets here.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

namespace {
constexpr tmol::Device MPS = tmol::Device::MPS;
template <typename T, int N>
using V = TView<T, N, MPS>;
using Coords = V<LJLKExternalVec<float, 3>, 1>;
using ElecGlobal = tmol::score::elec::potentials::ElecGlobalParams<float>;
}  // namespace

using MPSDispatch =
    LJLKAndElecPoseScoreDispatch<DeviceOperations, MPS, float, int>;

template auto MPSDispatch::build_rotamer_dispatch(
    ContextManager& mgr,
    Coords rot_coords,
    V<int, 1> rot_coord_offset,
    V<int, 2> first_rot_block_type,
    V<int, 1> block_type_ind_for_rot,
    V<int, 2> n_rots_for_block,
    V<int, 2> rot_offset_for_block,
    V<int, 2> lockstep_group_for_block,
    V<int, 1> block_type_n_atoms,
    float max_dis,
    int64_t candidate_begin,
    int64_t candidate_count) -> TPack<int, 2, MPS>;

template auto MPSDispatch::forward_weighted_rotamers(
    ContextManager& mgr,
    Coords rot_coords,
    V<int, 1> rot_coord_offset,
    V<int, 1> pose_ind_for_atom,
    V<int, 2> first_rot_for_block,
    V<int, 2> first_rot_block_type,
    V<int, 1> block_ind_for_rot,
    V<int, 1> pose_ind_for_rot,
    V<int, 1> block_type_ind_for_rot,
    V<int, 1> n_rots_for_pose,
    V<int, 1> rot_offset_for_pose,
    V<int, 2> n_rots_for_block,
    V<int, 2> rot_offset_for_block,
    V<int, 2> lockstep_group_for_block,
    int max_n_rots_per_pose,
    V<int, 4> pose_stack_near_blocks,
    V<int8_t, 5> pose_stack_inter_block_bondsep,
    V<int, 1> block_type_n_atoms,
    V<int, 2> block_type_atom_types,
    V<int, 1> block_type_n_interblock_bonds,
    V<int, 2> block_type_atoms_forming_chemical_bonds,
    V<int, 3> block_type_ljlk_path_distance,
    V<int, 1> block_type_all_atoms_ligand_typed,
    V<LJLKTypeParams<float>, 1> ljlk_type_params,
    V<LJGlobalParams<float>, 1> ljlk_global_params,
    V<float, 2> block_type_partial_charge,
    V<int, 3> block_type_elec_inter_repr_path_distance,
    V<int, 3> block_type_elec_intra_repr_path_distance,
    V<ElecGlobal, 1> elec_global_params,
    float max_dis,
    V<float, 1> score_weights,
    V<float, 1> output_gradients,
    TPack<int, 2, MPS> shared_dispatch_indices)
    -> std::tuple<
        TPack<float, 4, MPS>,
        TPack<LJLKExternalVec<float, 3>, 2, MPS>,
        TPack<int, 2, MPS>>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
