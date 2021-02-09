#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "simulated_annealing.hh"

#include <ctime>


namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

template <
  template <tmol::Device>
  class Dispatch,
  tmol::Device D,
  typename Real,
  typename Int
>
struct PickRotamers {
  static auto f(
    TView<Real, 4, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Int, 1, D> pose_id_for_context,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Real, 3, D> rotamer_coords,
    TView<Real, 3, D> alternate_coords,
    TView<Int, 2, D> alternate_id,
    TView<Int, 1, D> random_rots
  ) -> void
  {
    int const n_contexts = context_coords.size(0);
    int const max_n_blocks = context_coords.size(1);
    int const max_n_atoms = context_coords.size(2);
    int const n_poses = pose_id_for_context.size(0);
    int const n_rots = block_type_ind_for_rot.size(0);

    assert(context_coords.size(3) == 3);
    assert(context_block_type.size(0) == n_contexts);
    assert(context_block_type.size(1) == max_n_blocks);
    assert(n_rots_for_pose.size(0) == n_poses);
    assert(rot_offset_for_pose.size(0) == n_poses);
    assert(block_ind_for_rot.size(0) == n_rots);
    assert(rotamer_coords.size(0) == n_rots);
    assert(rotamer_coords.size(1) == max_n_atoms);
    assert(rotamer_coords.size(2) == 3);
    assert(random_rots.size(0) == n_contexts);
    assert(alternate_coords.size(0) == 2 * n_contexts);
    assert(alternate_coords.size(1) == max_n_atoms);
    assert(alternate_coords.size(2) == 3);
    assert(alternate_id.size(0) == 2 * n_contexts);
    assert(alternate_id.size(1) == 3);
    
    auto select_rotamer = [=] (int i) {
      Real rand_unif = ((Real) rand()) / RAND_MAX; // replace w/ call to Torch RNG
      Int i_pose = pose_id_for_context[i];
      Int i_n_rots = n_rots_for_pose[i_pose];
      if (i_n_rots == 0 ) {
	alternate_id[i*2][0] = -1;
	alternate_id[i*2][1] = -1;
	alternate_id[i*2][2] = -1;
	alternate_id[i*2 + 1][0] = -1;
	alternate_id[i*2 + 1][1] = -1;
	alternate_id[i*2 + 1][2] = -1;
	random_rots[i] = -1;
      } else {
	Int i_rot_local = i_n_rots * rand_unif;
	Int i_rot_global = i_rot_local + rot_offset_for_pose[i_pose];
	Int i_block = block_ind_for_rot[i_rot_global];
	random_rots[i] = i_rot_global;

	alternate_id[i*2][0] = i;
	alternate_id[i*2][1] = i_block;
	alternate_id[i*2][2] = context_block_type[i][i_block];
	alternate_id[i*2 + 1][0] = i;
	alternate_id[i*2 + 1][1] = i_block;
	alternate_id[i*2 + 1][2] = block_type_ind_for_rot[i_rot_global];
      }
    };

    Dispatch<D>::forall(n_contexts, select_rotamer);

    auto copy_rotamer_coords = [=] (int i) {
      Int alt_id = i / (max_n_atoms);
      Int atom_id = i % max_n_atoms;
      Int i_context = alternate_id[alt_id][0];
      Int i_block = alternate_id[alt_id][1];
      if ( i_block == -1 ) {
	return;
      }
      if (alt_id % 2 == 0) {
	alternate_coords[alt_id][atom_id][0] = context_coords[i_context][i_block][atom_id][0];
	alternate_coords[alt_id][atom_id][1] = context_coords[i_context][i_block][atom_id][1];
	alternate_coords[alt_id][atom_id][2] = context_coords[i_context][i_block][atom_id][2];
      } else {
	Int i_rot = random_rots[i_context];
	alternate_coords[alt_id][atom_id][0] = rotamer_coords[i_rot][atom_id][0];
	alternate_coords[alt_id][atom_id][1] = rotamer_coords[i_rot][atom_id][1];
	alternate_coords[alt_id][atom_id][2] = rotamer_coords[i_rot][atom_id][2];
      }
    };

    Dispatch<D>::forall(n_contexts* 2 * max_n_atoms, copy_rotamer_coords);

  }

};



template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetropolisAcceptReject {
  static auto f(
      TView<Real, 1, D> temperature,
      TView<Real, 4, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Real, 3, D> alternate_coords,
      TView<Int, 2, D> alternate_id,
      TView<Real, 2, D> rotamer_component_energies,
      TView<Int, 1, D> accept
  )
      -> void
  {
    int const n_contexts = context_coords.size(0);
    int const n_terms = rotamer_component_energies.size(0);
    int const max_n_atoms = context_coords.size(2);

    assert(rotamer_component_energies.size(1) == 2 * n_contexts);
    assert(alternate_coords.size(0) == 2 * n_contexts);
    assert(alternate_coords.size(1) == max_n_atoms);
    assert(alternate_coords.size(2) == 3);
    assert(alternate_id.size(0) == 2 * n_contexts);
    assert(accept.size(0) == n_contexts);

    auto accept_reject = [=] (int i) {
      Real altE = 0;
      Real currE = 0;
      for (int j = 0; j < n_terms; ++j) {
	currE += rotamer_component_energies[j][2*i];
	altE += rotamer_component_energies[j][2*i + 1];
	rotamer_component_energies[j][2*i] = 0;
	rotamer_component_energies[j][2*i + 1] = 0;
      }
      Real deltaE = altE - currE;
      Real rand_unif = ((Real) rand()) / RAND_MAX;
      Real temp = temperature[0];
      Real prob_accept = std::exp( -1 * deltaE / temp );
      accept[i] = deltaE < 0 || rand_unif < prob_accept;
      if (accept[i]) {
	int block_id = alternate_id[2*i+1][1];
	context_block_type[i][block_id] = alternate_id[2*i+1][2];
      }
    };

    Dispatch<D>::forall(n_contexts, accept_reject);

    auto copy_accepted_coords = [=] (int i) {
      int context_id = i / max_n_atoms;
      int atom_id = i % max_n_atoms;
      Int accepted = accept[context_id];      
      if (accepted) {
	int block_id = alternate_id[2*context_id+1][1];
	for (int j = 0; j < 3; ++j) {
	  context_coords[context_id][block_id][atom_id][j] =
	    alternate_coords[2*context_id + 1][atom_id][j];
	}
      }
    };

    Dispatch<D>::forall(n_contexts * max_n_atoms, copy_accepted_coords);
    return;
  }
};


template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct FinalOp {
  static auto f() -> void {
    }
};


template struct PickRotamers<score::common::ForallDispatch, tmol::Device::CPU, float, int32_t>;
template struct PickRotamers<score::common::ForallDispatch, tmol::Device::CPU, double, int32_t>;
template struct PickRotamers<score::common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct PickRotamers<score::common::ForallDispatch, tmol::Device::CPU, double, int64_t>;


template struct MetropolisAcceptReject<score::common::ForallDispatch, tmol::Device::CPU, float, int32_t>;
template struct MetropolisAcceptReject<score::common::ForallDispatch, tmol::Device::CPU, double, int32_t>;
template struct MetropolisAcceptReject<score::common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct MetropolisAcceptReject<score::common::ForallDispatch, tmol::Device::CPU, double, int64_t>;

template struct FinalOp<score::common::ForallDispatch, tmol::Device::CPU, float, int32_t>;
template struct FinalOp<score::common::ForallDispatch, tmol::Device::CPU, double, int32_t>;
template struct FinalOp<score::common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct FinalOp<score::common::ForallDispatch, tmol::Device::CPU, double, int64_t>;


} // namespace compiled
} // namespace sim_anneal
} // namespace pack
} // namespace tmol
