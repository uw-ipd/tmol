#include <torch/script.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/function.h> // ??
#include <torch/csrc/autograd/saved_variable.h> // ??
#include <torch/types.h>

#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include "annealer.hh"
#include "simulated_annealing.hh"

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

using torch::Tensor;

template < template <tmol::Device> class DispatchMethod >
std::vector< Tensor >
pick_random_rotamers(
  Tensor context_coords,
  Tensor context_block_type,
  Tensor pose_id_for_context,
  Tensor n_rots_for_pose,
  Tensor rot_offset_for_pose,
  Tensor block_type_ind_for_rot,
  Tensor block_ind_for_rot,
  Tensor rotamer_coords
)
{
  at::Tensor alternate_coords;
  at::Tensor alternate_id;
  at::Tensor random_rotamers;

  using Int = int32_t;

  try {
    TMOL_DISPATCH_FLOATING_DEVICE(
        context_coords.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
  
          auto result = PickRotamers<DispatchMethod, Dev, Real, Int>::f(
              TCAST(context_coords),
              TCAST(context_block_type),
              TCAST(pose_id_for_context),
              TCAST(n_rots_for_pose),
              TCAST(rot_offset_for_pose),
              TCAST(block_type_ind_for_rot),
              TCAST(block_ind_for_rot),
              TCAST(rotamer_coords));
  
          alternate_coords = std::get<0>(result).tensor;
          alternate_id = std::get<1>(result).tensor;
	  random_rotamers = std::get<2>(result).tensor;
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return {alternate_coords, alternate_id, random_rotamers};
}

static auto registry =
  torch::jit::RegisterOperators()
  .op("tmol::pick_random_rotamers", &pick_random_rotamers<tmol::score::common::ForallDispatch>);


} // namespace compiled
} // namespace sim_anneal
} // namespace pack
} // namespace tmol
