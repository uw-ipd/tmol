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

Tensor
create_sim_annealer(
  Tensor annealer
)
{
  try {
    auto annealer_tp = TPack<int64_t, 1, tmol::Device::CPU>(
      annealer,
      view_tensor<int64_t, 1, tmol::Device::CPU>(annealer, "annealer"));
    auto annealer_view = annealer_tp.view;

    SimAnnealer * sim_annealer = new SimAnnealer;
    int64_t sim_annealer_uint = reinterpret_cast<int64_t> (sim_annealer);
    annealer_view[0] = sim_annealer_uint;

  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return annealer;
}

Tensor
delete_sim_annealer(
  Tensor annealer
)
{
  try {
    auto annealer_tp = TPack<int64_t, 1, tmol::Device::CPU>(
      annealer,
      view_tensor<int64_t, 1, tmol::Device::CPU>(annealer, "annealer"));
    auto annealer_view = annealer_tp.view;

    int64_t sim_annealer_uint = annealer_view[0];
    SimAnnealer * sim_annealer = reinterpret_cast<SimAnnealer *> (sim_annealer_uint);
    delete sim_annealer;
    annealer_view[0] = 0;

  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return annealer;
}



Tensor
register_standard_random_rotamer_picker(
  Tensor context_coords,
  Tensor context_block_type,
  Tensor pose_id_for_context,
  Tensor n_rots_for_pose,
  Tensor rot_offset_for_pose,
  Tensor block_type_ind_for_rot,
  Tensor block_ind_for_rot,
  Tensor rotamer_coords,
  Tensor alternate_coords,
  Tensor alternate_id,
  Tensor random_rotamers,
  Tensor annealer
)
{
  try {
    auto annealer_tp = TPack<int64_t, 1, tmol::Device::CPU>(
      annealer,
      view_tensor<int64_t, 1, tmol::Device::CPU>(annealer, "annealer"));

    int64_t annealer_uint = annealer_tp.view[0];
    SimAnnealer * sim_annealer = reinterpret_cast<SimAnnealer *> (annealer_uint);
    sim_annealer->set_pick_rotamers_step(
      std::make_shared<PickRotamersStep>(
	context_coords,
	context_block_type,
	pose_id_for_context,
	n_rots_for_pose,
	rot_offset_for_pose,
	block_type_ind_for_rot,
	block_ind_for_rot,
	rotamer_coords,
	random_rotamers,
	alternate_coords,
	alternate_id
      )
    );
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return annealer;
}


Tensor
register_standard_metropolis_accept_or_rejector(
  Tensor temperature,
  Tensor context_coords,
  Tensor context_block_type,
  Tensor alternate_coords,
  Tensor alternate_ids,
  Tensor rotamer_component_energies,
  Tensor accepted,
  Tensor annealer
)
{
  try {
    auto annealer_tp = TPack<int64_t, 1, tmol::Device::CPU>(
      annealer,
      view_tensor<int64_t, 1, tmol::Device::CPU>(annealer, "annealer"));

    int64_t annealer_uint = annealer_tp.view[0];
    SimAnnealer * sim_annealer = reinterpret_cast<SimAnnealer *> (annealer_uint);
    sim_annealer->set_metropolis_accept_reject_step(
      std::make_shared<MetropolisAcceptRejectStep>(
        temperature,
        context_coords,
        context_block_type,
        alternate_coords,
        alternate_ids,
        rotamer_component_energies,
        accepted
      )
    );
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return annealer;
}



template < template <tmol::Device> class DispatchMethod >
Tensor
pick_random_rotamers(
  Tensor context_coords,
  Tensor context_block_type,
  Tensor pose_id_for_context,
  Tensor n_rots_for_pose,
  Tensor rot_offset_for_pose,
  Tensor block_type_ind_for_rot,
  Tensor block_ind_for_rot,
  Tensor rotamer_coords,
  Tensor alternate_coords,
  Tensor alternate_id,
  Tensor random_rotamers
)
{

  using Int = int32_t;

  try {
    TMOL_DISPATCH_FLOATING_DEVICE(
        context_coords.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
  
          PickRotamers<DispatchMethod, Dev, Real, Int>::f(
              TCAST(context_coords),
              TCAST(context_block_type),
              TCAST(pose_id_for_context),
              TCAST(n_rots_for_pose),
              TCAST(rot_offset_for_pose),
              TCAST(block_type_ind_for_rot),
              TCAST(block_ind_for_rot),
              TCAST(rotamer_coords),
	      TCAST(alternate_coords),
	      TCAST(alternate_id),
	      TCAST(random_rotamers)
	  );
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return random_rotamers;
}

template < template <tmol::Device> class DispatchMethod >
Tensor
metropolis_accept_reject(
  Tensor temperature,
  Tensor context_coords,
  Tensor context_block_type,
  Tensor alternate_coords,
  Tensor alternate_ids,
  Tensor rotamer_component_energies,
  Tensor accepted
)
{
  using Int = int32_t;

  try {
    TMOL_DISPATCH_FLOATING_DEVICE(
        context_coords.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
  
          MetropolisAcceptReject<DispatchMethod, Dev, Real, Int>::f(
              TCAST(temperature),
	      TCAST(context_coords),
              TCAST(context_block_type),
              TCAST(alternate_coords),
              TCAST(alternate_ids),
              TCAST(rotamer_component_energies),
	      TCAST(accepted)
	  );
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
  return accepted;
}


static auto registry =
  torch::jit::RegisterOperators()
  .op("tmol::pick_random_rotamers", &pick_random_rotamers<tmol::score::common::ForallDispatch>)
  .op("tmol::metropolis_accept_reject", &metropolis_accept_reject<tmol::score::common::ForallDispatch>)
  .op("tmol::create_sim_annealer", &create_sim_annealer)
  .op("tmol::delete_sim_annealer", &delete_sim_annealer)
  .op("tmol::register_standard_random_rotamer_picker", &register_standard_random_rotamer_picker)
  .op("tmol::register_standard_metropolis_accept_or_rejector", &register_standard_metropolis_accept_or_rejector);


} // namespace compiled
} // namespace sim_anneal
} // namespace pack
} // namespace tmol
