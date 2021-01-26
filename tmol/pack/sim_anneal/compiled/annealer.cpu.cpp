#include <tmol/pack/sim_anneal/compiled/annealer.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/utility/tensor/TensorCast.h>

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

PickRotamersStep::PickRotamersStep(
  torch::Tensor context_coords,
  torch::Tensor context_block_type,
  torch::Tensor pose_id_for_context,
  torch::Tensor n_rots_for_pose,
  torch::Tensor rot_offset_for_pose,
  torch::Tensor block_type_ind_for_rot,
  torch::Tensor block_ind_for_rot,
  torch::Tensor rotamer_coords,
  torch::Tensor random_rots,
  torch::Tensor alternate_coords,
  torch::Tensor alternate_id
):
  context_coords_(context_coords),
  context_block_type_(context_block_type),
  pose_id_for_context_(pose_id_for_context),
  n_rots_for_pose_(n_rots_for_pose),
  rot_offset_for_pose_(rot_offset_for_pose),
  block_type_ind_for_rot_(block_type_ind_for_rot),
  block_ind_for_rot_(block_ind_for_rot),
  rotamer_coords_(rotamer_coords),
  random_rots_(random_rots),
  alternate_coords_(alternate_coords),
  alternate_id_(alternate_id)
{}

PickRotamersStep::~PickRotamersStep() {}

void
PickRotamersStep::pick_rotamers()
{
  using Int = int32_t;
  
  try {
    TMOL_DISPATCH_FLOATING_DEVICE(
        context_coords_.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

	  using tmol::score::common::ForallDispatch;
          PickRotamers<ForallDispatch, Dev, Real, Int>::f(
              TCAST(context_coords_),
              TCAST(context_block_type_),
              TCAST(pose_id_for_context_),
              TCAST(n_rots_for_pose_),
              TCAST(rot_offset_for_pose_),
              TCAST(block_type_ind_for_rot_),
              TCAST(block_ind_for_rot_),
              TCAST(rotamer_coords_),
	      TCAST(alternate_coords_),
	      TCAST(alternate_id_),
	      TCAST(random_rots_)
	  );
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
}


MetropolisAcceptRejectStep::MetropolisAcceptRejectStep(
  torch::Tensor temperature,
  torch::Tensor context_coords,
  torch::Tensor context_block_type,
  torch::Tensor alternate_coords,
  torch::Tensor alternate_id,
  torch::Tensor rotamer_component_energies,
  torch::Tensor accept
) :
  temperature_(temperature),
  context_coords_(context_coords),
  context_block_type_(context_block_type),
  alternate_coords_(alternate_coords),
  alternate_id_(alternate_id),
  rotamer_component_energies_(rotamer_component_energies),
  accept_(accept)
{}

MetropolisAcceptRejectStep::~MetropolisAcceptRejectStep() {}
void MetropolisAcceptRejectStep::accept_reject()
{
  using Int = int32_t;

  try {
    TMOL_DISPATCH_FLOATING_DEVICE(
        context_coords_.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
  
	  using tmol::score::common::ForallDispatch;
          MetropolisAcceptReject<ForallDispatch, Dev, Real, Int>::f(
              TCAST(temperature_),
	      TCAST(context_coords_),
              TCAST(context_block_type_),
              TCAST(alternate_coords_),
              TCAST(alternate_id_),
              TCAST(rotamer_component_energies_),
	      TCAST(accept_)
	  );
        }));
  } catch (at::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  } catch (c10::Error err) {
    std::cerr << "caught exception:\n" << err.what_without_backtrace() << std::endl;
    throw err;
  }
}


SimAnnealer::SimAnnealer() {std::cout << "Annealer ctor" << std::endl;}
SimAnnealer::~SimAnnealer() {std::cout << "Annealer dstor" << std::endl;}

void SimAnnealer::set_pick_rotamers_step(
  std::shared_ptr<PickRotamersStep> pick_step
)
{
  std::cout << "Setting pick step " << pick_step << std::endl;
  pick_step_ = pick_step;
}

void SimAnnealer::set_metropolis_accept_reject_step(
  std::shared_ptr<MetropolisAcceptRejectStep> acc_rej_step
)
{
  std::cout << "Setting acc/rej step " << acc_rej_step << std::endl;
  acc_rej_step_ = acc_rej_step;
}


void SimAnnealer::add_score_component(
  std::shared_ptr<RPECalc> score_calculator
)
{
  score_calculators_.push_back(score_calculator);
}

void SimAnnealer::run_annealer()
{
  for ( int i = 0; i < 100; ++i ) {
    pick_step_->pick_rotamers();
    for (auto const & rpe_calc: score_calculators_) {
      rpe_calc->calc_energies();
    }
    acc_rej_step_->accept_reject();
  }
}

std::shared_ptr<PickRotamersStep>
SimAnnealer::pick_step()
{
  return pick_step_;
}

std::shared_ptr<MetropolisAcceptRejectStep>
SimAnnealer::acc_rej_step()
{
  return acc_rej_step_;
}


std::list<std::shared_ptr<RPECalc>> const &
SimAnnealer::score_calculators()
{
  return score_calculators_;
}


}
}
}
}
