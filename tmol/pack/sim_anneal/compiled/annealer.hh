#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
//#include <tmol/pack/compiled/params.hh>

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
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
      TView<Int, 1, D> random_rots) -> void;
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
      TView<Int, 1, D> accept) -> void;
};

class PickRotamersStep {
 public:
  PickRotamersStep(
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
      torch::Tensor alternate_id);
  virtual ~PickRotamersStep();
  virtual void pick_rotamers();

 private:
  torch::Tensor context_coords_;
  torch::Tensor context_block_type_;
  torch::Tensor pose_id_for_context_;
  torch::Tensor n_rots_for_pose_;
  torch::Tensor rot_offset_for_pose_;
  torch::Tensor block_type_ind_for_rot_;
  torch::Tensor block_ind_for_rot_;
  torch::Tensor rotamer_coords_;
  torch::Tensor random_rots_;
  torch::Tensor alternate_coords_;
  torch::Tensor alternate_id_;
};

class MetropolisAcceptRejectStep {
 public:
  MetropolisAcceptRejectStep(
      torch::Tensor temperature,
      torch::Tensor context_coords,
      torch::Tensor context_block_type,
      torch::Tensor alternate_coords,
      torch::Tensor alternate_id,
      torch::Tensor rotamer_component_energies,
      torch::Tensor accept);

  virtual ~MetropolisAcceptRejectStep();
  virtual void accept_reject();

 private:
  torch::Tensor temperature_;
  torch::Tensor context_coords_;
  torch::Tensor context_block_type_;
  torch::Tensor alternate_coords_;
  torch::Tensor alternate_id_;
  torch::Tensor rotamer_component_energies_;
  torch::Tensor accept_;
};

class RPECalc {
 public:
  virtual ~RPECalc() {}
  virtual void calc_energies() = 0;
};

class SimAnnealer {
 public:
  SimAnnealer();
  ~SimAnnealer();

  virtual void set_pick_rotamers_step(
      std::shared_ptr<PickRotamersStep> pick_step);

  virtual void set_metropolis_accept_reject_step(
      std::shared_ptr<MetropolisAcceptRejectStep> acc_rej_step);

  virtual void add_score_component(std::shared_ptr<RPECalc> score_calculator);

  virtual void run_annealer();

 protected:
  std::shared_ptr<PickRotamersStep> pick_step();
  std::shared_ptr<MetropolisAcceptRejectStep> acc_rej_step();
  std::list<std::shared_ptr<RPECalc>> const& score_calculators();

 private:
  std::shared_ptr<PickRotamersStep> pick_step_;
  std::shared_ptr<MetropolisAcceptRejectStep> acc_rej_step_;
  std::list<std::shared_ptr<RPECalc>> score_calculators_;
};

}  // namespace compiled
}  // namespace sim_anneal
}  // namespace pack
}  // namespace tmol
