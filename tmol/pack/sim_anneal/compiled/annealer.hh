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
      TView<Real, 3, D> context_coords,
      TView<Int, 2, D> context_coord_offsets,
      TView<Int, 2, D> context_block_type,
      TView<Int, 1, D> pose_id_for_context,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Real, 2, D> rotamer_coords,
      TView<Int, 1, D> rotamer_coord_offsets,
      TView<Real, 2, D> alternate_coords,
      TView<Int, 1, D> alternate_coord_offsets,
      TView<Int, 2, D> alternate_id,
      TView<Int, 1, D> random_rots,
      TView<Int, 1, D> block_type_n_atoms,
      Int max_n_atoms,
      TView<int64_t, 1, tmol::Device::CPU> annealer_event) -> void;
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetropolisAcceptReject {
  static auto f(
      TView<Real, 1, tmol::Device::CPU> temperature,
      TView<Real, 3, D> context_coords,
      TView<Int, 2, D> context_coord_offsets,
      TView<Int, 2, D> context_block_type,
      TView<Real, 2, D> alternate_coords,
      TView<Int, 1, D> alternate_coord_offsets,
      TView<Int, 2, D> alternate_id,
      TView<Real, 2, D> rotamer_component_energies,
      TView<Int, 1, D> accept,
      TView<Int, 1, D> block_type_n_atoms,
      Int max_n_atoms,
      TView<int64_t, 1, tmol::Device::CPU> score_events) -> void;
};

class TemperatureScheduler {
 public:
  TemperatureScheduler(int n_outer_iterations, float max_temp, float min_temp);

  virtual float temp(int outer_iteration) const;

  virtual bool quench(int outer_iteration) const;

 private:
  int n_iterations_;
  float max_temp_;
  float min_temp_;
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct FinalOp {
  static auto f() -> void;
};

class PickRotamersStep {
 public:
  virtual ~PickRotamersStep() = default;
  virtual int max_n_rotamers() const = 0;
  virtual void pick_rotamers() = 0;
};

class MetropolisAcceptRejectStep {
 public:
  virtual ~MetropolisAcceptRejectStep() = default;
  virtual void set_temperature_scheduler(
      std::shared_ptr<TemperatureScheduler> temp_sched) = 0;
  virtual void accept_reject(int outer_iteration) = 0;
  virtual void final_op() = 0;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct PickRotamersStepRegistrator {
  static void f(
      TView<Real, 3, D> context_coords,
      TView<Int, 2, D> context_coord_offsets,
      TView<Int, 2, D> context_block_type,
      TView<Int, 1, D> pose_id_for_context,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Real, 2, D> rotamer_coords,
      TView<Int, 1, D> rotamer_coord_offsets,
      TView<Real, 2, D> alternate_coords,
      TView<Int, 1, D> alternate_coord_offsets,
      TView<Int, 2, D> alternate_id,
      TView<Int, 1, D> random_rots,
      TView<Int, 1, D> block_type_n_atoms,
      Int max_n_atoms,
      TView<int64_t, 1, tmol::Device::CPU> annealer_event,
      TView<int64_t, 1, tmol::Device::CPU> annealer);
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetropolisAcceptRejectStepRegistrator {
  static void f(
      TView<Real, 1, tmol::Device::CPU> temperature,
      TView<Real, 3, D> context_coords,
      TView<Int, 2, D> context_coord_offsets,
      TView<Int, 2, D> context_block_type,
      TView<Real, 2, D> alternate_coords,
      TView<Int, 1, D> alternate_coord_offsets,
      TView<Int, 2, D> alternate_id,
      TView<Real, 2, D> rotamer_component_energies,
      TView<Int, 1, D> accept,
      TView<Int, 1, D> block_type_n_atoms,
      Int max_n_atoms,
      TView<int64_t, 1, tmol::Device::CPU> score_events,
      TView<int64_t, 1, tmol::Device::CPU> annealer);
};

// class PickRotamersStep {
//  public:
//   PickRotamersStep(
//       torch::Tensor context_coords,
//       torch::Tensor context_block_type,
//       torch::Tensor pose_id_for_context,
//       torch::Tensor n_rots_for_pose,
//       torch::Tensor rot_offset_for_pose,
//       torch::Tensor block_type_ind_for_rot,
//       torch::Tensor block_ind_for_rot,
//       torch::Tensor rotamer_coords,
//       torch::Tensor random_rots,
//       torch::Tensor alternate_coords,
//       torch::Tensor alternate_id,
//       torch::Tensor annealer_events
//   );
//   virtual ~PickRotamersStep();
//   virtual void pick_rotamers();
//
//  private:
//   torch::Tensor context_coords_;
//   torch::Tensor context_block_type_;
//   torch::Tensor pose_id_for_context_;
//   torch::Tensor n_rots_for_pose_;
//   torch::Tensor rot_offset_for_pose_;
//   torch::Tensor block_type_ind_for_rot_;
//   torch::Tensor block_ind_for_rot_;
//   torch::Tensor rotamer_coords_;
//   torch::Tensor random_rots_;
//   torch::Tensor alternate_coords_;
//   torch::Tensor alternate_id_;
//   torch::Tensor annealer_events_;
// };
//
// class MetropolisAcceptRejectStep {
//  public:
//   MetropolisAcceptRejectStep(
//       torch::Tensor temperature,
//       torch::Tensor context_coords,
//       torch::Tensor context_block_type,
//       torch::Tensor alternate_coords,
//       torch::Tensor alternate_id,
//       torch::Tensor rotamer_component_energies,
//       torch::Tensor accept,
//       torch::Tensor score_events
//   );
//
//   virtual ~MetropolisAcceptRejectStep();
//   virtual void accept_reject();
//   virtual void final_op();
//
//  private:
//   torch::Tensor temperature_;
//   torch::Tensor context_coords_;
//   torch::Tensor context_block_type_;
//   torch::Tensor alternate_coords_;
//   torch::Tensor alternate_id_;
//   torch::Tensor rotamer_component_energies_;
//   torch::Tensor accept_;
//   torch::Tensor score_events_;
// };

class RPECalc {
 public:
  virtual ~RPECalc() {}
  virtual void calc_energies() = 0;
  virtual void finalize() = 0;
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
