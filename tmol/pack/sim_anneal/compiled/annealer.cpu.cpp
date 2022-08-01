#include <tmol/pack/sim_anneal/compiled/annealer.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/utility/tensor/TensorCast.h>

#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include <chrono>

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

using tmol::score::common::ForallDispatch;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
class CPUPickRotamersStep : public PickRotamersStep {
 public:
  CPUPickRotamersStep(
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
      TView<int64_t, 1, tmol::Device::CPU> annealer_event)
      : context_coords_(context_coords),
        context_coord_offsets_(context_coord_offsets),
        context_block_type_(context_block_type),
        pose_id_for_context_(pose_id_for_context),
        n_rots_for_pose_(n_rots_for_pose),
        rot_offset_for_pose_(rot_offset_for_pose),
        block_type_ind_for_rot_(block_type_ind_for_rot),
        block_ind_for_rot_(block_ind_for_rot),
        rotamer_coords_(rotamer_coords),
        rotamer_coord_offsets_(rotamer_coord_offsets),
        alternate_coords_(alternate_coords),
        alternate_coord_offsets_(alternate_coord_offsets),
        alternate_id_(alternate_id),
        random_rots_(random_rots),
        block_type_n_atoms_(block_type_n_atoms),
        max_n_atoms_(max_n_atoms),
        annealer_event_(annealer_event) {
    std::cout << "CPUPickRotamersStep" << std::endl;
  }

  int max_n_rotamers() const override {
    int const n_poses = n_rots_for_pose_.size(0);
    int max_n_rots = -1;
    for (int i = 0; i < n_poses; ++i) {
      int i_nrots = n_rots_for_pose_[i];
      if (i_nrots < max_n_rots) {
        max_n_rots = i_nrots;
      }
    }
    return max_n_rots;
  }

  void pick_rotamers() override {
    PickRotamers<DeviceDispatch, D, Real, Int>::f(
        context_coords_,
        context_coord_offsets_,
        context_block_type_,
        pose_id_for_context_,
        n_rots_for_pose_,
        rot_offset_for_pose_,
        block_type_ind_for_rot_,
        block_ind_for_rot_,
        rotamer_coords_,
        rotamer_coord_offsets_,
        alternate_coords_,
        alternate_coord_offsets_,
        alternate_id_,
        random_rots_,
        block_type_n_atoms_,
        max_n_atoms_,
        annealer_event_);
  }

 private:
  TView<Real, 3, D> context_coords_;
  TView<Int, 2, D> context_coord_offsets_;
  TView<Int, 2, D> context_block_type_;
  TView<Int, 1, D> pose_id_for_context_;
  TView<Int, 1, D> n_rots_for_pose_;
  TView<Int, 1, D> rot_offset_for_pose_;
  TView<Int, 1, D> block_type_ind_for_rot_;
  TView<Int, 1, D> block_ind_for_rot_;
  TView<Real, 2, D> rotamer_coords_;
  TView<Int, 1, D> rotamer_coord_offsets_;
  TView<Real, 2, D> alternate_coords_;
  TView<Int, 1, D> alternate_coord_offsets_;
  TView<Int, 2, D> alternate_id_;
  TView<Int, 1, D> random_rots_;
  TView<Int, 1, D> block_type_n_atoms_;
  Int max_n_atoms_;
  TView<int64_t, 1, tmol::Device::CPU> annealer_event_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
class CPUMetropolisAcceptRejectStep : public MetropolisAcceptRejectStep {
 public:
  CPUMetropolisAcceptRejectStep(
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
      TView<int64_t, 1, tmol::Device::CPU> score_events)
      : temperature_(temperature),
        context_coords_(context_coords),
        context_coord_offsets_(context_coord_offsets),
        context_block_type_(context_block_type),
        alternate_coords_(alternate_coords),
        alternate_id_(alternate_id),
        rotamer_component_energies_(rotamer_component_energies),
        accept_(accept),
        block_type_n_atoms_(block_type_n_atoms),
        max_n_atoms_(max_n_atoms),
        score_events_(score_events),
        last_outer_iteration_(-1) {}

  void set_temperature_scheduler(
      std::shared_ptr<TemperatureScheduler> temp_sched) {
    temp_sched_ = temp_sched;
  }

  void accept_reject(int outer_iteration) override {
    if (outer_iteration != last_outer_iteration_) {
      Real temperature = temp_sched_->temp(outer_iteration);
      last_outer_iteration_ = outer_iteration;
      temperature_[0] = temperature;
    }

    MetropolisAcceptReject<ForallDispatch, D, Real, Int>::f(
        temperature_,
        context_coords_,
        context_coord_offsets_,
        context_block_type_,
        alternate_coords_,
        alternate_coord_offsets_,
        alternate_id_,
        rotamer_component_energies_,
        accept_,
        block_type_n_atoms_,
        max_n_atoms_,
        score_events_);
  }

  void final_op() override {}

 private:
  TView<Real, 1, tmol::Device::CPU> temperature_;
  TView<Real, 3, D> context_coords_;
  TView<Int, 2, D> context_coord_offsets_;
  TView<Int, 2, D> context_block_type_;
  TView<Real, 2, D> alternate_coords_;
  TView<Int, 1, D> alternate_coord_offsets_;
  TView<Int, 2, D> alternate_id_;
  TView<Real, 2, D> rotamer_component_energies_;
  TView<Int, 1, D> accept_;
  TView<Int, 1, D> block_type_n_atoms_;
  Int max_n_atoms_;
  TView<int64_t, 1, tmol::Device::CPU> score_events_;
  std::shared_ptr<TemperatureScheduler> temp_sched_;
  int last_outer_iteration_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void PickRotamersStepRegistrator<DeviceDispatch, D, Real, Int>::f(
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
    TView<int64_t, 1, tmol::Device::CPU> annealer) {
  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<PickRotamersStep> pick_step =
      std::make_shared<CPUPickRotamersStep<DeviceDispatch, D, Real, Int>>(
          context_coords,
          context_coord_offsets,
          context_block_type,
          pose_id_for_context,
          n_rots_for_pose,
          rot_offset_for_pose,
          block_type_ind_for_rot,
          block_ind_for_rot,
          rotamer_coords,
          rotamer_coord_offsets,
          alternate_coords,
          alternate_coord_offsets,
          alternate_id,
          random_rots,
          block_type_n_atoms,
          max_n_atoms,
          annealer_event);

  sim_annealer->set_pick_rotamers_step(pick_step);
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void MetropolisAcceptRejectStepRegistrator<DeviceDispatch, D, Real, Int>::f(
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
    TView<int64_t, 1, tmol::Device::CPU> annealer) {
  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<MetropolisAcceptRejectStep> metropolis_step =
      std::make_shared<
          CPUMetropolisAcceptRejectStep<DeviceDispatch, D, Real, Int>>(
          temperature,
          context_coords,
          context_coord_offsets,
          context_block_type,
          alternate_coords,
          alternate_coord_offsets,
          alternate_id,
          rotamer_component_energies,
          accept,
          block_type_n_atoms,
          max_n_atoms,
          score_events);

  sim_annealer->set_metropolis_accept_reject_step(metropolis_step);
}

TemperatureScheduler::TemperatureScheduler(
    int n_outer_iterations, float max_temp, float min_temp)
    : n_iterations_(n_outer_iterations),
      max_temp_(max_temp),
      min_temp_(min_temp) {}

float TemperatureScheduler::temp(int outer_iteration) const {
  if (quench(outer_iteration)) {
    return 0;
  }
  return min_temp_ + (max_temp_ - min_temp_) * std::exp(-1 * outer_iteration);
}

bool TemperatureScheduler::quench(int outer_iteration) const {
  return outer_iteration == n_iterations_ - 1;
}

SimAnnealer::SimAnnealer() {}
SimAnnealer::~SimAnnealer() {}

void SimAnnealer::set_pick_rotamers_step(
    std::shared_ptr<PickRotamersStep> pick_step) {
  pick_step_ = pick_step;
}

void SimAnnealer::set_metropolis_accept_reject_step(
    std::shared_ptr<MetropolisAcceptRejectStep> acc_rej_step) {
  acc_rej_step_ = acc_rej_step;
}

void SimAnnealer::add_score_component(
    std::shared_ptr<RPECalc> score_calculator) {
  score_calculators_.push_back(score_calculator);
}

void SimAnnealer::run_annealer() {
  int const max_n_rots = pick_step_->max_n_rotamers();

  int const n_outer_cycles = 20;
  int const n_inner_cycles = 5 * max_n_rots;

  auto temp_sched =
      std::make_shared<TemperatureScheduler>(n_outer_cycles, 100.0, 0.3);
  acc_rej_step_->set_temperature_scheduler(temp_sched);

  clock_t start_clock = clock();
  time_t start_time = time(NULL);
  using namespace std::chrono;
  auto start_chrono = high_resolution_clock::now();
  for (int i = 0; i < n_outer_cycles; ++i) {
    for (int j = 0; j < n_inner_cycles; ++j) {
      if (i * n_inner_cycles + j % 100 == 0) {
        pick_step_->pick_rotamers();
      }
      for (auto const &rpe_calc : score_calculators_) {
        rpe_calc->calc_energies();
      }
      // std::cout << "." << std::flush;
      acc_rej_step_->accept_reject(i);
    }
  }
  acc_rej_step_->final_op();

  clock_t stop_clock = clock();
  time_t stop_time = time(NULL);
  auto stop_chrono = high_resolution_clock::now();
  auto duration_us = duration_cast<microseconds>(stop_chrono - start_chrono);
  auto duration_ms = duration_cast<milliseconds>(stop_chrono - start_chrono);

  int const n_cycles = n_outer_cycles * n_inner_cycles;
  std::cout << n_cycles << " cycles of simA in ";
  std::cout << (double)duration_us.count() / n_cycles
            << " us per iteration (chrono) ";
  std::cout << (double)duration_ms.count() / 1000 << " s (chrono) "
            << std::endl;
  std::cout << ((double)stop_clock - start_clock) / (n_cycles * CLOCKS_PER_SEC)
            << " s (clock) ";
  std::cout << ((double)stop_time - start_time) / (n_cycles)
            << " s (wall time) " << std::endl;
}

std::shared_ptr<PickRotamersStep> SimAnnealer::pick_step() {
  return pick_step_;
}

std::shared_ptr<MetropolisAcceptRejectStep> SimAnnealer::acc_rej_step() {
  return acc_rej_step_;
}

std::list<std::shared_ptr<RPECalc>> const &SimAnnealer::score_calculators() {
  return score_calculators_;
}

template struct PickRotamersStepRegistrator<
    ForallDispatch,
    tmol::Device::CPU,
    float,
    int>;
template struct PickRotamersStepRegistrator<
    ForallDispatch,
    tmol::Device::CPU,
    double,
    int>;

template struct MetropolisAcceptRejectStepRegistrator<
    ForallDispatch,
    tmol::Device::CPU,
    float,
    int>;
template struct MetropolisAcceptRejectStepRegistrator<
    ForallDispatch,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace compiled
}  // namespace sim_anneal
}  // namespace pack
}  // namespace tmol
