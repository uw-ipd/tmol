#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/pack/sim_anneal/compiled/annealer.hh>
#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
#include <tmol/utility/function_dispatch/aten.hh>

#include <moderngpu/kernel_reduce.hxx>

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
class CUDAPickRotamersStep : public PickRotamersStep {
 public:
  CUDAPickRotamersStep(
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
        annealer_event_(annealer_event) {
    std::cout << "CUDAPickRotamersStep" << std::endl;
  }

  int max_n_rotamers() const override {
    int const n_poses = n_rots_for_pose_.size(0);

    using namespace mgpu;
    typedef launch_box_t<
        arch_20_cta<32, 3>,
        arch_35_cta<32, 3>,
        arch_52_cta<32, 3>>
        launch_t;
    auto max_n_rots_tp = TPack<Int, 1, D>::zeros({1});
    auto max_n_rots_tv = max_n_rots_tp.view;

    mgpu::standard_context_t context;
    mgpu::reduce<launch_t>(
        &n_rots_for_pose_[0],
        n_poses,
        &max_n_rots_tv[0],
        maximum_t<Int>(),
        context);
    Int max_n_rots;
    cudaMemcpy(
        &max_n_rots, &max_n_rots_tv[0], sizeof(Int), cudaMemcpyDeviceToHost);

    std::cout << "Max n rots: " << max_n_rots << " test" << std::endl;
    return max_n_rots;
  }

  void pick_rotamers() override {
    clear_old_events();
    create_new_event();

    PickRotamers<DeviceDispatch, D, Real, Int>::f(
        context_coords_,
        context_block_type_,
        pose_id_for_context_,
        n_rots_for_pose_,
        rot_offset_for_pose_,
        block_type_ind_for_rot_,
        block_ind_for_rot_,
        rotamer_coords_,
        alternate_coords_,
        alternate_id_,
        random_rots_,
        annealer_event_);
  }

  void clear_old_events() {
    if (previously_created_events_.size() >= 100) {
      // std::cout << "waiting on old event" << std::endl;
      cudaEvent_t event =
          reinterpret_cast<cudaEvent_t>(previously_created_events_.front());
      cudaEventSynchronize(event);
    }

    for (auto event_iter = previously_created_events_.begin();
         event_iter != previously_created_events_.end();
         /*no increment*/) {
      cudaEvent_t event = *event_iter;
      cudaError_t status = cudaEventQuery(event);
      auto event_iter_next = event_iter;
      ++event_iter_next;
      if (status == cudaSuccess) {
        // std::cout << "event " << event << " done" << std::endl;
        cudaEventDestroy(event);
        previously_created_events_.erase(event_iter);
      } else {
        // std::cout << "event " << event << " not yet ready" << std::endl;
      }

      event_iter = event_iter_next;
    }
  }

  void create_new_event() {
    cudaEvent_t event;
    cudaEventCreate(&event);
    annealer_event_[0] = reinterpret_cast<int64_t>(event);
    previously_created_events_.push_back(event);
    // std::cout << "Creating new event " << event << std::endl;
  }

 private:
  TView<Real, 4, D> context_coords_;
  TView<Int, 2, D> context_block_type_;
  TView<Int, 1, D> pose_id_for_context_;
  TView<Int, 1, D> n_rots_for_pose_;
  TView<Int, 1, D> rot_offset_for_pose_;
  TView<Int, 1, D> block_type_ind_for_rot_;
  TView<Int, 1, D> block_ind_for_rot_;
  TView<Real, 3, D> rotamer_coords_;
  TView<Real, 3, D> alternate_coords_;
  TView<Int, 2, D> alternate_id_;
  TView<Int, 1, D> random_rots_;
  TView<int64_t, 1, tmol::Device::CPU> annealer_event_;
  std::list<cudaEvent_t> previously_created_events_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
class CUDAMetropolisAcceptRejectStep : public MetropolisAcceptRejectStep {
 public:
  CUDAMetropolisAcceptRejectStep(
      TView<Real, 1, tmol::Device::CPU> temperature,
      TView<Real, 4, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Real, 3, D> alternate_coords,
      TView<Int, 2, D> alternate_id,
      TView<Real, 2, D> rotamer_component_energies,
      TView<Int, 1, D> accept,
      TView<int64_t, 1, tmol::Device::CPU> score_events)
      : temperature_(temperature),
        context_coords_(context_coords),
        context_block_type_(context_block_type),
        alternate_coords_(alternate_coords),
        alternate_id_(alternate_id),
        rotamer_component_energies_(rotamer_component_energies),
        accept_(accept),
        score_events_(score_events),
        last_outer_iteration_(-1) {}

  void set_temperature_scheduler(
      std::shared_ptr<TemperatureScheduler> temp_sched) {
    temp_sched_ = temp_sched;
  }

  void accept_reject(int outer_iteration) override {
    assert(temp_sched_);

    if (outer_iteration != last_outer_iteration_) {
      Real temperature = temp_sched_->temp(outer_iteration);
      std::cout << "Setting new temperature: " << temperature << std::endl;
      last_outer_iteration_ = outer_iteration;
      temperature_[0] = temperature;
    }

    MetropolisAcceptReject<ForallDispatch, D, Real, Int>::f(
        temperature_,
        context_coords_,
        context_block_type_,
        alternate_coords_,
        alternate_id_,
        rotamer_component_energies_,
        accept_,
        score_events_);
  }

  void final_op() override { FinalOp<ForallDispatch, D, Real, Int>::f(); }

 private:
  TView<Real, 1, tmol::Device::CPU> temperature_;
  TView<Real, 4, D> context_coords_;
  TView<Int, 2, D> context_block_type_;
  TView<Real, 3, D> alternate_coords_;
  TView<Int, 2, D> alternate_id_;
  TView<Real, 2, D> rotamer_component_energies_;
  TView<Int, 1, D> accept_;
  TView<int64_t, 1, tmol::Device::CPU> score_events_;
  // std::list<cudaEvent_t> events_;
  int last_outer_iteration_;
  std::shared_ptr<TemperatureScheduler> temp_sched_;
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void PickRotamersStepRegistrator<DeviceDispatch, D, Real, Int>::f(
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
    TView<Int, 1, D> random_rots,
    TView<int64_t, 1, tmol::Device::CPU> annealer_event,
    TView<int64_t, 1, tmol::Device::CPU> annealer) {
  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<PickRotamersStep> pick_step =
      std::make_shared<CUDAPickRotamersStep<DeviceDispatch, D, Real, Int>>(
          context_coords,
          context_block_type,
          pose_id_for_context,
          n_rots_for_pose,
          rot_offset_for_pose,
          block_type_ind_for_rot,
          block_ind_for_rot,
          rotamer_coords,
          alternate_coords,
          alternate_id,
          random_rots,
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
    TView<Real, 4, D> context_coords,
    TView<Int, 2, D> context_block_type,
    TView<Real, 3, D> alternate_coords,
    TView<Int, 2, D> alternate_id,
    TView<Real, 2, D> rotamer_component_energies,
    TView<Int, 1, D> accept,
    TView<int64_t, 1, tmol::Device::CPU> score_events,
    TView<int64_t, 1, tmol::Device::CPU> annealer) {
  int64_t annealer_uint = annealer[0];
  SimAnnealer *sim_annealer = reinterpret_cast<SimAnnealer *>(annealer_uint);
  std::shared_ptr<MetropolisAcceptRejectStep> metropolis_step =
      std::make_shared<
          CUDAMetropolisAcceptRejectStep<DeviceDispatch, D, Real, Int>>(
          temperature,
          context_coords,
          context_block_type,
          alternate_coords,
          alternate_id,
          rotamer_component_energies,
          accept,
          score_events);

  sim_annealer->set_metropolis_accept_reject_step(metropolis_step);
}

template struct PickRotamersStepRegistrator<
    ForallDispatch,
    tmol::Device::CUDA,
    float,
    int>;
template struct PickRotamersStepRegistrator<
    ForallDispatch,
    tmol::Device::CUDA,
    double,
    int>;
// template struct PickRotamersStepRegistrator<
//     ForallDispatch,
//     tmol::Device::CUDA,
//     float,
//     int>;
// template struct PickRotamersStepRegistrator<
//     ForallDispatch,
//     tmol::Device::CUDA,
//     double,
//     int>;

template struct MetropolisAcceptRejectStepRegistrator<
    ForallDispatch,
    tmol::Device::CUDA,
    float,
    int>;
template struct MetropolisAcceptRejectStepRegistrator<
    ForallDispatch,
    tmol::Device::CUDA,
    double,
    int>;
// template struct MetropolisAcceptRejectStepRegistrator<
//     ForallDispatch,
//     tmol::Device::CUDA,
//     float,
//     int>;
// template struct MetropolisAcceptRejectStepRegistrator<
//     ForallDispatch,
//     tmol::Device::CUDA,
//     double,
//     int>;

}  // namespace compiled
}  // namespace sim_anneal
}  // namespace pack
}  // namespace tmol
