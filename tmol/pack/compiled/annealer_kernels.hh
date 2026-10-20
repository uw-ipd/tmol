#pragma once

// Simulated annealing, one workgroup per trajectory.
//
// A translation of the CUDA annealer's per-warp body into the project's
// portable primitives, so one source serves CUDA, Metal and the host. The
// structure is CUDA's: one trajectory per workgroup, the lanes splitting each
// proposal's neighbour list and reducing the result, and a Metropolis
// accept/reject on the reduced delta.
//
// Three things differ from the CUDA original, each because Philox is
// counter-based where cuRAND's state is not:
//
//  * Every lane runs the same stream and therefore computes the same draws,
//    so the candidate and the acceptance roll need no broadcast. CUDA has
//    lane 0 draw and shuffle them out. This holds only while the control flow
//    is uniform across the workgroup, which it is: every lane makes the same
//    number of draws on every path.
//  * The initial random assignment keys its draw on the residue index rather
//    than drawing in sequence, so a residue's starting rotamer does not
//    depend on which lane happened to seed it.
//  * The quench visits rotamers by stepping a stride coprime with the rotamer
//    count from a random start. That covers every rotamer exactly once per
//    sweep, as CUDA's shuffled order does, without the per-trajectory
//    permutation buffer CUDA stores.
//
// There is no portable single-value broadcast among the workgroup primitives.
// Where one is needed, the value is held by lane 0 and zero elsewhere, and a
// sum-reduce-and-broadcast delivers it to every lane.

// Deliberately not device_operations.hh: it reaches the torch tensor layer.
// The DeviceDispatch template parameter is supplied by whichever front end
// instantiates this body, each of which has already included its own.
#include <tmol/pack/compiled/annealer_ctx.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/philox.hh>
#include <tmol/utility/portable/reduce_ops.hh>

namespace tmol {
namespace pack {
namespace compiled {

using tmol::portable::Philox4x32_10;
using tmol::score::common::WorkgroupLane;

// Phase selector, mirroring the three rounds of the CUDA funnel.
enum {
  kAnnealPhaseHitemp = 0,  // random start, high temperature, track best
  kAnnealPhaseLotemp = 1,  // seeded from a ranked survivor, low temperature
  kAnnealPhaseQuench = 2   // greedy descent from the incoming assignment
};

// No threadgroup struct: the only shared storage this body needs is the
// reduction scratch, which WorkgroupLane already carries. A trajectory's
// assignments stay in their global slices -- CUDA caches them in threadgroup
// memory when a pose's residues fit, which is an optimisation to add once
// this is measured, not a correctness requirement.

// Sum across the workgroup and return the total to every lane.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    int nt,
    typename Real>
TMOL_HD Real anneal_sum_bcast(Real val, WorkgroupLane<D> lane) {
  return DeviceDispatch<D>::template shuffle_reduce_and_broadcast_in_workgroup<
      nt>(val, tmol::portable::plus_t<Real>(), lane);
}

// Largest stride < n that is coprime with n, for the quench sweep. n is a
// rotamer count, so this loop runs a handful of times at most.
TMOL_HD inline int anneal_coprime_stride(int n, int start) {
  if (n <= 2) return 1;
  for (int s = start; s > 1; --s) {
    int a = s, b = n;
    while (b != 0) {
      int const t = a % b;
      a = b;
      b = t;
    }
    if (a == 1) return s;
  }
  return 1;
}

// One trajectory. `unit` indexes (pose, trajectory) as pose * n_traj + traj.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    int nt,
    typename Ctx>
TMOL_HD void annealer_run_trajectory(
    TMOL_THREAD Ctx const& ctx, int unit, WorkgroupLane<Dev> lane = {}) {
  int const pose = unit / ctx.n_traj;
  int const traj = unit - pose * ctx.n_traj;
  int const n_res = ctx.pose_n_res[pose];
  int const n_rotamers = ctx.pose_n_rotamers[pose];
  if (n_res <= 0 || n_rotamers <= 0) return;

  int const phase = ctx.phase;
  bool const track_best = phase != kAnnealPhaseQuench;

  uint64_t const seed =
      (uint64_t(uint32_t(ctx.seed_hi)) << 32) | uint64_t(uint32_t(ctx.seed_lo));
  // Distinct stream per (unit, phase); the phase occupies the low two bits so
  // a trajectory's three rounds never replay one another's draws.
  Philox4x32_10 rng(
      seed, (uint64_t(uint32_t(unit)) << 2) | uint64_t(phase & 3));

  auto current = ctx.current_assignments[pose][traj];
  auto best = ctx.best_assignments[pose][traj];

  // ── seed the assignment ────────────────────────────────────────────────
  auto seed_assignment = ([=] TMOL_DEVICE_FUNC(int tid) {
    if (phase == kAnnealPhaseHitemp) {
      for (int i = tid; i < n_res; i += nt) {
        int const i_n_rots = ctx.n_rotamers_for_res[pose][i];
        if (i_n_rots <= 0) {
          current[i] = -1;
          best[i] = -1;
          continue;
        }
        // Keyed on the residue so the start does not depend on the lane.
        Philox4x32_10 pick(seed, uint64_t(uint32_t(unit)), uint64_t(i) + 1);
        int const rot = pick.next_below(i_n_rots);
        current[i] = int32_t(rot);
        best[i] = int32_t(rot);
      }
    } else {
      // Continue from a ranked survivor's best assignment.
      int const source = ctx.seed_ranking[pose][traj];
      auto source_best = ctx.seed_assignments[pose][source];
      for (int i = tid; i < n_res; i += nt) {
        current[i] = source_best[i];
        best[i] = source_best[i];
      }
    }
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      seed_assignment, lane);
  DeviceDispatch<Dev>::synchronize_workgroup(lane);

  // ── initial exact energy ───────────────────────────────────────────────
  Real partial = 0;
  TMOL_THREAD_PTR(Real) partial_p = &partial;
  auto accumulate_initial = ([=] TMOL_DEVICE_FUNC(int tid) {
    *partial_p += ctx.partial_energy(pose, current, tid, nt);
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      accumulate_initial, lane);
  Real current_energy =
      anneal_sum_bcast<DeviceDispatch, Dev, nt>(partial, lane);
  Real best_energy = current_energy;

  float temperature = ctx.high_temp;
  if (phase == kAnnealPhaseQuench) temperature = 0.0f;

  // ── annealing ──────────────────────────────────────────────────────────
  for (int outer = 0; outer < ctx.n_outer_iterations; ++outer) {
    bool const quench =
        phase == kAnnealPhaseQuench || outer == ctx.n_outer_iterations - 1;
    int inner_iterations = quench ? n_rotamers : ctx.n_inner_iterations;
    int quench_start = 0;
    int quench_stride = 1;

    if (quench) {
      temperature = 0.0f;
      // Quench from the best assignment seen, not wherever the walk ended.
      auto restore_best = ([=] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_res; i += nt) {
          current[i] = best[i];
        }
      });
      DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
          restore_best, lane);
      DeviceDispatch<Dev>::synchronize_workgroup(lane);
      current_energy = best_energy;
      quench_start = rng.next_below(n_rotamers);
      quench_stride = anneal_coprime_stride(n_rotamers, n_rotamers - 1);
    }

    for (int step = 0; step < inner_iterations; ++step) {
      // Every lane draws the same values, so there is nothing to broadcast.
      int global_candidate;
      float accept_roll;
      if (quench) {
        global_candidate = (quench_start + step * quench_stride) % n_rotamers;
        accept_roll = 0.0f;
      } else {
        global_candidate = rng.next_below(n_rotamers);
        accept_roll = rng.next_uniform();
      }
      global_candidate += ctx.pose_rotamer_offset[pose];

      int const res = ctx.res_for_rot[global_candidate];
      int const res_offset = ctx.oneb_offsets[pose][res];
      int const new_rot = global_candidate - res_offset;
      int const prev_rot = current[res];
      if (prev_rot < 0 || new_rot == prev_rot) continue;

      Real dpartial = 0;
      TMOL_THREAD_PTR(Real) dpartial_p = &dpartial;
      auto accumulate_delta = ([=] TMOL_DEVICE_FUNC(int tid) {
        *dpartial_p += ctx.partial_delta_energy(
            pose, current, res, prev_rot, new_rot, tid, nt);
      });
      DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
          accumulate_delta, lane);
      Real const delta =
          anneal_sum_bcast<DeviceDispatch, Dev, nt>(dpartial, lane);

      bool accept;
      if (quench) {
        accept = delta < Real(0);
      } else {
        accept = delta < Real(0)
                 || accept_roll < tmol::portable::math::exp(
                        float(-delta) / temperature);
      }
      if (!accept) continue;

      // Every lane writes the same value, so no lane needs to be singled out;
      // the barrier is what the threadgroup cache requires, not the write.
      DeviceDispatch<Dev>::synchronize_workgroup(lane);
      current[res] = int32_t(new_rot);
      DeviceDispatch<Dev>::synchronize_workgroup(lane);

      current_energy += delta;
      if (track_best && current_energy < best_energy) {
        best_energy = current_energy;
        auto save_best = ([=] TMOL_DEVICE_FUNC(int tid) {
          for (int i = tid; i < n_res; i += nt) {
            best[i] = current[i];
          }
        });
        DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
            save_best, lane);
        DeviceDispatch<Dev>::synchronize_workgroup(lane);
      }
    }

    // Geometric cooling toward the low temperature, as CUDA does.
    temperature =
        ctx.cooling_factor * (temperature - ctx.low_temp) + ctx.low_temp;

    if (phase == kAnnealPhaseQuench) break;  // a quench is a single sweep
  }

  // ── exact rescore of the assignment that will be ranked ────────────────
  auto final_assignment = track_best ? best : current;
  Real final_partial = 0;
  TMOL_THREAD_PTR(Real) final_partial_p = &final_partial;
  auto accumulate_final = ([=] TMOL_DEVICE_FUNC(int tid) {
    *final_partial_p += ctx.partial_energy(pose, final_assignment, tid, nt);
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(
      accumulate_final, lane);
  Real const final_energy =
      anneal_sum_bcast<DeviceDispatch, Dev, nt>(final_partial, lane);

  auto write_out = ([=] TMOL_DEVICE_FUNC(int tid) {
    if (tid != 0) return;
    ctx.scores[pose][traj] = float(final_energy);
  });
  DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(write_out, lane);

  // The ranked assignment is what downstream consumes; make sure it is the
  // one in `best` for every phase.
  if (!track_best) {
    auto publish = ([=] TMOL_DEVICE_FUNC(int tid) {
      for (int i = tid; i < n_res; i += nt) {
        best[i] = current[i];
      }
    });
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(publish, lane);
    DeviceDispatch<Dev>::synchronize_workgroup(lane);
  }
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
