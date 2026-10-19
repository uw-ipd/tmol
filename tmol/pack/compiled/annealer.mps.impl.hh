#pragma once

// Simulated annealing on Metal: the host half.
//
// The kernel body lives in annealer_kernels.hh and is shared with the other
// backends; this file does what only the host can -- build the graph's
// transposed offset table, and drive the three rounds of the funnel, ranking
// between them.
//
// The funnel is CUDA's: many short high-temperature trajectories, the best
// quarter of which each seed several low-temperature trajectories, the best
// quarter of those fully quenched. Ranking happens here rather than on the
// device because it is three sorts of a few thousand floats per packing call,
// not per evaluation; mgpu::segmented_sort has no Metal equivalent and does
// not need one.

#include <tmol/pack/compiled/annealer.hh>
#include <tmol/pack/compiled/annealer_ctx_tensor.hh>
#include <tmol/pack/compiled/annealer_kernels.hh>
#include <tmol/pack/compiled/annealer_mps_launch.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/native.hh>

#include <algorithm>
#include <numeric>
#include <vector>

namespace tmol {
namespace pack {
namespace compiled {

// Funnel shape, matching the CUDA annealer.
// The funnel's shape is CUDA's; its width and length are not, and both were
// measured rather than inherited. Width 128 is one trajectory per threadgroup,
// two threadgroups per core on a 16-core M2 Pro: 256 is no better and 500 --
// CUDA's width -- only costs more. Length was then cut until energies stopped
// holding. At CUDA's 10 outer and 1.5/0.5 inner multipliers the annealer is
// 2.7x slower than the single-threaded CPU annealer it replaces for a tenth
// of an energy unit per pose; at 5 and 0.4/0.15 it is faster than the CPU on
// every size measured and still far better: over 100 ubiquitins, -2182.6
// against the CPU's -2171.4, and on a 753-residue pose 161.1 (worst of five
// 169.9) against the CPU's 168.3 to 326.0. Halving it again saves another
// 170 ms but grows a tail -- 198.8 on that pose -- which is the thing a
// packer cannot afford.
struct MpsAnnealSchedule {
  int hitemp_traj = 128;
  int lotemp_expansions = 10;
  float round1_cut = 0.25f;
  float round2_cut = 0.25f;
  int outer_hitemp = 5;
  int outer_lotemp = 5;
  float inner_mult_hitemp = 0.4f;
  float inner_mult_lotemp = 0.15f;
  float high_temp_initial = 30.0f;
  float low_temp_initial = 0.3f;
  float high_temp_later = 0.2f;
  float low_temp_later = 0.1f;
  float cooling_factor = 0.35f;
};

// Rank each pose's trajectories by score, best first, and return the ranking
// as [n_poses, n_keep] source indices for the next round.
inline TPack<int32_t, 2, tmol::Device::CPU> rank_trajectories(
    TPack<float, 2, tmol::Device::CPU> const& scores_cpu,
    int n_valid,
    int n_keep,
    int expansions) {
  auto scores = scores_cpu.view;
  int const n_poses = scores.size(0);
  // The score buffer is as wide as the widest round; only the trajectories
  // the round just run actually wrote to it.
  int const n_traj = n_valid;
  int const n_out = n_keep * expansions;
  auto ranking_t =
      TPack<int32_t, 2, tmol::Device::CPU>::zeros({n_poses, n_out});
  auto ranking = ranking_t.view;

  std::vector<int> order(n_traj);
  for (int pose = 0; pose < n_poses; ++pose) {
    std::iota(order.begin(), order.end(), 0);
    // Stable, so equal scores keep trajectory order and the ranking is
    // reproducible run to run.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return scores[pose][a] < scores[pose][b];
    });
    for (int k = 0; k < n_out; ++k) {
      // Survivor k / expansions seeds `expansions` trajectories each.
      ranking[pose][k] = int32_t(order[(k / expansions) % n_traj]);
    }
  }
  return ranking_t;
}


// The graph is fixed for the whole run, so resolve the transposed chunk-table
// offsets once per edge rather than binary-searching a neighbour row for
// every proposal in every trajectory. CUDA does this in a kernel; the
// neighbour arrays are small beside the energy tables, so doing it on the
// host costs one round trip per packing call and keeps the device side to a
// single entry point.
inline TPack<int64_t, 1, tmol::Device::CPU> build_reverse_chunk_offsets(
    TView<int64_t, 1, tmol::Device::CPU> neighbor_row_offsets,
    TView<int32_t, 1, tmol::Device::CPU> neighbor_blocks,
    TView<int64_t, 1, tmol::Device::CPU> neighbor_chunk_offset_offsets,
    int n_poses,
    int max_n_res) {
  int64_t const n_edges = neighbor_blocks.size(0);
  auto reverse_t = TPack<int64_t, 1, tmol::Device::CPU>::zeros({n_edges});
  auto reverse = reverse_t.view;

  auto find_edge = [&](int pose, int block, int neighbor) -> int64_t {
    int const row = pose * max_n_res + block;
    int64_t lower = neighbor_row_offsets[row];
    int64_t upper = neighbor_row_offsets[row + 1];
    while (lower < upper) {
      int64_t const middle = lower + (upper - lower) / 2;
      if (neighbor_blocks[middle] < neighbor) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    return lower;
  };

  for (int pose = 0; pose < n_poses; ++pose) {
    for (int block = 0; block < max_n_res; ++block) {
      int const row = pose * max_n_res + block;
      for (int64_t edge = neighbor_row_offsets[row];
           edge < neighbor_row_offsets[row + 1];
           ++edge) {
        int64_t const back = find_edge(pose, neighbor_blocks[edge], block);
        reverse[edge] = neighbor_chunk_offset_offsets[back];
      }
    }
  }
  return reverse_t;
}

// Drive the funnel.
//
// Unlike the CPU and CUDA backends this takes tensors rather than views: the
// host has to read the scores between rounds to rank them, and an MPS
// tensor's data pointer is an opaque buffer handle, not host-addressable
// memory. Only a tensor can cross the device boundary, so the Metal annealer
// is reached through its own entry point instead of AnnealerDispatch.
//
// Returns [n_poses, 1] scores and [n_poses, 1, max_n_res] assignments -- the
// same shape the CPU backend returns, which is what
// impose_top_rotamer_assignments consumes.
inline std::tuple<tmol::native::Tensor, tmol::native::Tensor> anneal_mps(
    int64_t max_n_rotamers_per_pose,
    tmol::native::Tensor pose_n_res,
    tmol::native::Tensor pose_n_rotamers,
    tmol::native::Tensor pose_rotamer_offset,
    tmol::native::Tensor n_rotamers_for_res,
    tmol::native::Tensor oneb_offsets,
    tmol::native::Tensor res_for_rot,
    int64_t chunk_size,
    tmol::native::Tensor neighbor_row_offsets,
    tmol::native::Tensor neighbor_blocks,
    tmol::native::Tensor neighbor_chunk_offset_offsets,
    tmol::native::Tensor chunk_offsets,
    tmol::native::Tensor energy1b,
    tmol::native::Tensor energy2b) {
  namespace nat = tmol::native;
  constexpr auto Dev = tmol::Device::MPS;
  constexpr auto Cpu = tmol::Device::CPU;
  auto const cpu_device = nat::Device(nat::kCPU);
  auto const device = energy1b.device();

  int const n_poses = int(pose_n_res.size(0));
  int const max_n_res = int(n_rotamers_for_res.size(1));
  int const max_n_rotamers = int(max_n_rotamers_per_pose);

  MpsAnnealSchedule const sched;
  int const n_hitemp = sched.hitemp_traj;
  int const n_lotemp =
      int(n_hitemp * sched.round1_cut) * sched.lotemp_expansions;
  int const n_quench = int(n_lotemp * sched.round2_cut);
  int const n_traj_max = std::max(std::max(n_hitemp, n_lotemp), n_quench);

  int const inner_hitemp = int(max_n_rotamers * sched.inner_mult_hitemp);
  int const inner_lotemp = int(max_n_rotamers * sched.inner_mult_lotemp);

  // The transposed chunk-offset table, built once on the host from the
  // neighbour arrays. They are small beside the energy tables.
  TPack<int64_t, 1, Cpu> const row_offsets_cpu(neighbor_row_offsets.to(cpu_device));
  TPack<int32_t, 1, Cpu> const blocks_cpu(neighbor_blocks.to(cpu_device));
  TPack<int64_t, 1, Cpu> const chunk_offset_offsets_cpu(
      neighbor_chunk_offset_offsets.to(cpu_device));
  auto reverse_cpu = build_reverse_chunk_offsets(
      row_offsets_cpu.view,
      blocks_cpu.view,
      chunk_offset_offsets_cpu.view,
      n_poses,
      max_n_res);
  TPack<int64_t, 1, Dev> const reverse_dev(reverse_cpu.tensor.to(device));

  // Per-trajectory state, allocated once at the widest round. A round narrower
  // than that launches fewer units and leaves the tail untouched; the views
  // carry their own strides, so indexing does not depend on the width in use.
  auto scores_t = TPack<float, 2, Dev>::zeros({n_poses, n_traj_max});
  auto current_t = TPack<int32_t, 3, Dev>::zeros({n_poses, n_traj_max, max_n_res});
  auto best_t = TPack<int32_t, 3, Dev>::zeros({n_poses, n_traj_max, max_n_res});
  auto seed_t = TPack<int32_t, 3, Dev>::zeros({n_poses, n_traj_max, max_n_res});
  auto ranking_t = TPack<int32_t, 2, Dev>::zeros({n_poses, n_traj_max});
  auto round_ranking = ranking_t;

  // One seed for the whole call; the kernel separates phases and trajectories
  // into their own Philox streams.
  auto seed_tensor = nat::random_seed_tensor({}, cpu_device);
  uint64_t const seed = static_cast<uint64_t>(seed_tensor.item<int64_t>());

  AnnealerCtx<float, int32_t, Dev> ctx{
      TPack<int32_t, 1, Dev>(pose_n_res).view,
      TPack<int32_t, 1, Dev>(pose_n_rotamers).view,
      TPack<int32_t, 1, Dev>(pose_rotamer_offset).view,
      TPack<int32_t, 2, Dev>(n_rotamers_for_res).view,
      TPack<int32_t, 2, Dev>(oneb_offsets).view,
      TPack<int32_t, 1, Dev>(res_for_rot).view,
      TPack<int64_t, 1, Dev>(neighbor_row_offsets).view,
      TPack<int32_t, 1, Dev>(neighbor_blocks).view,
      TPack<int64_t, 1, Dev>(neighbor_chunk_offset_offsets).view,
      reverse_dev.view,
      TPack<int64_t, 1, Dev>(chunk_offsets).view,
      TPack<float, 1, Dev>(energy1b).view,
      TPack<float, 1, Dev>(energy2b).view,
      scores_t.view,
      current_t.view,
      best_t.view,
      seed_t.view,
      int(chunk_size),
      max_n_res,
      n_hitemp,
      sched.outer_hitemp,
      inner_hitemp,
      int32_t(seed & 0xffffffffu),
      int32_t(seed >> 32),
      sched.high_temp_initial,
      sched.low_temp_initial,
      sched.cooling_factor,
      kAnnealPhaseHitemp,
      ranking_t.view};

  // Round 1: many short trajectories from random starts.
  annealer_mps_launch_if<Dev>(ctx, int64_t(n_poses) * n_hitemp);

  // Rank on the host. Three reads of a few thousand floats per packing call,
  // against one launch per trajectory, so the syncs do not show up.
  auto rank_and_seed = [&](int n_from, int n_keep, int expansions) {
    TPack<float, 2, Cpu> const scores_cpu(scores_t.tensor.to(cpu_device));
    auto ranking = rank_trajectories(scores_cpu, n_from, n_keep, expansions);
    round_ranking = TPack<int32_t, 2, Dev>(ranking.tensor.to(device));
    ctx.seed_ranking = round_ranking.view;
    // The survivors' best assignments become the next round's read-only seed
    // source; writing into `best_t` while trajectories read it would alias.
    auto destination = seed_t.tensor.handle();
    auto origin = best_t.tensor.handle();
    tmol::native::stable::copy_(destination, origin);
  };

  // Round 2: the best quarter each seed `lotemp_expansions` trajectories at
  // low temperature.
  rank_and_seed(n_hitemp, int(n_hitemp * sched.round1_cut), sched.lotemp_expansions);
  ctx.phase = kAnnealPhaseLotemp;
  ctx.n_traj = n_lotemp;
  ctx.n_outer_iterations = sched.outer_lotemp;
  ctx.n_inner_iterations = inner_lotemp;
  ctx.high_temp = sched.high_temp_later;
  ctx.low_temp = sched.low_temp_later;
  annealer_mps_launch_if<Dev>(ctx, int64_t(n_poses) * n_lotemp);

  // Round 3: the best quarter of those, quenched.
  rank_and_seed(n_lotemp, n_quench, 1);
  ctx.phase = kAnnealPhaseQuench;
  ctx.n_traj = n_quench;
  ctx.n_outer_iterations = 1;
  ctx.n_inner_iterations = max_n_rotamers;
  annealer_mps_launch_if<Dev>(ctx, int64_t(n_poses) * n_quench);

  // Keep the best quenched trajectory per pose, in the shape the CPU backend
  // returns.
  TPack<float, 2, Cpu> const final_scores_cpu(scores_t.tensor.to(cpu_device));
  TPack<int32_t, 3, Cpu> const final_best_cpu(best_t.tensor.to(cpu_device));
  auto out_scores = TPack<float, 2, Cpu>::zeros({n_poses, 1});
  auto out_assignments = TPack<int32_t, 3, Cpu>::zeros({n_poses, 1, max_n_res});
  for (int pose = 0; pose < n_poses; ++pose) {
    int argbest = 0;
    for (int traj = 1; traj < n_quench; ++traj) {
      if (final_scores_cpu.view[pose][traj]
          < final_scores_cpu.view[pose][argbest]) {
        argbest = traj;
      }
    }
    out_scores.view[pose][0] = final_scores_cpu.view[pose][argbest];
    for (int i = 0; i < max_n_res; ++i) {
      out_assignments.view[pose][0][i] =
          final_best_cpu.view[pose][argbest][i];
    }
  }

  return {out_scores.tensor.to(device), out_assignments.tensor.to(device)};
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
