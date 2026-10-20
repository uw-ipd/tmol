#pragma once

// The state the ported simulated-annealing kernels read.
//
// The CUDA annealer reaches its energy tables through an InteractionGraph
// struct of TViews and takes a cooperative_groups tile for its warp-wide
// reductions. Neither can appear in a Metal translation unit: TView lives in
// the torch tensor layer, and Metal has no cooperative groups. This context
// holds the same fields behind the project's View template, and the energy
// accessors below take a WorkgroupLane instead of a tile, so one body serves
// every backend.
//
// Templated on the view so the field list is declared once: host and CUDA
// instantiate with TView, a Metal translation unit with
// tmol::portable::MetalView.

// Deliberately not device_operations.hh: it reaches the torch tensor layer,
// which a Metal translation unit cannot include. The context is pure data
// plus accessors; the workgroup primitives live in the kernel body.
#include <tmol/score/common/mps_layout.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace pack {
namespace compiled {

// The interaction graph plus the per-trajectory scratch the annealer writes.
//
// `assignment` is reached through a raw pointer rather than a view: a
// trajectory's assignments live either in a threadgroup cache or in a slice of
// a global buffer, and the body should not care which.
template <template <typename, int> class View, typename Real, typename Int>
struct AnnealerCtxT {
  // ── interaction graph ──────────────────────────────────────────────────
  View<Int, 1> pose_n_res;
  View<Int, 1> pose_n_rotamers;
  View<Int, 1> pose_rotamer_offset;
  View<Int, 2> n_rotamers_for_res;
  View<Int, 2> oneb_offsets;
  View<Int, 1> res_for_rot;
  View<int64_t, 1> neighbor_row_offsets;
  View<int32_t, 1> neighbor_blocks;
  View<int64_t, 1> neighbor_chunk_offset_offsets;
  // The transposed companion: a move looks its neighbours up as
  // [neighbour][moved residue], which is the other triangle of the table.
  View<int64_t, 1> reverse_neighbor_chunk_offset_offsets;
  View<int64_t, 1> chunk_offsets;
  View<Real, 1> energy1b;
  View<Real, 1> energy2b;

  // ── per-trajectory state, [n_poses, n_traj, ...] ───────────────────────
  View<float, 2> scores;
  View<int32_t, 3> current_assignments;
  View<int32_t, 3> best_assignments;
  // The previous round's results, read-only. Separate from best_assignments
  // because a later phase seeds trajectory t from survivor s: with one array
  // a trajectory could read a slot another is concurrently overwriting.
  View<int32_t, 3> seed_assignments;

  // Launch shape and schedule, carried in the metadata buffer.
  int chunk_size;
  int max_n_res;
  int n_traj;
  int n_outer_iterations;
  int n_inner_iterations;
  int seed_lo;
  int seed_hi;
  float high_temp;
  float low_temp;
  float cooling_factor;
  // Phase selector: 0 = high temperature from a random start, 1 = low
  // temperature seeded from a ranked survivor, 2 = greedy quench.
  int phase;
  // For phase 1 and 2, the ranking that selects which trajectory each unit
  // continues from. Bound to `scores` when unused so no view is undefined.
  View<int32_t, 2> seed_ranking;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(pose_n_res);
    f(pose_n_rotamers);
    f(pose_rotamer_offset);
    f(n_rotamers_for_res);
    f(oneb_offsets);
    f(res_for_rot);
    f(neighbor_row_offsets);
    f(neighbor_blocks);
    f(neighbor_chunk_offset_offsets);
    f(reverse_neighbor_chunk_offset_offsets);
    f(chunk_offsets);
    f(energy1b);
    f(energy2b);
    f(scores);
    f(current_assignments);
    f(best_assignments);
    f(seed_assignments);
    f(seed_ranking);
  }

  enum { n_views = 18 };

  // Named slots for the launch scalars carried in the metadata buffer, so the
  // launcher and the Metal entry never repeat a bare index.
  enum {
    scalar_chunk_size = 0,
    scalar_max_n_res,
    scalar_n_traj,
    scalar_n_outer_iterations,
    scalar_n_inner_iterations,
    scalar_seed_lo,
    scalar_seed_hi,
    scalar_phase,
    scalar_high_temp,
    scalar_low_temp,
    scalar_cooling_factor,
    n_scalars
  };

  // ── portable interaction-graph accessors ───────────────────────────────

  // The two-body energy for (ires, irot) against (jres, jrot), or 0 when the
  // chunk pair carries no support. Factored out because the total energy and
  // the per-move delta both need exactly this lookup.
  TMOL_HD Real two_body(
      int pose, int ires, int irot_local, int64_t edge, int j, int jrot_local)
      const {
    int const irot_chunk = irot_local / chunk_size;
    int const irot_in_chunk = irot_local - chunk_size * irot_chunk;
    int const jrot_chunk = jrot_local / chunk_size;
    int const jrot_in_chunk = jrot_local - chunk_size * jrot_chunk;

    int const jres_n_rots = n_rotamers_for_res[pose][j];
    int const jres_n_chunks = (jres_n_rots - 1) / chunk_size + 1;
    int const jrot_chunk_size =
        chunk_size < jres_n_rots - chunk_size * jrot_chunk
            ? chunk_size
            : jres_n_rots - chunk_size * jrot_chunk;

    int64_t const ij_chunk_offset_offset = neighbor_chunk_offset_offsets[edge];
    int64_t const ij_chunk_offset = chunk_offsets
        [ij_chunk_offset_offset + irot_chunk * jres_n_chunks + jrot_chunk];
    if (ij_chunk_offset == -1) {
      return Real(0);
    }
    return energy2b
        [ij_chunk_offset + jrot_chunk_size * irot_in_chunk + jrot_in_chunk];
  }

  // This lane's share of the total energy of an assignment.
  //
  // Deliberately not the whole reduction: WorkgroupLane carries no lane index
  // on the host and CUDA backends -- lanes arrive as the argument of
  // for_each_in_workgroup -- so the body owns the striding and the reduction
  // and the context stays a plain data accessor. See annealer_kernels.hh.
  // Assignment is a template parameter rather than a pointer type: a
  // trajectory's assignments live either in a threadgroup cache or in a slice
  // of a global view, both of which index with operator[], and Metal folds
  // the address space into a pointer type so one spelling cannot serve both.
  template <typename Assignment>
  TMOL_HD Real partial_energy(
      int pose,
      TMOL_THREAD Assignment const& assignment,
      int lane,
      int n_lanes) const {
    Real totalE = 0;
    int const n_res = pose_n_res[pose];

    for (int i = lane; i < n_res; i += n_lanes) {
      int const irot_local = assignment[i];
      if (irot_local < 0) continue;
      totalE += energy1b[irot_local + oneb_offsets[pose][i]];

      int const row = pose * max_n_res + i;
      for (int64_t edge = neighbor_row_offsets[row];
           edge < neighbor_row_offsets[row + 1];
           ++edge) {
        int const j = neighbor_blocks[edge];
        if (j <= i) continue;  // count each pair once
        int const jrot_local = assignment[j];
        if (jrot_local < 0) continue;
        totalE += two_body(pose, i, irot_local, edge, j, jrot_local);
      }
    }
    return totalE;
  }

  // This lane's share of the energy change from moving `res` to `new_rot`.
  //
  // A faithful translation of the CUDA annealer's delta loop, including its
  // two subtleties: the lookup goes through the *reverse* chunk-offset table,
  // because a move indexes its neighbours as [neighbour][moved residue]; and
  // when the proposed and current rotamers share a chunk the two table
  // entries share one sparse offset, so the irregular global read is done
  // once rather than per neighbour.
  template <typename Assignment>
  TMOL_HD Real partial_delta_energy(
      int pose,
      TMOL_THREAD Assignment const& assignment,
      int res,
      int prev_rot,
      int new_rot,
      int lane,
      int n_lanes) const {
    int const res_n_rots = n_rotamers_for_res[pose][res];
    int const res_n_chunks = (res_n_rots - 1) / chunk_size + 1;

    int const new_chunk = new_rot / chunk_size;
    int const new_in_chunk = new_rot - chunk_size * new_chunk;
    int const new_chunk_size = chunk_size < res_n_rots - chunk_size * new_chunk
                                   ? chunk_size
                                   : res_n_rots - chunk_size * new_chunk;

    int const prev_chunk = prev_rot / chunk_size;
    int const prev_in_chunk = prev_rot - chunk_size * prev_chunk;
    int const prev_chunk_size =
        chunk_size < res_n_rots - chunk_size * prev_chunk
            ? chunk_size
            : res_n_rots - chunk_size * prev_chunk;

    int const base = oneb_offsets[pose][res];
    Real dE = (lane == 0) ? energy1b[new_rot + base] - energy1b[prev_rot + base]
                          : Real(0);

    int const row = pose * max_n_res + res;
    int64_t const row_begin = neighbor_row_offsets[row];
    int64_t const row_end = neighbor_row_offsets[row + 1];

    for (int64_t edge = row_begin + lane; edge < row_end; edge += n_lanes) {
      int const k = neighbor_blocks[edge];
      int const k_rot = assignment[k];
      if (k_rot < 0) continue;
      int const k_chunk = k_rot / chunk_size;
      int const k_in_chunk = k_rot - k_chunk * chunk_size;

      int64_t const k_offset_offset =
          reverse_neighbor_chunk_offset_offsets[edge];

      int64_t const k_new_chunk_offset =
          chunk_offsets[k_offset_offset + k_chunk * res_n_chunks + new_chunk];
      Real new_contrib = Real(0);
      if (k_new_chunk_offset >= 0) {
        new_contrib = energy2b
            [k_new_chunk_offset + new_chunk_size * k_in_chunk + new_in_chunk];
      }

      int64_t const k_prev_chunk_offset =
          prev_chunk == new_chunk
              ? k_new_chunk_offset
              : chunk_offsets
                    [k_offset_offset + k_chunk * res_n_chunks + prev_chunk];
      Real prev_contrib = Real(0);
      if (k_prev_chunk_offset >= 0) {
        prev_contrib = energy2b
            [k_prev_chunk_offset + prev_chunk_size * k_in_chunk
             + prev_in_chunk];
      }

      dE += new_contrib - prev_contrib;
    }
    return dE;
  }
};

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
