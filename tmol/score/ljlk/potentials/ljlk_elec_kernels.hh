#pragma once

// Fused LJ/LK + elec rotamer-pair kernel body, for the Metal backend.
//
// The CPU and CUDA backends fuse these two terms in ljlk_elec_pose_score
// .impl.hh, whose kernel is one large lambda capturing TView locals; Metal
// cannot take it as it stands (see ljlk_elec_ctx.hh). This is the packing
// variant of that kernel -- rotamer pairs, weighted, energies only -- written
// against an explicit context so a `.metal` entry can run it. It follows the
// fused kernel's arithmetic: one distance and one cutoff test per atom pair,
// then LJ/LK and elec from the same geometry, with their own count-pair
// separations, combined under the score weights into one value per rotamer
// pair.
//
// The shared tile differs from the CPU/CUDA one in one way: it stages atom
// type indices rather than LJLKTypeParams records, as the unfused Metal LJ/LK
// kernel does (see LJLKSingleResData). Threadgroup memory bounds occupancy on
// Metal, and the records would add 3 KB per workgroup.

#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_ctx.hh>
#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using tmol::score::common::WorkgroupLane;

enum { LJLK_ELEC_TILE = 32, LJLK_ELEC_MAX_N_CONN = 12 };

template <typename Real>
struct LJLKElecSingleResData {
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  TMOL_SHARED_PTR(Real) coords;
  TMOL_SHARED_PTR(short) atom_types;
  TMOL_SHARED_PTR(Real) charges;
  TMOL_SHARED_PTR(unsigned char) ljlk_path_dist;
  TMOL_SHARED_PTR(unsigned char) elec_path_dist;
};

template <typename Real>
struct LJLKElecScoringData {
  int pose_ind;
  int block_ind1;
  int block_ind2;
  LJLKElecSingleResData<Real> r1;
  LJLKElecSingleResData<Real> r2;
  int min_separation;
  bool in_count_pair_striking_dist;
  bool crossover_3full;
  TMOL_SHARED_PTR(unsigned char) conn_seps;
  LJGlobalParams<Real> ljlk_global_params;
  tmol::score::elec::potentials::ElecGlobalParams<Real> elec_global_params;
  Real w_ljatr;
  Real w_ljrep;
  Real w_lk;
  Real w_elec;
  Real total_weighted;
};

template <typename Real, int TILE, int MAX_CONN>
struct LJLKElecSharedData {
  Real coords1[TILE * 3];
  Real coords2[TILE * 3];
  short atom_types1[TILE];
  short atom_types2[TILE];
  Real charges1[TILE];
  Real charges2[TILE];
  unsigned char conn_ats1[MAX_CONN];
  unsigned char conn_ats2[MAX_CONN];
  unsigned char ljlk_path_dist1[MAX_CONN * TILE];
  unsigned char ljlk_path_dist2[MAX_CONN * TILE];
  unsigned char elec_path_dist1[MAX_CONN * TILE];
  unsigned char elec_path_dist2[MAX_CONN * TILE];
  unsigned char conn_seps[MAX_CONN * MAX_CONN];
};

template <typename Real, int TILE, int MAX_CONN, typename launch_t>
union LJLKElecSharedMem {
  CTA_REAL_REDUCE_T_TYPEDEF;

#if !TMOL_PORTABLE_METAL
  LJLKElecSharedMem() {}
#endif
  LJLKElecSharedData<Real, TILE, MAX_CONN> m;
  CTA_REAL_REDUCE_T_VARIABLE;
};

// Count-pair separation across two blocks, by either term's bond-path table.
template <int TILE, typename Real>
TMOL_HD int ljlk_elec_inter_separation(
    TMOL_THREAD LJLKElecScoringData<Real> const& data,
    int atom1,
    int atom2,
    bool elec_path) {
  int separation = data.min_separation;
  if (data.in_count_pair_striking_dist) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        4,
        atom1,
        atom2,
        data.r1.n_conn,
        data.r2.n_conn,
        elec_path ? data.r1.elec_path_dist : data.r1.ljlk_path_dist,
        elec_path ? data.r2.elec_path_dist : data.r2.ljlk_path_dist,
        data.conn_seps);
  }
  if (data.crossover_3full && (separation == 3 || separation == 4)) {
    separation = 5;
  }
  return separation;
}

// The weighted LJ/LK + elec energy of one atom pair whose squared distance
// is already known to lie inside the larger of the two cutoffs.
template <typename Real, typename TypeParamsView>
TMOL_HD Real ljlk_elec_weighted_pair_energy(
    int atom1,
    int atom2,
    TMOL_THREAD LJLKElecScoringData<Real> const& data,
    Real dist2,
    int ljlk_separation,
    int elec_separation,
    TypeParamsView type_params) {
  if (ljlk_separation < 4 && elec_separation < 4) {
    return Real(0);
  }
  Real const dist = tmol::portable::math::sqrt(dist2);
  Real total = 0;

  int const type1 = data.r1.atom_types[atom1];
  int const type2 = data.r2.atom_types[atom2];
  if (type1 >= 0 && type2 >= 0
      && dist2 < data.ljlk_global_params.max_dis
                     * data.ljlk_global_params.max_dis) {
    LJLKTypeParams<Real> const p1 = type_params[type1];
    LJLKTypeParams<Real> const p2 = type_params[type2];
    auto const lj = lj_score<Real>::V(
        dist,
        ljlk_separation,
        p1.lj_params(),
        p2.lj_params(),
        data.ljlk_global_params);
    total += data.w_ljatr * lj[0] + data.w_ljrep * lj[1];
    if (!p1.is_hydrogen && !p2.is_hydrogen) {
      total += data.w_lk
               * lk_isotropic_score<Real>::V(
                   dist,
                   ljlk_separation,
                   p1.lk_params(),
                   p2.lk_params(),
                   data.ljlk_global_params);
    }
  }

  total += data.w_elec
           * tmol::score::elec::potentials::elec(
               dist,
               data.r1.charges[atom1],
               data.r2.charges[atom2],
               Real(elec_separation),
               data.elec_global_params);
  return total;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void ljlk_elec_eval_weighted_rotamer_energies(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  enum { TILE = LJLK_ELEC_TILE };
  typedef LJLKElecScoringData<Real> Data;

  auto write_zero = ([&](int tid) {
    if (tid == 0) ctx.output[0][0][0][cta] = 0;
  });

  int const pose_ind = ctx.dispatch_indices[0][cta];
  int const rot_ind1 = ctx.dispatch_indices[1][cta];
  int const rot_ind2 = ctx.dispatch_indices[2][cta];
  if (rot_ind1 < 0 || rot_ind2 < 0) {
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(write_zero, lane);
    return;
  }
  int const block_ind1 = ctx.block_ind_for_rot[rot_ind1];
  int const block_ind2 = ctx.block_ind_for_rot[rot_ind2];
  int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
  int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];
  if (block_type1 < 0 || block_type2 < 0) {
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(write_zero, lane);
    return;
  }
  int const n_atoms1 = ctx.block_type_n_atoms[block_type1];
  int const n_atoms2 = ctx.block_type_n_atoms[block_type2];

  // Everything that does not depend on the tile pair.
  auto initialize_common = ([&](int p,
                                int r1,
                                int r2,
                                int b1,
                                int b2,
                                int bt1,
                                int bt2,
                                int na1,
                                int na2,
                                TMOL_THREAD Data& data,
                                TMOL_SHARED_REF(Shared) sm) {
    data.pose_ind = p;
    data.block_ind1 = b1;
    data.block_ind2 = b2;
    data.r1.block_type = bt1;
    data.r2.block_type = bt2;
    data.r1.rot_coord_offset = ctx.rot_coord_offset[r1];
    data.r2.rot_coord_offset = ctx.rot_coord_offset[r2];
    data.r1.n_atoms = na1;
    data.r2.n_atoms = na2;
    data.r1.coords = sm.m.coords1;
    data.r2.coords = sm.m.coords2;
    data.r1.atom_types = sm.m.atom_types1;
    data.r2.atom_types = sm.m.atom_types2;
    data.r1.charges = sm.m.charges1;
    data.r2.charges = sm.m.charges2;
    data.r1.ljlk_path_dist = sm.m.ljlk_path_dist1;
    data.r2.ljlk_path_dist = sm.m.ljlk_path_dist2;
    data.r1.elec_path_dist = sm.m.elec_path_dist1;
    data.r2.elec_path_dist = sm.m.elec_path_dist2;
    data.conn_seps = sm.m.conn_seps;
    data.crossover_3full = ctx.block_type_all_atoms_ligand_typed[bt1]
                           && ctx.block_type_all_atoms_ligand_typed[bt2];
    data.ljlk_global_params = ctx.ljlk_global_params[0];
    data.elec_global_params = ctx.elec_global_params[0];
    data.w_ljatr = ctx.score_weights[0];
    data.w_ljrep = ctx.score_weights[1];
    data.w_lk = ctx.score_weights[2];
    data.w_elec = ctx.score_weights[3];
    data.total_weighted = 0;
  });

  // Stage one tile of one residue: coordinates, atom types, charges, and --
  // when the blocks are close enough in bonds to need count-pair -- the
  // path distances from each connection atom under both terms' tables.
  auto load_tile = ([&](int start,
                        int count,
                        bool count_pair,
                        TMOL_THREAD LJLKElecSingleResData<Real> const& res,
                        TMOL_SHARED_PTR(unsigned char) conn_ats) {
    DeviceDispatch<D>::template copy_contiguous_data<nt, 3>(
        res.coords,
        reinterpret_cast<TMOL_DEVICE_PTR(Real)>(
            &ctx.rot_coords[res.rot_coord_offset + start]),
        count * 3,
        lane);
    auto load_atoms = ([&](int tid) {
      for (int atom = tid; atom < count; atom += nt) {
        int const atid = start + atom;
        res.atom_types[atom] =
            short(ctx.block_type_atom_types[res.block_type][atid]);
        res.charges[atom] = ctx.block_type_partial_charge[res.block_type][atid];
        if (!count_pair) continue;
        for (int conn = 0; conn < res.n_conn; ++conn) {
          int const conn_atom = conn_ats[conn];
          res.ljlk_path_dist[conn * TILE + atom] =
              ctx.block_type_ljlk_path_distance[res.block_type][conn_atom]
                                               [atid];
          res.elec_path_dist[conn * TILE + atom] =
              ctx.block_type_elec_inter_repr_path_distance[res.block_type]
                                                          [conn_atom][atid];
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(load_atoms, lane);
  });

  // Sum the weighted energy of one tile pair into the running total.
  auto eval_tile_pair =
      ([&](TMOL_THREAD Data& data, int start1, int start2, bool intra) {
        auto pair_energy = ([&](int pair_start1,
                                int pair_start2,
                                int atom1,
                                int atom2,
                                TMOL_THREAD Data const& pair_data) {
          tmol::portable::array<Real, 1> result = {};
          tmol::portable::Vec<Real, 3> const delta =
              common::coord_from_shared<Real>(pair_data.r1.coords, atom1)
              - common::coord_from_shared<Real>(pair_data.r2.coords, atom2);
          Real const dist2 = delta.squaredNorm();
          Real const max_dis =
              max(pair_data.ljlk_global_params.max_dis,
                  pair_data.elec_global_params.max_dis);
          // Atom pairs in overlapping rotamer spheres can still lie beyond the
          // cutoff; reject them before reading either bond-path table.
          if (dist2 >= max_dis * max_dis) return result;
          int ljlk_sep;
          int elec_sep;
          if (intra) {
            int const global1 = pair_start1 + atom1;
            int const global2 = pair_start2 + atom2;
            ljlk_sep =
                ctx.block_type_ljlk_path_distance[pair_data.r1.block_type]
                                                 [global1][global2];
            elec_sep = ctx.block_type_elec_intra_repr_path_distance
                           [pair_data.r1.block_type][global1][global2];
          } else {
            ljlk_sep = ljlk_elec_inter_separation<TILE>(
                pair_data, atom1, atom2, false);
            elec_sep =
                ljlk_elec_inter_separation<TILE>(pair_data, atom1, atom2, true);
          }
          result[0] = ljlk_elec_weighted_pair_energy<Real>(
              atom1,
              atom2,
              pair_data,
              dist2,
              ljlk_sep,
              elec_sep,
              ctx.ljlk_type_params);
          return result;
        });
        auto evaluate = ([&](int tid) {
          tmol::portable::array<Real, 1> partial;
          if (intra) {
            partial = common::IntraResBlockEvaluation<
                LJLKElecScoringData,
                common::AllAtomPairSelector,
                D,
                TILE,
                nt,
                1,
                Real,
                Int>::
                eval_intrares_atom_pairs(
                    tid, start1, start2, pair_energy, data);
          } else {
            partial = common::InterResBlockEvaluation<
                LJLKElecScoringData,
                common::AllAtomPairSelector,
                D,
                TILE,
                nt,
                1,
                Real,
                Int>::
                eval_interres_atom_pair(tid, start1, start2, pair_energy, data);
          }
          data.total_weighted += partial[0];
        });
        DeviceDispatch<D>::template for_each_in_workgroup<nt>(evaluate, lane);
      });

  // Inter-block: tile-invariant setup, including the count-pair connection
  // data, which is shared by every tile pair.
  auto load_tile_invariant_inter = ([&](int p,
                                        int r1,
                                        int r2,
                                        int b1,
                                        int b2,
                                        int bt1,
                                        int bt2,
                                        int na1,
                                        int na2,
                                        TMOL_THREAD Data& data,
                                        TMOL_SHARED_REF(Shared) sm) {
    initialize_common(p, r1, r2, b1, b2, bt1, bt2, na1, na2, data, sm);
    data.r1.n_conn = ctx.block_type_n_interblock_bonds[bt1];
    data.r2.n_conn = ctx.block_type_n_interblock_bonds[bt2];
    int const near_slot = common::count_pair::near_block_slot(
        ctx.pose_stack_near_blocks[p][b1], b2);
    data.min_separation = ctx.pose_stack_near_blocks[p][b1][near_slot][1];
    data.in_count_pair_striking_dist = data.min_separation <= 4;
    if (!data.in_count_pair_striking_dist) return;
    auto load_connections = ([&](int tid) {
      int const n1 = data.r1.n_conn;
      int const n2 = data.r2.n_conn;
      int const total = n1 + n2 + n1 * n2;
      for (int index = tid; index < total; index += nt) {
        if (index < n1) {
          sm.m.conn_ats1[index] =
              ctx.block_type_atoms_forming_chemical_bonds[bt1][index];
        } else if (index < n1 + n2) {
          sm.m.conn_ats2[index - n1] =
              ctx.block_type_atoms_forming_chemical_bonds[bt2][index - n1];
        } else {
          int const pair_index = index - n1 - n2;
          sm.m.conn_seps[pair_index] =
              ctx.pose_stack_inter_block_bondsep[p][b1][near_slot]
                                                [pair_index / n2]
                                                [pair_index % n2];
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(
        load_connections, lane);
  });
  auto load_inter1 = ([&](int,
                          int start,
                          int count,
                          TMOL_THREAD Data& data,
                          TMOL_SHARED_REF(Shared) sm) {
    load_tile(
        start,
        count,
        data.in_count_pair_striking_dist,
        data.r1,
        sm.m.conn_ats1);
  });
  auto load_inter2 = ([&](int,
                          int start,
                          int count,
                          TMOL_THREAD Data& data,
                          TMOL_SHARED_REF(Shared) sm) {
    load_tile(
        start,
        count,
        data.in_count_pair_striking_dist,
        data.r2,
        sm.m.conn_ats2);
  });
  auto finish_inter_load =
      ([&](int, int, TMOL_SHARED_REF(Shared), TMOL_THREAD Data&) {});
  auto eval_inter = ([&](TMOL_THREAD Data& data, int start1, int start2) {
    eval_tile_pair(data, start1, start2, false);
  });

  // Intra-block: separations come straight from the block type's tables, so
  // no connection data is staged.
  auto load_tile_invariant_intra = ([&](int p,
                                        int r1,
                                        int b1,
                                        int bt1,
                                        int na1,
                                        TMOL_THREAD Data& data,
                                        TMOL_SHARED_REF(Shared) sm) {
    initialize_common(p, r1, r1, b1, b1, bt1, bt1, na1, na1, data, sm);
    data.r1.n_conn = 0;
    data.r2.n_conn = 0;
    data.min_separation = 0;
    data.in_count_pair_striking_dist = false;
  });
  auto load_intra1 = ([&](int,
                          int start,
                          int count,
                          TMOL_THREAD Data& data,
                          TMOL_SHARED_REF(Shared) sm) {
    data.r1.coords = sm.m.coords1;
    data.r1.atom_types = sm.m.atom_types1;
    data.r1.charges = sm.m.charges1;
    load_tile(start, count, false, data.r1, sm.m.conn_ats1);
  });
  auto load_intra2 = ([&](int,
                          int start,
                          int count,
                          TMOL_THREAD Data& data,
                          TMOL_SHARED_REF(Shared) sm) {
    // A same-tile pass may have aliased r2 onto the "1" arrays.
    data.r2.coords = sm.m.coords2;
    data.r2.atom_types = sm.m.atom_types2;
    data.r2.charges = sm.m.charges2;
    load_tile(start, count, false, data.r2, sm.m.conn_ats2);
  });
  auto finish_intra_load = ([&](int tile1,
                                int tile2,
                                TMOL_SHARED_REF(Shared) sm,
                                TMOL_THREAD Data& data) {
    // A tile paired with itself is staged once, in the "1" arrays.
    bool const same_tile = tile1 == tile2;
    data.r1.coords = sm.m.coords1;
    data.r1.atom_types = sm.m.atom_types1;
    data.r1.charges = sm.m.charges1;
    data.r2.coords = same_tile ? sm.m.coords1 : sm.m.coords2;
    data.r2.atom_types = same_tile ? sm.m.atom_types1 : sm.m.atom_types2;
    data.r2.charges = same_tile ? sm.m.charges1 : sm.m.charges2;
  });
  auto eval_intra = ([&](TMOL_THREAD Data& data, int start1, int start2) {
    eval_tile_pair(data, start1, start2, true);
  });

  auto store_energy = ([&](TMOL_THREAD Data& data, TMOL_SHARED_REF(Shared) sm) {
    auto reduce = ([&](int tid) {
      Real const total = DeviceDispatch<D>::template reduce_in_workgroup<nt>(
          data.total_weighted, sm, tmol::portable::plus_t<Real>(), lane);
      if (tid == 0) ctx.output[0][0][0][cta] = total;
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce, lane);
  });

  common::tile_evaluate_rot_pair<DeviceDispatch, D, Data, Data, Real, TILE>(
      shared,
      pose_ind,
      rot_ind1,
      rot_ind2,
      block_ind1,
      block_ind2,
      block_type1,
      block_type2,
      n_atoms1,
      n_atoms2,
      load_tile_invariant_inter,
      load_inter1,
      load_inter2,
      finish_inter_load,
      eval_inter,
      store_energy,
      load_tile_invariant_intra,
      load_intra1,
      load_intra2,
      finish_intra_load,
      eval_intra,
      store_energy,
      lane);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
