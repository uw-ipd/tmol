#pragma once

#include <cmath>

#include <moderngpu/operators.hxx>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>

#include "ljlk_elec_pose_score.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

namespace ljlk_elec_detail {

constexpr int tile_size = 32;
constexpr int max_n_conn = 4;

template <typename Real>
struct SingleResData {
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  Real* coords;
  LJLKTypeParams<Real>* ljlk_params;
  Real* charges;
  unsigned char* ljlk_path_dist;
  unsigned char* elec_path_dist;
};

template <typename Real>
struct ScoringData {
  int pose_ind;
  int output_ind;
  int block_ind1;
  int block_ind2;
  SingleResData<Real> r1;
  SingleResData<Real> r2;
  int min_separation;
  bool in_count_pair_striking_dist;
  unsigned char* conn_seps;
  LJGlobalParams<Real> ljlk_global_params;
  tmol::score::elec::potentials::ElecGlobalParams<Real> elec_global_params;
  Real total_ljatr;
  Real total_ljrep;
  Real total_lk;
  Real total_elec;
  Real total_weighted;
};

template <typename Real>
struct SharedData {
  Real coords1[tile_size * 3];
  Real coords2[tile_size * 3];
  LJLKTypeParams<Real> ljlk_params1[tile_size];
  LJLKTypeParams<Real> ljlk_params2[tile_size];
  Real charges1[tile_size];
  Real charges2[tile_size];
  unsigned char conn_ats1[max_n_conn];
  unsigned char conn_ats2[max_n_conn];
  unsigned char ljlk_path_dist1[max_n_conn * tile_size];
  unsigned char ljlk_path_dist2[max_n_conn * tile_size];
  unsigned char elec_path_dist1[max_n_conn * tile_size];
  unsigned char elec_path_dist2[max_n_conn * tile_size];
  unsigned char conn_seps[max_n_conn * max_n_conn];
};

template <typename Real>
TMOL_DEVICE_FUNC int inter_separation(
    ScoringData<Real> const& data,
    int atom1,
    int atom2,
    bool elec_path,
    bool crossover_3full) {
  int separation = data.min_separation;
  if (data.in_count_pair_striking_dist) {
    auto path1 = elec_path ? data.r1.elec_path_dist : data.r1.ljlk_path_dist;
    auto path2 = elec_path ? data.r2.elec_path_dist : data.r2.ljlk_path_dist;
    separation =
        common::count_pair::shared_mem_inter_block_separation<tile_size>(
            4,
            atom1,
            atom2,
            data.r1.n_conn,
            data.r2.n_conn,
            path1,
            path2,
            data.conn_seps);
  }
  if (crossover_3full && (separation == 3 || separation == 4)) {
    separation = 5;
  }
  return separation;
}

template <bool require_gradient, bool weighted, typename Real, tmol::Device D>
TMOL_DEVICE_FUNC std::array<Real, 4> score_atom_pair(
    int atom1,
    int atom2,
    int start1,
    int start2,
    ScoringData<Real> const& data,
    int ljlk_separation,
    int elec_separation,
    TView<Eigen::Matrix<Real, 3, 1>, 2, D> dV_dcoords,
    TView<Real, 1, D> score_weights,
    Real derivative_scale) {
  if (ljlk_separation < 4 && elec_separation < 4) {
    return {0, 0, 0, 0};
  }
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  Real3 const coord1 = coord_from_shared(data.r1.coords, atom1);
  Real3 const coord2 = coord_from_shared(data.r2.coords, atom2);
  Real3 const delta = coord1 - coord2;
  Real const dist2 = delta.squaredNorm();
  Real const max_dis =
      max(data.ljlk_global_params.max_dis, data.elec_global_params.max_dis);
  if (dist2 >= max_dis * max_dis) return {0, 0, 0, 0};

  Real const dist = std::sqrt(dist2);
  auto const& p1 = data.r1.ljlk_params[atom1];
  auto const& p2 = data.r2.ljlk_params[atom2];

  Real ljatr = 0;
  Real ljrep = 0;
  Real lk_value = 0;
  Real ljatr_deriv = 0;
  Real ljrep_deriv = 0;
  Real lk_deriv = 0;
  if (dist2
      < data.ljlk_global_params.max_dis * data.ljlk_global_params.max_dis) {
    if (require_gradient) {
      auto const lj = lj_score<Real>::V_dV(
          dist,
          ljlk_separation,
          p1.lj_params(),
          p2.lj_params(),
          data.ljlk_global_params);
      ljatr = lj.Vatr;
      ljrep = lj.Vrep;
      ljatr_deriv = lj.dVatr_ddist;
      ljrep_deriv = lj.dVrep_ddist;
      if (!p1.is_hydrogen && !p2.is_hydrogen) {
        auto const lk = lk_isotropic_score<Real>::V_dV(
            dist,
            ljlk_separation,
            p1.lk_params(),
            p2.lk_params(),
            data.ljlk_global_params);
        lk_value = lk.V;
        lk_deriv = lk.dV_ddist;
      }
    } else {
      auto const lj = lj_score<Real>::V(
          dist,
          ljlk_separation,
          p1.lj_params(),
          p2.lj_params(),
          data.ljlk_global_params);
      ljatr = lj[0];
      ljrep = lj[1];
      if (!p1.is_hydrogen && !p2.is_hydrogen) {
        lk_value = lk_isotropic_score<Real>::V(
            dist,
            ljlk_separation,
            p1.lk_params(),
            p2.lk_params(),
            data.ljlk_global_params);
      }
    }
  }

  Real elec_value = 0;
  Real elec_deriv = 0;
  if (require_gradient) {
    common::tie(elec_value, elec_deriv) =
        tmol::score::elec::potentials::elec_delec_ddist(
            dist,
            data.r1.charges[atom1],
            data.r2.charges[atom2],
            Real(elec_separation),
            data.elec_global_params);
  } else {
    elec_value = tmol::score::elec::potentials::elec(
        dist,
        data.r1.charges[atom1],
        data.r2.charges[atom2],
        Real(elec_separation),
        data.elec_global_params);
  }

  if (require_gradient && dist != 0) {
    Real3 const unit_delta = delta / dist;
    Real const radial_derivs[4] = {
        ljatr_deriv, ljrep_deriv, lk_deriv, elec_deriv};
    if constexpr (weighted) {
      Real const weighted_deriv = derivative_scale
                                  * (score_weights[0] * radial_derivs[0]
                                     + score_weights[1] * radial_derivs[1]
                                     + score_weights[2] * radial_derivs[2]
                                     + score_weights[3] * radial_derivs[3]);
      Real3 const dxyz1 = weighted_deriv * unit_delta;
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        if (dxyz1[axis] == 0) continue;
        accumulate<D, Real>::add(
            dV_dcoords[0][data.r1.rot_coord_offset + start1 + atom1][axis],
            dxyz1[axis]);
        accumulate<D, Real>::add(
            dV_dcoords[0][data.r2.rot_coord_offset + start2 + atom2][axis],
            -dxyz1[axis]);
      }
    } else {
#pragma unroll
      for (int score_type = 0; score_type < 4; ++score_type) {
        Real3 const dxyz1 = radial_derivs[score_type] * unit_delta;
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
          if (dxyz1[axis] == 0) continue;
          accumulate<D, Real>::add(
              dV_dcoords[score_type][data.r1.rot_coord_offset + start1 + atom1]
                        [axis],
              dxyz1[axis]);
          accumulate<D, Real>::add(
              dV_dcoords[score_type][data.r2.rot_coord_offset + start2 + atom2]
                        [axis],
              -dxyz1[axis]);
        }
      }
    }
  }
  return {ljatr, ljrep, lk_value, elec_value};
}

template <
    bool weighted,
    bool rotamer_pairs,
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    int nt,
    typename Real,
    typename Shared>
TMOL_DEVICE_FUNC void store_score_totals(
    int tid,
    ScoringData<Real>& data,
    Shared& shared,
    TView<Real, 4, D> output) {
  if constexpr (weighted) {
    Real const total = DeviceOperations<D>::template reduce_in_workgroup<nt>(
        data.total_weighted, shared, mgpu::plus_t<Real>());
    if (tid == 0) {
      if constexpr (rotamer_pairs) {
        output[0][0][0][data.output_ind] = total;
      } else {
        accumulate<D, Real>::add(output[0][data.pose_ind][0][0], total);
      }
    }
  } else {
    Real const totals[4] = {
        DeviceOperations<D>::template reduce_in_workgroup<nt>(
            data.total_ljatr, shared, mgpu::plus_t<Real>()),
        DeviceOperations<D>::template reduce_in_workgroup<nt>(
            data.total_ljrep, shared, mgpu::plus_t<Real>()),
        DeviceOperations<D>::template reduce_in_workgroup<nt>(
            data.total_lk, shared, mgpu::plus_t<Real>()),
        DeviceOperations<D>::template reduce_in_workgroup<nt>(
            data.total_elec, shared, mgpu::plus_t<Real>())};
    if (tid == 0) {
      for (int score_type = 0; score_type < 4; ++score_type) {
        accumulate<D, Real>::add(
            output[score_type][data.pose_ind][0][0], totals[score_type]);
      }
    }
  }
}

}  // namespace ljlk_elec_detail

template <
    bool require_gradient,
    bool weighted,
    bool rotamer_pairs,
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto ljlk_elec_forward_impl(
    ContextManager& mgr,
    TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,
    TView<Int, 3, D> pose_stack_min_bond_separation,
    TView<Int, 5, D> pose_stack_inter_block_bondsep,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Int, 2, D> block_type_atom_types,
    TView<Int, 1, D> block_type_n_interblock_bonds,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
    TView<Int, 3, D> block_type_ljlk_path_distance,
    TView<Int, 1, D> block_type_is_ligand_fragment,
    TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
    TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
    TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
    TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
        elec_global_params,
    TView<Int, 1, D> shared_compact_block_neighbors,
    TView<Int, 2, D> rotamer_dispatch_indices,
    TView<Real, 1, D> score_weights,
    TView<Real, 1, D> output_gradients)
    -> std::tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>> {
  using namespace ljlk_elec_detail;
  using Real3 = Eigen::Matrix<Real, 3, 1>;
  static_assert(!require_gradient || !rotamer_pairs || weighted);

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "fused LJ/LK-electrostatics block-pair dispatch");
  (void)pose_ind_for_atom;
  (void)first_rot_block_type;
  (void)block_ind_for_rot;
  (void)pose_ind_for_rot;
  (void)n_rots_for_pose;
  (void)rot_offset_for_pose;
  (void)n_rots_for_block;
  (void)max_n_rots_per_pose;

  int constexpr n_output_scores = weighted ? 1 : 4;
  int const n_outputs =
      rotamer_pairs ? rotamer_dispatch_indices.size(1) : n_poses;
  auto output_t =
      rotamer_pairs
          ? TPack<Real, 4, D>::empty({n_output_scores, 1, 1, n_outputs})
          : TPack<Real, 4, D>::zeros({n_output_scores, n_outputs, 1, 1});
  auto output = output_t.view;
  auto dV_dcoords_t =
      require_gradient ? TPack<Real3, 2, D>::zeros({n_output_scores, n_atoms})
                       : TPack<Real3, 2, D>::empty({n_output_scores, 0});
  auto dV_dcoords = dV_dcoords_t.view;

  LAUNCH_BOX_32_OCC_AS(launch_t, 32);
  CTA_REAL_REDUCE_T_TYPEDEF;
  // CPU workgroup lanes run serially; one lane traverses each tile directly.
  constexpr int score_nt = D == tmol::Device::CPU ? 1 : nt;

  auto eval_neighbor = ([=] TMOL_DEVICE_FUNC(int candidate) {
    Real derivative_scale = 1;
    if (require_gradient && rotamer_pairs) {
      derivative_scale = output_gradients[candidate];
    }
    int pose_ind;
    int rot_ind1;
    int rot_ind2;
    int block_ind1;
    int block_ind2;
    // A normal branch is intentional here. NVCC rejects first capture of a
    // view from an extended device lambda inside constexpr-if; rotamer_pairs
    // is a template constant, so optimized code still contains one path.
    if (rotamer_pairs) {
      pose_ind = rotamer_dispatch_indices[0][candidate];
      rot_ind1 = rotamer_dispatch_indices[1][candidate];
      rot_ind2 = rotamer_dispatch_indices[2][candidate];
      block_ind1 = block_ind_for_rot[rot_ind1];
      block_ind2 = block_ind_for_rot[rot_ind2];
    } else {
      pose_ind = candidate / max_n_upper_triangle_inds;
      auto const pair = common::upper_triangle_inds_from_linear_index(
          candidate % max_n_upper_triangle_inds, max_n_blocks + 1);
      block_ind1 = common::get<0>(pair);
      block_ind2 = common::get<1>(pair) - 1;
      rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
      rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];
    }
    if (rot_ind1 < 0 || rot_ind2 < 0) {
      if (rotamer_pairs) output[0][0][0][candidate] = 0;
      return;
    }
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];
    if (block_type1 < 0 || block_type2 < 0) {
      if (rotamer_pairs) output[0][0][0][candidate] = 0;
      return;
    }
    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      SharedData<Real> m;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    auto initialize_common = ([=] TMOL_DEVICE_FUNC(
                                  int p,
                                  int r1,
                                  int r2,
                                  int b1,
                                  int b2,
                                  int bt1,
                                  int bt2,
                                  int na1,
                                  int na2,
                                  ScoringData<Real>& data,
                                  shared_mem_union& sm) {
      data.pose_ind = p;
      data.output_ind = candidate;
      data.block_ind1 = b1;
      data.block_ind2 = b2;
      data.r1.block_type = bt1;
      data.r2.block_type = bt2;
      data.r1.rot_coord_offset = rot_coord_offset[r1];
      data.r2.rot_coord_offset = rot_coord_offset[r2];
      data.r1.n_atoms = na1;
      data.r2.n_atoms = na2;
      data.r1.n_conn = block_type_n_interblock_bonds[bt1];
      data.r2.n_conn = block_type_n_interblock_bonds[bt2];
      data.min_separation = pose_stack_min_bond_separation[p][b1][b2];
      data.in_count_pair_striking_dist = data.min_separation <= 4;
      data.r1.coords = sm.m.coords1;
      data.r2.coords = sm.m.coords2;
      data.r1.ljlk_params = sm.m.ljlk_params1;
      data.r2.ljlk_params = sm.m.ljlk_params2;
      data.r1.charges = sm.m.charges1;
      data.r2.charges = sm.m.charges2;
      data.r1.ljlk_path_dist = sm.m.ljlk_path_dist1;
      data.r2.ljlk_path_dist = sm.m.ljlk_path_dist2;
      data.r1.elec_path_dist = sm.m.elec_path_dist1;
      data.r2.elec_path_dist = sm.m.elec_path_dist2;
      data.conn_seps = sm.m.conn_seps;
      data.ljlk_global_params = ljlk_global_params[0];
      data.elec_global_params = elec_global_params[0];
      data.total_ljatr = 0;
      data.total_ljrep = 0;
      data.total_lk = 0;
      data.total_elec = 0;
      data.total_weighted = 0;
    });

    auto load_tile = ([=] TMOL_DEVICE_FUNC(
                          int start,
                          int count,
                          ScoringData<Real>& data,
                          SingleResData<Real>& residue,
                          Real* coord_dst,
                          LJLKTypeParams<Real>* params_dst,
                          Real* charge_dst,
                          unsigned char* lj_path_dst,
                          unsigned char* elec_path_dst,
                          unsigned char* conn_ats) {
      DeviceOperations<D>::template copy_contiguous_data<nt, 3>(
          coord_dst,
          reinterpret_cast<Real*>(
              &rot_coords[residue.rot_coord_offset + start]),
          count * 3);
      auto load_atom_data = ([=](int tid) {
        for (int atom = tid; atom < count; atom += score_nt) {
          int const atid = start + atom;
          int const atom_type = block_type_atom_types[residue.block_type][atid];
          if (atom_type >= 0) params_dst[atom] = ljlk_type_params[atom_type];
          charge_dst[atom] =
              block_type_partial_charge[residue.block_type][atid];
          if (!data.in_count_pair_striking_dist) continue;
          for (int conn = 0; conn < residue.n_conn; ++conn) {
            int const conn_atom = conn_ats[conn];
            lj_path_dst[conn * tile_size + atom] =
                block_type_ljlk_path_distance[residue.block_type][conn_atom]
                                             [atid];
            elec_path_dst[conn * tile_size + atom] =
                block_type_elec_inter_repr_path_distance[residue.block_type]
                                                        [conn_atom][atid];
          }
        }
      });
      DeviceOperations<D>::template for_each_in_workgroup<score_nt>(
          load_atom_data);
    });

    auto load_tile_invariant_inter = ([=] TMOL_DEVICE_FUNC(
                                          int p,
                                          int r1,
                                          int r2,
                                          int b1,
                                          int b2,
                                          int bt1,
                                          int bt2,
                                          int na1,
                                          int na2,
                                          ScoringData<Real>& data,
                                          shared_mem_union& sm) {
      initialize_common(p, r1, r2, b1, b2, bt1, bt2, na1, na2, data, sm);
      if (!data.in_count_pair_striking_dist) return;
      auto load_connections = ([&](int tid) {
        int const n1 = data.r1.n_conn;
        int const n2 = data.r2.n_conn;
        int const total = n1 + n2 + n1 * n2;
        for (int index = tid; index < total; index += score_nt) {
          if (index < n1) {
            sm.m.conn_ats1[index] =
                block_type_atoms_forming_chemical_bonds[bt1][index];
          } else if (index < n1 + n2) {
            sm.m.conn_ats2[index - n1] =
                block_type_atoms_forming_chemical_bonds[bt2][index - n1];
          } else {
            int const pair_index = index - n1 - n2;
            sm.m.conn_seps[pair_index] =
                pose_stack_inter_block_bondsep[p][b1][b2][pair_index / n2]
                                              [pair_index % n2];
          }
        }
      });
      DeviceOperations<D>::template for_each_in_workgroup<score_nt>(
          load_connections);
    });

    auto load_inter1 = ([=] TMOL_DEVICE_FUNC(
                            int,
                            int start,
                            int count,
                            ScoringData<Real>& data,
                            shared_mem_union& sm) {
      load_tile(
          start,
          count,
          data,
          data.r1,
          sm.m.coords1,
          sm.m.ljlk_params1,
          sm.m.charges1,
          sm.m.ljlk_path_dist1,
          sm.m.elec_path_dist1,
          sm.m.conn_ats1);
    });
    auto load_inter2 = ([=] TMOL_DEVICE_FUNC(
                            int,
                            int start,
                            int count,
                            ScoringData<Real>& data,
                            shared_mem_union& sm) {
      load_tile(
          start,
          count,
          data,
          data.r2,
          sm.m.coords2,
          sm.m.ljlk_params2,
          sm.m.charges2,
          sm.m.ljlk_path_dist2,
          sm.m.elec_path_dist2,
          sm.m.conn_ats2);
    });
    auto finish_inter_load =
        ([=] TMOL_DEVICE_FUNC(int, int, shared_mem_union&, ScoringData<Real>&) {
        });

    auto eval_pairs = ([=] TMOL_DEVICE_FUNC(
                           ScoringData<Real> & data,
                           int start1,
                           int start2,
                           bool intra) {
      bool const crossover_3full =
          block_type_is_ligand_fragment[data.r1.block_type]
          && block_type_is_ligand_fragment[data.r2.block_type];
      auto pair_score = ([=] TMOL_DEVICE_FUNC(
                             int pair_start1,
                             int pair_start2,
                             int atom1,
                             int atom2,
                             ScoringData<Real> const& pair_data) {
        int lj_sep;
        int elec_sep;
        if (intra) {
          int const global_atom1 = pair_start1 + atom1;
          int const global_atom2 = pair_start2 + atom2;
          lj_sep = block_type_ljlk_path_distance[pair_data.r1.block_type]
                                                [global_atom1][global_atom2];
          elec_sep =
              block_type_elec_intra_repr_path_distance[pair_data.r1.block_type]
                                                      [global_atom1]
                                                      [global_atom2];
        } else {
          lj_sep =
              inter_separation(pair_data, atom1, atom2, false, crossover_3full);
          elec_sep =
              inter_separation(pair_data, atom1, atom2, true, crossover_3full);
        }
        return score_atom_pair<require_gradient, weighted, Real, D>(
            atom1,
            atom2,
            pair_start1,
            pair_start2,
            pair_data,
            lj_sep,
            elec_sep,
            dV_dcoords,
            score_weights,
            derivative_scale);
      });
      auto evaluate = ([&](int tid) {
        std::array<Real, 4> scores = {};
        if constexpr (D == tmol::Device::CPU) {
          // The CPU workgroup has one lane. Traverse directly so GCC keeps
          // score_atom_pair inline and avoids flattened-index division.
          int const n_atoms1 =
              min(int(tile_size), int(data.r1.n_atoms - start1));
          int const n_atoms2 =
              min(int(tile_size), int(data.r2.n_atoms - start2));
          bool const same_tile = intra && start1 == start2;
          for (int atom1 = 0; atom1 < n_atoms1; ++atom1) {
            int const first_atom2 = same_tile ? atom1 + 1 : 0;
            for (int atom2 = first_atom2; atom2 < n_atoms2; ++atom2) {
              auto pair_scores = pair_score(start1, start2, atom1, atom2, data);
              common::for_<4>([&](auto term) {
                scores[term.value] += pair_scores[term.value];
              });
            }
          }
        } else {
          if (intra) {
            scores = common::IntraResBlockEvaluation<
                ScoringData,
                common::AllAtomPairSelector,
                D,
                tile_size,
                score_nt,
                4,
                Real,
                Int>::
                eval_intrares_atom_pairs(tid, start1, start2, pair_score, data);
          } else {
            scores = common::InterResBlockEvaluation<
                ScoringData,
                common::AllAtomPairSelector,
                D,
                tile_size,
                score_nt,
                4,
                Real,
                Int>::
                eval_interres_atom_pair(tid, start1, start2, pair_score, data);
          }
        }
        if constexpr (weighted) {
          data.total_weighted +=
              score_weights[0] * scores[0] + score_weights[1] * scores[1]
              + score_weights[2] * scores[2] + score_weights[3] * scores[3];
        } else {
          data.total_ljatr += scores[0];
          data.total_ljrep += scores[1];
          data.total_lk += scores[2];
          data.total_elec += scores[3];
        }
      });
      DeviceOperations<D>::template for_each_in_workgroup<score_nt>(evaluate);
    });
    auto eval_inter = ([=] TMOL_DEVICE_FUNC(
                           ScoringData<Real> & data, int start1, int start2) {
      eval_pairs(data, start1, start2, false);
    });

    auto load_tile_invariant_intra = ([=] TMOL_DEVICE_FUNC(
                                          int p,
                                          int r1,
                                          int b1,
                                          int bt1,
                                          int na1,
                                          ScoringData<Real>& data,
                                          shared_mem_union& sm) {
      initialize_common(p, r1, r1, b1, b1, bt1, bt1, na1, na1, data, sm);
      data.r1.n_conn = 0;
      data.r2.n_conn = 0;
      data.in_count_pair_striking_dist = false;
    });
    auto load_intra1 = ([=] TMOL_DEVICE_FUNC(
                            int,
                            int start,
                            int count,
                            ScoringData<Real>& data,
                            shared_mem_union& sm) {
      load_tile(
          start,
          count,
          data,
          data.r1,
          sm.m.coords1,
          sm.m.ljlk_params1,
          sm.m.charges1,
          sm.m.ljlk_path_dist1,
          sm.m.elec_path_dist1,
          sm.m.conn_ats1);
    });
    auto load_intra2 = ([=] TMOL_DEVICE_FUNC(
                            int,
                            int start,
                            int count,
                            ScoringData<Real>& data,
                            shared_mem_union& sm) {
      data.r2.coords = sm.m.coords2;
      data.r2.ljlk_params = sm.m.ljlk_params2;
      data.r2.charges = sm.m.charges2;
      load_tile(
          start,
          count,
          data,
          data.r2,
          sm.m.coords2,
          sm.m.ljlk_params2,
          sm.m.charges2,
          sm.m.ljlk_path_dist2,
          sm.m.elec_path_dist2,
          sm.m.conn_ats2);
    });
    auto finish_intra_load = ([=] TMOL_DEVICE_FUNC(
                                  int tile1,
                                  int tile2,
                                  shared_mem_union& sm,
                                  ScoringData<Real>& data) {
      bool const same_tile = tile1 == tile2;
      data.r1.coords = sm.m.coords1;
      data.r1.ljlk_params = sm.m.ljlk_params1;
      data.r1.charges = sm.m.charges1;
      data.r2.coords = same_tile ? sm.m.coords1 : sm.m.coords2;
      data.r2.ljlk_params = same_tile ? sm.m.ljlk_params1 : sm.m.ljlk_params2;
      data.r2.charges = same_tile ? sm.m.charges1 : sm.m.charges2;
    });
    auto eval_intra = ([=] TMOL_DEVICE_FUNC(
                           ScoringData<Real> & data, int start1, int start2) {
      eval_pairs(data, start1, start2, true);
    });

    auto store_energies =
        ([=] TMOL_DEVICE_FUNC(ScoringData<Real> & data, shared_mem_union & sm) {
          auto reduce = ([&](int tid) {
            store_score_totals<
                weighted,
                rotamer_pairs,
                DeviceOperations,
                D,
                score_nt>(tid, data, sm, output);
          });
          DeviceOperations<D>::template for_each_in_workgroup<score_nt>(reduce);
        });

    common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        ScoringData<Real>,
        ScoringData<Real>,
        Real,
        tile_size>(
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
        store_energies,
        load_tile_invariant_intra,
        load_intra1,
        load_intra2,
        finish_intra_load,
        eval_intra,
        store_energies);
  });

  if constexpr (rotamer_pairs) {
    if constexpr (require_gradient) {
      // Rotamer pairs share coordinate outputs. Keep CPU workgroups serial;
      // CUDA accumulation is atomic, so its workgroups remain concurrent.
      DeviceOperations<D>::template foreach_workgroup<launch_t>(
          mgr, n_outputs, eval_neighbor);
    } else {
      DeviceOperations<D>::template foreach_independent_workgroup<launch_t>(
          mgr, n_outputs, eval_neighbor);
    }
  } else {
    common::sphere_overlap::
        launch_precomputed_block_neighbors<DeviceOperations, D, launch_t, Int>(
            mgr, shared_compact_block_neighbors, eval_neighbor);
  }
  return {output_t, dV_dcoords_t};
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::forward(
    ContextManager& mgr,
    TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,
    TView<Int, 3, D> pose_stack_min_bond_separation,
    TView<Int, 5, D> pose_stack_inter_block_bondsep,
    TView<Int, 1, D> block_type_n_atoms,
    TView<Int, 2, D> block_type_atom_types,
    TView<Int, 1, D> block_type_n_interblock_bonds,
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
    TView<Int, 3, D> block_type_ljlk_path_distance,
    TView<Int, 1, D> block_type_is_ligand_fragment,
    TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
    TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
    TView<Real, 2, D> block_type_partial_charge,
    TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
    TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
    TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
        elec_global_params,
    TView<Int, 1, D> shared_compact_block_neighbors,
    bool require_gradient)
    -> std::tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>> {
#define TMOL_LJLK_ELEC_FORWARD_ARGS                                           \
  mgr, rot_coords, rot_coord_offset, pose_ind_for_atom, first_rot_for_block,  \
      first_rot_block_type, block_ind_for_rot, pose_ind_for_rot,              \
      block_type_ind_for_rot, n_rots_for_pose, rot_offset_for_pose,           \
      n_rots_for_block, rot_offset_for_block, max_n_rots_per_pose,            \
      pose_stack_min_bond_separation, pose_stack_inter_block_bondsep,         \
      block_type_n_atoms, block_type_atom_types,                              \
      block_type_n_interblock_bonds, block_type_atoms_forming_chemical_bonds, \
      block_type_ljlk_path_distance, block_type_is_ligand_fragment,           \
      ljlk_type_params, ljlk_global_params, block_type_partial_charge,        \
      block_type_elec_inter_repr_path_distance,                               \
      block_type_elec_intra_repr_path_distance, elec_global_params,           \
      shared_compact_block_neighbors
  auto empty_weights = TPack<Real, 1, D>::empty({0});
  auto empty_rotamer_dispatch = TPack<Int, 2, D>::empty({0, 0});
  if (require_gradient) {
    return ljlk_elec_forward_impl<
        true,
        false,
        false,
        DeviceOperations,
        D,
        Real,
        Int>(
        TMOL_LJLK_ELEC_FORWARD_ARGS,
        empty_rotamer_dispatch.view,
        empty_weights.view,
        empty_weights.view);
  }
  return ljlk_elec_forward_impl<
      false,
      false,
      false,
      DeviceOperations,
      D,
      Real,
      Int>(
      TMOL_LJLK_ELEC_FORWARD_ARGS,
      empty_rotamer_dispatch.view,
      empty_weights.view,
      empty_weights.view);
#undef TMOL_LJLK_ELEC_FORWARD_ARGS
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    forward_weighted(
        ContextManager& mgr,
        TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 1, D> pose_ind_for_atom,
        TView<Int, 2, D> first_rot_for_block,
        TView<Int, 2, D> first_rot_block_type,
        TView<Int, 1, D> block_ind_for_rot,
        TView<Int, 1, D> pose_ind_for_rot,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 1, D> n_rots_for_pose,
        TView<Int, 1, D> rot_offset_for_pose,
        TView<Int, 2, D> n_rots_for_block,
        TView<Int, 2, D> rot_offset_for_block,
        Int max_n_rots_per_pose,
        TView<Int, 3, D> pose_stack_min_bond_separation,
        TView<Int, 5, D> pose_stack_inter_block_bondsep,
        TView<Int, 1, D> block_type_n_atoms,
        TView<Int, 2, D> block_type_atom_types,
        TView<Int, 1, D> block_type_n_interblock_bonds,
        TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
        TView<Int, 3, D> block_type_ljlk_path_distance,
        TView<Int, 1, D> block_type_is_ligand_fragment,
        TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
        TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
        TView<Real, 2, D> block_type_partial_charge,
        TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
        TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
        TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
            elec_global_params,
        TView<Int, 1, D> shared_compact_block_neighbors,
        TView<Real, 1, D> score_weights,
        bool require_gradient) -> std::
        tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>> {
#define TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS                                  \
  mgr, rot_coords, rot_coord_offset, pose_ind_for_atom, first_rot_for_block,  \
      first_rot_block_type, block_ind_for_rot, pose_ind_for_rot,              \
      block_type_ind_for_rot, n_rots_for_pose, rot_offset_for_pose,           \
      n_rots_for_block, rot_offset_for_block, max_n_rots_per_pose,            \
      pose_stack_min_bond_separation, pose_stack_inter_block_bondsep,         \
      block_type_n_atoms, block_type_atom_types,                              \
      block_type_n_interblock_bonds, block_type_atoms_forming_chemical_bonds, \
      block_type_ljlk_path_distance, block_type_is_ligand_fragment,           \
      ljlk_type_params, ljlk_global_params, block_type_partial_charge,        \
      block_type_elec_inter_repr_path_distance,                               \
      block_type_elec_intra_repr_path_distance, elec_global_params,           \
      shared_compact_block_neighbors
  auto empty_rotamer_dispatch = TPack<Int, 2, D>::empty({0, 0});
  auto empty_output_gradients = TPack<Real, 1, D>::empty({0});
  if (require_gradient) {
    return ljlk_elec_forward_impl<
        true,
        true,
        false,
        DeviceOperations,
        D,
        Real,
        Int>(
        TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS,
        empty_rotamer_dispatch.view,
        score_weights,
        empty_output_gradients.view);
  }
  return ljlk_elec_forward_impl<
      false,
      true,
      false,
      DeviceOperations,
      D,
      Real,
      Int>(
      TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS,
      empty_rotamer_dispatch.view,
      score_weights,
      empty_output_gradients.view);
#undef TMOL_LJLK_ELEC_WEIGHTED_FORWARD_ARGS
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    forward_weighted_rotamers(
        ContextManager& mgr,
        TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 1, D> pose_ind_for_atom,
        TView<Int, 2, D> first_rot_for_block,
        TView<Int, 2, D> first_rot_block_type,
        TView<Int, 1, D> block_ind_for_rot,
        TView<Int, 1, D> pose_ind_for_rot,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 1, D> n_rots_for_pose,
        TView<Int, 1, D> rot_offset_for_pose,
        TView<Int, 2, D> n_rots_for_block,
        TView<Int, 2, D> rot_offset_for_block,
        Int max_n_rots_per_pose,
        TView<Int, 3, D> pose_stack_min_bond_separation,
        TView<Int, 5, D> pose_stack_inter_block_bondsep,
        TView<Int, 1, D> block_type_n_atoms,
        TView<Int, 2, D> block_type_atom_types,
        TView<Int, 1, D> block_type_n_interblock_bonds,
        TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
        TView<Int, 3, D> block_type_ljlk_path_distance,
        TView<Int, 1, D> block_type_is_ligand_fragment,
        TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
        TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
        TView<Real, 2, D> block_type_partial_charge,
        TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
        TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
        TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
            elec_global_params,
        Real max_dis,
        TView<Real, 1, D> score_weights,
        TView<Real, 1, D> output_gradients,
        TPack<Int, 2, D> shared_dispatch_indices)
        -> std::tuple<
            TPack<Real, 4, D>,
            TPack<LJLKExternalVec<Real, 3>, 2, D>,
            TPack<Int, 2, D>> {
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  TPack<Int, 2, D> rotamer_dispatch_indices;
  if (shared_dispatch_indices.size(0) == 3) {
    rotamer_dispatch_indices = shared_dispatch_indices;
  } else {
    auto scratch_rot_spheres_t = D == Device::CPU
                                     ? TPack<Real, 2, D>::zeros({n_rots, 4})
                                     : TPack<Real, 2, D>::empty({n_rots, 4});
    auto scratch_block_spheres_t =
        D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                         : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
    auto scratch_block_neighbors_t =
        D == Device::CPU
            ? TPack<Int, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks})
            : TPack<Int, 3, D>::empty({n_poses, max_n_blocks, max_n_blocks});
    score::common::sphere_overlap::
        compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres_t.view);
    score::common::sphere_overlap::
        compute_block_spheres_from_rot_spheres<DeviceOperations, D, Real, Int>::
            f(mgr,
              scratch_rot_spheres_t.view,
              n_rots_for_block,
              rot_offset_for_block,
              scratch_block_spheres_t.view);
    score::common::sphere_overlap::
        detect_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_block_type,
            scratch_block_spheres_t.view,
            scratch_block_neighbors_t.view,
            max_dis);
    rotamer_dispatch_indices = score::common::sphere_overlap::
        rot_neighbor_indices_from_block_neighbors<
            DeviceOperations,
            D,
            Real,
            Int>::
            f(mgr,
              scratch_block_neighbors_t.view,
              n_rots_for_block,
              rot_offset_for_block,
              scratch_rot_spheres_t.view,
              max_dis);
  }
  assert(
      output_gradients.size(0) == 0
      || output_gradients.size(0) == rotamer_dispatch_indices.view.size(1));
  auto empty_compact_block_neighbors = TPack<Int, 1, D>::empty({0});
#define TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS                                  \
  mgr, rot_coords, rot_coord_offset, pose_ind_for_atom, first_rot_for_block,  \
      first_rot_block_type, block_ind_for_rot, pose_ind_for_rot,              \
      block_type_ind_for_rot, n_rots_for_pose, rot_offset_for_pose,           \
      n_rots_for_block, rot_offset_for_block, max_n_rots_per_pose,            \
      pose_stack_min_bond_separation, pose_stack_inter_block_bondsep,         \
      block_type_n_atoms, block_type_atom_types,                              \
      block_type_n_interblock_bonds, block_type_atoms_forming_chemical_bonds, \
      block_type_ljlk_path_distance, block_type_is_ligand_fragment,           \
      ljlk_type_params, ljlk_global_params, block_type_partial_charge,        \
      block_type_elec_inter_repr_path_distance,                               \
      block_type_elec_intra_repr_path_distance, elec_global_params,           \
      empty_compact_block_neighbors.view, rotamer_dispatch_indices.view,      \
      score_weights, output_gradients
  if (shared_dispatch_indices.size(0) == 3) {
    auto result = ljlk_elec_forward_impl<
        true,
        true,
        true,
        DeviceOperations,
        D,
        Real,
        Int>(TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS);
    return {std::get<0>(result), std::get<1>(result), rotamer_dispatch_indices};
  }
  auto result =
      ljlk_elec_forward_impl<false, true, true, DeviceOperations, D, Real, Int>(
          TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS);
#undef TMOL_LJLK_ELEC_WEIGHTED_ROTAMER_ARGS
  return {std::get<0>(result), std::get<1>(result), rotamer_dispatch_indices};
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    reduce_weighted_scores(
        ContextManager& mgr,
        TView<Real, 2, D> score_lanes,
        TView<Real, 2, D> score_weights,
        Int fused_weight_begin,
        Int fused_weight_width) -> TPack<Real, 1, D> {
  int const n_score_lanes = score_lanes.size(0);
  int const n_poses = score_lanes.size(1);
  auto output_t = TPack<Real, 1, D>::empty({n_poses});
  auto output = output_t.view;
  LAUNCH_BOX_32_OCC_AS(launch_t, 32);
  DeviceOperations<D>::template forall<launch_t>(
      mgr, n_poses, [=] TMOL_DEVICE_FUNC(int pose) {
        Real total = 0;
        for (int lane = 0; lane < n_score_lanes; ++lane) {
          if (lane == fused_weight_begin) {
            total += score_lanes[lane][pose];
            continue;
          }
          int weight_index = lane;
          if (lane > fused_weight_begin) {
            weight_index += fused_weight_width - 1;
          }
          total += score_weights[weight_index][0] * score_lanes[lane][pose];
        }
        output[pose] = total;
      });
  return output_t;
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKAndElecPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    reduce_weighted_score_gradients(
        ContextManager& mgr,
        TView<Real, 1, D> output_gradient,
        TView<Real, 2, D> score_weights,
        Int n_score_lanes,
        Int fused_weight_begin,
        Int fused_weight_width) -> TPack<Real, 2, D> {
  int const n_poses = output_gradient.size(0);
  auto lane_gradients_t = TPack<Real, 2, D>::empty({n_score_lanes, n_poses});
  auto lane_gradients = lane_gradients_t.view;
  LAUNCH_BOX_32_OCC_AS(launch_t, 32);
  DeviceOperations<D>::template forall<launch_t>(
      mgr, n_score_lanes * n_poses, [=] TMOL_DEVICE_FUNC(int index) {
        int const lane = index / n_poses;
        int const pose = index - lane * n_poses;
        Real multiplier = 1;
        if (lane != fused_weight_begin) {
          int weight_index = lane;
          if (lane > fused_weight_begin) {
            weight_index += fused_weight_width - 1;
          }
          multiplier = score_weights[weight_index][0];
        }
        lane_gradients[lane][pose] = multiplier * output_gradient[pose];
      });
  return lane_gradients_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
