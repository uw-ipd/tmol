#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/bonded_atom.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <tmol/score/hbond/identification.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

#define HBOND_BASES_GEN_TILE_SIZE 32

// Per-residue data carried in stack registers while a workgroup processes
// one rotamer.  Shared-memory pointers (coords, *_tile_inds,
// acc_hybridization) point into HBondBasesGenSharedData below.
template <typename Real>
class HBondBasesGenSingleResData {
 public:
  int rot_ind;
  int block_ind;
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  Real* coords;
  unsigned char n_donH;
  unsigned char n_acc;
  unsigned char* donH_tile_inds;
  unsigned char* acc_tile_inds;
  unsigned char* acc_hybridization;
};

// Cross-residue context (TViews into global memory) needed when the bond
// walk for an acceptor base crosses a residue boundary.
template <tmol::Device Dev, typename Real, typename Int>
class HBondBasesGenPoseContextData {
 public:
  int pose_ind;

  TView<Vec<Real, 3>, 1, Dev> rot_coords;
  TView<Int, 1, Dev> rot_coord_offset;
  TView<Int, 1, Dev> block_type_ind_for_rot;

  TView<Int, 2, Dev> first_rot_for_block;
  TView<Int, 2, Dev> first_rot_block_type;

  TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections;

  TView<Int, 1, Dev> block_type_n_all_bonds;
  TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds;
  TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges;
  TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds;
  TView<Int, 2, Dev> block_type_atom_is_hydrogen;
};

template <tmol::Device Dev, typename Real, typename Int>
class HBondBasesGenData {
 public:
  HBondBasesGenSingleResData<Real> r_dat;
  HBondBasesGenPoseContextData<Dev, Real, Int> pose_context;
};

template <typename Real, int TILE_SIZE>
struct HBondBasesGenSharedData {
  Real coords[TILE_SIZE * 3];
  unsigned char n_donH;
  unsigned char n_acc;
  unsigned char donH_tile_inds[TILE_SIZE];
  unsigned char acc_tile_inds[TILE_SIZE];
  unsigned char acc_hybridization[TILE_SIZE];
};

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Int,
    typename Real,
    int TILE_SIZE>
void TMOL_DEVICE_FUNC hbond_bases_gen_load_tile_invariant_data(
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 1, Dev> block_type_ind_for_rot,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,
    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen,

    int pose_ind,
    int rot_ind,
    int block_ind,
    int block_type,
    int n_atoms,

    HBondBasesGenData<Dev, Real, Int>& dat,
    HBondBasesGenSharedData<Real, TILE_SIZE>& shared_m) {
  dat.pose_context.pose_ind = pose_ind;
  dat.r_dat.rot_ind = rot_ind;
  dat.r_dat.block_ind = block_ind;
  dat.r_dat.block_type = block_type;
  dat.r_dat.rot_coord_offset = rot_coord_offset[rot_ind];
  dat.r_dat.n_atoms = n_atoms;
  dat.r_dat.n_conn = block_type_n_interblock_bonds[block_type];

  dat.r_dat.coords = shared_m.coords;
  dat.r_dat.donH_tile_inds = shared_m.donH_tile_inds;
  dat.r_dat.acc_tile_inds = shared_m.acc_tile_inds;
  dat.r_dat.acc_hybridization = shared_m.acc_hybridization;

  dat.pose_context.rot_coords = rot_coords;
  dat.pose_context.first_rot_for_block = first_rot_for_block;
  dat.pose_context.first_rot_block_type = first_rot_block_type;
  dat.pose_context.rot_coord_offset = rot_coord_offset;
  dat.pose_context.block_type_ind_for_rot = block_type_ind_for_rot;
  dat.pose_context.pose_stack_inter_residue_connections =
      pose_stack_inter_residue_connections;
  dat.pose_context.block_type_n_all_bonds = block_type_n_all_bonds;
  dat.pose_context.block_type_all_bonds = block_type_all_bonds;
  dat.pose_context.block_type_atom_all_bond_ranges =
      block_type_atom_all_bond_ranges;
  dat.pose_context.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  dat.pose_context.block_type_atom_is_hydrogen = block_type_atom_is_hydrogen;
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    int nt,
    typename Real,
    typename Int>
void TMOL_DEVICE_FUNC hbond_bases_gen_load_block_coords_and_params_into_shared(
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    int tile_ind,
    HBondBasesGenSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom) {
  r_dat.n_donH = block_type_tile_n_donH[r_dat.block_type][tile_ind];
  r_dat.n_acc = block_type_tile_n_acc[r_dat.block_type][tile_ind];

  DeviceDispatch<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<Real*>(&rot_coords[r_dat.rot_coord_offset + start_atom]),
      n_atoms_to_load * 3);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.donH_tile_inds,
      &block_type_tile_donH_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_donH);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_tile_inds,
      &block_type_tile_acc_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
  DeviceDispatch<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_hybridization,
      &block_type_tile_hybridization[r_dat.block_type][tile_ind][0],
      r_dat.n_acc);
}

// Resolve a BlockCentricAtom to its global pose-atom index (the same
// index that addresses rot_coords / dV_dcoords).  Returns -1 if bcat
// is invalid.
template <tmol::Device Dev, typename Real, typename Int>
TMOL_DEVICE_FUNC int hbond_bases_gen_pose_atom_ind(
    bonded_atom::BlockCentricAtom<Int> const& bcat,
    HBondBasesGenSingleResData<Real> const& r_dat,
    HBondBasesGenPoseContextData<Dev, Real, Int> const& context_dat) {
  if (bcat.atom < 0) {
    return -1;
  }
  int coord_offset =
      (bcat.block == r_dat.block_ind
           ? r_dat.rot_coord_offset
           : context_dat
                 .rot_coord_offset[context_dat.first_rot_for_block
                                       [context_dat.pose_ind][bcat.block]]);
  return coord_offset + bcat.atom;
}

// Resolve a BlockCentricAtom to its coordinate, preferring the shared
// in-tile coords array when possible and falling back to global memory
// otherwise.  Mirrors load_coord in lk_ball/water.hh.
template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
TMOL_DEVICE_FUNC Vec<Real, 3> hbond_bases_gen_load_coord(
    bonded_atom::BlockCentricAtom<Int> const& bcat,
    HBondBasesGenSingleResData<Real> const& r_dat,
    HBondBasesGenPoseContextData<Dev, Real, Int> const& context_dat,
    int tile_start) {
  Vec<Real, 3> xyz{Real(0), Real(0), Real(0)};
  if (bcat.atom == -1) {
    return xyz;
  }
  if (bcat.block == r_dat.block_ind) {
    int bcat_tile_ind = bcat.atom - tile_start;
    if (bcat_tile_ind >= 0 && bcat_tile_ind < TILE_SIZE) {
      return common::coord_from_shared(r_dat.coords, bcat_tile_ind);
    }
  }
  int coord_offset =
      (bcat.block == r_dat.block_ind
           ? r_dat.rot_coord_offset
           : context_dat
                 .rot_coord_offset[context_dat.first_rot_for_block
                                       [context_dat.pose_ind][bcat.block]]);
  return context_dat.rot_coords[bcat.atom + coord_offset];
}

// Build the donor base entry for the donH at intra-tile index don_h_ind.
template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
void TMOL_DEVICE_FUNC build_donor_base(
    TView<Vec<Real, 3>, 2, Dev> derived_coords,
    TView<Int, 2, Dev> derived_atom_inds,
    HBondBasesGenData<Dev, Real, Int> const& dat,
    int tile_start,
    int don_h_ind  // [0..n_donH)
) {
  using bonded_atom::BlockCentricAtom;
  using bonded_atom::RotamerCentricIndexedBonds;

  auto const& r_dat = dat.r_dat;
  auto const& ctx = dat.pose_context;

  int const don_h_atom_tile_ind = r_dat.donH_tile_inds[don_h_ind];
  int const don_h_atom_in_block = tile_start + don_h_atom_tile_ind;

  BlockCentricAtom<Int> H{
      r_dat.block_ind, r_dat.block_type, don_h_atom_in_block};
  RotamerCentricIndexedBonds<Int, Dev> bonds{
      r_dat.block_ind,
      r_dat.block_type,
      ctx.pose_stack_inter_residue_connections[ctx.pose_ind],
      ctx.first_rot_block_type[ctx.pose_ind],
      ctx.block_type_n_all_bonds,
      ctx.block_type_all_bonds,
      ctx.block_type_atom_all_bond_ranges,
      ctx.block_type_atoms_forming_chemical_bonds};
  auto don_bases = RotamerCentricDonorBase<Int>::for_polar_H(
      H, bonds, ctx.block_type_atom_is_hydrogen);

  Vec<Real, 3> Dxyz = hbond_bases_gen_load_coord<TILE_SIZE>(
      don_bases.D, r_dat, ctx, tile_start);
  int const D_pose_atom_ind =
      hbond_bases_gen_pose_atom_ind(don_bases.D, r_dat, ctx);

  int const H_pose_atom_ind = r_dat.rot_coord_offset + don_h_atom_in_block;
  derived_coords[H_pose_atom_ind][0] = Dxyz;
  derived_atom_inds[H_pose_atom_ind][0] = D_pose_atom_ind;
}

// Build the two acceptor base entries (B, B0) for the acceptor at
// intra-tile index acc_ind.
template <int TILE_SIZE, tmol::Device Dev, typename Real, typename Int>
void TMOL_DEVICE_FUNC build_acceptor_bases(
    TView<Vec<Real, 3>, 2, Dev> derived_coords,
    TView<Int, 2, Dev> derived_atom_inds,
    HBondBasesGenData<Dev, Real, Int> const& dat,
    int tile_start,
    int acc_ind  // [0..n_acc)
) {
  using bonded_atom::BlockCentricAtom;
  using bonded_atom::RotamerCentricIndexedBonds;

  auto const& r_dat = dat.r_dat;
  auto const& ctx = dat.pose_context;

  int const acc_atom_tile_ind = r_dat.acc_tile_inds[acc_ind];
  int const acc_atom_in_block = tile_start + acc_atom_tile_ind;

  BlockCentricAtom<Int> A{r_dat.block_ind, r_dat.block_type, acc_atom_in_block};
  RotamerCentricIndexedBonds<Int, Dev> bonds{
      r_dat.block_ind,
      r_dat.block_type,
      ctx.pose_stack_inter_residue_connections[ctx.pose_ind],
      ctx.first_rot_block_type[ctx.pose_ind],
      ctx.block_type_n_all_bonds,
      ctx.block_type_all_bonds,
      ctx.block_type_atom_all_bond_ranges,
      ctx.block_type_atoms_forming_chemical_bonds};
  auto acc_bases = RotamerCentricAcceptorBases<Int>::for_acceptor(
      A,
      r_dat.acc_hybridization[acc_ind],
      bonds,
      ctx.block_type_atom_is_hydrogen);

  Vec<Real, 3> Bxyz = hbond_bases_gen_load_coord<TILE_SIZE>(
      acc_bases.B, r_dat, ctx, tile_start);
  Vec<Real, 3> B0xyz = hbond_bases_gen_load_coord<TILE_SIZE>(
      acc_bases.B0, r_dat, ctx, tile_start);
  int const B_pose_atom_ind =
      hbond_bases_gen_pose_atom_ind(acc_bases.B, r_dat, ctx);
  int const B0_pose_atom_ind =
      hbond_bases_gen_pose_atom_ind(acc_bases.B0, r_dat, ctx);

  int const A_pose_atom_ind = r_dat.rot_coord_offset + acc_atom_in_block;
  derived_coords[A_pose_atom_ind][1] = Bxyz;
  derived_coords[A_pose_atom_ind][2] = B0xyz;
  derived_atom_inds[A_pose_atom_ind][1] = B_pose_atom_ind;
  derived_atom_inds[A_pose_atom_ind][2] = B0_pose_atom_ind;
}

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto GenerateHBondBases<DeviceOps, Dev, Real, Int>::forward(
    TView<Vec<Real, 3>, 1, Dev> rot_coords,
    TView<Int, 1, Dev> rot_coord_offset,
    TView<Int, 2, Dev> first_rot_for_block,
    TView<Int, 2, Dev> first_rot_block_type,
    TView<Int, 1, Dev> block_ind_for_rot,
    TView<Int, 1, Dev> pose_ind_for_rot,
    TView<Int, 1, Dev> block_type_ind_for_rot,

    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

    TView<Int, 1, Dev> block_type_n_atoms,
    TView<Int, 1, Dev> block_type_n_interblock_bonds,
    TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

    TView<Int, 1, Dev> block_type_n_all_bonds,
    TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
    TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

    TView<Int, 2, Dev> block_type_tile_n_donH,
    TView<Int, 2, Dev> block_type_tile_n_acc,
    TView<Int, 3, Dev> block_type_tile_donH_inds,
    TView<Int, 3, Dev> block_type_tile_acc_inds,
    TView<Int, 3, Dev> block_type_tile_hybridization,
    TView<Int, 2, Dev> block_type_atom_is_hydrogen)
    -> std::tuple<TPack<Vec<Real, 3>, 2, Dev>, TPack<Int, 2, Dev> > {
  int const n_rots = rot_coord_offset.size(0);
  int const n_atoms = rot_coords.size(0);

  NVTXRange _function(__FUNCTION__);

  constexpr int TILE_SIZE = HBOND_BASES_GEN_TILE_SIZE;

  nvtx_range_push("hbond_bases::setup");
  auto derived_coords_t = TPack<Vec<Real, 3>, 2, Dev>::full({n_atoms, 3}, NAN);
  auto derived_coords = derived_coords_t.view;
  auto derived_atom_inds_t = TPack<Int, 2, Dev>::full({n_atoms, 3}, -1);
  auto derived_atom_inds = derived_atom_inds_t.view;
  nvtx_range_pop();

  nvtx_range_push("hbond_bases::gen");
  LAUNCH_BOX_32;
  CTA_LAUNCH_T_PARAMS;

  auto f_basesgen = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_type = block_type_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    if (block_type == -1) {
      return;
    }

    int const n_atoms_in_block = block_type_n_atoms[block_type];

    SHARED_MEMORY HBondBasesGenSharedData<Real, TILE_SIZE> shared_m;
    HBondBasesGenData<Dev, Real, Int> dat;

    hbond_bases_gen_load_tile_invariant_data<DeviceOps, Dev, nt>(
        rot_coords,
        first_rot_for_block,
        first_rot_block_type,
        rot_coord_offset,
        block_type_ind_for_rot,
        pose_stack_inter_residue_connections,
        block_type_n_all_bonds,
        block_type_all_bonds,
        block_type_atom_all_bond_ranges,
        block_type_n_interblock_bonds,
        block_type_atoms_forming_chemical_bonds,
        block_type_atom_is_hydrogen,
        pose_ind,
        rot_ind,
        block_ind,
        block_type,
        n_atoms_in_block,
        dat,
        shared_m);

    int const n_iterations = (n_atoms_in_block - 1) / TILE_SIZE + 1;
    for (int tile_ind = 0; tile_ind < n_iterations; ++tile_ind) {
      int const n_atoms_to_load =
          min(TILE_SIZE, n_atoms_in_block - TILE_SIZE * tile_ind);

      if (tile_ind != 0) {
        DeviceOps<Dev>::synchronize_workgroup();
      }

      hbond_bases_gen_load_block_coords_and_params_into_shared<
          DeviceOps,
          Dev,
          nt>(
          rot_coords,
          block_type_tile_n_donH,
          block_type_tile_n_acc,
          block_type_tile_donH_inds,
          block_type_tile_acc_inds,
          block_type_tile_hybridization,
          tile_ind,
          dat.r_dat,
          n_atoms_to_load,
          tile_ind * TILE_SIZE);
      DeviceOps<Dev>::synchronize_workgroup();

      auto gen_tile_bases = ([&] TMOL_DEVICE_FUNC(int tid) {
        int const n_items = dat.r_dat.n_donH + dat.r_dat.n_acc;
        for (int i = tid; i < n_items; i += nt) {
          if (i < dat.r_dat.n_donH) {
            build_donor_base<TILE_SIZE>(
                derived_coords,
                derived_atom_inds,
                dat,
                tile_ind * TILE_SIZE,
                i);
          } else {
            int const acc_i = i - dat.r_dat.n_donH;
            build_acceptor_bases<TILE_SIZE>(
                derived_coords,
                derived_atom_inds,
                dat,
                tile_ind * TILE_SIZE,
                acc_i);
          }
        }
      });
      DeviceOps<Dev>::template for_each_in_workgroup<nt>(gen_tile_bases);
    }
  });

  DeviceOps<Dev>::template foreach_workgroup<launch_t>(n_rots, f_basesgen);
  nvtx_range_pop();

  return {derived_coords_t, derived_atom_inds_t};
}

#undef HBOND_BASES_GEN_TILE_SIZE

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
