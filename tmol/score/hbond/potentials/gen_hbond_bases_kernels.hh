#pragma once

// The derived-atom pre-pass kernel body.
//
// Separated from gen_hbond_bases.impl.hh -- which holds the dispatch entry
// point and therefore reaches the torch tensor layer -- so that a Metal
// translation unit can include just the kernel. The bond walk it drives
// (bonded_atom.hh, identification.hh) was already view-generic; this header
// supplies the surrounding tile machinery in the same form.

#include <tmol/score/bonded_atom.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/hbond/identification.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_ctx.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/metal_view.hh>
#include <tmol/utility/portable/qualifiers.hh>

#define HBOND_BASES_GEN_TILE_SIZE 32

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::portable::Vec;
using tmol::score::common::WorkgroupLane;

// Per-residue data carried in stack registers while a workgroup processes one
// rotamer. The tile pointers address the shared block below, so they carry the
// shared address space on backends that have one; see metal_view.hh.
template <typename Real>
class HBondBasesGenSingleResData {
 public:
  int rot_ind;
  int block_ind;
  int block_type;
  int rot_coord_offset;
  int n_atoms;
  int n_conn;
  TMOL_SHARED_PTR(Real) coords;
  unsigned char n_donH;
  unsigned char n_acc;
  TMOL_SHARED_PTR(unsigned char) donH_tile_inds;
  TMOL_SHARED_PTR(unsigned char) acc_tile_inds;
  TMOL_SHARED_PTR(unsigned char) acc_hybridization;
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
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    int nt,
    typename Real,
    typename Ctx>
void TMOL_DEVICE_FUNC hbond_bases_gen_load_block_coords_and_params_into_shared(
    TMOL_THREAD Ctx const& ctx,
    int tile_ind,
    TMOL_THREAD HBondBasesGenSingleResData<Real>& r_dat,
    int n_atoms_to_load,
    int start_atom,
    // Trailing lane handle: empty and omissible on CPU and CUDA, required by
    // the Metal backend. See workgroup_lane.hh.
    WorkgroupLane<Dev> lane = {}) {
  r_dat.n_donH = ctx.block_type_tile_n_donH[r_dat.block_type][tile_ind];
  r_dat.n_acc = ctx.block_type_tile_n_acc[r_dat.block_type][tile_ind];

  DeviceOps<Dev>::template copy_contiguous_data<nt, 3>(
      r_dat.coords,
      reinterpret_cast<TMOL_DEVICE_PTR(Real)>(
          &ctx.rot_coords[r_dat.rot_coord_offset + start_atom]),
      n_atoms_to_load * 3,
      lane);
  DeviceOps<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.donH_tile_inds,
      &ctx.block_type_tile_donH_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_donH,
      lane);
  DeviceOps<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_tile_inds,
      &ctx.block_type_tile_acc_inds[r_dat.block_type][tile_ind][0],
      r_dat.n_acc,
      lane);
  DeviceOps<Dev>::template copy_contiguous_data_and_cast<nt, 1>(
      r_dat.acc_hybridization,
      &ctx.block_type_tile_hybridization[r_dat.block_type][tile_ind][0],
      r_dat.n_acc,
      lane);
}

// Resolve a BlockCentricAtom to its global pose-atom index -- the same index
// that addresses rot_coords and dV_dcoords. Returns -1 if bcat is invalid.
template <typename Real, typename Int, typename Ctx>
TMOL_DEVICE_FUNC int hbond_bases_gen_pose_atom_ind(
    TMOL_THREAD bonded_atom::BlockCentricAtom<Int> const& bcat,
    TMOL_THREAD HBondBasesGenSingleResData<Real> const& r_dat,
    TMOL_THREAD Ctx const& ctx,
    int pose_ind) {
  if (bcat.atom < 0) {
    return -1;
  }
  int coord_offset =
      (bcat.block == r_dat.block_ind
           ? r_dat.rot_coord_offset
           : ctx.rot_coord_offset[ctx.first_rot_for_block[pose_ind]
                                                         [bcat.block]]);
  return coord_offset + bcat.atom;
}

// Resolve a BlockCentricAtom to its coordinate, preferring the shared in-tile
// coords array when possible and falling back to global memory otherwise.
// Mirrors load_coord in lk_ball/water.hh.
template <int TILE_SIZE, typename Real, typename Int, typename Ctx>
TMOL_DEVICE_FUNC Vec<Real, 3> hbond_bases_gen_load_coord(
    TMOL_THREAD bonded_atom::BlockCentricAtom<Int> const& bcat,
    TMOL_THREAD HBondBasesGenSingleResData<Real> const& r_dat,
    TMOL_THREAD Ctx const& ctx,
    int pose_ind,
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
           : ctx.rot_coord_offset[ctx.first_rot_for_block[pose_ind]
                                                         [bcat.block]]);
  return ctx.rot_coords[bcat.atom + coord_offset];
}

// The bond graph as the walk sees it, rebuilt from the context.
//
// A local rather than a member of the per-residue data: it holds const
// members, so it cannot be assigned after construction, and spelling it here
// keeps the view type out of the data struct.
template <typename Int, typename Ctx>
TMOL_DEVICE_FUNC
    bonded_atom::RotamerCentricIndexedBonds<Int, Ctx::template ViewT>
    hbond_bases_gen_bonds(
        TMOL_THREAD Ctx const& ctx,
        int pose_ind,
        int block_ind,
        int block_type) {
  return bonded_atom::RotamerCentricIndexedBonds<Int, Ctx::template ViewT>{
      block_ind,
      block_type,
      pose_ind,
      ctx.pose_stack_inter_residue_connections,
      ctx.first_rot_block_type,
      ctx.block_type_n_all_bonds,
      ctx.block_type_all_bonds,
      ctx.block_type_atom_all_bond_ranges,
      ctx.block_type_atoms_forming_chemical_bonds};
}

// Build the donor base entry for the donH at intra-tile index don_h_ind.
template <int TILE_SIZE, typename Real, typename Int, typename Ctx>
void TMOL_DEVICE_FUNC build_donor_base(
    TMOL_THREAD Ctx const& ctx,
    TMOL_THREAD HBondBasesGenSingleResData<Real> const& r_dat,
    int pose_ind,
    int tile_start,
    int don_h_ind  // [0..n_donH)
) {
  using bonded_atom::BlockCentricAtom;

  int const don_h_atom_tile_ind = r_dat.donH_tile_inds[don_h_ind];
  int const don_h_atom_in_block = tile_start + don_h_atom_tile_ind;

  BlockCentricAtom<Int> H{
      r_dat.block_ind, r_dat.block_type, don_h_atom_in_block};
  auto bonds = hbond_bases_gen_bonds<Int>(
      ctx, pose_ind, r_dat.block_ind, r_dat.block_type);
  auto don_bases = RotamerCentricDonorBase<Int>::for_polar_H(
      H, bonds, ctx.block_type_atom_is_hydrogen);

  Vec<Real, 3> Dxyz = hbond_bases_gen_load_coord<TILE_SIZE>(
      don_bases.D, r_dat, ctx, pose_ind, tile_start);
  int const D_pose_atom_ind =
      hbond_bases_gen_pose_atom_ind(don_bases.D, r_dat, ctx, pose_ind);

  int const H_pose_atom_ind = r_dat.rot_coord_offset + don_h_atom_in_block;
  ctx.derived_coords[H_pose_atom_ind][0] = Dxyz;
  ctx.derived_atom_inds[H_pose_atom_ind][0] = D_pose_atom_ind;
}

// Build the two acceptor base entries (B, B0) for the acceptor at intra-tile
// index acc_ind.
template <int TILE_SIZE, typename Real, typename Int, typename Ctx>
void TMOL_DEVICE_FUNC build_acceptor_bases(
    TMOL_THREAD Ctx const& ctx,
    TMOL_THREAD HBondBasesGenSingleResData<Real> const& r_dat,
    int pose_ind,
    int tile_start,
    int acc_ind  // [0..n_acc)
) {
  using bonded_atom::BlockCentricAtom;

  int const acc_atom_tile_ind = r_dat.acc_tile_inds[acc_ind];
  int const acc_atom_in_block = tile_start + acc_atom_tile_ind;

  BlockCentricAtom<Int> A{r_dat.block_ind, r_dat.block_type, acc_atom_in_block};
  auto bonds = hbond_bases_gen_bonds<Int>(
      ctx, pose_ind, r_dat.block_ind, r_dat.block_type);
  auto acc_bases = RotamerCentricAcceptorBases<Int>::for_acceptor(
      A,
      r_dat.acc_hybridization[acc_ind],
      bonds,
      ctx.block_type_atom_is_hydrogen);

  Vec<Real, 3> Bxyz = hbond_bases_gen_load_coord<TILE_SIZE>(
      acc_bases.B, r_dat, ctx, pose_ind, tile_start);
  Vec<Real, 3> B0xyz = hbond_bases_gen_load_coord<TILE_SIZE>(
      acc_bases.B0, r_dat, ctx, pose_ind, tile_start);
  int const B_pose_atom_ind =
      hbond_bases_gen_pose_atom_ind(acc_bases.B, r_dat, ctx, pose_ind);
  int const B0_pose_atom_ind =
      hbond_bases_gen_pose_atom_ind(acc_bases.B0, r_dat, ctx, pose_ind);

  int const A_pose_atom_ind = r_dat.rot_coord_offset + acc_atom_in_block;
  ctx.derived_coords[A_pose_atom_ind][1] = Bxyz;
  ctx.derived_coords[A_pose_atom_ind][2] = B0xyz;
  ctx.derived_atom_inds[A_pose_atom_ind][1] = B_pose_atom_ind;
  ctx.derived_atom_inds[A_pose_atom_ind][2] = B0_pose_atom_ind;
}

// One workgroup per rotamer: tile the rotamer's atoms, and for each tile
// resolve the donor and acceptor bases of the polar atoms it holds.
//
// Extracted from a device lambda so that a backend whose entry points cannot
// be closures -- Metal's `kernel void` functions -- can call the identical
// body. Each rotamer writes only its own atoms' rows of derived_coords and
// derived_atom_inds, so there are no atomics.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void hbond_bases_gen(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared_m,
    int rot_ind,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  constexpr int TILE_SIZE = HBOND_BASES_GEN_TILE_SIZE;

  int const pose_ind = ctx.pose_ind_for_rot[rot_ind];
  int const block_type = ctx.block_type_ind_for_rot[rot_ind];
  int const block_ind = ctx.block_ind_for_rot[rot_ind];
  if (block_type == -1) {
    return;
  }

  int const n_atoms_in_block = ctx.block_type_n_atoms[block_type];

  HBondBasesGenSingleResData<Real> r_dat;
  r_dat.rot_ind = rot_ind;
  r_dat.block_ind = block_ind;
  r_dat.block_type = block_type;
  r_dat.rot_coord_offset = ctx.rot_coord_offset[rot_ind];
  r_dat.n_atoms = n_atoms_in_block;
  r_dat.n_conn = ctx.block_type_n_interblock_bonds[block_type];
  r_dat.coords = shared_m.coords;
  r_dat.donH_tile_inds = shared_m.donH_tile_inds;
  r_dat.acc_tile_inds = shared_m.acc_tile_inds;
  r_dat.acc_hybridization = shared_m.acc_hybridization;

  int const n_iterations = (n_atoms_in_block - 1) / TILE_SIZE + 1;
  for (int tile_ind = 0; tile_ind < n_iterations; ++tile_ind) {
    int const n_atoms_to_load =
        min(TILE_SIZE, n_atoms_in_block - TILE_SIZE * tile_ind);

    if (tile_ind != 0) {
      DeviceOps<Dev>::synchronize_workgroup(lane);
    }

    hbond_bases_gen_load_block_coords_and_params_into_shared<
        DeviceOps,
        Dev,
        nt,
        Real>(
        ctx, tile_ind, r_dat, n_atoms_to_load, tile_ind * TILE_SIZE, lane);
    DeviceOps<Dev>::synchronize_workgroup(lane);

    auto gen_tile_bases = ([=] TMOL_DEVICE_FUNC(int tid) {
      int const n_items = r_dat.n_donH + r_dat.n_acc;
      for (int i = tid; i < n_items; i += nt) {
        if (i < r_dat.n_donH) {
          build_donor_base<TILE_SIZE, Real, Int>(
              ctx, r_dat, pose_ind, tile_ind * TILE_SIZE, i);
        } else {
          int const acc_i = i - r_dat.n_donH;
          build_acceptor_bases<TILE_SIZE, Real, Int>(
              ctx, r_dat, pose_ind, tile_ind * TILE_SIZE, acc_i);
        }
      }
    });
    DeviceOps<Dev>::template for_each_in_workgroup<nt>(gen_tile_bases, lane);
  }
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
