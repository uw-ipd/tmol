#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <tmol/score/hbond/potentials/gen_hbond_bases.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_ctx_tensor.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_kernels.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_mps_launch.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto GenerateHBondBases<DeviceOps, Dev, Real, Int>::forward(
    ContextManager& mgr,
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

  // State the kernel body reads, named explicitly rather than captured
  // implicitly by [=]; the Metal backend has nowhere for a closure to live.
  HBondBasesCtx<Real, Int, Dev> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.first_rot_for_block = first_rot_for_block;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.pose_stack_inter_residue_connections =
      pose_stack_inter_residue_connections;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_n_all_bonds = block_type_n_all_bonds;
  ctx.block_type_all_bonds = block_type_all_bonds;
  ctx.block_type_atom_all_bond_ranges = block_type_atom_all_bond_ranges;
  ctx.block_type_atom_is_hydrogen = block_type_atom_is_hydrogen;
  ctx.block_type_tile_n_donH = block_type_tile_n_donH;
  ctx.block_type_tile_n_acc = block_type_tile_n_acc;
  ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
  ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
  ctx.block_type_tile_hybridization = block_type_tile_hybridization;
  ctx.derived_coords = derived_coords;
  ctx.derived_atom_inds = derived_atom_inds;

  auto f_basesgen = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
    SHARED_MEMORY HBondBasesGenSharedData<Real, TILE_SIZE> shared_m;
    hbond_bases_gen<DeviceOps, Dev, Real, Int, launch_t>(
        ctx, shared_m, rot_ind);
  });

  if (Dev == tmol::Device::MPS) {
    // Metal cannot launch a device lambda, so the identical body is reached
    // through a named entry instead. Each rotamer writes only its own atoms'
    // rows, so there is nothing to make deterministic.
    hbond_bases_mps_launch_if<Dev>("hbond_bases_gen_f32", ctx, int64_t(n_rots));
  } else {
    DeviceOps<Dev>::template foreach_workgroup<launch_t>(
        mgr, n_rots, f_basesgen);
  }
  nvtx_range_pop();

  return {derived_coords_t, derived_atom_inds_t};
}

#undef HBOND_BASES_GEN_TILE_SIZE

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
