#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/grid_build.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

namespace grid_build_detail {

using tmol::score::ljlk::potentials::LJGlobalParams;
using tmol::score::ljlk::potentials::LJLKTypeParams;

// Grid type rows carry the raw well depth and no LK factors; the shared pair
// potentials consume the derived form.
template <tmol::Device D, typename Real>
EIGEN_DEVICE_FUNC LJLKTypeParams<Real> type_row(
    TView<Real, 2, D> const& rows, int64_t type) {
  return LJLKTypeParams<Real>{
      rows[type][0],
      std::sqrt(rows[type][1]),
      rows[type][2],
      rows[type][3],
      rows[type][4],
      rows[type][5],
      rows[type][6],
      rows[type][7],
      rows[type][8],
      rows[type][9],
      Real(0),  // is_hydrogen is unused by the grid pair potentials
      -rows[type][2] / (2 * Real(5.56832799683) * rows[type][3]),
      1 / (rows[type][3] * rows[type][3])};
}

template <typename Real>
EIGEN_DEVICE_FUNC Real ramp_fraction(Real delta, Real width) {
  if (delta < 0) return 1;
  if (delta >= width) return 0;
  Real const scaled = delta / width;
  Real const shoulder = 1 - scaled * scaled;
  return shoulder * shoulder;
}

template <tmol::Device D, typename Real>
EIGEN_DEVICE_FUNC bool finite_water(
    TView<Real, 3, D> const& waters, int64_t atom, int water) {
  return std::isfinite(waters[atom][water][0])
         && std::isfinite(waters[atom][water][1])
         && std::isfinite(waters[atom][water][2]);
}

template <tmol::Device D, typename Real>
EIGEN_DEVICE_FUNC bool has_water(
    TView<Real, 3, D> const& waters, int64_t atom) {
  for (int water = 0; water < 4; ++water) {
    if (finite_water<D, Real>(waters, atom, water)) return true;
  }
  return false;
}

// Smallest (squared distance to a finite water of `atom`) - offset, capped.
template <tmol::Device D, typename Real>
EIGEN_DEVICE_FUNC Real min_water_delta(
    TView<Real, 3, D> const& waters,
    int64_t atom,
    Real x,
    Real y,
    Real z,
    Real offset,
    Real cap) {
  Real best = cap;
  for (int water = 0; water < 4; ++water) {
    if (!finite_water<D, Real>(waters, atom, water)) continue;
    Real const dx = x - waters[atom][water][0];
    Real const dy = y - waters[atom][water][1];
    Real const dz = z - waters[atom][water][2];
    best = std::min(best, dx * dx + dy * dy + dz * dz - offset);
  }
  return best;
}

template <typename Real>
EIGEN_DEVICE_FUNC Real directional_lk(
    Real distance,
    LJLKTypeParams<Real> const& desolvated,
    LJLKTypeParams<Real> const& occluding,
    LJGlobalParams<Real> const& global) {
  Real const sigma = tmol::score::ljlk::potentials::lj_sigma<Real>(
      desolvated, occluding, global);
  return tmol::score::ljlk::potentials::lk_isotropic_pair<Real>::V(
      distance,
      Real(5),
      sigma,
      desolvated.lj_radius,
      occluding.lk_volume,
      desolvated.lk_coeff,
      desolvated.lk_inv_lambda2,
      global.max_dis,
      desolvated.is_carbon_lk && occluding.is_carbon_lk);
}

template <typename Real>
struct PairSums {
  Real atr, rep, sol, ball, water, bridge;
};

template <tmol::Device D, typename Real, typename Int>
EIGEN_DEVICE_FUNC void voxel_xyz(
    TView<Real, 1, D> const& origin,
    Real spacing,
    Int n_y,
    Int n_z,
    Int voxel,
    Real* xyz) {
  xyz[0] = origin[0] + spacing * (voxel / (n_y * n_z));
  xyz[1] = origin[1] + spacing * ((voxel / n_z) % n_y);
  xyz[2] = origin[2] + spacing * (voxel % n_z);
}

// Calls visit(atom) for every receptor atom in the 27 cells around (x, y, z).
template <tmol::Device D, typename Real, typename Int, typename Visit>
EIGEN_DEVICE_FUNC void for_neighbors(
    Real x,
    Real y,
    Real z,
    Real cell_width,
    TView<Real, 1, D> const& cell_origin,
    TView<Int, 1, D> const& cell_dims,
    TView<Int, 1, D> const& cell_offsets,
    TView<Int, 1, D> const& cell_indices,
    Visit visit) {
  int64_t const cx = std::floor((x - cell_origin[0]) / cell_width);
  int64_t const cy = std::floor((y - cell_origin[1]) / cell_width);
  int64_t const cz = std::floor((z - cell_origin[2]) / cell_width);
  for (int64_t i = std::max<int64_t>(cx - 1, 0);
       i <= std::min<int64_t>(cx + 1, cell_dims[0] - 1);
       ++i) {
    for (int64_t j = std::max<int64_t>(cy - 1, 0);
         j <= std::min<int64_t>(cy + 1, cell_dims[1] - 1);
         ++j) {
      for (int64_t k = std::max<int64_t>(cz - 1, 0);
           k <= std::min<int64_t>(cz + 1, cell_dims[2] - 1);
           ++k) {
        int64_t const cell = (i * cell_dims[1] + j) * cell_dims[2] + k;
        for (int64_t slot = cell_offsets[cell]; slot < cell_offsets[cell + 1];
             ++slot) {
          visit(cell_indices[slot]);
        }
      }
    }
  }
}

}  // namespace grid_build_detail

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
void GridBuildDispatch<DeviceOps, D, Real, Int>::build(
    ContextManager& mgr,
    TView<Real, 2, D> receptor_coords,
    TView<Int, 1, D> receptor_types,
    TView<Real, 1, D> receptor_charges,
    TView<Real, 3, D> receptor_waters,
    TView<Int, 1, D> probe_types,
    TView<Real, 2, D> type_params,
    tmol::score::ljlk::potentials::LJGlobalParams<Real> lj_global,
    tmol::score::elec::potentials::ElecGlobalParams<Real> elec_global,
    GridBuildLKWeights<Real> lk_weights,
    TView<Int, 1, D> cell_offsets,
    TView<Int, 1, D> cell_indices,
    TView<Real, 1, D> cell_origin,
    TView<Int, 1, D> cell_dims,
    TView<Real, 1, D> grid_origin,
    Int n_x,
    Int n_y,
    Int n_z,
    Real voxel_spacing,
    bool build_lk_ball,
    TView<Real, 2, D> atr,
    TView<Real, 2, D> rep,
    TView<Real, 2, D> sol,
    TView<Real, 2, D> lk_ball,
    TView<Real, 2, D> lk_water,
    TView<Real, 2, D> lk_bridge,
    TView<Real, 1, D> elec) {
  using namespace grid_build_detail;
  using tmol::score::ljlk::potentials::lj_score;
  using tmol::score::ljlk::potentials::lk_isotropic_score;

  Int const n_voxels = n_x * n_y * n_z;
  Int const n_probes = probe_types.size(0);
  Real const max_dis = std::max(lj_global.max_dis, elec_global.max_dis);
  Real const lj_max2 = lj_global.max_dis * lj_global.max_dis;
  Real const elec_max2 = elec_global.max_dis * elec_global.max_dis;
  // Rosetta GridScorer bakes the LK-ball weights into three non-additive grid
  // families; the hard fractions use the grid point itself (water_xyz).
  constexpr Real heavy_water = 2.65;
  constexpr Real ramp_width = 3.709;
  constexpr Real overlap_gap = 0.5;
  constexpr Real overlap_width = 2.6;
  Real const water_water = heavy_water * 4 * Real(1.4142135623730951) / 3;

  LAUNCH_BOX_32;

  // Per-probe running sums over one voxel's receptor neighbors.
  auto accumulate = [=] TMOL_DEVICE_FUNC(
                        LJLKTypeParams<Real> const& probe_params,
                        LJLKTypeParams<Real> const& receptor,
                        Int atom,
                        bool has_waters,
                        Real dist2,
                        Real x,
                        Real y,
                        Real z,
                        PairSums<Real>& sums) {
    Real const dist = std::sqrt(dist2);
    auto const lj = lj_score<Real>::V(
        dist,
        Real(5),
        probe_params.lj_params(),
        receptor.lj_params(),
        lj_global);
    sums.atr += lj[0];
    sums.rep += lj[1];
    sums.sol += lk_isotropic_score<Real>::V(
        dist,
        Real(5),
        probe_params.lk_params(),
        receptor.lk_params(),
        lj_global);
    if (!build_lk_ball) return;

    Real const fasol1 = directional_lk(dist, probe_params, receptor, lj_global);
    Real const fasol2 = directional_lk(dist, receptor, probe_params, lj_global);
    if (has_waters) {
      Real const d2_low = std::max(
          Real(0),
          (Real(1.4) + probe_params.lj_radius)
                  * (Real(1.4) + probe_params.lj_radius)
              - ramp_width);
      Real const hard_fraction = ramp_fraction<Real>(
          min_water_delta<D, Real>(
              receptor_waters, atom, x, y, z, d2_low, ramp_width),
          ramp_width);
      sums.ball +=
          lk_weights.ball * fasol2 * hard_fraction + lk_weights.iso * fasol2;
    }
    if (probe_params.is_donor == 0 && probe_params.is_acceptor == 0) return;

    sums.ball += lk_weights.iso * fasol1;
    Real const fasol1_water =
        directional_lk(heavy_water, probe_params, receptor, lj_global);
    Real const receptor_d2_low = std::max(
        Real(0),
        (Real(1.4) + receptor.lj_radius) * (Real(1.4) + receptor.lj_radius)
            - ramp_width);
    sums.water += lk_weights.ball * fasol1_water
                  * ramp_fraction<Real>(dist2 - receptor_d2_low, ramp_width);

    if (!has_waters
        || (lk_weights.bridge == 0 && lk_weights.bridge_uncoupled == 0)) {
      return;
    }
    Real const bridge_fraction = ramp_fraction<Real>(
        min_water_delta<D, Real>(
            receptor_waters, atom, x, y, z, overlap_gap, overlap_width),
        overlap_width);
    sums.bridge +=
        (lk_weights.bridge
             * (directional_lk(water_water, probe_params, receptor, lj_global)
                + directional_lk(
                    water_water, receptor, probe_params, lj_global))
         + lk_weights.bridge_uncoupled)
        * bridge_fraction;
  };

  auto store = [=] TMOL_DEVICE_FUNC(
                   Int probe, Int voxel, PairSums<Real> const& sums) {
    atr[probe][voxel] = sums.atr;
    rep[probe][voxel] = std::pow(sums.rep + 1, Real(-0.25));
    sol[probe][voxel] = sums.sol;
    if (build_lk_ball) {
      lk_ball[probe][voxel] = sums.ball;
      lk_water[probe][voxel] = sums.water;
      lk_bridge[probe][voxel] = sums.bridge;
    }
  };

  // One work item per (voxel, tile of probes): the receptor neighbors of a
  // voxel (and their distances) are shared by the probes of a tile.
  constexpr Int tile_size = 8;
  Int const n_tiles = (n_probes + tile_size - 1) / tile_size;
  auto tile_sums = [=] TMOL_DEVICE_FUNC(int work) {
    Int const voxel = work / n_tiles;
    Int const first = (work % n_tiles) * tile_size;
    Int const count = std::min<Int>(tile_size, n_probes - first);
    Real xyz[3];
    voxel_xyz<D, Real, Int>(grid_origin, voxel_spacing, n_y, n_z, voxel, xyz);
    LJLKTypeParams<Real> probes[tile_size];
    PairSums<Real> sums[tile_size] = {};
    for (Int p = 0; p < count; ++p) {
      probes[p] = type_row<D, Real>(type_params, probe_types[first + p]);
    }
    Real elec_sum = 0;
    for_neighbors<D, Real, Int>(
        xyz[0],
        xyz[1],
        xyz[2],
        max_dis,
        cell_origin,
        cell_dims,
        cell_offsets,
        cell_indices,
        [&](Int atom) {
          Real const dx = xyz[0] - receptor_coords[atom][0];
          Real const dy = xyz[1] - receptor_coords[atom][1];
          Real const dz = xyz[2] - receptor_coords[atom][2];
          Real const dist2 = dx * dx + dy * dy + dz * dz;
          if (first == 0 && dist2 <= elec_max2) {
            elec_sum += tmol::score::elec::potentials::elec<Real>(
                std::sqrt(dist2),
                Real(1),
                receptor_charges[atom],
                Real(5),
                elec_global);
          }
          if (dist2 > lj_max2) return;
          auto const receptor =
              type_row<D, Real>(type_params, receptor_types[atom]);
          bool const has_waters =
              build_lk_ball && has_water<D, Real>(receptor_waters, atom);
          for (Int p = 0; p < count; ++p) {
            accumulate(
                probes[p],
                receptor,
                atom,
                has_waters,
                dist2,
                xyz[0],
                xyz[1],
                xyz[2],
                sums[p]);
          }
        });
    if (first == 0) elec[voxel] = elec_sum;
    for (Int p = 0; p < count; ++p) {
      store(first + p, voxel, sums[p]);
    }
  };
  DeviceOps<D>::template forall_independent<launch_t>(
      mgr, n_voxels * n_tiles, tile_sums);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
