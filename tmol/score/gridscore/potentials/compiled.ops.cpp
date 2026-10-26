#include <tmol/utility/tensor/native_library.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/score/gridscore/potentials/grid_build.hh>
#include <tmol/score/gridscore/potentials/grid_interp_pose_score.hh>
#include <tmol/score/gridscore/potentials/grid_lk_ball.hh>
#include <tmol/score/gridscore/potentials/live_hbond.hh>
#include <tmol/score/gridscore/potentials/live_lk_ball.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

using tmol::native::Tensor;
using tmol::score::elec::potentials::deps_ddist;
using tmol::score::elec::potentials::elec;
using tmol::score::elec::potentials::ElecGlobalParams;
using tmol::score::elec::potentials::eps;
using tmol::score::ljlk::potentials::lj_score;
using tmol::score::ljlk::potentials::lj_sigma;
using tmol::score::ljlk::potentials::LJGlobalParams;
using tmol::score::ljlk::potentials::LJLKTypeParams;
using tmol::score::ljlk::potentials::lk_isotropic_pair;
using tmol::score::ljlk::potentials::lk_isotropic_score;

ContextManager interpolation_context;

namespace {

template <typename Real>
ElecGlobalParams<Real> load_elec_global_params(double const* g) {
  Real const D = g[0], D0 = g[1], S = g[2], min_dis = g[3], max_dis = g[4];
  Real const cutoff_offset =
      322.0637 / (max_dis * eps<Real>(max_dis, D, D0, S));
  Real const low_end = min_dis + 0.25;
  Real const low_eps = eps<Real>(low_end, D, D0, S);
  Real const high_start = max_dis - 1.0;
  Real const high_eps = eps<Real>(high_start, D, D0, S);
  return ElecGlobalParams<Real>{
      D,
      D0,
      S,
      min_dis,
      max_dis,
      cutoff_offset,
      322.0637 / (min_dis * eps<Real>(min_dis, D, D0, S)) - cutoff_offset,
      322.0637 / (low_end * low_eps) - cutoff_offset,
      -322.0637 * (low_eps + low_end * deps_ddist<Real>(low_end, D, D0, S))
          / (low_end * low_end * low_eps * low_eps),
      322.0637 / (high_start * high_eps) - cutoff_offset,
      -322.0637
          * (high_eps + high_start * deps_ddist<Real>(high_start, D, D0, S))
          / (high_start * high_start * high_eps * high_eps)};
}

}  // namespace

// Receptor-vs-probe grids for every probe atom type, on the device of the
// receptor tensors. Outputs are flattened over space: [probe, nx*ny*nz] and
// [nx*ny*nz]. Cutoff-wide cells (a CSR cell list over the receptor atoms)
// bound the neighbor search; host-side scalars (global parameters, LK-ball
// weights, grid dimensions) are CPU tensors. Without LK-ball the last three
// outputs are omitted.
std::vector<Tensor> build_grid(
    Tensor receptor_coords,
    Tensor receptor_atom_types,
    Tensor receptor_charges,
    Tensor receptor_waters,
    Tensor probe_atom_types,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor elec_global_params,
    Tensor lk_ball_weights,
    Tensor cell_offsets,
    Tensor cell_indices,
    Tensor cell_origin,
    Tensor cell_dims,
    Tensor origin,
    Tensor dims,
    double voxel_spacing,
    bool build_lk_ball) {
  TORCH_CHECK(
      receptor_coords.dim() == 2 && receptor_coords.size(1) == 3
          && receptor_coords.size(0) > 0,
      "receptor_coords must have shape [n_atoms > 0, 3]");
  TORCH_CHECK(
      probe_atom_types.size(0) > 0, "at least one probe type is required");
  TORCH_CHECK(
      std::isfinite(voxel_spacing) && voxel_spacing > 0,
      "voxel_spacing must be finite and positive");
  int64_t const* dims_ptr = dims.data_ptr<int64_t>();
  int64_t const nx = dims_ptr[0], ny = dims_ptr[1], nz = dims_ptr[2];
  int64_t const n_voxels = nx * ny * nz;
  int64_t const n_probes = probe_atom_types.size(0);
  TORCH_CHECK(
      nx > 0 && ny > 0 && nz > 0
          && n_probes * n_voxels < std::numeric_limits<int32_t>::max(),
      "grid dimensions must be positive and fit 32-bit work indices");

  auto options = receptor_coords.options();
  int64_t const lk_probes = build_lk_ball ? n_probes : 0;
  int64_t const lk_voxels = build_lk_ball ? n_voxels : 0;
  Tensor atr = tmol::native::empty({n_probes, n_voxels}, options);
  Tensor rep = tmol::native::empty({n_probes, n_voxels}, options);
  Tensor sol = tmol::native::empty({n_probes, n_voxels}, options);
  Tensor elec_grid = tmol::native::empty({n_voxels}, options);
  Tensor lk_ball = tmol::native::empty({lk_probes, lk_voxels}, options);
  Tensor lk_water = tmol::native::empty({lk_probes, lk_voxels}, options);
  Tensor lk_bridge = tmol::native::empty({lk_probes, lk_voxels}, options);
  if (!build_lk_ball) receptor_waters = tmol::native::empty({1, 4, 3}, options);

  double const* lj = ljlk_global_params.data_ptr<double>();
  double const* weights = lk_ball_weights.data_ptr<double>();
  TMOL_DISPATCH_FLOATING_DEVICE(
      options, "build_grid", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        GridBuildDispatch<common::DeviceOperations, Dev, Real, int64_t>::build(
            interpolation_context,
            TCAST(receptor_coords),
            TCAST(receptor_atom_types),
            TCAST(receptor_charges),
            TCAST(receptor_waters),
            TCAST(probe_atom_types),
            TCAST(ljlk_type_params),
            LJGlobalParams<Real>{
                Real(lj[0]),
                Real(lj[1]),
                Real(lj[2]),
                Real(lj[3]),
                Real(lj[4])},
            load_elec_global_params<Real>(
                elec_global_params.data_ptr<double>()),
            GridBuildLKWeights<Real>{
                Real(weights[0]),
                Real(weights[1]),
                Real(weights[2]),
                Real(weights[3])},
            TCAST(cell_offsets),
            TCAST(cell_indices),
            TCAST(cell_origin),
            TCAST(cell_dims),
            TCAST(origin),
            nx,
            ny,
            nz,
            Real(voxel_spacing),
            build_lk_ball,
            TCAST(atr),
            TCAST(rep),
            TCAST(sol),
            TCAST(lk_ball),
            TCAST(lk_water),
            TCAST(lk_bridge),
            TCAST(elec_grid));
      }));
  if (build_lk_ball) {
    return {atr, rep, sol, elec_grid, lk_ball, lk_water, lk_bridge};
  }
  return {atr, rep, sol, elec_grid};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> interpolate_with_derivatives(
    Tensor coefficients,
    Tensor indices,
    Tensor grid_indices,
    double out_of_bounds_penalty) {
  TORCH_CHECK(
      coefficients.dim() == 4,
      "coefficients must have shape [n_grids, nx, ny, nz]");
  TORCH_CHECK(
      indices.dim() == 2 && indices.size(1) == 3,
      "indices must have shape [n_points, 3]");
  TORCH_CHECK(
      grid_indices.dim() == 1 && grid_indices.size(0) == indices.size(0),
      "grid_indices must have shape [n_points]");
  TORCH_CHECK(
      coefficients.is_contiguous() && indices.is_contiguous()
          && grid_indices.is_contiguous(),
      "interpolation inputs must be contiguous");
  TORCH_CHECK(
      coefficients.scalar_type() == indices.scalar_type(),
      "coefficients and indices must have the same dtype");
  TORCH_CHECK(
      grid_indices.scalar_type() == tmol::native::ScalarType::Long,
      "grid_indices must be int64");
  TORCH_CHECK(
      coefficients.device() == indices.device()
          && coefficients.device() == grid_indices.device(),
      "interpolation inputs must be on the same device");
  TORCH_CHECK(
      coefficients.size(0) > 0 && coefficients.size(1) > 0
          && coefficients.size(2) > 0 && coefficients.size(3) > 0,
      "coefficient dimensions must be positive");
  TORCH_CHECK(
      indices.size(0) <= std::numeric_limits<int>::max(),
      "too many interpolation points");
  TORCH_CHECK(
      std::isfinite(out_of_bounds_penalty) && out_of_bounds_penalty >= 0.0,
      "out_of_bounds_penalty must be finite and nonnegative");

  Tensor values;
  Tensor derivatives;
  Tensor penalties;
  Tensor penalty_derivatives;
  TMOL_DISPATCH_FLOATING_DEVICE(
      coefficients.options(), "grid_interp_derivatives", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            GridInterpDispatch<common::DeviceOperations, Dev, Real, int64_t>::
                forward_with_derivatives(
                    interpolation_context,
                    TCAST(coefficients),
                    TCAST(indices),
                    TCAST(grid_indices),
                    Real(out_of_bounds_penalty));
        values = std::get<0>(result).tensor;
        derivatives = std::get<1>(result).tensor;
        penalties = std::get<2>(result).tensor;
        penalty_derivatives = std::get<3>(result).tensor;
      }));
  return std::make_tuple(values, derivatives, penalties, penalty_derivatives);
}

std::tuple<Tensor, Tensor> live_hbond(
    Tensor ligand_coords,
    Tensor ligand_donor_atoms,
    Tensor ligand_donor_types,
    Tensor ligand_acceptor_atoms,
    Tensor ligand_acceptor_types,
    Tensor receptor_donors,
    Tensor receptor_donor_types,
    Tensor receptor_acceptors,
    Tensor receptor_acceptor_types,
    Tensor cell_origin,
    Tensor cell_dims,
    Tensor receptor_donor_cell_offsets,
    Tensor receptor_donor_cell_indices,
    Tensor receptor_acceptor_cell_offsets,
    Tensor receptor_acceptor_cell_indices,
    Tensor pair_params,
    Tensor pair_polynomials,
    Tensor global_params) {
  auto const device = ligand_coords.device();
  auto const dtype = ligand_coords.scalar_type();
  auto check_real = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(
        value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(
        value.scalar_type() == dtype, name, " must match coordinate dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  auto check_int = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(
        value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(
        value.scalar_type() == tmol::native::ScalarType::Long,
        name,
        " must be int64");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  check_real(ligand_coords, "ligand_coords");
  check_real(receptor_donors, "receptor_donors");
  check_real(receptor_acceptors, "receptor_acceptors");
  check_real(cell_origin, "cell_origin");
  check_real(pair_params, "pair_params");
  check_real(global_params, "global_params");
  TORCH_CHECK(
      pair_polynomials.device() == device
          && pair_polynomials.scalar_type() == tmol::native::kFloat64
          && pair_polynomials.is_contiguous(),
      "pair_polynomials must be contiguous float64 on the coordinate device");
  check_int(ligand_donor_atoms, "ligand_donor_atoms");
  check_int(ligand_donor_types, "ligand_donor_types");
  check_int(ligand_acceptor_atoms, "ligand_acceptor_atoms");
  check_int(ligand_acceptor_types, "ligand_acceptor_types");
  check_int(receptor_donor_types, "receptor_donor_types");
  check_int(receptor_acceptor_types, "receptor_acceptor_types");
  check_int(cell_dims, "cell_dims");
  check_int(receptor_donor_cell_offsets, "receptor_donor_cell_offsets");
  check_int(receptor_donor_cell_indices, "receptor_donor_cell_indices");
  check_int(receptor_acceptor_cell_offsets, "receptor_acceptor_cell_offsets");
  check_int(receptor_acceptor_cell_indices, "receptor_acceptor_cell_indices");

  TORCH_CHECK(
      ligand_coords.dim() == 3 && ligand_coords.size(2) == 3
          && ligand_coords.size(0) > 0 && ligand_coords.size(1) > 0,
      "ligand_coords must have shape [n_poses, n_atoms, 3]");
  TORCH_CHECK(
      ligand_donor_atoms.dim() == 2 && ligand_donor_atoms.size(1) == 2
          && ligand_donor_types.dim() == 1
          && ligand_donor_types.size(0) == ligand_donor_atoms.size(0),
      "ligand donor arrays have inconsistent shapes");
  TORCH_CHECK(
      ligand_acceptor_atoms.dim() == 2 && ligand_acceptor_atoms.size(1) == 3
          && ligand_acceptor_types.dim() == 1
          && ligand_acceptor_types.size(0) == ligand_acceptor_atoms.size(0),
      "ligand acceptor arrays have inconsistent shapes");
  TORCH_CHECK(
      receptor_donors.dim() == 3 && receptor_donors.size(1) == 2
          && receptor_donors.size(2) == 3 && receptor_donor_types.dim() == 1
          && receptor_donor_types.size(0) == receptor_donors.size(0),
      "receptor donor arrays have inconsistent shapes");
  TORCH_CHECK(
      receptor_acceptors.dim() == 3 && receptor_acceptors.size(1) == 3
          && receptor_acceptors.size(2) == 3
          && receptor_acceptor_types.dim() == 1
          && receptor_acceptor_types.size(0) == receptor_acceptors.size(0),
      "receptor acceptor arrays have inconsistent shapes");
  TORCH_CHECK(
      cell_origin.dim() == 1 && cell_origin.size(0) == 3 && cell_dims.dim() == 1
          && cell_dims.size(0) == 3,
      "cell origin and dimensions must have shape [3]");
  TORCH_CHECK(
      pair_params.dim() == 3 && pair_params.size(2) == 3
          && pair_polynomials.dim() == 3 && pair_polynomials.size(2) == 48
          && pair_polynomials.size(0) == pair_params.size(0)
          && pair_polynomials.size(1) == pair_params.size(1),
      "hbond pair parameter tables have inconsistent shapes");
  TORCH_CHECK(
      global_params.dim() == 2 && global_params.size(0) == 1
          && global_params.size(1) == 6,
      "global_params must have shape [1, 6]");
  TORCH_CHECK(
      ligand_coords.size(0) <= std::numeric_limits<int>::max()
          && ligand_coords.size(1) <= std::numeric_limits<int>::max(),
      "live hbond dimensions exceed kernel integer range");

  int64_t const n_cells = receptor_donor_cell_offsets.size(0) - 1;
  TORCH_CHECK(
      receptor_donor_cell_offsets.dim() == 1 && n_cells > 0
          && receptor_acceptor_cell_offsets.dim() == 1
          && receptor_acceptor_cell_offsets.size(0) == n_cells + 1,
      "receptor cell offset arrays must have matching positive lengths");
  TORCH_CHECK(
      receptor_donor_cell_indices.dim() == 1
          && receptor_donor_cell_indices.size(0) == receptor_donors.size(0)
          && receptor_acceptor_cell_indices.dim() == 1
          && receptor_acceptor_cell_indices.size(0)
                 == receptor_acceptors.size(0),
      "cell index arrays must contain every receptor site once");

  Tensor energies;
  Tensor gradients;
  TMOL_DISPATCH_FLOATING_DEVICE(
      ligand_coords.options(), "live_hbond", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            LiveHBondDispatch<common::DeviceOperations, Dev, Real, int64_t>::
                forward(
                    interpolation_context,
                    TCAST(ligand_coords),
                    TCAST(ligand_donor_atoms),
                    TCAST(ligand_donor_types),
                    TCAST(ligand_acceptor_atoms),
                    TCAST(ligand_acceptor_types),
                    TCAST(receptor_donors),
                    TCAST(receptor_donor_types),
                    TCAST(receptor_acceptors),
                    TCAST(receptor_acceptor_types),
                    TCAST(cell_origin),
                    TCAST(cell_dims),
                    TCAST(receptor_donor_cell_offsets),
                    TCAST(receptor_donor_cell_indices),
                    TCAST(receptor_acceptor_cell_offsets),
                    TCAST(receptor_acceptor_cell_indices),
                    TCAST(pair_params),
                    TCAST(pair_polynomials),
                    TCAST(global_params));
        energies = std::get<0>(result).tensor;
        gradients = std::get<1>(result).tensor;
      }));
  return std::make_tuple(energies, gradients);
}

std::tuple<Tensor, Tensor> live_lk_ball(
    Tensor ligand_coords,
    Tensor ligand_atom_params,
    Tensor ligand_heavy_atoms,
    Tensor ligand_polar_atoms,
    Tensor ligand_atom_to_polar,
    Tensor ligand_water_kinds,
    Tensor ligand_water_atoms,
    Tensor ligand_water_params,
    Tensor ligand_water_is_ring,
    Tensor receptor_coords,
    Tensor receptor_params,
    Tensor receptor_waters,
    Tensor receptor_polar_atoms,
    Tensor cell_origin,
    Tensor cell_dims,
    Tensor receptor_cell_offsets,
    Tensor receptor_cell_indices,
    Tensor receptor_polar_cell_offsets,
    Tensor receptor_polar_cell_indices,
    Tensor global_params,
    bool use_cell_list) {
  auto const device = ligand_coords.device();
  auto const dtype = ligand_coords.scalar_type();
  auto check_real = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(
        value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(
        value.scalar_type() == dtype, name, " must match coordinate dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  auto check_int = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(
        value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(
        value.scalar_type() == tmol::native::ScalarType::Long,
        name,
        " must be int64");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  check_real(ligand_coords, "ligand_coords");
  check_real(ligand_atom_params, "ligand_atom_params");
  check_real(ligand_water_params, "ligand_water_params");
  check_real(receptor_coords, "receptor_coords");
  check_real(receptor_params, "receptor_params");
  check_real(receptor_waters, "receptor_waters");
  check_real(cell_origin, "cell_origin");
  check_real(global_params, "global_params");
  check_int(ligand_heavy_atoms, "ligand_heavy_atoms");
  check_int(ligand_polar_atoms, "ligand_polar_atoms");
  check_int(ligand_atom_to_polar, "ligand_atom_to_polar");
  check_int(ligand_water_kinds, "ligand_water_kinds");
  check_int(ligand_water_atoms, "ligand_water_atoms");
  check_int(ligand_water_is_ring, "ligand_water_is_ring");
  check_int(receptor_polar_atoms, "receptor_polar_atoms");
  check_int(cell_dims, "cell_dims");
  check_int(receptor_cell_offsets, "receptor_cell_offsets");
  check_int(receptor_cell_indices, "receptor_cell_indices");
  check_int(receptor_polar_cell_offsets, "receptor_polar_cell_offsets");
  check_int(receptor_polar_cell_indices, "receptor_polar_cell_indices");

  TORCH_CHECK(
      ligand_coords.dim() == 3 && ligand_coords.size(2) == 3
          && ligand_coords.size(0) > 0 && ligand_coords.size(1) > 0,
      "ligand_coords must have shape [n_poses, n_atoms, 3]");
  int64_t const n_atoms = ligand_coords.size(1);
  int64_t const n_ligand_polar = ligand_polar_atoms.size(0);
  int64_t const n_receptor = receptor_coords.size(0);
  int64_t const n_receptor_polar = receptor_polar_atoms.size(0);
  TORCH_CHECK(
      ligand_atom_params.dim() == 2 && ligand_atom_params.size(0) == n_atoms
          && ligand_atom_params.size(1) == 9,
      "ligand_atom_params must have shape [n_atoms, 9]");
  TORCH_CHECK(
      ligand_heavy_atoms.dim() == 1 && ligand_polar_atoms.dim() == 1
          && ligand_atom_to_polar.dim() == 1
          && ligand_atom_to_polar.size(0) == n_atoms,
      "ligand atom index arrays have inconsistent shapes");
  TORCH_CHECK(
      ligand_water_kinds.dim() == 2
          && ligand_water_kinds.size(0) == n_ligand_polar
          && ligand_water_kinds.size(1) == 4 && ligand_water_atoms.dim() == 3
          && ligand_water_atoms.size(0) == n_ligand_polar
          && ligand_water_atoms.size(1) == 4 && ligand_water_atoms.size(2) == 3
          && ligand_water_params.dim() == 3
          && ligand_water_params.sizes() == ligand_water_atoms.sizes()
          && ligand_water_is_ring.dim() == 2
          && ligand_water_is_ring.sizes() == ligand_water_kinds.sizes(),
      "ligand water arrays have inconsistent shapes");
  TORCH_CHECK(
      receptor_coords.dim() == 2 && receptor_coords.size(1) == 3
          && n_receptor > 0 && receptor_params.dim() == 2
          && receptor_params.size(0) == n_receptor
          && receptor_params.size(1) == 9 && receptor_waters.dim() == 3
          && receptor_waters.size(0) == n_receptor
          && receptor_waters.size(1) == 4 && receptor_waters.size(2) == 3
          && receptor_polar_atoms.dim() == 1,
      "receptor LK-ball arrays have inconsistent shapes");
  TORCH_CHECK(
      cell_origin.dim() == 1 && cell_origin.size(0) == 3 && cell_dims.dim() == 1
          && cell_dims.size(0) == 3,
      "cell origin and dimensions must have shape [3]");
  int64_t const n_cells = receptor_cell_offsets.size(0) - 1;
  TORCH_CHECK(
      receptor_cell_offsets.dim() == 1 && n_cells > 0
          && receptor_polar_cell_offsets.dim() == 1
          && receptor_polar_cell_offsets.size(0) == n_cells + 1,
      "receptor cell offset arrays must have matching positive lengths");
  TORCH_CHECK(
      receptor_cell_indices.dim() == 1
          && receptor_cell_indices.size(0) == n_receptor
          && receptor_polar_cell_indices.dim() == 1
          && receptor_polar_cell_indices.size(0) == n_receptor_polar,
      "receptor cell index arrays have inconsistent shapes");
  TORCH_CHECK(
      global_params.dim() == 2 && global_params.size(0) == 1
          && global_params.size(1) == 5,
      "global_params must have shape [1, 5]");
  TORCH_CHECK(
      ligand_coords.size(0) <= std::numeric_limits<int>::max()
          && n_atoms <= std::numeric_limits<int>::max()
          && n_ligand_polar <= std::numeric_limits<int>::max(),
      "live LK-ball dimensions exceed kernel integer range");

  Tensor energies;
  Tensor gradients;
  TMOL_DISPATCH_FLOATING_DEVICE(
      ligand_coords.options(), "live_lk_ball", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            LiveLKBallDispatch<common::DeviceOperations, Dev, Real, int64_t>::
                forward(
                    interpolation_context,
                    TCAST(ligand_coords),
                    TCAST(ligand_atom_params),
                    TCAST(ligand_heavy_atoms),
                    TCAST(ligand_polar_atoms),
                    TCAST(ligand_atom_to_polar),
                    TCAST(ligand_water_kinds),
                    TCAST(ligand_water_atoms),
                    TCAST(ligand_water_params),
                    TCAST(ligand_water_is_ring),
                    TCAST(receptor_coords),
                    TCAST(receptor_params),
                    TCAST(receptor_waters),
                    TCAST(receptor_polar_atoms),
                    TCAST(cell_origin),
                    TCAST(cell_dims),
                    TCAST(receptor_cell_offsets),
                    TCAST(receptor_cell_indices),
                    TCAST(receptor_polar_cell_offsets),
                    TCAST(receptor_polar_cell_indices),
                    TCAST(global_params),
                    use_cell_list);
        energies = std::get<0>(result).tensor;
        gradients = std::get<1>(result).tensor;
      }));
  return std::make_tuple(energies, gradients);
}

void validate_grid_lk_ball_water_inputs(
    Tensor const& ligand_coords,
    Tensor const& ligand_polar_atoms,
    Tensor const& ligand_water_kinds,
    Tensor const& ligand_water_atoms,
    Tensor const& ligand_water_params,
    Tensor const& ligand_water_is_ring,
    double water_distance) {
  auto const device = ligand_coords.device();
  auto const dtype = ligand_coords.scalar_type();
  auto check_real = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(
        value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(
        value.scalar_type() == dtype, name, " must match coordinate dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  auto check_int = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(
        value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(
        value.scalar_type() == tmol::native::ScalarType::Long,
        name,
        " must be int64");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  check_real(ligand_coords, "ligand_coords");
  check_real(ligand_water_params, "ligand_water_params");
  check_int(ligand_polar_atoms, "ligand_polar_atoms");
  check_int(ligand_water_kinds, "ligand_water_kinds");
  check_int(ligand_water_atoms, "ligand_water_atoms");
  check_int(ligand_water_is_ring, "ligand_water_is_ring");
  TORCH_CHECK(
      ligand_coords.dim() == 3 && ligand_coords.size(2) == 3
          && ligand_coords.size(0) > 0 && ligand_coords.size(1) > 0,
      "ligand_coords must have shape [n_poses, n_atoms, 3]");
  int64_t const n_polar = ligand_polar_atoms.size(0);
  TORCH_CHECK(
      ligand_coords.size(0) <= std::numeric_limits<int>::max()
          && ligand_coords.size(1) <= std::numeric_limits<int>::max()
          && n_polar <= std::numeric_limits<int>::max(),
      "gridded LK-ball water dimensions exceed kernel integer range");
  TORCH_CHECK(
      ligand_polar_atoms.dim() == 1 && ligand_water_kinds.dim() == 2
          && ligand_water_kinds.size(0) == n_polar
          && ligand_water_kinds.size(1) == 4 && ligand_water_atoms.dim() == 3
          && ligand_water_atoms.size(0) == n_polar
          && ligand_water_atoms.size(1) == 4 && ligand_water_atoms.size(2) == 3
          && ligand_water_params.dim() == 3
          && ligand_water_params.sizes() == ligand_water_atoms.sizes()
          && ligand_water_is_ring.sizes() == ligand_water_kinds.sizes(),
      "ligand water arrays have inconsistent shapes");
  TORCH_CHECK(
      std::isfinite(water_distance) && water_distance > 0,
      "water_distance must be finite and positive");
}

Tensor build_lk_ball_waters(
    Tensor ligand_coords,
    Tensor ligand_polar_atoms,
    Tensor ligand_water_kinds,
    Tensor ligand_water_atoms,
    Tensor ligand_water_params,
    Tensor ligand_water_is_ring,
    double water_distance) {
  validate_grid_lk_ball_water_inputs(
      ligand_coords,
      ligand_polar_atoms,
      ligand_water_kinds,
      ligand_water_atoms,
      ligand_water_params,
      ligand_water_is_ring,
      water_distance);
  Tensor waters;
  TMOL_DISPATCH_FLOATING_DEVICE(
      ligand_coords.options(), "build_lk_ball_waters", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        waters =
            GridLKBallDispatch<common::DeviceOperations, Dev, Real, int64_t>::
                build_waters(
                    interpolation_context,
                    TCAST(ligand_coords),
                    TCAST(ligand_polar_atoms),
                    TCAST(ligand_water_kinds),
                    TCAST(ligand_water_atoms),
                    TCAST(ligand_water_params),
                    TCAST(ligand_water_is_ring),
                    Real(water_distance))
                    .tensor;
      }));
  return waters;
}

Tensor propagate_lk_ball_water_gradients(
    Tensor ligand_coords,
    Tensor ligand_polar_atoms,
    Tensor ligand_water_kinds,
    Tensor ligand_water_atoms,
    Tensor ligand_water_params,
    Tensor ligand_water_is_ring,
    Tensor water_gradients,
    double water_distance) {
  validate_grid_lk_ball_water_inputs(
      ligand_coords,
      ligand_polar_atoms,
      ligand_water_kinds,
      ligand_water_atoms,
      ligand_water_params,
      ligand_water_is_ring,
      water_distance);
  TORCH_CHECK(
      water_gradients.device() == ligand_coords.device()
          && water_gradients.scalar_type() == ligand_coords.scalar_type()
          && water_gradients.is_contiguous(),
      "water_gradients must match coordinates and be contiguous");
  TORCH_CHECK(
      water_gradients.dim() == 4
          && water_gradients.size(0) == ligand_coords.size(0)
          && water_gradients.size(1) == ligand_polar_atoms.size(0)
          && water_gradients.size(2) == 4 && water_gradients.size(3) == 3,
      "water_gradients must have shape [n_poses, n_polar, 4, 3]");
  Tensor gradients;
  TMOL_DISPATCH_FLOATING_DEVICE(
      ligand_coords.options(), "propagate_lk_ball_water_gradients", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        gradients =
            GridLKBallDispatch<common::DeviceOperations, Dev, Real, int64_t>::
                propagate_water_gradients(
                    interpolation_context,
                    TCAST(ligand_coords),
                    TCAST(ligand_polar_atoms),
                    TCAST(ligand_water_kinds),
                    TCAST(ligand_water_atoms),
                    TCAST(ligand_water_params),
                    TCAST(ligand_water_is_ring),
                    TCAST(water_gradients),
                    Real(water_distance))
                    .tensor;
      }));
  return gradients;
}

TMOL_LIBRARY(tmol_gridscore, m) {
  m.def("build_grid", TMOL_FN(&build_grid));
  m.def("interpolate_with_derivatives", TMOL_FN(&interpolate_with_derivatives));
  m.def("live_hbond", TMOL_FN(&live_hbond));
  m.def("live_lk_ball", TMOL_FN(&live_lk_ball));
  m.def("build_lk_ball_waters", TMOL_FN(&build_lk_ball_waters));
  m.def(
      "propagate_lk_ball_water_gradients",
      TMOL_FN(&propagate_lk_ball_water_gradients));
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
