#include <ATen/Parallel.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/score/gridscore/potentials/grid_interp_pose_score.hh>
#include <tmol/score/gridscore/potentials/live_hbond.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/tensor/TensorCast.h>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

using torch::Tensor;
using tmol::score::elec::potentials::elec;
using tmol::score::ljlk::potentials::LJGlobalParams;
using tmol::score::ljlk::potentials::LJLKTypeParams;
using tmol::score::ljlk::potentials::lj_score;
using tmol::score::ljlk::potentials::lk_isotropic_score;

ContextManager interpolation_context;

namespace {

struct BinKey {
  int64_t x;
  int64_t y;
  int64_t z;

  bool operator==(BinKey const& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct BinKeyHash {
  std::size_t operator()(BinKey const& key) const {
    std::size_t seed = std::hash<int64_t>{}(key.x);
    seed ^= std::hash<int64_t>{}(key.y) + 0x9e3779b9 + (seed << 6)
            + (seed >> 2);
    seed ^= std::hash<int64_t>{}(key.z) + 0x9e3779b9 + (seed << 6)
            + (seed >> 2);
    return seed;
  }
};

using ReceptorBins =
    std::unordered_map<BinKey, std::vector<int64_t>, BinKeyHash>;

void check_double_cpu_contiguous(Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat64, name, " must be float64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int64_cpu_contiguous(Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

LJLKTypeParams<double> load_type_params(double const* type_params, int64_t type) {
  double const* row = type_params + type * 10;
  return LJLKTypeParams<double>{
      row[0], row[1], row[2], row[3], row[4],
      row[5], row[6], row[7], row[8], row[9]};
}

LJGlobalParams<double> load_global_params(double const* global_params) {
  return LJGlobalParams<double>{
      global_params[0],
      global_params[1],
      global_params[2],
      global_params[3],
      global_params[4]};
}

BinKey bin_for_point(double x, double y, double z, double bin_width) {
  return BinKey{
      static_cast<int64_t>(std::floor(x / bin_width)),
      static_cast<int64_t>(std::floor(y / bin_width)),
      static_cast<int64_t>(std::floor(z / bin_width))};
}

void validate_common_inputs(
    Tensor const& receptor_atom_types,
    Tensor const& receptor_charges,
    Tensor const& probe_atom_types,
    Tensor const& ljlk_type_params,
    Tensor const& ljlk_global_params,
    Tensor const& elec_global_params) {
  check_int64_cpu_contiguous(receptor_atom_types, "receptor_atom_types");
  check_double_cpu_contiguous(receptor_charges, "receptor_charges");
  check_int64_cpu_contiguous(probe_atom_types, "probe_atom_types");
  check_double_cpu_contiguous(ljlk_type_params, "ljlk_type_params");
  check_double_cpu_contiguous(ljlk_global_params, "ljlk_global_params");
  check_double_cpu_contiguous(elec_global_params, "elec_global_params");
  TORCH_CHECK(receptor_atom_types.dim() == 1, "receptor_atom_types must be 1D");
  TORCH_CHECK(receptor_charges.dim() == 1, "receptor_charges must be 1D");
  TORCH_CHECK(probe_atom_types.dim() == 1, "probe_atom_types must be 1D");
  TORCH_CHECK(
      receptor_atom_types.size(0) == receptor_charges.size(0),
      "receptor atom arrays must have matching lengths");
  TORCH_CHECK(
      ljlk_type_params.dim() == 2 && ljlk_type_params.size(1) == 10,
      "ljlk_type_params must have shape [n_types, 10]");
  TORCH_CHECK(
      ljlk_global_params.dim() == 2 && ljlk_global_params.size(0) == 1
          && ljlk_global_params.size(1) == 5,
      "ljlk_global_params must have shape [1, 5]");
  TORCH_CHECK(
      elec_global_params.dim() == 1 && elec_global_params.size(0) == 5,
      "elec_global_params must have shape [5]");

  auto receptor_types = receptor_atom_types.accessor<int64_t, 1>();
  auto probe_types = probe_atom_types.accessor<int64_t, 1>();
  int64_t const n_types = ljlk_type_params.size(0);
  auto type_params = ljlk_type_params.accessor<double, 2>();
  auto check_type = [&](int64_t type, char const* kind, int64_t index) {
    TORCH_CHECK(
        type >= 0 && type < n_types,
        kind,
        " atom-type index out of range at index ",
        index);
    for (int64_t parameter = 0; parameter < 10; ++parameter) {
      TORCH_CHECK(
          std::isfinite(type_params[type][parameter]),
          kind,
          " atom-type parameters are non-finite at index ",
          index);
    }
  };
  for (int64_t i = 0; i < receptor_atom_types.size(0); ++i) {
    check_type(receptor_types[i], "receptor", i);
  }
  for (int64_t i = 0; i < probe_atom_types.size(0); ++i) {
    check_type(probe_types[i], "probe", i);
  }
  auto lj_global = ljlk_global_params.accessor<double, 2>();
  auto elec_global = elec_global_params.accessor<double, 1>();
  for (int64_t parameter = 0; parameter < 5; ++parameter) {
    TORCH_CHECK(
        std::isfinite(lj_global[0][parameter]),
        "ljlk_global_params must be finite");
    TORCH_CHECK(
        std::isfinite(elec_global[parameter]),
        "elec_global_params must be finite");
  }
}

}  // namespace

std::vector<Tensor> build_grid(
    Tensor receptor_coords,
    Tensor receptor_atom_types,
    Tensor receptor_charges,
    Tensor probe_atom_types,
    Tensor origin,
    Tensor dims,
    double voxel_spacing,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor elec_global_params) {
  check_double_cpu_contiguous(receptor_coords, "receptor_coords");
  check_double_cpu_contiguous(origin, "origin");
  check_int64_cpu_contiguous(dims, "dims");
  validate_common_inputs(
      receptor_atom_types,
      receptor_charges,
      probe_atom_types,
      ljlk_type_params,
      ljlk_global_params,
      elec_global_params);
  TORCH_CHECK(
      receptor_coords.dim() == 2 && receptor_coords.size(1) == 3,
      "receptor_coords must have shape [n_atoms, 3]");
  TORCH_CHECK(
      receptor_coords.size(0) == receptor_atom_types.size(0),
      "receptor coordinate and atom arrays must have matching lengths");
  TORCH_CHECK(origin.dim() == 1 && origin.size(0) == 3, "origin must have shape [3]");
  TORCH_CHECK(dims.dim() == 1 && dims.size(0) == 3, "dims must have shape [3]");
  TORCH_CHECK(
      std::isfinite(voxel_spacing) && voxel_spacing > 0,
      "voxel_spacing must be finite and positive");

  auto dims_a = dims.accessor<int64_t, 1>();
  int64_t const nx = dims_a[0];
  int64_t const ny = dims_a[1];
  int64_t const nz = dims_a[2];
  TORCH_CHECK(nx > 0 && ny > 0 && nz > 0, "grid dimensions must be positive");
  TORCH_CHECK(
      nx <= std::numeric_limits<int64_t>::max() / ny
          && nx * ny <= std::numeric_limits<int64_t>::max() / nz,
      "grid dimensions overflow int64");
  int64_t const n_voxels = nx * ny * nz;
  int64_t const n_probes = probe_atom_types.size(0);
  int64_t const n_receptor = receptor_coords.size(0);
  TORCH_CHECK(n_probes > 0, "at least one probe atom type is required");
  TORCH_CHECK(n_receptor > 0, "at least one receptor atom is required");
  TORCH_CHECK(
      n_probes <= std::numeric_limits<int64_t>::max() / n_voxels,
      "typed grid dimensions overflow int64");

  double const* lj_global_ptr = ljlk_global_params.data_ptr<double>();
  LJGlobalParams<double> const lj_global = load_global_params(lj_global_ptr);
  double const* elec_global = elec_global_params.data_ptr<double>();
  double const elec_max_dis = elec_global[4];
  double const max_dis = std::max(lj_global.max_dis, elec_max_dis);
  TORCH_CHECK(
      std::isfinite(max_dis) && max_dis > 0,
      "interaction cutoff must be finite and positive");
  double const max_dis2 = max_dis * max_dis;
  double const lj_max_dis2 = lj_global.max_dis * lj_global.max_dis;
  double const elec_max_dis2 = elec_max_dis * elec_max_dis;

  auto coords = receptor_coords.accessor<double, 2>();
  auto charges = receptor_charges.accessor<double, 1>();
  ReceptorBins bins;
  bins.reserve(static_cast<std::size_t>(n_receptor));
  for (int64_t atom = 0; atom < n_receptor; ++atom) {
    TORCH_CHECK(
        std::isfinite(coords[atom][0]) && std::isfinite(coords[atom][1])
            && std::isfinite(coords[atom][2]),
        "receptor coordinates must be finite");
    TORCH_CHECK(
        std::isfinite(charges[atom]),
        "receptor charges must be finite");
    bins[bin_for_point(coords[atom][0], coords[atom][1], coords[atom][2], max_dis)]
        .push_back(atom);
  }
  auto origin_a = origin.accessor<double, 1>();
  TORCH_CHECK(
      std::isfinite(origin_a[0]) && std::isfinite(origin_a[1])
          && std::isfinite(origin_a[2]),
      "origin must be finite");
  ReceptorBins const& receptor_bins = bins;

  std::vector<int64_t> typed_shape{n_probes, nx, ny, nz};
  std::vector<int64_t> shared_shape{nx, ny, nz};
  auto options = receptor_coords.options().requires_grad(false);
  Tensor fa_atr = torch::empty(typed_shape, options);
  Tensor fa_rep = torch::empty(typed_shape, options);
  Tensor fa_sol = torch::empty(typed_shape, options);
  Tensor fa_elec = torch::empty(shared_shape, options);

  double const* coord_ptr = receptor_coords.data_ptr<double>();
  int64_t const* receptor_type_ptr = receptor_atom_types.data_ptr<int64_t>();
  double const* charge_ptr = receptor_charges.data_ptr<double>();
  int64_t const* probe_type_ptr = probe_atom_types.data_ptr<int64_t>();
  double const* origin_ptr = origin.data_ptr<double>();
  double const* type_param_ptr = ljlk_type_params.data_ptr<double>();
  double* atr_ptr = fa_atr.data_ptr<double>();
  double* rep_ptr = fa_rep.data_ptr<double>();
  double* sol_ptr = fa_sol.data_ptr<double>();
  double* elec_ptr = fa_elec.data_ptr<double>();

  at::parallel_for(0, n_voxels, 64, [&](int64_t begin, int64_t end) {
    std::vector<double> atr_sum(static_cast<std::size_t>(n_probes));
    std::vector<double> rep_sum(static_cast<std::size_t>(n_probes));
    std::vector<double> sol_sum(static_cast<std::size_t>(n_probes));
    for (int64_t voxel = begin; voxel < end; ++voxel) {
      std::fill(atr_sum.begin(), atr_sum.end(), 0.0);
      std::fill(rep_sum.begin(), rep_sum.end(), 0.0);
      std::fill(sol_sum.begin(), sol_sum.end(), 0.0);
      double elec_sum = 0.0;

      int64_t const ix = voxel / (ny * nz);
      int64_t const remainder = voxel % (ny * nz);
      int64_t const iy = remainder / nz;
      int64_t const iz = remainder % nz;
      double const x = origin_ptr[0] + voxel_spacing * ix;
      double const y = origin_ptr[1] + voxel_spacing * iy;
      double const z = origin_ptr[2] + voxel_spacing * iz;
      BinKey const center_bin = bin_for_point(x, y, z, max_dis);

      for (int64_t bx = center_bin.x - 1; bx <= center_bin.x + 1; ++bx) {
        for (int64_t by = center_bin.y - 1; by <= center_bin.y + 1; ++by) {
          for (int64_t bz = center_bin.z - 1; bz <= center_bin.z + 1; ++bz) {
            auto bin = receptor_bins.find(BinKey{bx, by, bz});
            if (bin == receptor_bins.end()) continue;
            for (int64_t atom : bin->second) {
              double const dx = x - coord_ptr[3 * atom];
              double const dy = y - coord_ptr[3 * atom + 1];
              double const dz = z - coord_ptr[3 * atom + 2];
              double const dist2 = dx * dx + dy * dy + dz * dz;
              if (dist2 > max_dis2) continue;
              double const dist = std::sqrt(dist2);

              if (dist2 <= elec_max_dis2) {
                elec_sum += elec<double>(
                    dist,
                    1.0,
                    charge_ptr[atom],
                    5.0,
                    elec_global[0],
                    elec_global[1],
                    elec_global[2],
                    elec_global[3],
                    elec_global[4]);
              }
              if (dist2 > lj_max_dis2) continue;

              auto receptor_params =
                  load_type_params(type_param_ptr, receptor_type_ptr[atom]);
              for (int64_t probe = 0; probe < n_probes; ++probe) {
                auto probe_params =
                    load_type_params(type_param_ptr, probe_type_ptr[probe]);
                auto lj = lj_score<double>::V(
                    dist,
                    5.0,
                    probe_params.lj_params(),
                    receptor_params.lj_params(),
                    lj_global);
                atr_sum[probe] += lj[0];
                rep_sum[probe] += lj[1];
                sol_sum[probe] += lk_isotropic_score<double>::V(
                    dist,
                    5.0,
                    probe_params.lk_params(),
                    receptor_params.lk_params(),
                    lj_global);
              }
            }
          }
        }
      }

      elec_ptr[voxel] = elec_sum;
      for (int64_t probe = 0; probe < n_probes; ++probe) {
        int64_t const output_index = probe * n_voxels + voxel;
        atr_ptr[output_index] = atr_sum[probe];
        rep_ptr[output_index] = std::pow(rep_sum[probe] + 1.0, -0.25);
        sol_ptr[output_index] = sol_sum[probe];
      }
    }
  });

  return {fa_atr, fa_rep, fa_sol, fa_elec};
}

Tensor evaluate_pairs(
    Tensor distances,
    int64_t probe_atom_type,
    Tensor receptor_atom_types,
    Tensor receptor_charges,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor elec_global_params) {
  check_double_cpu_contiguous(distances, "distances");
  Tensor probe_types = torch::tensor(
      {probe_atom_type}, torch::TensorOptions().dtype(torch::kInt64));
  validate_common_inputs(
      receptor_atom_types,
      receptor_charges,
      probe_types,
      ljlk_type_params,
      ljlk_global_params,
      elec_global_params);
  TORCH_CHECK(distances.dim() == 1, "distances must be 1D");
  TORCH_CHECK(
      distances.size(0) == receptor_atom_types.size(0),
      "distances and receptor arrays must have matching lengths");

  int64_t const n_pairs = distances.size(0);
  auto distances_a = distances.accessor<double, 1>();
  for (int64_t pair = 0; pair < n_pairs; ++pair) {
    TORCH_CHECK(
        std::isfinite(distances_a[pair]) && distances_a[pair] >= 0,
        "distances must be finite and nonnegative");
  }
  Tensor result = torch::empty({n_pairs, 4}, distances.options());
  auto receptor_types_a = receptor_atom_types.accessor<int64_t, 1>();
  auto charges_a = receptor_charges.accessor<double, 1>();
  auto result_a = result.accessor<double, 2>();
  double const* type_param_ptr = ljlk_type_params.data_ptr<double>();
  auto probe_params = load_type_params(type_param_ptr, probe_atom_type);
  LJGlobalParams<double> const lj_global =
      load_global_params(ljlk_global_params.data_ptr<double>());
  double const* elec_global = elec_global_params.data_ptr<double>();

  at::parallel_for(0, n_pairs, 256, [&](int64_t begin, int64_t end) {
    for (int64_t pair = begin; pair < end; ++pair) {
      double const dist = distances_a[pair];
      auto receptor_params =
          load_type_params(type_param_ptr, receptor_types_a[pair]);
      auto lj = lj_score<double>::V(
          dist,
          5.0,
          probe_params.lj_params(),
          receptor_params.lj_params(),
          lj_global);
      result_a[pair][0] = lj[0];
      result_a[pair][1] = lj[1];
      result_a[pair][2] = lk_isotropic_score<double>::V(
          dist,
          5.0,
          probe_params.lk_params(),
          receptor_params.lk_params(),
          lj_global);
      result_a[pair][3] = elec<double>(
          dist,
          1.0,
          charges_a[pair],
          5.0,
          elec_global[0],
          elec_global[1],
          elec_global[2],
          elec_global[3],
          elec_global[4]);
    }
  });
  return result;
}

Tensor interpolate(Tensor coefficients, Tensor indices, Tensor grid_indices) {
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
      grid_indices.scalar_type() == torch::kInt64,
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

  Tensor output;
  TMOL_DISPATCH_FLOATING_DEVICE(
      coefficients.options(), "grid_interp", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        output = GridInterpDispatch<
                     common::DeviceOperations,
                     Dev,
                     Real,
                     int64_t>::forward(
                     interpolation_context,
                     TCAST(coefficients),
                     TCAST(indices),
                     TCAST(grid_indices))
                     .tensor;
      }));
  return output;
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
      grid_indices.scalar_type() == torch::kInt64,
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
      std::isfinite(out_of_bounds_penalty)
          && out_of_bounds_penalty >= 0.0,
      "out_of_bounds_penalty must be finite and nonnegative");

  Tensor values;
  Tensor derivatives;
  Tensor penalties;
  Tensor penalty_derivatives;
  TMOL_DISPATCH_FLOATING_DEVICE(
      coefficients.options(), "grid_interp_derivatives", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result = GridInterpDispatch<
                          common::DeviceOperations,
                          Dev,
                          Real,
                          int64_t>::forward_with_derivatives(
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
  return std::make_tuple(
      values, derivatives, penalties, penalty_derivatives);
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
    TORCH_CHECK(value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(value.scalar_type() == dtype, name, " must match coordinate dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  auto check_int = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(value.scalar_type() == torch::kInt64, name, " must be int64");
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
          && pair_polynomials.scalar_type() == torch::kFloat64
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
      cell_origin.dim() == 1 && cell_origin.size(0) == 3
          && cell_dims.dim() == 1 && cell_dims.size(0) == 3,
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
          && receptor_acceptor_cell_indices.size(0) == receptor_acceptors.size(0),
      "cell index arrays must contain every receptor site once");

  Tensor energies;
  Tensor gradients;
  TMOL_DISPATCH_FLOATING_DEVICE(
      ligand_coords.options(), "live_hbond", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result = LiveHBondDispatch<
                          common::DeviceOperations,
                          Dev,
                          Real,
                          int64_t>::forward(
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

TORCH_LIBRARY(tmol_gridscore, m) {
  m.def("build_grid", &build_grid);
  m.def("evaluate_pairs", &evaluate_pairs);
  m.def("interpolate", &interpolate);
  m.def("interpolate_with_derivatives", &interpolate_with_derivatives);
  m.def("live_hbond", &live_hbond);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
