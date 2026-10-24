#include <tmol/utility/tensor/native_library.hh>

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
using tmol::score::ljlk::potentials::LJGlobalParams;
using tmol::score::ljlk::potentials::LJLKTypeParams;
using tmol::score::ljlk::potentials::lj_score;
using tmol::score::ljlk::potentials::lk_isotropic_score;
using tmol::score::ljlk::potentials::lj_sigma;
using tmol::score::ljlk::potentials::lk_isotropic_pair;

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
  TORCH_CHECK(tensor.scalar_type() == tmol::native::kFloat64, name, " must be float64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int64_cpu_contiguous(Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.scalar_type() == tmol::native::ScalarType::Long, name, " must be int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

LJLKTypeParams<double> load_type_params(
    double const* type_params, int64_t type) {
  double const* row = type_params + type * 10;
  // The grid API still supplies raw well depths and LK parameters; the shared
  // potentials now consume square-root well depths and precomputed LK factors.
  return LJLKTypeParams<double>{
      row[0],
      std::sqrt(row[1]),
      row[2],
      row[3],
      row[4],
      row[5],
      row[6],
      row[7],
      row[8],
      row[9],
      0.0,  // is_hydrogen is unused by the grid pair potentials.
      -row[2] / (2.0 * 5.56832799683 * row[3]),
      1.0 / (row[3] * row[3])};
}

LJGlobalParams<double> load_global_params(double const* global_params) {
  return LJGlobalParams<double>{
      global_params[0],
      global_params[1],
      global_params[2],
      global_params[3],
      global_params[4]};
}

ElecGlobalParams<double> load_elec_global_params(double const* global_params) {
  double const D = global_params[0];
  double const D0 = global_params[1];
  double const S = global_params[2];
  double const min_dis = global_params[3];
  double const max_dis = global_params[4];
  double const cutoff_offset =
      322.0637 / (max_dis * eps<double>(max_dis, D, D0, S));
  double const low_end = min_dis + 0.25;
  double const low_eps = eps<double>(low_end, D, D0, S);
  double const high_start = max_dis - 1.0;
  double const high_eps = eps<double>(high_start, D, D0, S);
  return ElecGlobalParams<double>{
      D,
      D0,
      S,
      min_dis,
      max_dis,
      cutoff_offset,
      322.0637 / (min_dis * eps<double>(min_dis, D, D0, S)) - cutoff_offset,
      322.0637 / (low_end * low_eps) - cutoff_offset,
      -322.0637
          * (low_eps + low_end * deps_ddist<double>(low_end, D, D0, S))
          / (low_end * low_end * low_eps * low_eps),
      322.0637 / (high_start * high_eps) - cutoff_offset,
      -322.0637
          * (high_eps + high_start * deps_ddist<double>(high_start, D, D0, S))
          / (high_start * high_start * high_eps * high_eps)};
}

bool finite_water(double const* waters, int64_t atom, int water) {
  double const* xyz = waters + (atom * 4 + water) * 3;
  return std::isfinite(xyz[0]) && std::isfinite(xyz[1])
         && std::isfinite(xyz[2]);
}

bool has_finite_water(double const* waters, int64_t atom) {
  for (int water = 0; water < 4; ++water) {
    if (finite_water(waters, atom, water)) return true;
  }
  return false;
}

double hard_lk_fraction(
    double x,
    double y,
    double z,
    double const* waters,
    int64_t atom,
    double occluder_radius) {
  constexpr double ramp_width = 3.709;
  double const d2_low = std::max(
      0.0,
      (1.4 + occluder_radius) * (1.4 + occluder_radius) - ramp_width);
  double best_delta = ramp_width;
  for (int water = 0; water < 4; ++water) {
    if (!finite_water(waters, atom, water)) continue;
    double const* water_xyz = waters + (atom * 4 + water) * 3;
    double const dx = x - water_xyz[0];
    double const dy = y - water_xyz[1];
    double const dz = z - water_xyz[2];
    best_delta = std::min(
        best_delta, dx * dx + dy * dy + dz * dz - d2_low);
  }
  if (best_delta < 0.0) return 1.0;
  if (best_delta >= ramp_width) return 0.0;
  double const scaled = best_delta / ramp_width;
  double const shoulder = 1.0 - scaled * scaled;
  return shoulder * shoulder;
}

double hard_single_lk_fraction(
    double receptor_x,
    double receptor_y,
    double receptor_z,
    double grid_x,
    double grid_y,
    double grid_z,
    double receptor_radius) {
  constexpr double ramp_width = 3.709;
  double const d2_low = std::max(
      0.0,
      (1.4 + receptor_radius) * (1.4 + receptor_radius) - ramp_width);
  double const dx = receptor_x - grid_x;
  double const dy = receptor_y - grid_y;
  double const dz = receptor_z - grid_z;
  double const delta = dx * dx + dy * dy + dz * dz - d2_low;
  if (delta < 0.0) return 1.0;
  if (delta >= ramp_width) return 0.0;
  double const scaled = delta / ramp_width;
  double const shoulder = 1.0 - scaled * scaled;
  return shoulder * shoulder;
}

double hard_bridge_fraction(
    double x, double y, double z, double const* waters, int64_t atom) {
  constexpr double overlap_gap = 0.5;
  constexpr double overlap_width = 2.6;
  double best_delta = overlap_width;
  for (int water = 0; water < 4; ++water) {
    if (!finite_water(waters, atom, water)) continue;
    double const* water_xyz = waters + (atom * 4 + water) * 3;
    double const dx = x - water_xyz[0];
    double const dy = y - water_xyz[1];
    double const dz = z - water_xyz[2];
    best_delta = std::min(
        best_delta, dx * dx + dy * dy + dz * dz - overlap_gap);
  }
  if (best_delta < 0.0) return 1.0;
  if (best_delta >= overlap_width) return 0.0;
  double const scaled = best_delta / overlap_width;
  double const shoulder = 1.0 - scaled * scaled;
  return shoulder * shoulder;
}

double directional_lk(
    double distance,
    LJLKTypeParams<double> const& desolvated,
    LJLKTypeParams<double> const& occluding,
    LJGlobalParams<double> const& global) {
  double const sigma = lj_sigma<double>(desolvated, occluding, global);
  return lk_isotropic_pair<double>::V(
      distance,
      5.0,
      sigma,
      desolvated.lj_radius,
      occluding.lk_volume,
      desolvated.lk_coeff,
      desolvated.lk_inv_lambda2,
      global.max_dis,
      desolvated.is_carbon_lk && occluding.is_carbon_lk);
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

  TView<int64_t, 1, tmol::Device::CPU> receptor_types = TCAST(receptor_atom_types);
  TView<int64_t, 1, tmol::Device::CPU> probe_types = TCAST(probe_atom_types);
  int64_t const n_types = ljlk_type_params.size(0);
  TView<double, 2, tmol::Device::CPU> type_params = TCAST(ljlk_type_params);
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
  TView<double, 2, tmol::Device::CPU> lj_global = TCAST(ljlk_global_params);
  TView<double, 1, tmol::Device::CPU> elec_global = TCAST(elec_global_params);
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

std::vector<Tensor> build_grid_impl(
    Tensor receptor_coords,
    Tensor receptor_atom_types,
    Tensor receptor_charges,
    Tensor probe_atom_types,
    Tensor origin,
    Tensor dims,
    double voxel_spacing,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor elec_global_params,
    Tensor receptor_waters,
    Tensor lk_ball_weights,
    bool build_lk_ball) {
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
  if (build_lk_ball) {
    check_double_cpu_contiguous(receptor_waters, "receptor_waters");
    check_double_cpu_contiguous(lk_ball_weights, "lk_ball_weights");
    TORCH_CHECK(
        receptor_waters.dim() == 3
            && receptor_waters.size(0) == receptor_coords.size(0)
            && receptor_waters.size(1) == 4 && receptor_waters.size(2) == 3,
        "receptor_waters must have shape [n_atoms, 4, 3]");
    TORCH_CHECK(
        lk_ball_weights.dim() == 1 && lk_ball_weights.size(0) == 4,
        "lk_ball_weights must have shape [4]");
    for (int64_t atom = 0; atom < receptor_waters.size(0); ++atom) {
      for (int water = 0; water < 4; ++water) {
        double const* xyz =
            receptor_waters.data_ptr<double>() + (atom * 4 + water) * 3;
        bool const all_finite = std::isfinite(xyz[0])
                                && std::isfinite(xyz[1])
                                && std::isfinite(xyz[2]);
        bool const all_nan = std::isnan(xyz[0]) && std::isnan(xyz[1])
                             && std::isnan(xyz[2]);
        TORCH_CHECK(
            all_finite || all_nan,
            "receptor water entries must be finite triples or all-NaN");
      }
    }
    for (int weight = 0; weight < 4; ++weight) {
      TORCH_CHECK(
          std::isfinite(lk_ball_weights.data_ptr<double>()[weight]),
          "lk_ball_weights must be finite");
    }
  }
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

  TView<int64_t, 1, tmol::Device::CPU> dims_a = TCAST(dims);
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
  ElecGlobalParams<double> const elec_global =
      load_elec_global_params(elec_global_params.data_ptr<double>());
  double const elec_max_dis = elec_global.max_dis;
  double const max_dis = std::max(lj_global.max_dis, elec_max_dis);
  TORCH_CHECK(
      std::isfinite(max_dis) && max_dis > 0,
      "interaction cutoff must be finite and positive");
  double const max_dis2 = max_dis * max_dis;
  double const lj_max_dis2 = lj_global.max_dis * lj_global.max_dis;
  double const elec_max_dis2 = elec_max_dis * elec_max_dis;

  TView<double, 2, tmol::Device::CPU> coords = TCAST(receptor_coords);
  TView<double, 1, tmol::Device::CPU> charges = TCAST(receptor_charges);
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
  TView<double, 1, tmol::Device::CPU> origin_a = TCAST(origin);
  TORCH_CHECK(
      std::isfinite(origin_a[0]) && std::isfinite(origin_a[1])
          && std::isfinite(origin_a[2]),
      "origin must be finite");
  ReceptorBins const& receptor_bins = bins;

  std::vector<int64_t> typed_shape{n_probes, nx, ny, nz};
  std::vector<int64_t> shared_shape{nx, ny, nz};
  auto options = receptor_coords.options();
  Tensor fa_atr = tmol::native::empty(typed_shape, options);
  Tensor fa_rep = tmol::native::empty(typed_shape, options);
  Tensor fa_sol = tmol::native::empty(typed_shape, options);
  Tensor fa_elec = tmol::native::empty(shared_shape, options);
  Tensor lk_ball;
  Tensor lk_ball_water;
  Tensor lk_bridge;
  if (build_lk_ball) {
    lk_ball = tmol::native::zeros(typed_shape, options);
    lk_ball_water = tmol::native::zeros(typed_shape, options);
    lk_bridge = tmol::native::zeros(typed_shape, options);
  }

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
  double* lk_ball_ptr = build_lk_ball ? lk_ball.data_ptr<double>() : nullptr;
  double* lk_water_ptr =
      build_lk_ball ? lk_ball_water.data_ptr<double>() : nullptr;
  double* lk_bridge_ptr = build_lk_ball ? lk_bridge.data_ptr<double>() : nullptr;
  double const* receptor_water_ptr =
      build_lk_ball ? receptor_waters.data_ptr<double>() : nullptr;
  double const* lk_weight_ptr =
      build_lk_ball ? lk_ball_weights.data_ptr<double>() : nullptr;

  tmol::native::parallel_for(0, n_voxels, 64, [&](int64_t begin, int64_t end) {
    std::vector<double> atr_sum(static_cast<std::size_t>(n_probes));
    std::vector<double> rep_sum(static_cast<std::size_t>(n_probes));
    std::vector<double> sol_sum(static_cast<std::size_t>(n_probes));
    std::size_t const n_lk_probes =
        build_lk_ball ? static_cast<std::size_t>(n_probes) : 0;
    std::vector<double> lk_ball_sum(n_lk_probes);
    std::vector<double> lk_water_sum(n_lk_probes);
    std::vector<double> lk_bridge_sum(n_lk_probes);
    for (int64_t voxel = begin; voxel < end; ++voxel) {
      std::fill(atr_sum.begin(), atr_sum.end(), 0.0);
      std::fill(rep_sum.begin(), rep_sum.end(), 0.0);
      std::fill(sol_sum.begin(), sol_sum.end(), 0.0);
      std::fill(lk_ball_sum.begin(), lk_ball_sum.end(), 0.0);
      std::fill(lk_water_sum.begin(), lk_water_sum.end(), 0.0);
      std::fill(lk_bridge_sum.begin(), lk_bridge_sum.end(), 0.0);
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
                    elec_global);
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
                if (!build_lk_ball) continue;

                // Rosetta GridScorer bakes the four LK-ball weights into three
                // non-additive grid families. Its synthetic separations affect
                // only the analytic LK halves; the hard fractions use the grid
                // point itself (water_xyz in GridScorer.cc).
                double const weight_iso = lk_weight_ptr[0];
                double const weight_ball = lk_weight_ptr[1];
                double const weight_bridge = lk_weight_ptr[2];
                double const weight_bridge_uncpl = lk_weight_ptr[3];
                bool const receptor_has_waters =
                    has_finite_water(receptor_water_ptr, atom);
                bool const probe_is_polar = probe_params.is_donor != 0
                                            || probe_params.is_acceptor != 0;
                double const fasol1 = directional_lk(
                    dist, probe_params, receptor_params, lj_global);
                double const fasol2 = directional_lk(
                    dist, receptor_params, probe_params, lj_global);

                if (receptor_has_waters) {
                  double const hard_fraction = hard_lk_fraction(
                      x,
                      y,
                      z,
                      receptor_water_ptr,
                      atom,
                      probe_params.lj_radius);
                  lk_ball_sum[probe] +=
                      weight_ball * fasol2 * hard_fraction
                      + weight_iso * fasol2;
                }
                if (!probe_is_polar) continue;

                lk_ball_sum[probe] += weight_iso * fasol1;
                constexpr double heavy_water_distance = 2.65;
                double const fasol1_water = directional_lk(
                    heavy_water_distance,
                    probe_params,
                    receptor_params,
                    lj_global);
                double const water_fraction = hard_single_lk_fraction(
                    coord_ptr[3 * atom],
                    coord_ptr[3 * atom + 1],
                    coord_ptr[3 * atom + 2],
                    x,
                    y,
                    z,
                    receptor_params.lj_radius);
                lk_water_sum[probe] +=
                    weight_ball * fasol1_water * water_fraction;

                if (!receptor_has_waters
                    || (weight_bridge == 0.0
                        && weight_bridge_uncpl == 0.0)) {
                  continue;
                }
                constexpr double water_water_distance =
                    heavy_water_distance * 4.0 * 1.4142135623730951 / 3.0;
                double const fasol1_bridge = directional_lk(
                    water_water_distance,
                    probe_params,
                    receptor_params,
                    lj_global);
                double const fasol2_bridge = directional_lk(
                    water_water_distance,
                    receptor_params,
                    probe_params,
                    lj_global);
                double const bridge_fraction = hard_bridge_fraction(
                    x, y, z, receptor_water_ptr, atom);
                lk_bridge_sum[probe] +=
                    (weight_bridge * (fasol1_bridge + fasol2_bridge)
                     + weight_bridge_uncpl)
                    * bridge_fraction;
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
        if (build_lk_ball) {
          lk_ball_ptr[output_index] = lk_ball_sum[probe];
          lk_water_ptr[output_index] = lk_water_sum[probe];
          lk_bridge_ptr[output_index] = lk_bridge_sum[probe];
        }
      }
    }
  });

  if (build_lk_ball) {
    return {
        fa_atr,
        fa_rep,
        fa_sol,
        fa_elec,
        lk_ball,
        lk_ball_water,
        lk_bridge};
  }
  return {fa_atr, fa_rep, fa_sol, fa_elec};
}

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
  return build_grid_impl(
      receptor_coords,
      receptor_atom_types,
      receptor_charges,
      probe_atom_types,
      origin,
      dims,
      voxel_spacing,
      ljlk_type_params,
      ljlk_global_params,
      elec_global_params,
      Tensor(),
      Tensor(),
      false);
}

std::vector<Tensor> build_grid_lk_ball(
    Tensor receptor_coords,
    Tensor receptor_atom_types,
    Tensor receptor_charges,
    Tensor probe_atom_types,
    Tensor origin,
    Tensor dims,
    double voxel_spacing,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor elec_global_params,
    Tensor receptor_waters,
    Tensor lk_ball_weights) {
  return build_grid_impl(
      receptor_coords,
      receptor_atom_types,
      receptor_charges,
      probe_atom_types,
      origin,
      dims,
      voxel_spacing,
      ljlk_type_params,
      ljlk_global_params,
      elec_global_params,
      receptor_waters,
      lk_ball_weights,
      true);
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
    TORCH_CHECK(value.scalar_type() == tmol::native::ScalarType::Long, name, " must be int64");
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
    TORCH_CHECK(value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(value.scalar_type() == dtype, name, " must match coordinate dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  auto check_int = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(value.scalar_type() == tmol::native::ScalarType::Long, name, " must be int64");
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
          && ligand_water_kinds.size(1) == 4
          && ligand_water_atoms.dim() == 3
          && ligand_water_atoms.size(0) == n_ligand_polar
          && ligand_water_atoms.size(1) == 4
          && ligand_water_atoms.size(2) == 3
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
      cell_origin.dim() == 1 && cell_origin.size(0) == 3
          && cell_dims.dim() == 1 && cell_dims.size(0) == 3,
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
        auto result = LiveLKBallDispatch<
                          common::DeviceOperations,
                          Dev,
                          Real,
                          int64_t>::forward(
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
    TORCH_CHECK(value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(value.scalar_type() == dtype, name, " must match coordinate dtype");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
  };
  auto check_int = [&](Tensor const& value, char const* name) {
    TORCH_CHECK(value.device() == device, name, " must match coordinate device");
    TORCH_CHECK(value.scalar_type() == tmol::native::ScalarType::Long, name, " must be int64");
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
          && ligand_water_atoms.size(1) == 4
          && ligand_water_atoms.size(2) == 3
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
        waters = GridLKBallDispatch<
                     common::DeviceOperations,
                     Dev,
                     Real,
                     int64_t>::build_waters(
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
        gradients = GridLKBallDispatch<
                        common::DeviceOperations,
                        Dev,
                        Real,
                        int64_t>::propagate_water_gradients(
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
  m.def("build_grid_lk_ball", TMOL_FN(&build_grid_lk_ball));
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
