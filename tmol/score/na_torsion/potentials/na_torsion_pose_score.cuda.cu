#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cmath>

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {

constexpr int N_TORSION = 10;
constexpr int N_PUCKER = 10;
constexpr int ALPHA = 0;
constexpr int BETA = 1;
constexpr int GAMMA = 2;
constexpr int DELTA = 3;
constexpr int EPSILON = 4;
constexpr int ZETA = 5;
constexpr int CHI = 9;

template <typename Real>
struct Vec3 {
  Real x, y, z;
};

template <typename Real>
__device__ __forceinline__ Vec3<Real> operator+(Vec3<Real> a, Vec3<Real> b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename Real>
__device__ __forceinline__ Vec3<Real> operator-(Vec3<Real> a, Vec3<Real> b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename Real>
__device__ __forceinline__ Vec3<Real> operator*(Vec3<Real> a, Real s) {
  return {a.x * s, a.y * s, a.z * s};
}

template <typename Real>
__device__ __forceinline__ Vec3<Real>& operator+=(Vec3<Real>& a, Vec3<Real> b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

template <typename Real>
__device__ __forceinline__ Vec3<Real>& operator-=(Vec3<Real>& a, Vec3<Real> b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

template <typename Real>
__device__ __forceinline__ Real dot(Vec3<Real> a, Vec3<Real> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename Real>
__device__ __forceinline__ Real norm(Vec3<Real> value) {
  return sqrt(dot(value, value));
}

template <typename Real>
__device__ __forceinline__ Vec3<Real> cross(Vec3<Real> a, Vec3<Real> b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template <typename Real>
__device__ __forceinline__ Vec3<Real> unit(Vec3<Real> value) {
  Real magnitude = norm(value);
  Real inv = Real(1) / (magnitude > Real(1e-9) ? magnitude : Real(1e-9));
  return value * inv;
}

template <typename Real>
__device__ __forceinline__ Vec3<Real> reverse_unit(
    Vec3<Real> value, Vec3<Real> normalized, Vec3<Real> d_normalized) {
  Real magnitude = norm(value);
  if (magnitude <= Real(1e-9)) return d_normalized * Real(1e9);
  return (d_normalized - normalized * dot(normalized, d_normalized))
         * (Real(1) / magnitude);
}

template <typename Real>
__device__ __forceinline__ Vec3<Real> load_coord(
    Real const* coords, int64_t atom) {
  Real const* xyz = coords + atom * 3;
  return {xyz[0], xyz[1], xyz[2]};
}

template <typename Real>
__device__ __forceinline__ Real mod360(Real angle) {
  angle = fmod(angle, Real(360));
  return angle < Real(0) ? angle + Real(360) : angle;
}

template <typename Real>
__device__ __forceinline__ Real wrap_degrees(Real angle) {
  return mod360(angle + Real(180)) - Real(180);
}

template <typename Real>
__device__ __forceinline__ Real sigmoid(Real value) {
  return Real(1) / (Real(1) + exp(-value));
}

template <typename Real>
__device__ __forceinline__ Real
dihedral_degrees(Vec3<Real> p0, Vec3<Real> p1, Vec3<Real> p2, Vec3<Real> p3) {
  Vec3<Real> b0 = p0 - p1;
  Vec3<Real> b1 = unit(p2 - p1);
  Vec3<Real> b2 = p3 - p2;
  Vec3<Real> v = b0 - b1 * dot(b0, b1);
  Vec3<Real> w = b2 - b1 * dot(b2, b1);
  Real y = dot(cross(b1, v), w);
  Real x = dot(v, w);
  return mod360(atan2(y, x) * Real(180.0 / 3.14159265358979323846));
}

template <typename Real>
__device__ __forceinline__ Real torsion_angle(
    Real const* coords,
    int64_t const* torsion_indices,
    bool const* torsion_ok,
    int flat_block,
    int torsion) {
  if (!torsion_ok[flat_block * N_TORSION + torsion]) return Real(0);
  int offset = (flat_block * N_TORSION + torsion) * 4;
  return dihedral_degrees(
      load_coord(coords, torsion_indices[offset]),
      load_coord(coords, torsion_indices[offset + 1]),
      load_coord(coords, torsion_indices[offset + 2]),
      load_coord(coords, torsion_indices[offset + 3]));
}

template <typename Real>
__device__ __forceinline__ void pucker_weights(
    Real const* coords,
    int64_t const* ring_indices,
    bool const* ring_ok,
    int flat_block,
    Real temperature,
    Real* pucker) {
  Vec3<Real> ring[5];
#pragma unroll
  for (int atom = 0; atom < 5; ++atom) {
    int offset = flat_block * 5 + atom;
    ring[atom] = ring_ok[offset] ? load_coord(coords, ring_indices[offset])
                                 : Vec3<Real>{Real(0), Real(0), Real(0)};
  }

  Real plane[5];
  Real exxo[5];
  Real min_plane = Real(1e20);
#pragma unroll
  for (int rotation = 0; rotation < 5; ++rotation) {
    Vec3<Real> a0 = ring[rotation];
    Vec3<Real> a1 = ring[(rotation + 1) % 5];
    Vec3<Real> a2 = ring[(rotation + 2) % 5];
    Vec3<Real> a3 = ring[(rotation + 3) % 5];
    Vec3<Real> a4 = ring[(rotation + 4) % 5];
    Vec3<Real> normal = unit(cross(a1 - a0, a2 - a1));
    plane[rotation] = abs(dot(normal, unit(a3 - a2)));
    exxo[rotation] = dot(normal, unit(a4 - (a3 + a0) * Real(0.5)));
    min_plane = plane[rotation] < min_plane ? plane[rotation] : min_plane;
  }

  Real rotation_weight[5];
  Real weight_sum = Real(0);
#pragma unroll
  for (int rotation = 0; rotation < 5; ++rotation) {
    rotation_weight[rotation] =
        exp(-(plane[rotation] - min_plane) / temperature);
    weight_sum += rotation_weight[rotation];
  }

  constexpr int slot[10] = {9, 0, 6, 2, 8, 4, 5, 1, 7, 3};
#pragma unroll
  for (int rotation = 0; rotation < 5; ++rotation) {
    Real weight = rotation_weight[rotation] / weight_sum;
    Real endo = sigmoid(Real(-2) * exxo[rotation] / temperature);
    pucker[slot[rotation]] = weight * endo;
    pucker[slot[rotation + 5]] = weight * (Real(1) - endo);
  }
}

template <typename Real>
__device__ __forceinline__ void triple_bin_weights(
    Real angle, Real const* means, Real sdev, Real* weights) {
  Real total = Real(0);
#pragma unroll
  for (int bin = 0; bin < 3; ++bin) {
    Real dev = wrap_degrees(angle - means[bin]);
    weights[bin] = exp(-dev * dev / (Real(2) * sdev * sdev));
    total += weights[bin];
  }
#pragma unroll
  for (int bin = 0; bin < 3; ++bin) weights[bin] /= total;
}

template <typename Real>
__device__ __forceinline__ void triple_bin_weights_deriv(
    Real angle,
    Real const* means,
    Real sdev,
    Real* weights,
    Real* derivatives) {
  triple_bin_weights(angle, means, sdev, weights);
  Real mean_log_deriv = Real(0);
#pragma unroll
  for (int bin = 0; bin < 3; ++bin) {
    Real dev = wrap_degrees(angle - means[bin]);
    derivatives[bin] = -dev / (sdev * sdev);
    mean_log_deriv += weights[bin] * derivatives[bin];
  }
#pragma unroll
  for (int bin = 0; bin < 3; ++bin) {
    derivatives[bin] = weights[bin] * (derivatives[bin] - mean_log_deriv);
  }
}

template <typename Real>
__device__ __forceinline__ Real
blended_devsq(Real angle, Real const* means, Real const* weights, int n_bins) {
  Real value = Real(0);
  for (int bin = 0; bin < n_bins; ++bin) {
    Real dev = wrap_degrees(angle - means[bin]);
    value += weights[bin] * dev * dev;
  }
  return value;
}

template <typename Real>
__device__ __forceinline__ Real blended_devsq_deriv(
    Real angle,
    Real const* means,
    Real const* weights,
    Real const* weight_derivatives,
    int n_bins) {
  Real derivative = Real(0);
  for (int bin = 0; bin < n_bins; ++bin) {
    Real dev = wrap_degrees(angle - means[bin]);
    derivative +=
        weight_derivatives[bin] * dev * dev + Real(2) * weights[bin] * dev;
  }
  return derivative;
}

template <typename Real>
__device__ __forceinline__ Real
bi_bii_weight_deriv(Real epsilon, Real zeta, Real weight) {
  constexpr Real rad = Real(3.14159265358979323846 / 180.0);
  Real delta = wrap_degrees(epsilon - zeta) * rad;
  return weight * (Real(1) - weight) * Real(-40) * cos(delta) * rad;
}

template <typename Real>
__device__ __forceinline__ Real syn_weight_deriv(Real chi, Real& weight) {
  Real wrapped = mod360(chi);
  Real lower = sigmoid((wrapped - Real(20)) / Real(5));
  Real upper = sigmoid((Real(100) - wrapped) / Real(5));
  weight = lower * upper;
  return lower * (Real(1) - lower) * upper / Real(5)
         - lower * upper * (Real(1) - upper) / Real(5);
}

template <typename Real>
struct DihedralDeriv {
  Vec3<Real> atom[4];
};

template <typename Real>
__device__ __forceinline__ DihedralDeriv<Real> dihedral_deriv(
    Vec3<Real> i, Vec3<Real> j, Vec3<Real> k, Vec3<Real> l) {
  Vec3<Real> f = i - j;
  Vec3<Real> g = j - k;
  Vec3<Real> h = l - k;
  Vec3<Real> a = cross(f, g);
  Vec3<Real> b = cross(h, g);
  Real a2 = dot(a, a);
  Real b2 = dot(b, b);
  Real gnorm = norm(g);
  Real fg = dot(f, g);
  Real hg = dot(h, g);
  DihedralDeriv<Real> result;
  result.atom[0] = a * (-gnorm / a2);
  result.atom[1] =
      a * (gnorm / a2 + fg / (a2 * gnorm)) - b * (hg / (b2 * gnorm));
  result.atom[2] =
      b * (-gnorm / b2 + hg / (b2 * gnorm)) - a * (fg / (a2 * gnorm));
  result.atom[3] = b * (gnorm / b2);
  return result;
}

template <typename Real>
__device__ __forceinline__ void add_torsion_gradient(
    Real const* coords,
    int64_t const* torsion_indices,
    bool const* torsion_ok,
    int flat_block,
    int torsion,
    Real harmonic_deriv,
    Real well_deriv,
    int n_atoms,
    Real* derivatives) {
  if (!torsion_ok[flat_block * N_TORSION + torsion]) return;
  int offset = (flat_block * N_TORSION + torsion) * 4;
  int64_t atom[4];
  Vec3<Real> xyz[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    atom[i] = torsion_indices[offset + i];
    xyz[i] = load_coord(coords, atom[i]);
  }
  auto d_angle = dihedral_deriv(xyz[0], xyz[1], xyz[2], xyz[3]);
  constexpr Real degrees_per_radian = Real(180.0 / 3.14159265358979323846);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int64_t coord = atom[i] * 3;
    Vec3<Real> d = d_angle.atom[i] * degrees_per_radian;
    atomicAdd(derivatives + coord, harmonic_deriv * d.x);
    atomicAdd(derivatives + coord + 1, harmonic_deriv * d.y);
    atomicAdd(derivatives + coord + 2, harmonic_deriv * d.z);
    atomicAdd(derivatives + n_atoms * 3 + coord, well_deriv * d.x);
    atomicAdd(derivatives + n_atoms * 3 + coord + 1, well_deriv * d.y);
    atomicAdd(derivatives + n_atoms * 3 + coord + 2, well_deriv * d.z);
  }
}

template <typename Real>
__device__ __forceinline__ void add_pucker_gradient(
    Real const* coords,
    int64_t const* ring_indices,
    bool const* ring_ok,
    int flat_block,
    Real temperature,
    Real const* pucker,
    Real const d_energy_dpucker[2][N_PUCKER],
    int n_atoms,
    Real* derivatives) {
  constexpr int slot[N_PUCKER] = {9, 0, 6, 2, 8, 4, 5, 1, 7, 3};
  Vec3<Real> ring[5];
#pragma unroll
  for (int atom = 0; atom < 5; ++atom) {
    int offset = flat_block * 5 + atom;
    ring[atom] = ring_ok[offset] ? load_coord(coords, ring_indices[offset])
                                 : Vec3<Real>{Real(0), Real(0), Real(0)};
  }

  Real mean_coeff[2] = {Real(0), Real(0)};
#pragma unroll
  for (int rotation = 0; rotation < 5; ++rotation) {
    Real endo = pucker[slot[rotation]];
    Real exo = pucker[slot[rotation + 5]];
    Real weight = endo + exo;
    Real p_endo = endo / weight;
#pragma unroll
    for (int score = 0; score < 2; ++score) {
      Real coeff =
          p_endo * d_energy_dpucker[score][slot[rotation]]
          + (Real(1) - p_endo) * d_energy_dpucker[score][slot[rotation + 5]];
      mean_coeff[score] += weight * coeff;
    }
  }

  Vec3<Real> ring_gradient[2][5];
#pragma unroll
  for (int score = 0; score < 2; ++score) {
#pragma unroll
    for (int atom = 0; atom < 5; ++atom) {
      ring_gradient[score][atom] = {Real(0), Real(0), Real(0)};
    }
  }

#pragma unroll
  for (int rotation = 0; rotation < 5; ++rotation) {
    int i0 = rotation;
    int i1 = (rotation + 1) % 5;
    int i2 = (rotation + 2) % 5;
    int i3 = (rotation + 3) % 5;
    int i4 = (rotation + 4) % 5;
    Vec3<Real> u = ring[i1] - ring[i0];
    Vec3<Real> v = ring[i2] - ring[i1];
    Vec3<Real> c = cross(u, v);
    Vec3<Real> n = unit(c);
    Vec3<Real> edge = ring[i3] - ring[i2];
    Vec3<Real> m = unit(edge);
    Real q = dot(n, m);
    Vec3<Real> apex = ring[i4] - (ring[i3] + ring[i0]) * Real(0.5);
    Vec3<Real> h = unit(apex);
    Real exxo = dot(n, h);
    Real endo = pucker[slot[rotation]];
    Real exo = pucker[slot[rotation + 5]];
    Real weight = endo + exo;
    Real p_endo = endo / weight;

#pragma unroll
    for (int score = 0; score < 2; ++score) {
      Real endo_coeff = d_energy_dpucker[score][slot[rotation]];
      Real exo_coeff = d_energy_dpucker[score][slot[rotation + 5]];
      Real coeff = p_endo * endo_coeff + (Real(1) - p_endo) * exo_coeff;
      Real d_plane = -weight * (coeff - mean_coeff[score]) / temperature;
      Real d_exxo = weight * (endo_coeff - exo_coeff) * p_endo
                    * (Real(1) - p_endo) * Real(-2) / temperature;
      Real abs_deriv =
          q > Real(0) ? Real(1) : (q < Real(0) ? Real(-1) : Real(0));
      Real d_q = d_plane * abs_deriv;

      Vec3<Real> d_n = m * d_q + h * d_exxo;
      Vec3<Real> d_m = n * d_q;
      Vec3<Real> d_h = n * d_exxo;
      Vec3<Real> d_c = reverse_unit(c, n, d_n);
      Vec3<Real> d_u = cross(v, d_c);
      Vec3<Real> d_v = cross(d_c, u);
      Vec3<Real> d_edge = reverse_unit(edge, m, d_m);
      Vec3<Real> d_apex = reverse_unit(apex, h, d_h);

      ring_gradient[score][i0] -= d_u + d_apex * Real(0.5);
      ring_gradient[score][i1] += d_u - d_v;
      ring_gradient[score][i2] += d_v - d_edge;
      ring_gradient[score][i3] += d_edge - d_apex * Real(0.5);
      ring_gradient[score][i4] += d_apex;
    }
  }

#pragma unroll
  for (int atom = 0; atom < 5; ++atom) {
    int offset = flat_block * 5 + atom;
    if (!ring_ok[offset]) continue;
    int64_t coord = ring_indices[offset] * 3;
#pragma unroll
    for (int score = 0; score < 2; ++score) {
      Real* out = derivatives + score * n_atoms * 3 + coord;
      atomicAdd(out, ring_gradient[score][atom].x);
      atomicAdd(out + 1, ring_gradient[score][atom].y);
      atomicAdd(out + 2, ring_gradient[score][atom].z);
    }
  }
}

template <typename Real>
__global__ void na_torsion_forward_kernel(
    Real const* coords,
    int64_t const* base,
    bool const* is_na,
    int64_t const* torsion_indices,
    bool const* torsion_ok,
    int64_t const* ring_indices,
    bool const* ring_ok,
    int64_t const* prev,
    Real const* backbone_means,
    Real const* backbone_sdev,
    Real const* sugar_means,
    Real const* chi_means,
    Real const* sdev_sugar,
    Real const* sdev_chi,
    Real const* well_pucker,
    Real const* well_alpha_gamma,
    Real const* well_bibii_pucker,
    Real const* well_alphanext_bibii,
    Real const* well_chi_syn,
    bool const* is_north,
    Real const* weight_bb,
    Real const* weight_chi,
    Real const* weight_sugar,
    Real pucker_temperature,
    Real bin_blend_sdev,
    int n_poses,
    int max_n_blocks,
    Real* output) {
  int flat_block = blockIdx.x * blockDim.x + threadIdx.x;
  int n_blocks = n_poses * max_n_blocks;
  if (flat_block >= n_blocks || !is_na[flat_block]) return;

  int pose = flat_block / max_n_blocks;
  int base_ind = int(base[flat_block]);
  int polymer = base_ind >> 2;
  Real torsion[N_TORSION];
#pragma unroll
  for (int tor = 0; tor < N_TORSION; ++tor) {
    torsion[tor] =
        torsion_angle(coords, torsion_indices, torsion_ok, flat_block, tor);
  }
  Real pucker[N_PUCKER];
  pucker_weights(
      coords, ring_indices, ring_ok, flat_block, pucker_temperature, pucker);

  Real const* means = backbone_means + polymer * 6 * 3;
  Real const* sdev_bb = backbone_sdev + polymer * 6;
  Real zero = Real(0);
  Real e_bb = zero;
  Real alpha_w[3], gamma_w[3];
  triple_bin_weights(
      torsion[ALPHA], means + ALPHA * 3, bin_blend_sdev, alpha_w);
  triple_bin_weights(
      torsion[GAMMA], means + GAMMA * 3, bin_blend_sdev, gamma_w);
  if (torsion_ok[flat_block * N_TORSION + ALPHA]) {
    e_bb += blended_devsq(torsion[ALPHA], means + ALPHA * 3, alpha_w, 3)
            / (sdev_bb[ALPHA] * sdev_bb[ALPHA]);
  }
  if (torsion_ok[flat_block * N_TORSION + GAMMA]) {
    e_bb += blended_devsq(torsion[GAMMA], means + GAMMA * 3, gamma_w, 3)
            / (sdev_bb[GAMMA] * sdev_bb[GAMMA]);
  }

  bool both = torsion_ok[flat_block * N_TORSION + EPSILON]
              && torsion_ok[flat_block * N_TORSION + ZETA];
  Real w_bi = sigmoid(
      Real(-40)
      * sin(
          wrap_degrees(torsion[EPSILON] - torsion[ZETA])
          * Real(3.14159265358979323846 / 180.0)));
  Real bibii_w[2] = {w_bi, Real(1) - w_bi};
  if (both) {
    for (int tor = EPSILON; tor <= ZETA; ++tor) {
      e_bb += blended_devsq(torsion[tor], means + tor * 3, bibii_w, 2)
              / (sdev_bb[tor] * sdev_bb[tor]);
    }
  }

  int prev_block = int(prev[flat_block]);
  bool prev_ok = prev_block >= 0 && torsion_ok[prev_block * N_TORSION + EPSILON]
                 && torsion_ok[prev_block * N_TORSION + ZETA];
  Real w_beta = Real(1);
  if (prev_ok) {
    Real prev_epsilon =
        torsion_angle(coords, torsion_indices, torsion_ok, prev_block, EPSILON);
    Real prev_zeta =
        torsion_angle(coords, torsion_indices, torsion_ok, prev_block, ZETA);
    w_beta = sigmoid(
        Real(-40)
        * sin(
            wrap_degrees(prev_epsilon - prev_zeta)
            * Real(3.14159265358979323846 / 180.0)));
  }
  Real beta_w[2] = {w_beta, Real(1) - w_beta};
  if (torsion_ok[flat_block * N_TORSION + BETA]) {
    e_bb += blended_devsq(torsion[BETA], means + BETA * 3, beta_w, 2)
            / (sdev_bb[BETA] * sdev_bb[BETA]);
  }

  Real chi = torsion[CHI];
  Real chi_mod = mod360(chi);
  Real w_syn = sigmoid((chi_mod - Real(20)) / Real(5))
               * sigmoid((Real(100) - chi_mod) / Real(5));
  Real e_chi = zero;
  if (torsion_ok[flat_block * N_TORSION + CHI]) {
    Real dev_syn = wrap_degrees(chi - Real(50));
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      Real dev = wrap_degrees(chi - chi_means[base_ind * N_PUCKER + puck]);
      e_chi += pucker[puck]
               * ((Real(1) - w_syn) * dev * dev + w_syn * dev_syn * dev_syn);
    }
    e_chi /= sdev_chi[polymer] * sdev_chi[polymer];
  }

  constexpr int sugar_torsion[4] = {DELTA, 6, 7, 8};
  Real e_sugar = zero;
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    int tor = sugar_torsion[slot];
    if (!torsion_ok[flat_block * N_TORSION + tor]) continue;
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      Real dev = wrap_degrees(
          torsion[tor] - sugar_means[(polymer * N_PUCKER + puck) * 4 + slot]);
      e_sugar += pucker[puck] * dev * dev;
    }
  }
  e_sugar /= sdev_sugar[polymer] * sdev_sugar[polymer];

  Real e_well = zero;
#pragma unroll
  for (int puck = 0; puck < N_PUCKER; ++puck) {
    e_well += pucker[puck] * well_pucker[polymer * N_PUCKER + puck];
  }
  if (torsion_ok[flat_block * N_TORSION + ALPHA]
      && torsion_ok[flat_block * N_TORSION + GAMMA]) {
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int g = 0; g < 3; ++g) {
        e_well += alpha_w[a] * well_alpha_gamma[(polymer * 3 + a) * 3 + g]
                  * gamma_w[g];
      }
    }
  }
  if (both) {
    Real north = zero;
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      if (is_north[puck]) north += pucker[puck];
    }
    Real ns[2] = {north, Real(1) - north};
#pragma unroll
    for (int bi = 0; bi < 2; ++bi) {
#pragma unroll
      for (int state = 0; state < 2; ++state) {
        e_well += bibii_w[bi]
                  * well_bibii_pucker[(polymer * 2 + bi) * 2 + state]
                  * ns[state];
      }
    }
  }
  if (prev_ok && torsion_ok[flat_block * N_TORSION + ALPHA]) {
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int bi = 0; bi < 2; ++bi) {
        e_well += alpha_w[a] * well_alphanext_bibii[(polymer * 3 + a) * 2 + bi]
                  * beta_w[bi];
      }
    }
  }
  if (torsion_ok[flat_block * N_TORSION + CHI]) {
#pragma unroll
    for (int syn = 0; syn < 2; ++syn) {
      Real syn_w = syn == 0 ? Real(1) - w_syn : w_syn;
#pragma unroll
      for (int puck = 0; puck < N_PUCKER; ++puck) {
        e_well += syn_w * well_chi_syn[(syn * N_PUCKER + puck) * 8 + base_ind]
                  * pucker[puck];
      }
    }
  }

  Real harmonic = weight_bb[polymer] * e_bb + weight_chi[polymer] * e_chi
                  + weight_sugar[polymer] * e_sugar;
  atomicAdd(output + pose, harmonic);
  atomicAdd(output + n_poses + pose, e_well);
}

template <typename Real>
// The gradient path accumulates the two energies while their shared torsion,
// pucker, and bin intermediates are live, avoiding a separate forward kernel.
__global__ void na_torsion_derivative_kernel(
    Real const* coords,
    int64_t const* base,
    bool const* is_na,
    int64_t const* torsion_indices,
    bool const* torsion_ok,
    int64_t const* ring_indices,
    bool const* ring_ok,
    int64_t const* prev,
    Real const* backbone_means,
    Real const* backbone_sdev,
    Real const* sugar_means,
    Real const* chi_means,
    Real const* sdev_sugar,
    Real const* sdev_chi,
    Real const* well_pucker,
    Real const* well_alpha_gamma,
    Real const* well_bibii_pucker,
    Real const* well_alphanext_bibii,
    Real const* well_chi_syn,
    bool const* is_north,
    Real const* weight_bb,
    Real const* weight_chi,
    Real const* weight_sugar,
    Real pucker_temperature,
    Real bin_blend_sdev,
    int n_poses,
    int max_n_blocks,
    int n_atoms,
    Real* derivatives,
    Real* output) {
  int flat_block = blockIdx.x * blockDim.x + threadIdx.x;
  int n_blocks = n_poses * max_n_blocks;
  if (flat_block >= n_blocks || !is_na[flat_block]) return;

  int pose = flat_block / max_n_blocks;
  int base_ind = int(base[flat_block]);
  int polymer = base_ind >> 2;
  Real torsion[N_TORSION];
#pragma unroll
  for (int tor = 0; tor < N_TORSION; ++tor) {
    torsion[tor] =
        torsion_angle(coords, torsion_indices, torsion_ok, flat_block, tor);
  }
  Real pucker[N_PUCKER];
  pucker_weights(
      coords, ring_indices, ring_ok, flat_block, pucker_temperature, pucker);

  Real d_angle[2][N_TORSION] = {};
  Real d_pucker[2][N_PUCKER] = {};
  Real d_prev[2][2] = {};
  Real energy[2] = {};
  Real const* means = backbone_means + polymer * 6 * 3;
  Real const* sdev_bb = backbone_sdev + polymer * 6;
  Real bb_weight = weight_bb[polymer];
  Real chi_weight = weight_chi[polymer];
  Real sugar_weight = weight_sugar[polymer];

  Real alpha_w[3], alpha_dw[3], gamma_w[3], gamma_dw[3];
  triple_bin_weights_deriv(
      torsion[ALPHA], means + ALPHA * 3, bin_blend_sdev, alpha_w, alpha_dw);
  triple_bin_weights_deriv(
      torsion[GAMMA], means + GAMMA * 3, bin_blend_sdev, gamma_w, gamma_dw);
  if (torsion_ok[flat_block * N_TORSION + ALPHA]) {
    Real inv_var = Real(1) / (sdev_bb[ALPHA] * sdev_bb[ALPHA]);
    energy[0] += bb_weight
                 * blended_devsq(torsion[ALPHA], means + ALPHA * 3, alpha_w, 3)
                 * inv_var;
    d_angle[0][ALPHA] +=
        bb_weight
        * blended_devsq_deriv(
            torsion[ALPHA], means + ALPHA * 3, alpha_w, alpha_dw, 3)
        * inv_var;
  }
  if (torsion_ok[flat_block * N_TORSION + GAMMA]) {
    Real inv_var = Real(1) / (sdev_bb[GAMMA] * sdev_bb[GAMMA]);
    energy[0] += bb_weight
                 * blended_devsq(torsion[GAMMA], means + GAMMA * 3, gamma_w, 3)
                 * inv_var;
    d_angle[0][GAMMA] +=
        bb_weight
        * blended_devsq_deriv(
            torsion[GAMMA], means + GAMMA * 3, gamma_w, gamma_dw, 3)
        * inv_var;
  }

  bool both = torsion_ok[flat_block * N_TORSION + EPSILON]
              && torsion_ok[flat_block * N_TORSION + ZETA];
  Real w_bi = sigmoid(
      Real(-40)
      * sin(
          wrap_degrees(torsion[EPSILON] - torsion[ZETA])
          * Real(3.14159265358979323846 / 180.0)));
  Real dw_bi = bi_bii_weight_deriv(torsion[EPSILON], torsion[ZETA], w_bi);
  if (both) {
    Real weight_derivative = Real(0);
    for (int tor = EPSILON; tor <= ZETA; ++tor) {
      Real dev_bi = wrap_degrees(torsion[tor] - means[tor * 3]);
      Real dev_bii = wrap_degrees(torsion[tor] - means[tor * 3 + 1]);
      Real inv_var = Real(1) / (sdev_bb[tor] * sdev_bb[tor]);
      energy[0] +=
          bb_weight
          * (w_bi * dev_bi * dev_bi + (Real(1) - w_bi) * dev_bii * dev_bii)
          * inv_var;
      d_angle[0][tor] += bb_weight * Real(2)
                         * (w_bi * dev_bi + (Real(1) - w_bi) * dev_bii)
                         * inv_var;
      weight_derivative += (dev_bi * dev_bi - dev_bii * dev_bii) * inv_var;
    }
    d_angle[0][EPSILON] += bb_weight * weight_derivative * dw_bi;
    d_angle[0][ZETA] -= bb_weight * weight_derivative * dw_bi;
  }

  int prev_block = int(prev[flat_block]);
  bool prev_ok = prev_block >= 0 && torsion_ok[prev_block * N_TORSION + EPSILON]
                 && torsion_ok[prev_block * N_TORSION + ZETA];
  Real prev_w_bi = Real(1);
  Real prev_dw_bi = Real(0);
  if (prev_ok) {
    Real prev_epsilon =
        torsion_angle(coords, torsion_indices, torsion_ok, prev_block, EPSILON);
    Real prev_zeta =
        torsion_angle(coords, torsion_indices, torsion_ok, prev_block, ZETA);
    prev_w_bi = sigmoid(
        Real(-40)
        * sin(
            wrap_degrees(prev_epsilon - prev_zeta)
            * Real(3.14159265358979323846 / 180.0)));
    prev_dw_bi = bi_bii_weight_deriv(prev_epsilon, prev_zeta, prev_w_bi);
  }
  if (torsion_ok[flat_block * N_TORSION + BETA]) {
    Real dev_bi = wrap_degrees(torsion[BETA] - means[BETA * 3]);
    Real dev_bii = wrap_degrees(torsion[BETA] - means[BETA * 3 + 1]);
    Real inv_var = Real(1) / (sdev_bb[BETA] * sdev_bb[BETA]);
    energy[0] += bb_weight
                 * (prev_w_bi * dev_bi * dev_bi
                    + (Real(1) - prev_w_bi) * dev_bii * dev_bii)
                 * inv_var;
    d_angle[0][BETA] += bb_weight * Real(2)
                        * (prev_w_bi * dev_bi + (Real(1) - prev_w_bi) * dev_bii)
                        * inv_var;
    if (prev_ok) {
      Real through_weight = bb_weight * (dev_bi * dev_bi - dev_bii * dev_bii)
                            * inv_var * prev_dw_bi;
      d_prev[0][0] += through_weight;
      d_prev[0][1] -= through_weight;
    }
  }

  Real w_syn;
  Real dw_syn = syn_weight_deriv(torsion[CHI], w_syn);
  if (torsion_ok[flat_block * N_TORSION + CHI]) {
    Real dev_syn = wrap_degrees(torsion[CHI] - Real(50));
    Real inv_var = Real(1) / (sdev_chi[polymer] * sdev_chi[polymer]);
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      Real dev =
          wrap_degrees(torsion[CHI] - chi_means[base_ind * N_PUCKER + puck]);
      Real coeff =
          ((Real(1) - w_syn) * dev * dev + w_syn * dev_syn * dev_syn) * inv_var;
      energy[0] += chi_weight * pucker[puck] * coeff;
      d_pucker[0][puck] += chi_weight * coeff;
      d_angle[0][CHI] +=
          chi_weight * pucker[puck]
          * (Real(2) * ((Real(1) - w_syn) * dev + w_syn * dev_syn)
             + dw_syn * (dev_syn * dev_syn - dev * dev))
          * inv_var;
    }
  }

  constexpr int sugar_torsion[4] = {DELTA, 6, 7, 8};
  Real sugar_inv_var = Real(1) / (sdev_sugar[polymer] * sdev_sugar[polymer]);
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    int tor = sugar_torsion[slot];
    if (!torsion_ok[flat_block * N_TORSION + tor]) continue;
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      Real dev = wrap_degrees(
          torsion[tor] - sugar_means[(polymer * N_PUCKER + puck) * 4 + slot]);
      energy[0] += sugar_weight * pucker[puck] * dev * dev * sugar_inv_var;
      d_pucker[0][puck] += sugar_weight * dev * dev * sugar_inv_var;
      d_angle[0][tor] +=
          sugar_weight * pucker[puck] * Real(2) * dev * sugar_inv_var;
    }
  }

#pragma unroll
  for (int puck = 0; puck < N_PUCKER; ++puck) {
    Real well = well_pucker[polymer * N_PUCKER + puck];
    energy[1] += pucker[puck] * well;
    d_pucker[1][puck] += well;
  }

  bool alpha_gamma_ok = torsion_ok[flat_block * N_TORSION + ALPHA]
                        && torsion_ok[flat_block * N_TORSION + GAMMA];
  if (alpha_gamma_ok) {
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int g = 0; g < 3; ++g) {
        Real table = well_alpha_gamma[(polymer * 3 + a) * 3 + g];
        energy[1] += alpha_w[a] * table * gamma_w[g];
        d_angle[1][ALPHA] += alpha_dw[a] * table * gamma_w[g];
        d_angle[1][GAMMA] += alpha_w[a] * table * gamma_dw[g];
      }
    }
  }

  if (both) {
    Real north = Real(0);
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      if (is_north[puck]) north += pucker[puck];
    }
    Real ns[2] = {north, Real(1) - north};
    Real state_value[2] = {Real(0), Real(0)};
    Real weight_derivative = Real(0);
#pragma unroll
    for (int state = 0; state < 2; ++state) {
      Real bi = well_bibii_pucker[(polymer * 2) * 2 + state];
      Real bii = well_bibii_pucker[(polymer * 2 + 1) * 2 + state];
      state_value[state] = w_bi * bi + (Real(1) - w_bi) * bii;
      energy[1] += ns[state] * state_value[state];
      weight_derivative += ns[state] * (bi - bii);
    }
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      if (is_north[puck]) {
        d_pucker[1][puck] += state_value[0] - state_value[1];
      }
    }
    d_angle[1][EPSILON] += weight_derivative * dw_bi;
    d_angle[1][ZETA] -= weight_derivative * dw_bi;
  }

  if (prev_ok && torsion_ok[flat_block * N_TORSION + ALPHA]) {
    Real prev_weight_derivative = Real(0);
#pragma unroll
    for (int a = 0; a < 3; ++a) {
      Real bi = well_alphanext_bibii[(polymer * 3 + a) * 2];
      Real bii = well_alphanext_bibii[(polymer * 3 + a) * 2 + 1];
      Real state_value = prev_w_bi * bi + (Real(1) - prev_w_bi) * bii;
      energy[1] += alpha_w[a] * state_value;
      d_angle[1][ALPHA] += alpha_dw[a] * state_value;
      prev_weight_derivative += alpha_w[a] * (bi - bii);
    }
    Real through_weight = prev_weight_derivative * prev_dw_bi;
    d_prev[1][0] += through_weight;
    d_prev[1][1] -= through_weight;
  }

  if (torsion_ok[flat_block * N_TORSION + CHI]) {
    Real chi_weight_derivative = Real(0);
#pragma unroll
    for (int puck = 0; puck < N_PUCKER; ++puck) {
      Real anti = well_chi_syn[puck * 8 + base_ind];
      Real syn = well_chi_syn[(N_PUCKER + puck) * 8 + base_ind];
      Real state_value = (Real(1) - w_syn) * anti + w_syn * syn;
      energy[1] += pucker[puck] * state_value;
      d_pucker[1][puck] += state_value;
      chi_weight_derivative += pucker[puck] * (syn - anti);
    }
    d_angle[1][CHI] += dw_syn * chi_weight_derivative;
  }

  atomicAdd(output + pose, energy[0]);
  atomicAdd(output + n_poses + pose, energy[1]);

#pragma unroll
  for (int tor = 0; tor < N_TORSION; ++tor) {
    add_torsion_gradient(
        coords,
        torsion_indices,
        torsion_ok,
        flat_block,
        tor,
        d_angle[0][tor],
        d_angle[1][tor],
        n_atoms,
        derivatives);
  }
  if (prev_ok) {
    add_torsion_gradient(
        coords,
        torsion_indices,
        torsion_ok,
        prev_block,
        EPSILON,
        d_prev[0][0],
        d_prev[1][0],
        n_atoms,
        derivatives);
    add_torsion_gradient(
        coords,
        torsion_indices,
        torsion_ok,
        prev_block,
        ZETA,
        d_prev[0][1],
        d_prev[1][1],
        n_atoms,
        derivatives);
  }
  add_pucker_gradient(
      coords,
      ring_indices,
      ring_ok,
      flat_block,
      pucker_temperature,
      pucker,
      d_pucker,
      n_atoms,
      derivatives);
}

std::tuple<at::Tensor, at::Tensor> na_torsion_pose_score_cuda(
    at::Tensor coords,
    at::Tensor base,
    at::Tensor is_na,
    at::Tensor torsion_indices,
    at::Tensor torsion_ok,
    at::Tensor ring_indices,
    at::Tensor ring_ok,
    at::Tensor prev,
    at::Tensor backbone_means,
    at::Tensor backbone_sdev,
    at::Tensor sugar_means,
    at::Tensor chi_means,
    at::Tensor sdev_sugar,
    at::Tensor sdev_chi,
    at::Tensor well_pucker,
    at::Tensor well_alpha_gamma,
    at::Tensor well_bibii_pucker,
    at::Tensor well_alphanext_bibii,
    at::Tensor well_chi_syn,
    at::Tensor is_north,
    at::Tensor weight_bb,
    at::Tensor weight_chi,
    at::Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev,
    bool compute_derivs) {
  TORCH_CHECK(coords.is_cuda(), "na_torsion_pose_score requires CUDA tensors");
  TORCH_CHECK(coords.is_contiguous(), "coords must be contiguous");
  int n_poses = int(base.size(0));
  int max_n_blocks = int(base.size(1));
  auto output = at::zeros({2, n_poses}, coords.options());
  auto derivatives = compute_derivs
                         ? at::zeros({2, coords.size(0), 3}, coords.options())
                         : at::empty({0}, coords.options());
  c10::cuda::CUDAGuard guard(coords.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 128;
  int blocks = (n_poses * max_n_blocks + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(
      coords.scalar_type(), "na_torsion_pose_score_cuda", [&] {
        if (!compute_derivs) {
          na_torsion_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              coords.const_data_ptr<scalar_t>(),
              base.const_data_ptr<int64_t>(),
              is_na.const_data_ptr<bool>(),
              torsion_indices.const_data_ptr<int64_t>(),
              torsion_ok.const_data_ptr<bool>(),
              ring_indices.const_data_ptr<int64_t>(),
              ring_ok.const_data_ptr<bool>(),
              prev.const_data_ptr<int64_t>(),
              backbone_means.const_data_ptr<scalar_t>(),
              backbone_sdev.const_data_ptr<scalar_t>(),
              sugar_means.const_data_ptr<scalar_t>(),
              chi_means.const_data_ptr<scalar_t>(),
              sdev_sugar.const_data_ptr<scalar_t>(),
              sdev_chi.const_data_ptr<scalar_t>(),
              well_pucker.const_data_ptr<scalar_t>(),
              well_alpha_gamma.const_data_ptr<scalar_t>(),
              well_bibii_pucker.const_data_ptr<scalar_t>(),
              well_alphanext_bibii.const_data_ptr<scalar_t>(),
              well_chi_syn.const_data_ptr<scalar_t>(),
              is_north.const_data_ptr<bool>(),
              weight_bb.const_data_ptr<scalar_t>(),
              weight_chi.const_data_ptr<scalar_t>(),
              weight_sugar.const_data_ptr<scalar_t>(),
              scalar_t(pucker_temperature),
              scalar_t(bin_blend_sdev),
              n_poses,
              max_n_blocks,
              output.mutable_data_ptr<scalar_t>());
        } else {
          na_torsion_derivative_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  coords.const_data_ptr<scalar_t>(),
                  base.const_data_ptr<int64_t>(),
                  is_na.const_data_ptr<bool>(),
                  torsion_indices.const_data_ptr<int64_t>(),
                  torsion_ok.const_data_ptr<bool>(),
                  ring_indices.const_data_ptr<int64_t>(),
                  ring_ok.const_data_ptr<bool>(),
                  prev.const_data_ptr<int64_t>(),
                  backbone_means.const_data_ptr<scalar_t>(),
                  backbone_sdev.const_data_ptr<scalar_t>(),
                  sugar_means.const_data_ptr<scalar_t>(),
                  chi_means.const_data_ptr<scalar_t>(),
                  sdev_sugar.const_data_ptr<scalar_t>(),
                  sdev_chi.const_data_ptr<scalar_t>(),
                  well_pucker.const_data_ptr<scalar_t>(),
                  well_alpha_gamma.const_data_ptr<scalar_t>(),
                  well_bibii_pucker.const_data_ptr<scalar_t>(),
                  well_alphanext_bibii.const_data_ptr<scalar_t>(),
                  well_chi_syn.const_data_ptr<scalar_t>(),
                  is_north.const_data_ptr<bool>(),
                  weight_bb.const_data_ptr<scalar_t>(),
                  weight_chi.const_data_ptr<scalar_t>(),
                  weight_sugar.const_data_ptr<scalar_t>(),
                  scalar_t(pucker_temperature),
                  scalar_t(bin_blend_sdev),
                  n_poses,
                  max_n_blocks,
                  int(coords.size(0)),
                  derivatives.mutable_data_ptr<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>());
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, derivatives};
}

template <typename Real>
__global__ void na_torsion_backward_kernel(
    Real const* derivatives,
    Real const* grad_output,
    int64_t grad_stride_score,
    int64_t grad_stride_pose,
    int n_poses,
    int n_atoms,
    Real* grad_coords) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int n_values = n_atoms * 3;
  if (index >= n_values) return;
  int atom = index / 3;
  int atoms_per_pose = n_atoms / n_poses;
  int pose = atom / atoms_per_pose;
  Real grad_harmonic = grad_output[pose * grad_stride_pose];
  Real grad_well = grad_output[grad_stride_score + pose * grad_stride_pose];
  grad_coords[index] = grad_harmonic * derivatives[index]
                       + grad_well * derivatives[n_values + index];
}

at::Tensor na_torsion_pose_score_backward_cuda(
    at::Tensor derivatives, at::Tensor grad_output) {
  TORCH_CHECK(derivatives.is_cuda(), "NA torsion derivatives must be CUDA");
  TORCH_CHECK(
      grad_output.is_cuda(), "NA torsion output gradients must be CUDA");
  TORCH_CHECK(
      derivatives.dim() == 3 && derivatives.size(0) == 2,
      "expected derivatives shaped [2, n_atoms, 3]");
  TORCH_CHECK(
      grad_output.dim() == 2 && grad_output.size(0) == 2,
      "expected output gradients shaped [2, n_poses]");
  int n_poses = int(grad_output.size(1));
  int n_atoms = int(derivatives.size(1));
  TORCH_CHECK(n_atoms % n_poses == 0, "atoms must divide evenly among poses");
  auto grad_coords = at::empty({n_atoms, 3}, derivatives.options());
  c10::cuda::CUDAGuard guard(derivatives.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;
  int blocks = (n_atoms * 3 + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(
      derivatives.scalar_type(), "na_torsion_pose_score_backward_cuda", [&] {
        na_torsion_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            derivatives.const_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            grad_output.stride(0),
            grad_output.stride(1),
            n_poses,
            n_atoms,
            grad_coords.mutable_data_ptr<scalar_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_coords;
}

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
