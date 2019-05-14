#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/accumulate.hh>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/tuple.hxx>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
struct CartBondedLengthDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedLengthParams<Int>, 1, D> atom_indices,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int pari = atom_indices[i].param_index;
      auto cblength = cblength_V_dV(
          coords[ati], coords[atj], param_table[pari].K, param_table[pari].x0);

      accumulate<D, Real>::add(V[0], mgpu::get<0>(cblength));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], mgpu::get<1>(cblength));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], mgpu::get<2>(cblength));
    });

    mgpu::standard_context_t context;
    mgpu::transform([=] MGPU_DEVICE(int idx) { f_i(idx); }, num_Vs, context);

    return {V_t, dV_dx_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedAngleDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedAngleParams<Int>, 1, D> atom_indices,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int atk = atom_indices[i].atom_index_k;
      Int pari = atom_indices[i].param_index;
      auto cbangle = cbangle_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          param_table[pari].K,
          param_table[pari].x0);

      accumulate<D, Real>::add(V[0], mgpu::get<0>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], mgpu::get<1>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], mgpu::get<2>(cbangle));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atk], mgpu::get<3>(cbangle));
    });

    mgpu::standard_context_t context;
    mgpu::transform([=] MGPU_DEVICE(int idx) { f_i(idx); }, num_Vs, context);

    return {V_t, dV_dx_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedTorsionParams<Int>, 1, D> atom_indices,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int atk = atom_indices[i].atom_index_k;
      Int atl = atom_indices[i].atom_index_l;
      Int pari = atom_indices[i].param_index;
      auto cbtorsion = cbtorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          param_table[pari].K,
          param_table[pari].x0,
          param_table[pari].period);

      accumulate<D, Real>::add(V[0], mgpu::get<0>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], mgpu::get<1>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], mgpu::get<2>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atk], mgpu::get<3>(cbtorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atl], mgpu::get<4>(cbtorsion));
    });

    mgpu::standard_context_t context;
    mgpu::transform([=] MGPU_DEVICE(int idx) { f_i(idx); }, num_Vs, context);

    return {V_t, dV_dx_t};
  }
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedHxlTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedTorsionParams<Int>, 1, D> atom_indices,
      TView<CartBondedSinusoidalTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
    auto num_Vs = atom_indices.size(0);

    auto V_t = TPack<Real, 1, D>::zeros({1});
    auto dV_dx_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0)});

    auto V = V_t.view;
    auto dV_dx = dV_dx_t.view;

    auto f_i = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_indices[i].atom_index_i;
      Int atj = atom_indices[i].atom_index_j;
      Int atk = atom_indices[i].atom_index_k;
      Int atl = atom_indices[i].atom_index_l;
      Int pari = atom_indices[i].param_index;
      auto cbhxltorsion = cbhxltorsion_V_dV(
          coords[ati],
          coords[atj],
          coords[atk],
          coords[atl],
          param_table[pari].k1,
          param_table[pari].k2,
          param_table[pari].k3,
          param_table[pari].phi1,
          param_table[pari].phi2,
          param_table[pari].phi3);

      accumulate<D, Real>::add(V[0], mgpu::get<0>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[ati], mgpu::get<1>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atj], mgpu::get<2>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atk], mgpu::get<3>(cbhxltorsion));
      accumulate<D, Vec<Real, 3>>::add(dV_dx[atl], mgpu::get<4>(cbhxltorsion));
    });

    mgpu::standard_context_t context;
    mgpu::transform([=] MGPU_DEVICE(int idx) { f_i(idx); }, num_Vs, context);

    return {V_t, dV_dx_t};
  }
};

template struct CartBondedLengthDispatch<tmol::Device::CUDA, float, int64_t>;
template struct CartBondedLengthDispatch<tmol::Device::CUDA, double, int64_t>;
template struct CartBondedAngleDispatch<tmol::Device::CUDA, float, int64_t>;
template struct CartBondedAngleDispatch<tmol::Device::CUDA, double, int64_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CUDA, float, int64_t>;
template struct CartBondedTorsionDispatch<tmol::Device::CUDA, double, int64_t>;
template struct CartBondedHxlTorsionDispatch<
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct CartBondedHxlTorsionDispatch<
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
