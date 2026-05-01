#include <tmol/utility/tensor/TensorPack.h>

#include "lbfgs.hh"

namespace tmol {
namespace optimization {
namespace compiled {

template <tmol::Device D, typename Real>
auto LbfgsTwoLoop<D, Real>::f(
    TView<Real, 1, D> grad,
    TView<Real, 2, D> dirs,
    TView<Real, 2, D> stps,
    TView<Real, 1, D> ro) -> TPack<Real, 1, D> {
  int m_i = dirs.size(0);
  int N_i = dirs.size(1);

  auto al_tp = TPack<Real, 1, D>::zeros({m_i});
  auto result_tp = TPack<Real, 1, D>::empty({N_i});
  auto al = al_tp.view;
  auto result = result_tp.view;

  for (int j = 0; j < N_i; j++) result[j] = -grad[j];

  for (int i = m_i - 1; i >= 0; i--) {
    Real dot = Real(0);
    for (int j = 0; j < N_i; j++) dot += stps[i][j] * result[j];
    al[i] = ro[i] * dot;
    for (int j = 0; j < N_i; j++) result[j] -= al[i] * dirs[i][j];
  }

  for (int i = 0; i < m_i; i++) {
    Real dot = Real(0);
    for (int j = 0; j < N_i; j++) dot += dirs[i][j] * result[j];
    Real coeff = al[i] - ro[i] * dot;
    for (int j = 0; j < N_i; j++) result[j] += coeff * stps[i][j];
  }

  return result_tp;
}

template struct LbfgsTwoLoop<tmol::Device::CPU, float>;
template struct LbfgsTwoLoop<tmol::Device::CPU, double>;

}  // namespace compiled
}  // namespace optimization
}  // namespace tmol
