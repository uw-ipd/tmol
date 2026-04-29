#include <ATen/cuda/CUDAContext.h>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/transform.hxx>

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

  auto stream = at::cuda::getCurrentCUDAStream();
  mgpu::standard_context_t context(false, stream);

  constexpr int NT = 256;
  typedef mgpu::cta_reduce_t<NT, Real> reduce_t;

  mgpu::cta_launch<NT>(
      [=] MGPU_DEVICE(int tid, int cta) {
        __shared__ typename reduce_t::storage_t shared;

        for (int j = tid; j < N_i; j += NT) result[j] = -grad[j];
        __syncthreads();

        for (int i = m_i - 1; i >= 0; i--) {
          Real local = Real(0);
          for (int j = tid; j < N_i; j += NT) local += stps[i][j] * result[j];

          Real al_i = ro[i] * reduce_t().reduce(tid, local, shared);
          if (tid == 0) al[i] = al_i;

          for (int j = tid; j < N_i; j += NT) result[j] -= al_i * dirs[i][j];
          __syncthreads();
        }

        for (int i = 0; i < m_i; i++) {
          Real local = Real(0);
          for (int j = tid; j < N_i; j += NT) local += dirs[i][j] * result[j];

          Real coeff = al[i] - ro[i] * reduce_t().reduce(tid, local, shared);
          for (int j = tid; j < N_i; j += NT) result[j] += coeff * stps[i][j];
          __syncthreads();
        }
      },
      1,
      context);

  return result_tp;
}

template struct LbfgsTwoLoop<tmol::Device::CUDA, float>;
template struct LbfgsTwoLoop<tmol::Device::CUDA, double>;

}  // namespace compiled
}  // namespace optimization
}  // namespace tmol
