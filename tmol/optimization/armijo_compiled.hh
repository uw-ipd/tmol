#pragma once

#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace optimization {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
struct ArmijoCompiledDispatch {
  static TPack<Real, 1, D> start(
      ContextManager& mgr,
      TView<bool, 1, D> searching,
      TView<Real, 1, D> alpha0);

  static std::tuple<
      TPack<Real, 1, D>,
      TPack<Real, 1, D>,
      TPack<int64_t, 1, D>,
      TPack<bool, 1, D>>
  classify(
      ContextManager& mgr,
      TView<bool, 1, D> searching,
      TView<Real, 1, D> alpha,
      TView<Real, 1, D> phi,
      TView<Real, 1, D> phi0,
      TView<Real, 1, D> derphi0,
      Real sigma_increase,
      Real sigma_decrease);

  static TPack<Real, 1, D> trial(
      ContextManager& mgr,
      TView<int64_t, 1, D> status,
      TView<Real, 1, D> alpha,
      TView<Real, 1, D> accepted,
      Real factor);

  static std::tuple<
      TPack<Real, 1, D>,
      TPack<Real, 1, D>,
      TPack<int64_t, 1, D>,
      TPack<bool, 1, D>,
      TPack<bool, 1, D>>
  update(
      ContextManager& mgr,
      TView<int64_t, 1, D> status,
      TView<Real, 1, D> trial,
      TView<Real, 1, D> accepted,
      TView<Real, 1, D> phi_accepted,
      TView<Real, 1, D> phi,
      TView<Real, 1, D> phi_trial,
      TView<Real, 1, D> phi0,
      TView<Real, 1, D> derphi0,
      Real sigma_decrease,
      Real minstep);

  static std::tuple<TPack<Real, 1, D>, TPack<bool, 1, D>> finalize(
      ContextManager& mgr,
      TView<int64_t, 1, D> status,
      TView<Real, 1, D> derphi0,
      TView<Real, 1, D> accepted,
      TView<Real, 1, D> start,
      TView<bool, 1, D> searching,
      Real minstep);
};

}  // namespace optimization
}  // namespace tmol
