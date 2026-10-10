#include <torch/script.h>
#include <torch/torch.h>

#include <tmol/score/common/device_operations.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>

#include "density_score.hh"

namespace tmol {
namespace score {
namespace density {
namespace potentials {

ContextManager mgr;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

class DensityScoreOp : public torch::autograd::Function<DensityScoreOp> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor coeffs,
      Tensor origin,
      Tensor inv_basis,
      Tensor pad,
      bool periodic,
      Tensor atom_index,
      Tensor atom_group,
      Tensor atom_weight,
      int64_t n_groups) {
    Tensor energy, dE_dx;
    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.options(), "density_score", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
          auto result =
              DensityScoreDispatch<common::DeviceOperations, Dev, Real>::
                  forward(
                      mgr,
                      TCAST(coords),
                      TCAST(coeffs),
                      TCAST(origin),
                      TCAST(inv_basis),
                      TCAST(pad),
                      periodic,
                      TCAST(atom_index),
                      TCAST(atom_group),
                      TCAST(atom_weight),
                      n_groups);
          energy = std::get<0>(result).tensor;
          dE_dx = std::get<1>(result).tensor;
        }));
    ctx->save_for_backward({dE_dx, atom_index, atom_group});
    ctx->saved_data["n_coords"] = coords.size(0);
    return energy;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    Tensor dE_dx = saved[0], atom_index = saved[1], atom_group = saved[2];
    int64_t n_coords = ctx->saved_data["n_coords"].toInt();
    Tensor group_grad = grad_outputs[0].contiguous();
    Tensor grad_coords;
    TMOL_DISPATCH_FLOATING_DEVICE(
        dE_dx.options(), "density_score_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
          grad_coords =
              DensityScoreDispatch<common::DeviceOperations, Dev, Real>::
                  backward(
                      mgr,
                      TCAST(dE_dx),
                      TCAST(atom_index),
                      TCAST(atom_group),
                      TCAST(group_grad),
                      n_coords)
                      .tensor;
        }));
    return {
        grad_coords,
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor()};
  }
};

Tensor density_score(
    Tensor coords,
    Tensor coeffs,
    Tensor origin,
    Tensor inv_basis,
    Tensor pad,
    bool periodic,
    Tensor atom_index,
    Tensor atom_group,
    Tensor atom_weight,
    int64_t n_groups) {
  return DensityScoreOp::apply(
      coords,
      coeffs,
      origin,
      inv_basis,
      pad,
      periodic,
      atom_index,
      atom_group,
      atom_weight,
      n_groups);
}

TORCH_LIBRARY(tmol_density, m) { m.def("density_score", &density_score); }

}  // namespace potentials
}  // namespace density
}  // namespace score
}  // namespace tmol
