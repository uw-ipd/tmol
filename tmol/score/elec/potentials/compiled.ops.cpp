#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "dispatch.hh"
#include "elec_pose_score.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <
    template <
        template <tmol::Device>
        class SingleDispatch,
        template <tmol::Device>
        class PairDispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class SingleDispatchMethod,
    template <tmol::Device>
    class PairDispatchMethod>
class ScoreOp
    : public torch::autograd::Function<
          ScoreOp<ScoreDispatch, SingleDispatchMethod, PairDispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor I,
      Tensor charge_I,
      Tensor J,
      Tensor charge_J,
      Tensor bonded_path_lengths,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dScore_dI;
    at::Tensor dScore_dJ;

    using Int = int64_t;

    TMOL_DISPATCH_FLOATING_DEVICE(I.type(), "score_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result = ScoreDispatch<
                                        SingleDispatchMethod,
                                        PairDispatchMethod,
                                        Dev,
                                        Real,
                                        Int>::
                                        f(TCAST(I),
                                          TCAST(charge_I),
                                          TCAST(J),
                                          TCAST(charge_J),
                                          TCAST(bonded_path_lengths),
                                          TCAST(global_params));

                                    score = std::get<0>(result).tensor;
                                    dScore_dI = std::get<1>(result).tensor;
                                    dScore_dJ = std::get<2>(result).tensor;
                                  }));

    ctx->save_for_backward({dScore_dI, dScore_dJ});
    return score;

    // return connect_backward_pass({I, J}, score, [&]() {
    //  return StackedSavedGradsBackward::create({dScore_dI, dScore_dJ});
    //});
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved_grads = ctx->get_saved_variables();

    tensor_list result;

    for (auto& saved_grad : saved_grads) {
      auto ingrad = grad_outputs[0];
      while (ingrad.dim() < saved_grad.dim()) {
        ingrad = ingrad.unsqueeze(-1);
      }

      result.emplace_back(saved_grad * ingrad);
    }

    int i = 0;
    auto dI = result[i++];
    auto dJ = result[i++];

    return {
        dI,
        torch::Tensor(),
        dJ,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
  }
};

template <
    template <
        template <tmol::Device>
        class SingleDispatch,
        template <tmol::Device>
        class PairDispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class SingleDispatchMethod,
    template <tmol::Device>
    class PairDispatchMethod>
Tensor score_op(
    Tensor I,
    Tensor charge_I,
    Tensor J,
    Tensor charge_J,
    Tensor bonded_path_lengths,
    Tensor global_params) {
  return ScoreOp<ScoreDispatch, SingleDispatchMethod, PairDispatchMethod>::
      apply(I, charge_I, J, charge_J, bonded_path_lengths, global_params);
}

template <template <tmol::Device> class DispatchMethod>
class ElecPoseScoreOp
    : public torch::autograd::Function<ElecPoseScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_min_bond_separation,
      Tensor pose_stack_inter_block_bondsep,

      Tensor block_type_n_atoms,
      Tensor block_type_partial_charge,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_inter_repr_path_distance,

      Tensor block_type_intra_repr_path_distance,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "elec_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              ElecPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
                  TCAST(coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),

                  TCAST(block_type_n_atoms),
                  TCAST(block_type_partial_charge),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_inter_repr_path_distance),

                  TCAST(block_type_intra_repr_path_distance),
                  TCAST(global_params),
                  coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({dscore_dcoords});
    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved_grads = ctx->get_saved_variables();

    tensor_list result;

    for (auto& saved_grad : saved_grads) {
      auto ingrad = grad_outputs[0];
      while (ingrad.dim() < saved_grad.dim()) {
        ingrad = ingrad.unsqueeze(-1);
      }

      result.emplace_back(saved_grad * ingrad);
    }

    int i = 0;
    auto dscore_dcoords = result[i++];

    return {
        dscore_dcoords,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(),
        torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor elec_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_min_bond_separation,
    Tensor pose_stack_inter_block_bondsep,

    Tensor block_type_n_atoms,
    Tensor block_type_partial_charge,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_inter_repr_path_distance,

    Tensor block_type_intra_repr_path_distance,
    Tensor global_params) {
  return ElecPoseScoreOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_min_bond_separation,
      pose_stack_inter_block_bondsep,

      block_type_n_atoms,
      block_type_partial_charge,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_inter_repr_path_distance,

      block_type_intra_repr_path_distance,
      global_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def(
      "score_elec",
      &score_op<
          ElecDispatch,
          tmol::score::common::ForallDispatch,
          tmol::score::common::AABBDispatch>);
  m.def(
      "score_elec_triu",
      &score_op<
          ElecDispatch,
          tmol::score::common::ForallDispatch,
          tmol::score::common::AABBTriuDispatch>);
  m.def("elec_pose_scores", &elec_pose_scores_op<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
