#pragma once
#include <torch/torch.h>

// #include <Eigen/Core>
// #include <Eigen/Geometry>

// #include <tmol/utility/tensor/TensorAccessor.h>
// #include <tmol/utility/tensor/TensorPack.h>
// #include <tmol/utility/tensor/TensorStruct.h>
// #include <tmol/utility/tensor/TensorUtil.h>
// #include <tmol/utility/nvtx.hh>

// #include <tmol/score/common/accumulate.hh>
// #include <tmol/score/common/geom.hh>
// #include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

class PoseScoreFusionModule {
 public:
  virtual ~PoseScoreFusionModule() = default;
  virtual void prepare_for_scoring(torch::Tensor coords) = 0;
  virtual void forward(torch::Tensor coords, torch::Tensor V) = 0;
  virtual void backward(
      torch::Tensor coords, torch::Tensor dTdV, torch::Tensor dVdxyz) = 0;
  virtual int n_terms() const = 0;
  virtual int n_poses() const = 0;
  virtual int max_n_blocks() const = 0;
  virtual bool output_block_pair_energies() const = 0;
};

}  // namespace common
}  // namespace score
}  // namespace tmol