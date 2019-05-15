#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
#include <tmol/utility/nvtx.hh>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/tuple.hxx>

#include "dispatch.impl.hh"
#include "params.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template struct CartBondedLengthDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct CartBondedLengthDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;
template struct CartBondedAngleDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct CartBondedAngleDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;
template struct CartBondedTorsionDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct CartBondedTorsionDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;
template struct CartBondedHxlTorsionDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct CartBondedHxlTorsionDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
