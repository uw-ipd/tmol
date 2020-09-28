#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/complex_dispatch.cuda.impl.cuh>
#include <tmol/score/common/geom.hh>

#include <ATen/Tensor.h>

#include "dispatch.impl.hh"

namespace tmol {
namespace pack {
namespace rotamer {

template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::CUDA,
    float,
    int32_t>;
template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::CUDA,
    double,
    int32_t>;
template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
