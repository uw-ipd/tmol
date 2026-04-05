// compiled.mps.mm — MPS instantiation for Dunbrack rotamer sampler.
//
// ComplexDispatch<D> in complex_dispatch.cpu.impl.hh is a primary template
// (device-agnostic CPU loops).  We include it directly for MPS.

#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/complex_dispatch.cpu.impl.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/numeric/bspline_compiled/bspline.hh>

#include <ATen/Tensor.h>

#include "dispatch.impl.hh"

namespace tmol {
namespace pack {
namespace rotamer {
namespace dunbrack {

template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::MPS,
    float,
    int32_t>;
template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::MPS,
    double,
    int32_t>;
template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::MPS,
    float,
    int64_t>;
template struct DunbrackChiSampler<
    score::common::ComplexDispatch,
    tmol::Device::MPS,
    double,
    int64_t>;

}  // namespace dunbrack
}  // namespace rotamer
}  // namespace pack
}  // namespace tmol
