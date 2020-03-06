#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/pack/compiled/params.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <tmol::Device D>
struct OneStageAnnealerDispatch
{
  static
  auto
  forward(
    TView<SimAParams<float>, 1, tmol::Device::CPU> simA_params,
    TView<int, 1, D> nrotamers_for_res,
    TView<int, 1, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    TView<int, 2, D> nenergies,
    TView<int64_t, 2, D> twob_offsets,
    TView<float, 2, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 2, D>,
      TPack<int, 2, D>,
      TPack<int, 1, D> >;
};

template <tmol::Device D>
struct MultiStageAnnealerDispatch
{
  static
  auto
  forward(
    TView<SimAParams<float>, 1, tmol::Device::CPU> simA_params,
    TView<int, 1, D> nrotamers_for_res,
    TView<int, 1, D> oneb_offsets,
    TView<int, 1, D> res_for_rot,
    TView<int, 2, D> nenergies,
    TView<int64_t, 2, D> twob_offsets,
    TView<float, 2, D> energy1b,
    TView<float, 1, D> energy2b
  )
    -> std::tuple<
      TPack<float, 2, D>,
      TPack<int, 2, D>,
      TPack<int, 1, D> >;
};

} // namespace compiled
} // namespace pack
} // namespace core
