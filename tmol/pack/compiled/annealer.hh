#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

namespace tmol {
namespace pack {
namespace compiled {

template <
  template<tmol::Deivce>
class Dispatch,
  tmol::Device D,
  typename Real,
  typename Int>
struct AnnealerDispatch
{
  static
  auto
  forward(
    TView<Int, 1, D> nrotamers_for_res,
    TView<Int, 1, D> oneb_offsets,
    TView<Int, 1, D> res_for_rot,
    TView<Int, 1, D> nenergies,
    TView<Int, 2, D> twob_offsets,
    TView<Real, 1, D> energy1b,
    TView<Real, 1, D> energy2b
  )
    -> std::tuple<
      TPack<Real, 1, D>,
      TPack<Int, 2, D> >;
};

} // namespace compiled
} // namespace pack
} // namespace core
