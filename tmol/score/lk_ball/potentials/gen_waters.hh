#include <tmol/score/common/dispatch.cpu.impl.hh>
#include <tmol/score/hbond/identification.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

//#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
#define def auto

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    int MAX_WATER>
struct GenerateWaters {
  static def forward(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Int, 1, D> atom_type_i,
      TView<Vec<Int, 2>, 1, D> indexed_bond_bonds,
      TView<Vec<Int, 2>, 1, D> indexed_bond_spans,
      TView<LKBallWaterGenTypeParams<Int>, 1, D> type_params,
      TView<LKBallWaterGenGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> sp2_water_tors,
      TView<Real, 1, D> sp3_water_tors,
      TView<Real, 1, D> ring_water_tors)
      ->TPack<Vec<Real, 3>, 2, D>;

  static def backward(
      TView<Vec<Real, 3>, 2, D> dE_dW,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Int, 1, D> atom_type_i,
      TView<Vec<Int, 2>, 1, D> indexed_bond_bonds,
      TView<Vec<Int, 2>, 1, D> indexed_bond_spans,
      TView<LKBallWaterGenTypeParams<Int>, 1, D> type_params,
      TView<LKBallWaterGenGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> sp2_water_tors,
      TView<Real, 1, D> sp3_water_tors,
      TView<Real, 1, D> ring_water_tors)
      ->TPack<Vec<Real, 3>, 1, D>;
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
