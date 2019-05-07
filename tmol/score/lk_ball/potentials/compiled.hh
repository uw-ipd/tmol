#include <tmol/score/common/dispatch.cpu.impl.hh>

#include <tmol/score/hbond/identification.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <tmol::Device D, typename Real, typename Int, int MAX_WATER>
struct attached_waters {
  static def forward(
      TView<Vec<Real, 3>, 1, D> coords,
      tmol::score::bonded_atom::IndexedBonds<Int, D> indexed_bonds,
      AtomTypes<D> atom_types,
      LKBallGlobalParameters<Real, D> global_params)
      ->TPack<Vec<Real, 3>, 2, D>;

  static def backward(
      TView<Vec<Real, 3>, 2, D> dE_dW,
      TView<Vec<Real, 3>, 1, D> coords,
      tmol::score::bonded_atom::IndexedBonds<Int, D> indexed_bonds,
      AtomTypes<D> atom_types,
      LKBallGlobalParameters<Real, D> global_params)
      ->TPack<Vec<Real, 3>, 1, D>;
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
