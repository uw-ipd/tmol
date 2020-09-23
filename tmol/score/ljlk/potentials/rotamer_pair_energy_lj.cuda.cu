#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lj.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJRPEDispatch<ForallDispatch, tmol::Device::CUDA, float, int>;
template struct LJRPEDispatch<ForallDispatch, tmol::Device::CUDA, double, int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
