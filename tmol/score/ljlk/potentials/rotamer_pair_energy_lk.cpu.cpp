#include <tmol/score/ljlk/potentials/rotamer_pair_energy_lk.impl.hh>
#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LKRPEDispatch<ForallDispatch, tmol::Device::CPU, float, int>;
template struct LKRPEDispatch<ForallDispatch, tmol::Device::CPU, double, int>;


}
}
}
}
