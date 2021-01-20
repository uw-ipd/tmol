#include <tmol/score/lk_ball/potentials/rotamer_pair_energy_lkball.impl.hh>
#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallRPEDispatch<common::ForallDispatch, tmol::Device::CPU, float, int, 4>;
template struct LKBallRPEDispatch<common::ForallDispatch, tmol::Device::CPU, double, int, 4>;


}
}
}
}
