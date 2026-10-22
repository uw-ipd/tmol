// Tier 2 guard: math that still reads through TView/TensorAccessor.
//
// These cannot yet be compiled standalone -- the tensor layer needs torch on
// the host, and a Metal build will need a Metal-specific accessor before they
// can be shared. What is asserted here is the property that already holds and
// must not regress: no Eigen reaches any of tmol's kernel-evaluated math.

#include <tmol/score/backbone_torsion/potentials/potentials.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/gen_coord.hh>
#include <tmol/score/disulfide/potentials/potentials.hh>
#include <tmol/score/dunbrack/potentials/potentials.hh>
#include <tmol/score/elec/potentials/elec.hh>
#include <tmol/score/hbond/potentials/hbond.hh>
#include <tmol/score/hbond/potentials/potentials.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/lk_ball/potentials/water.hh>
#include <tmol/score/metal/potentials/potentials.hh>

int main() { return 0; }
