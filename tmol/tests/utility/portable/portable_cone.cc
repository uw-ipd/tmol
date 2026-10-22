// Tier 1 guard: scalar math that is already freestanding.
//
// These headers compile against nothing but the portable layer -- no Eigen, no
// torch, no C++ standard library containers -- which is the state a Metal
// translation unit requires. Built with a poison Eigen/Core ahead of the real
// one, so a reintroduced dependency fails loudly rather than silently
// re-coupling the portable layer.
//
// Headers that read through TView/TensorAccessor are covered by tier 2
// (portable_cone_tensor.cc): they are Eigen-free, but still depend on the
// tensor layer and so cannot be Metal-compiled until TView gains a Metal
// variant.

#include <tmol/score/cartbonded/potentials/potentials.hh>
#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/polynomial.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>
#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/genbonded/potentials/potentials.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/tuple.hh>

namespace {

using tmol::portable::Vec;

// Instantiate templates so the guard covers function bodies, not merely
// declarations: a header can be free of Eigen in its signatures while a
// template body still names it.
float exercise() {
  Vec<float, 3> a{1.f, 2.f, 3.f};
  Vec<float, 3> b{0.5f, -1.f, 2.f};

  float v = tmol::score::common::distance<float>::V(a, b);
  auto dv = tmol::score::common::distance<float>::V_dV(a, b);
  v += dv.V + dv.dV_dA[0];
  v += tmol::score::common::interior_angle<float>::V(a, b);
  v += tmol::score::common::cos_interior_angle<float>::V(a, b);
  v += tmol::score::common::dihedral_angle<float>::V(a, b, a, b);
  v += tmol::score::elec::potentials::eps<float>(2.f, 80.f, 6.f, 0.4f);
  v += tmol::score::elec::potentials::connectivity_weight<float>(5.f);
  v += tmol::portable::math::pi<float>();

  auto t = tmol::portable::make_tuple(1.0f, 2.0f);
  float x, y;
  tmol::portable::tie(x, y) = t;
  return v + x + y;
}

}  // namespace

int main() { return exercise() == 12345.f ? 1 : 0; }
