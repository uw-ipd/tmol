// #include <tmol/pack/rotamer/dispatch.hh>
#include <tmol/pack/rotamer/dispatch.impl.hh>
#include <tmol/score/common/complex_dispatch.cpu.impl.hh>
#include "test.impl.hh"

namespace tmol {

template struct DunbrackChiSamplerTester<
  tmol::score::common::ComplexDispatch,
  Device::CPU,
  float,
  int32_t>;

}
