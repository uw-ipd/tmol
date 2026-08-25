#include <tmol/score/bonded_atom.hh>
#include <tmol/score/common/device_operations.cuda.impl.cuh>

#include <tmol/tests/score/bonded_atom/test.impl.hh>

namespace tmol {
namespace tests {
namespace score {
namespace bonded_atom {

template struct BondedAtomTests<
    tmol::score::common::DeviceOperations,
    Device::CUDA,
    int32_t>;

}  // namespace bonded_atom
}  // namespace score
}  // namespace tests
}  // namespace tmol
