#pragma once

// Forward declaration of the workgroup-primitive dispatcher.
//
// Kernel bodies take DeviceOperations as a template template parameter, which
// requires the primary template to be visible but not defined. The definition
// in device_operations.hh declares host-side launch helpers that return TPack
// and so reaches the torch tensor layer; a `.metal` translation unit includes
// this header instead and supplies its own specialization.

#include <tmol/utility/portable/device.hh>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct DeviceOperations;

}  // namespace common
}  // namespace score
}  // namespace tmol
