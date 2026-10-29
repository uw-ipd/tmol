#pragma once

namespace tmol {

// The backends kernel code is templated on.
//
// Kept apart from TensorAccessor.h so that device-side machinery -- which is
// templated on the device but never touches a tensor view -- can name this
// without pulling in the torch tensor layer. That matters for Metal, whose
// translation units cannot include torch at all.
//
// MPS is the Apple Metal backend.
enum class Device { CPU, CUDA, MPS };

enum class PtrTag { Restricted, Unrestricted };

}  // namespace tmol
