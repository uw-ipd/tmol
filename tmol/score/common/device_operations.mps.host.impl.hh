#pragma once

// Host-side stand-in for DeviceOperations<MPS>.
//
// The real Metal primitives live in device_operations.mps.impl.hh and are
// compiled only by the Metal front end. But the generic dispatch bodies --
// elec_pose_score.impl.hh and friends -- are ordinary C++ templates, and
// instantiating one with D = MPS instantiates every device lambda in it, each
// of which names DeviceOperations<MPS>. Those instantiations are dead code:
// at run time the MPS branch hands the work to a named Metal entry point
// instead. They still have to compile and link.
//
// Serial CPU semantics are the right stand-in. A workgroup's lanes run one
// after another, which is exactly what DeviceOperations<CPU> does, so this
// borrows it rather than restating it. The inheritance is safe here because
// this header is host-only; Metal is the toolchain that forbids derived
// classes, and it never sees this file.
//
// The host launchers (foreach_workgroup and friends) come along with the base
// and would run the kernel bodies on the CPU if anything called them. Nothing
// does -- the MPS code path branches to the Metal launcher before reaching
// them -- and TMOL_MPS_UNREACHABLE below documents the few places that rely
// on that.

#include <tmol/score/common/device_operations.cpu.impl.hh>

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::MPS>
    : DeviceOperations<tmol::Device::CPU> {};

}  // namespace common
}  // namespace score
}  // namespace tmol
