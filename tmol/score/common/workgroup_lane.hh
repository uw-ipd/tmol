#pragma once

#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/qualifiers.hh>

namespace tmol {
namespace score {
namespace common {

// A thread's identity within its workgroup.
//
// CUDA reads the lane from a hardware register and the CPU backend carries it
// in a loop variable, so on those two backends this handle is empty and the
// workgroup primitives default-construct it -- call sites that do not pass one
// are unaffected.
//
// Metal has neither route. A thread's position in its threadgroup arrives as a
// kernel-entry attribute, and MSL forbids program-scope variables outside the
// constant address space, so there is no way for a called function to reach it
// implicitly: it has to be an argument. The Metal specialization therefore
// carries the lane and is deliberately *not* default-constructible, which
// turns "this kernel has not been threaded through yet" into a compile error
// in the Metal translation unit and a no-op everywhere else.
template <tmol::Device D>
struct WorkgroupLane {};

// Threads per Metal threadgroup. 32 matches the simdgroup width on Apple
// silicon, so a workgroup reduction is a single simd_sum. Named here rather
// than taken from launch_t because the host and the shader compile different
// launch_t specializations and both sides must agree on the launch shape.
//
// Not widenable as things stand. The workgroup primitives handle any width --
// the reductions fall back to a threadgroup-scratch pass past 32 -- but the
// tile evaluators in tile_atom_pair_evaluation.hh and the term kernels that
// build on them assume one thread per tile slot, and TILE_SIZE is 32. Built
// at 128 the energies came out 7e-03 wrong and the kernels were slower, so
// widening means auditing the tile algorithms first, not just this constant.
enum { TMOL_MPS_THREADGROUP_SIZE = 32 };

#if TMOL_PORTABLE_METAL
template <>
struct WorkgroupLane<tmol::Device::MPS> {
  uint lane;
  uint lanes_per_group;
  threadgroup float* reduce_scratch;  // one float per simdgroup

  WorkgroupLane() = delete;
  WorkgroupLane(uint lane_, uint lanes_per_group_, threadgroup float* scratch)
      : lane(lane_),
        lanes_per_group(lanes_per_group_),
        reduce_scratch(scratch) {}
};
#endif

}  // namespace common
}  // namespace score
}  // namespace tmol
