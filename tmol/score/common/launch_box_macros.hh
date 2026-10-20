#pragma once

#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/portable/qualifiers.hh>

// Common macros for MGPU launches. Uniform configurations need no per-GPU list.
#ifdef __NVCC__
#include <moderngpu/launch_box.hxx>

// Names are qualified rather than pulled in with a using-directive: these
// macros expand inside kernel bodies, so an unqualified `using namespace mgpu`
// would inject moderngpu into every scope that launches work -- and obscure
// exactly which code depends on it, which is what the Metal port needs to know.
#define LAUNCH_BOX_32 \
  typedef mgpu::launch_box_t<mgpu::arch_20_cta<32, 1>> launch_t;
#define LAUNCH_BOX_64 \
  typedef mgpu::launch_box_t<mgpu::arch_20_cta<64, 1>> launch_t;
#define LAUNCH_BOX_128 \
  typedef mgpu::launch_box_t<mgpu::arch_20_cta<128, 1>> launch_t;

// Apply the requested occupancy on Volta and newer; older GPUs inherit the
// unconstrained configuration. MGPU propagates each setting to newer targets.
#define LAUNCH_BOX_32_OCC_AS(name, occ)                                        \
  typedef mgpu::                                                               \
      launch_box_t<mgpu::arch_20_cta<32, 1>, mgpu::arch_70_cta<32, 1, 1, occ>> \
          name;
#else
// The launch shape a non-CUDA backend reports. CPU workgroups traverse their
// lanes serially, so one lane is sufficient there; Metal runs a real
// threadgroup, so it reports the threadgroup width instead.
template <int NT, int VT>
struct launch_t_cpu {
  struct sm_ptx {
    enum { nt = NT, vt = VT, vt0 = VT };
  };
};

#if TMOL_PORTABLE_METAL
// One lane per Metal thread. 32 matches the simdgroup width on Apple silicon,
// so a workgroup reduction is a single simd_sum with no threadgroup traffic.
#define TMOL_LAUNCH_NT tmol::score::common::TMOL_MPS_THREADGROUP_SIZE
#else
#define TMOL_LAUNCH_NT 1
#endif

#define LAUNCH_BOX_32 typedef launch_t_cpu<TMOL_LAUNCH_NT, 1> launch_t;
#define LAUNCH_BOX_64 typedef launch_t_cpu<TMOL_LAUNCH_NT, 1> launch_t;
#define LAUNCH_BOX_128 typedef launch_t_cpu<TMOL_LAUNCH_NT, 1> launch_t;
#define LAUNCH_BOX_32_OCC_AS(name, occ) \
  typedef launch_t_cpu<TMOL_LAUNCH_NT, 1> name;
#endif

#define LAUNCH_BOX_32_OCC(occ) LAUNCH_BOX_32_OCC_AS(launch_t, occ)
