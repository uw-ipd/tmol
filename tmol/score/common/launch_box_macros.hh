#pragma once

// Common macros for MGPU launches. Uniform configurations need no per-GPU list.
#ifdef __NVCC__
#include <moderngpu/launch_box.hxx>

#define LAUNCH_BOX_32   \
  using namespace mgpu; \
  typedef launch_box_t<arch_20_cta<32, 1>> launch_t;
#define LAUNCH_BOX_64   \
  using namespace mgpu; \
  typedef launch_box_t<arch_20_cta<64, 1>> launch_t;
#define LAUNCH_BOX_128  \
  using namespace mgpu; \
  typedef launch_box_t<arch_20_cta<128, 1>> launch_t;

// Apply the requested occupancy on Volta and newer; older GPUs inherit the
// unconstrained configuration. MGPU propagates each setting to newer targets.
#define LAUNCH_BOX_32_OCC_AS(name, occ) \
  using namespace mgpu;                 \
  typedef launch_box_t<arch_20_cta<32, 1>, arch_70_cta<32, 1, 1, occ>> name;
#else
// CPU workgroups traverse their lanes serially, so one lane is sufficient.
template <int NT, int VT>
struct launch_t_cpu {
  struct sm_ptx {
    enum { nt = NT, vt = VT, vt0 = VT };
  };
};

#define LAUNCH_BOX_32 typedef launch_t_cpu<1, 1> launch_t;
#define LAUNCH_BOX_64 typedef launch_t_cpu<1, 1> launch_t;
#define LAUNCH_BOX_128 typedef launch_t_cpu<1, 1> launch_t;
#define LAUNCH_BOX_32_OCC_AS(name, occ) typedef launch_t_cpu<1, 1> name;
#endif

#define LAUNCH_BOX_32_OCC(occ) LAUNCH_BOX_32_OCC_AS(launch_t, occ)
