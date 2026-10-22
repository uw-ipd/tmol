#pragma once

#include <tmol/utility/portable/qualifiers.hh>

#ifdef __NVCC__
#include <tmol/utility/portable/math.hh>
#include <moderngpu/cta_reduce.hxx>
// #include <moderngpu/transform.hxx>
#endif

// Common macros for creating x.impl.hh files that complete
// the execution diamond:
//
//       vestibule
//         /   \
//       CPU   GPU
//         \   /
//        x.impl.hh

// Macro for the declaration of a shared memory struct inside
// the outer lambda
#ifdef __NVCC__
#define SHARED_MEMORY __shared__
#else
#define SHARED_MEMORY
#endif

// Macro for the declaration of all device lambdas;
// NOTE: TMOL_HD declares functions
// __host__ __device__
// which is unfortunate if you want to put a lambda inside
// a lambda, which we do.
#ifdef __NVCC__
#define TMOL_DEVICE_FUNC __device__
#else
#define TMOL_DEVICE_FUNC
#endif

// Retrieve the architecture-specific nt/vt params
// from the launch_t struct; requires a launch_box
// specification (perhaps from launch_box_macros.hh)
// Identical on GPU and CPU, though, the implementation
// of launch_t differs
#define CTA_LAUNCH_T_PARAMS                   \
  typedef typename launch_t::sm_ptx params_t; \
  enum {                                      \
    nt = params_t::nt,                        \
    vt = params_t::vt,                        \
    vt0 = params_t::vt0,                      \
    nv = nt * vt                              \
  }

// Macros for declaring a reduction variable inside the outer lambda
// within a shared-memory struct
#ifdef __NVCC__
#define CTA_REAL_REDUCE_T_TYPEDEF \
  CTA_LAUNCH_T_PARAMS;            \
  typedef mgpu::cta_reduce_t<nt, Real> reduce_t
#define CTA_REAL_REDUCE_T_VARIABLE typename reduce_t::storage_t reduce
#else

#define CTA_REAL_REDUCE_T_TYPEDEF CTA_LAUNCH_T_PARAMS
#define CTA_REAL_REDUCE_T_VARIABLE

#endif

// nvcc and the Metal standard library both supply min/max as builtins; the
// host build has to pull them in from <algorithm>. Metal translation units
// must not see libc++ at all, so the include is excluded there as well.
// TO DO: find a better place for this
#if !defined(__NVCC__) && !TMOL_PORTABLE_METAL
#include <algorithm>

using std::max;
using std::min;

#endif
