#pragma once

// Common macros for working with MGPU launch_box
#ifdef __NVCC__
#include <moderngpu/launch_box.hxx>
#endif

#ifndef __NVCC__
// A stub for a CPU launch_t mimicing the
// one provided by mgpu::launch_box
template <int NT, int VT>
struct launch_t_cpu {
  struct sm_ptx {
    enum { nt = NT, vt = VT, vt0 = VT };
  };
};
#endif

#ifdef __NVCC__
// Create a launch box that sets nt to 32 for all (supported) architectures
#define LAUNCH_BOX_32     \
  using namespace mgpu;   \
  typedef launch_box_t<   \
      arch_20_cta<32, 1>, \
      arch_35_cta<32, 1>, \
      arch_52_cta<32, 1>, \
      arch_70_cta<32, 1>, \
      arch_75_cta<32, 1>> \
      launch_t;

#else
// On the CPU, an "ntreads" of 1 is faster because there
// is only one set of threads
#define LAUNCH_BOX_32 typedef launch_t_cpu<1, 1> launch_t;
#endif

#ifdef __NVCC__
// Create a launch box that sets nt to 64 for all (supported) architectures
#define LAUNCH_BOX_64     \
  using namespace mgpu;   \
  typedef launch_box_t<   \
      arch_20_cta<64, 1>, \
      arch_35_cta<64, 1>, \
      arch_52_cta<64, 1>, \
      arch_70_cta<64, 1>, \
      arch_75_cta<64, 1>> \
      launch_t;

#else
// On the CPU, an "ntreads" of 1 is faster because there
// is only one set of threads
#define LAUNCH_BOX_64 typedef launch_t_cpu<1, 1> launch_t;
#endif

#ifdef __NVCC__
// Create a launch box that sets nt to 64 for all (supported) architectures
#define LAUNCH_BOX_128     \
  using namespace mgpu;    \
  typedef launch_box_t<    \
      arch_20_cta<128, 1>, \
      arch_35_cta<128, 1>, \
      arch_52_cta<128, 1>, \
      arch_70_cta<128, 1>, \
      arch_75_cta<128, 1>> \
      launch_t;

#else
// On the CPU, an "ntreads" of 1 is faster because there
// is only one set of threads
#define LAUNCH_BOX_128 typedef launch_t_cpu<1, 1> launch_t;
#endif
