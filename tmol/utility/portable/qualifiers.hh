#pragma once

// Portable function qualifiers for device-side math.
//
// tmol's scalar math is compiled by three different toolchains: the host C++
// compiler, nvcc, and (for the MPS backend) Apple's Metal compiler. Headers
// that are shared by all three may not use Eigen, the C++ standard library, or
// any CUDA-specific attribute, so they qualify their functions with TMOL_HD
// rather than EIGEN_DEVICE_FUNC.

#if defined(__METAL_VERSION__)
#define TMOL_PORTABLE_METAL 1
#else
#define TMOL_PORTABLE_METAL 0
#endif

#if defined(__CUDACC__)
// Match EIGEN_DEVICE_FUNC + EIGEN_STRONG_INLINE so that migrated call sites
// keep the inlining behavior they had under Eigen.
#define TMOL_HD __host__ __device__ __forceinline__
#else
#define TMOL_HD inline
#endif

// Explicit thread-address-space qualifier.
//
// MSL makes the address space part of every pointer and reference type. The
// newest language version infers `thread` where it is omitted, but tmol's
// shaders are compiled through torch's MPS shader API, which pins an older
// version that does not -- there, an unqualified `T&` is an error, not a
// thread-space reference. Portable headers therefore spell it out.
// Everywhere else this expands to nothing, because there is only one address
// space to name.
#if TMOL_PORTABLE_METAL
#define TMOL_THREAD thread
#else
#define TMOL_THREAD
#endif
