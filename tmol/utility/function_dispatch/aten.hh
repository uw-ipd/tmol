#pragma once

#include <ATen/Dispatch.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <iostream>

// ---------------------------------------------------------------------------
// TMOL_DISPATCH_FLOATING_DEVICE
// Dispatches on tensor device (CPU / CUDA / MPS) and floating scalar type.
// Each WITH_* guard is independent; multiple backends can coexist in one build.
// ---------------------------------------------------------------------------

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)                 \
  [&] {                                                                \
    if (TYPE.device().is_cpu()) {                                      \
      constexpr tmol::Device device_t = tmol::Device::CPU;             \
      AT_DISPATCH_FLOATING_TYPES(                                      \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                              \
      IF_WITH_CUDA(                                                    \
        constexpr tmol::Device device_t = tmol::Device::CUDA;          \
        AT_DISPATCH_FLOATING_TYPES(                                    \
            c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
      , AT_ERROR("Unsupported cuda tensor, non-cuda build."); )        \
    } else if (TYPE.device().is_mps()) {                               \
      IF_WITH_MPS(                                                     \
        constexpr tmol::Device device_t = tmol::Device::MPS;           \
        AT_DISPATCH_FLOATING_TYPES(                                    \
            c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
      , AT_ERROR("Unsupported MPS tensor, non-MPS build."); )          \
    } else {                                                           \
      AT_ERROR("Unsupported tensor device type.");                     \
    }                                                                  \
  }();

// ---------------------------------------------------------------------------
// TMOL_DISPATCH_INDEX_DEVICE
// Same pattern for integer index types.
// ---------------------------------------------------------------------------

#define TMOL_DISPATCH_INDEX_DEVICE(TYPE, NAME, ...)                    \
  [&] {                                                                \
    if (TYPE.device().is_cpu()) {                                      \
      constexpr tmol::Device device_t = tmol::Device::CPU;             \
      AT_DISPATCH_INDEX_TYPES(                                         \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                              \
      IF_WITH_CUDA(                                                    \
        constexpr tmol::Device device_t = tmol::Device::CUDA;          \
        AT_DISPATCH_INDEX_TYPES(                                       \
            c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
      , AT_ERROR("Unsupported cuda tensor, non-cuda build."); )        \
    } else if (TYPE.device().is_mps()) {                               \
      IF_WITH_MPS(                                                     \
        constexpr tmol::Device device_t = tmol::Device::MPS;           \
        AT_DISPATCH_INDEX_TYPES(                                       \
            c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
      , AT_ERROR("Unsupported MPS tensor, non-MPS build."); )          \
    } else {                                                           \
      AT_ERROR("Unsupported tensor device type.");                     \
    }                                                                  \
  }();

// ---------------------------------------------------------------------------
// Backend-presence helpers used by the macros above.
// Each translates to "execute A" when the backend is compiled in, or "B" when
// it is not — without requiring nested #ifdef inside a macro argument.
// ---------------------------------------------------------------------------

#ifdef WITH_CUDA
#  define IF_WITH_CUDA(A, B) A
#else
#  define IF_WITH_CUDA(A, B) B
#endif

#ifdef WITH_MPS
#  define IF_WITH_MPS(A, B) A
#else
#  define IF_WITH_MPS(A, B) B
#endif
