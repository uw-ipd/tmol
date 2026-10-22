#include <tmol/utility/tensor/native.hh>
#pragma once

#undef AT_DISPATCH_FLOATING_TYPES
#undef AT_DISPATCH_INDEX_TYPES
#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)      \
  [&] {                                                  \
    switch (TYPE) {                                      \
      case tmol::native::ScalarType::Float: {            \
        using scalar_t = float;                          \
        (__VA_ARGS__)();                                 \
        break;                                           \
      }                                                  \
      case tmol::native::ScalarType::Double: {           \
        using scalar_t = double;                         \
        (__VA_ARGS__)();                                 \
        break;                                           \
      }                                                  \
      default:                                           \
        AT_ERROR(NAME, ": expected float32 or float64"); \
    }                                                    \
  }()
#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)     \
  [&] {                                              \
    switch (TYPE) {                                  \
      case tmol::native::ScalarType::Int: {          \
        using scalar_t = int32_t;                    \
        (__VA_ARGS__)();                             \
        break;                                       \
      }                                              \
      case tmol::native::ScalarType::Long: {         \
        using scalar_t = int64_t;                    \
        (__VA_ARGS__)();                             \
        break;                                       \
      }                                              \
      default:                                       \
        AT_ERROR(NAME, ": expected int32 or int64"); \
    }                                                \
  }()

#include <tmol/utility/tensor/TensorAccessor.h>
#include <iostream>

#ifdef WITH_CUDA

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)                      \
  [&] {                                                                     \
    if (TYPE.device().is_cpu()) {                                           \
      constexpr tmol::Device device_t = tmol::Device::CPU;                  \
      AT_DISPATCH_FLOATING_TYPES(                                           \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                                   \
      constexpr tmol::Device device_t = tmol::Device::CUDA;                 \
      AT_DISPATCH_FLOATING_TYPES(                                           \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().type() == tmol::native::DeviceType::MPS) {     \
      AT_ERROR(NAME, ": this term has no MPS backend yet.");                \
    } else {                                                                \
      AT_ERROR("Unsupported tensor device type.");                          \
    }                                                                       \
  }();

#else

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)                      \
  [&] {                                                                     \
    if (TYPE.device().is_cpu()) {                                           \
      constexpr tmol::Device device_t = tmol::Device::CPU;                  \
      AT_DISPATCH_FLOATING_TYPES(                                           \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                                   \
      AT_ERROR("Unsupported cuda tensor, non-cuda build.");                 \
    } else if (TYPE.device().type() == tmol::native::DeviceType::MPS) {     \
      AT_ERROR(NAME, ": this term has no MPS backend yet.");                \
    } else {                                                                \
      AT_ERROR("Unsupported device type.");                                 \
    }                                                                       \
  }();

#endif

#ifdef WITH_CUDA

#define TMOL_DISPATCH_INDEX_DEVICE(TYPE, NAME, ...)                         \
  [&] {                                                                     \
    if (TYPE.device().is_cpu()) {                                           \
      constexpr tmol::Device device_t = tmol::Device::CPU;                  \
      AT_DISPATCH_INDEX_TYPES(                                              \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                                   \
      constexpr tmol::Device device_t = tmol::Device::CUDA;                 \
      AT_DISPATCH_INDEX_TYPES(                                              \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().type() == tmol::native::DeviceType::MPS) {     \
      AT_ERROR(NAME, ": this term has no MPS backend yet.");                \
    } else {                                                                \
      AT_ERROR("Unsupported tensor device type.");                          \
    }                                                                       \
  }();

#else

#define TMOL_DISPATCH_INDEX_DEVICE(TYPE, NAME, ...)                         \
  [&] {                                                                     \
    if (TYPE.device().is_cpu()) {                                           \
      constexpr tmol::Device device_t = tmol::Device::CPU;                  \
      AT_DISPATCH_INDEX_TYPES(                                              \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                                   \
      AT_ERROR("Unsupported cuda tensor, non-cuda build.");                 \
    } else if (TYPE.device().type() == tmol::native::DeviceType::MPS) {     \
      AT_ERROR(NAME, ": this term has no MPS backend yet.");                \
    } else {                                                                \
      AT_ERROR("Unsupported device type.");                                 \
    }                                                                       \
  }();

#endif

// ── Opting a term into the Metal backend ────────────────────────────────────
// The dispatch macros above reject MPS tensors, because reaching
// Dispatch<..., Device::MPS, ...> from an operator whose term has no MPS
// instantiation compiles (pybind modules link with -undefined
// dynamic_lookup) and then fails at call time with a missing symbol. A term
// that has been ported uses the _MPS variant instead, so the set of terms
// that accept an MPS pose is visible in the operator definitions rather than
// discovered at run time.
#ifdef WITH_MPS

#define TMOL_DISPATCH_FLOATING_DEVICE_MPS(TYPE, NAME, ...)                  \
  [&] {                                                                     \
    if (TYPE.device().is_cpu()) {                                           \
      constexpr tmol::Device device_t = tmol::Device::CPU;                  \
      AT_DISPATCH_FLOATING_TYPES(                                           \
          tmol::native::scalar_type_from_options(TYPE), NAME, __VA_ARGS__); \
    } else if (TYPE.device().type() == tmol::native::DeviceType::MPS) {     \
      constexpr tmol::Device device_t = tmol::Device::MPS;                  \
      /* Metal has no double, so only float32 reaches the GPU. */           \
      if (tmol::native::scalar_type_from_options(TYPE)                      \
          != tmol::native::ScalarType::Float) {                             \
        AT_ERROR(NAME, ": the MPS backend is float32 only.");               \
      }                                                                     \
      using scalar_t = float;                                               \
      (__VA_ARGS__)();                                                      \
    } else {                                                                \
      TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, __VA_ARGS__);               \
    }                                                                       \
  }();

#else

#define TMOL_DISPATCH_FLOATING_DEVICE_MPS(TYPE, NAME, ...) \
  TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, __VA_ARGS__)

#endif
