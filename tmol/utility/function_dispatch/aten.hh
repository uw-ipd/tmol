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
    } else {                                                                \
      AT_ERROR("Unsupported device type.");                                 \
    }                                                                       \
  }();

#endif
