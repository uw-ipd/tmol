#pragma once

#include <ATen/Dispatch.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <iostream>

#ifdef WITH_CUDA

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)                 \
  [&] {                                                                \
    if (TYPE.device().is_cpu()) {                                      \
      constexpr tmol::Device device_t = tmol::Device::CPU;             \
      AT_DISPATCH_FLOATING_TYPES(                                      \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                              \
      constexpr tmol::Device device_t = tmol::Device::CUDA;            \
      AT_DISPATCH_FLOATING_TYPES(                                      \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else {                                                           \
      AT_ERROR("Unsupported tensor device type.");                     \
    }                                                                  \
  }();

#else

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)                 \
  [&] {                                                                \
    if (TYPE.device().is_cpu()) {                                      \
      constexpr tmol::Device device_t = tmol::Device::CPU;             \
      AT_DISPATCH_FLOATING_TYPES(                                      \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                              \
      AT_ERROR("Unsupported cuda tensor, non-cuda build.");            \
    } else {                                                           \
      AT_ERROR("Unsupported device type.");                            \
    }                                                                  \
  }();

#endif

#ifdef WITH_CUDA

#define TMOL_DISPATCH_INDEX_DEVICE(TYPE, NAME, ...)                    \
  [&] {                                                                \
    if (TYPE.device().is_cpu()) {                                      \
      constexpr tmol::Device device_t = tmol::Device::CPU;             \
      AT_DISPATCH_INDEX_TYPES(                                         \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                              \
      constexpr tmol::Device device_t = tmol::Device::CUDA;            \
      AT_DISPATCH_INDEX_TYPES(                                         \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else {                                                           \
      AT_ERROR("Unsupported tensor device type.");                     \
    }                                                                  \
  }();

#else

#define TMOL_DISPATCH_INDEX_DEVICE(TYPE, NAME, ...)                    \
  [&] {                                                                \
    if (TYPE.device().is_cpu()) {                                      \
      constexpr tmol::Device device_t = tmol::Device::CPU;             \
      AT_DISPATCH_INDEX_TYPES(                                         \
          c10::typeMetaToScalarType(TYPE.dtype()), NAME, __VA_ARGS__); \
    } else if (TYPE.device().is_cuda()) {                              \
      AT_ERROR("Unsupported cuda tensor, non-cuda build.");            \
    } else {                                                           \
      AT_ERROR("Unsupported device type.");                            \
    }                                                                  \
  }();

#endif
