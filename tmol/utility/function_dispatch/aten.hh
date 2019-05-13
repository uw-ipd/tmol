#pragma once

#include <ATen/Dispatch.h>
#include <tmol/utility/tensor/TensorAccessor.h>

#ifdef WITH_CUDA

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)       \
  [&] {                                                      \
    if (TYPE.device_type() == at::DeviceType::CPU) {         \
      constexpr tmol::Device device_t = tmol::Device::CPU;   \
                                                             \
      AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);   \
    } else if (TYPE.device_type() == at::DeviceType::CUDA) { \
      constexpr tmol::Device device_t = tmol::Device::CUDA;  \
                                                             \
      AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);   \
    } else {                                                 \
      AT_ERROR("Unsupported tensor device type.");           \
    }                                                        \
  }();

#else

#define TMOL_DISPATCH_FLOATING_DEVICE(TYPE, NAME, ...)       \
  [&] {                                                      \
    if (TYPE.device_type() == at::DeviceType::CPU) {         \
      constexpr tmol::Device device_t = tmol::Device::CPU;   \
                                                             \
      AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);   \
    } else if (TYPE.device_type() == at::DeviceType::CUDA) { \
      AT_ERROR("Unsupported cuda tensor, non-cuda build.");  \
    } else {                                                 \
      AT_ERROR("Unsupported device type.");                  \
    }                                                        \
  }();

#endif
