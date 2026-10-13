#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TMOL_DISULFIDE_CUDA_BUILD)
#define TMOL_DISULFIDE_CUDA_API __declspec(dllexport)
#else
#define TMOL_DISULFIDE_CUDA_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define TMOL_DISULFIDE_CUDA_API __attribute__((visibility("default")))
#else
#define TMOL_DISULFIDE_CUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TMOL_DISULFIDE_CUDA_ABI_VERSION 1u

enum tmol_disulfide_cuda_status {
  TMOL_DISULFIDE_CUDA_SUCCESS = 0,
  TMOL_DISULFIDE_CUDA_INVALID_MODE = 1,
  TMOL_DISULFIDE_CUDA_INVALID_SHAPE = 2,
  TMOL_DISULFIDE_CUDA_INVALID_INDEX = 3,
  TMOL_DISULFIDE_CUDA_MISSING_DERIVATIVE_OUTPUT = 4,
  TMOL_DISULFIDE_CUDA_INSUFFICIENT_WORKSPACE = 5,
  TMOL_DISULFIDE_CUDA_LAUNCH_ERROR = 6
};

enum tmol_disulfide_cuda_forward_mode {
  TMOL_DISULFIDE_CUDA_INFERENCE = 0,
  TMOL_DISULFIDE_CUDA_TRACKED = 1
};

enum tmol_disulfide_cuda_vjp_mode {
  TMOL_DISULFIDE_CUDA_MAY_OVERLAP = 0,
  TMOL_DISULFIDE_CUDA_DISJOINT_PAIRS = 1
};

typedef struct tmol_disulfide_cuda_workspace {
  size_t bytes;
  size_t alignment;
} tmol_disulfide_cuda_workspace;

/*
 * Pointers address device memory. stream is the caller's cudaStream_t encoded
 * as uintptr_t; zero selects the legacy default stream. Calls enqueue work and
 * never synchronize. device_status is initialized asynchronously and receives
 * data-dependent errors such as invalid prepared indices; inspect it only
 * after the submitted stream completes.
 *
 * Strides are measured in logical xyz, eight-int pair, or 33-scalar parameter
 * records. Every record itself is tightly packed.
 */
#define TMOL_DISULFIDE_DECLARE_CUDA_ABI(Suffix, Real)                      \
  typedef struct tmol_disulfide_cuda_##Suffix##_inputs {                   \
    const Real* coords;                                                    \
    int64_t n_atoms;                                                       \
    int64_t coord_stride;                                                  \
    const int32_t* pairs;                                                  \
    int64_t n_pairs;                                                       \
    int64_t pair_stride;                                                   \
    const Real* params;                                                    \
    int64_t n_params;                                                      \
    int64_t param_stride;                                                  \
  } tmol_disulfide_cuda_##Suffix##_inputs;                                 \
  typedef struct tmol_disulfide_cuda_##Suffix##_forward_outputs {          \
    Real* energies;                                                        \
    int64_t energy_stride;                                                 \
    Real* coordinate_derivatives;                                          \
    int64_t derivative_pair_stride;                                        \
    int64_t derivative_atom_stride;                                        \
  } tmol_disulfide_cuda_##Suffix##_forward_outputs;                        \
  TMOL_DISULFIDE_CUDA_API tmol_disulfide_cuda_workspace                    \
  tmol_disulfide_cuda_##Suffix##_forward_workspace(int64_t n_pairs);       \
  TMOL_DISULFIDE_CUDA_API tmol_disulfide_cuda_workspace                    \
  tmol_disulfide_cuda_##Suffix##_recompute_vjp_workspace(int64_t n_pairs); \
  TMOL_DISULFIDE_CUDA_API int32_t tmol_disulfide_cuda_##Suffix##_forward(  \
      uintptr_t stream,                                                    \
      tmol_disulfide_cuda_##Suffix##_inputs inputs,                        \
      int32_t mode,                                                        \
      tmol_disulfide_cuda_##Suffix##_forward_outputs outputs,              \
      int32_t* device_status);                                             \
  TMOL_DISULFIDE_CUDA_API int32_t tmol_disulfide_cuda_##Suffix##_vjp(      \
      uintptr_t stream,                                                    \
      int32_t mode,                                                        \
      const int32_t* pairs,                                                \
      int64_t n_pairs,                                                     \
      int64_t pair_stride,                                                 \
      const Real* coordinate_derivatives,                                  \
      int64_t derivative_pair_stride,                                      \
      int64_t derivative_atom_stride,                                      \
      const Real* cotangents,                                              \
      int64_t cotangent_stride,                                            \
      Real* coordinate_gradient,                                           \
      int64_t n_atoms,                                                     \
      int64_t gradient_stride,                                             \
      int32_t* device_status);                                             \
  TMOL_DISULFIDE_CUDA_API int32_t                                          \
  tmol_disulfide_cuda_##Suffix##_recompute_vjp(                            \
      uintptr_t stream,                                                    \
      int32_t mode,                                                        \
      tmol_disulfide_cuda_##Suffix##_inputs inputs,                        \
      const Real* cotangents,                                              \
      int64_t cotangent_stride,                                            \
      Real* coordinate_gradient,                                           \
      int64_t gradient_stride,                                             \
      void* workspace,                                                     \
      size_t workspace_bytes,                                              \
      int32_t* device_status)

TMOL_DISULFIDE_DECLARE_CUDA_ABI(f32, float);
TMOL_DISULFIDE_DECLARE_CUDA_ABI(f64, double);
#undef TMOL_DISULFIDE_DECLARE_CUDA_ABI

TMOL_DISULFIDE_CUDA_API uint32_t tmol_disulfide_cuda_abi_version(void);
TMOL_DISULFIDE_CUDA_API const char* tmol_disulfide_cuda_status_message(
    int32_t status);

#ifdef __cplusplus
}
#endif
