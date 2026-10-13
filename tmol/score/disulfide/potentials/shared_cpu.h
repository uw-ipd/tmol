#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TMOL_DISULFIDE_CPU_BUILD)
#define TMOL_DISULFIDE_CPU_API __declspec(dllexport)
#else
#define TMOL_DISULFIDE_CPU_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define TMOL_DISULFIDE_CPU_API __attribute__((visibility("default")))
#else
#define TMOL_DISULFIDE_CPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TMOL_DISULFIDE_CPU_ABI_VERSION 1u

enum tmol_disulfide_cpu_status {
  TMOL_DISULFIDE_CPU_SUCCESS = 0,
  TMOL_DISULFIDE_CPU_INVALID_EXECUTOR = 1,
  TMOL_DISULFIDE_CPU_INVALID_MODE = 2,
  TMOL_DISULFIDE_CPU_INVALID_SHAPE = 3,
  TMOL_DISULFIDE_CPU_INVALID_INDEX = 4,
  TMOL_DISULFIDE_CPU_MISSING_DERIVATIVE_OUTPUT = 5,
  TMOL_DISULFIDE_CPU_INSUFFICIENT_WORKSPACE = 6
};

enum tmol_disulfide_cpu_forward_mode {
  TMOL_DISULFIDE_CPU_INFERENCE = 0,
  TMOL_DISULFIDE_CPU_TRACKED = 1
};

enum tmol_disulfide_cpu_vjp_mode {
  TMOL_DISULFIDE_CPU_MAY_OVERLAP = 0,
  TMOL_DISULFIDE_CPU_DISJOINT_PAIRS = 1
};

typedef void (*tmol_disulfide_cpu_chunk)(
    void* task_context, int64_t begin, int64_t end);
typedef void (*tmol_disulfide_cpu_parallel_for)(
    void* executor_context,
    int64_t count,
    tmol_disulfide_cpu_chunk chunk,
    void* task_context);

typedef struct tmol_disulfide_cpu_executor {
  void* context;
  tmol_disulfide_cpu_parallel_for parallel_for;
} tmol_disulfide_cpu_executor;

typedef struct tmol_disulfide_cpu_workspace {
  size_t bytes;
  size_t alignment;
} tmol_disulfide_cpu_workspace;

/* Strides are measured in logical records: xyz, eight-int pair, or 33-scalar
 * parameter records. Every xyz/pair/parameter record is tightly packed.
 * Every stride of a nonempty borrowed view must be positive. */
#define TMOL_DISULFIDE_DECLARE_C_ABI(Suffix, Real)                            \
  typedef struct tmol_disulfide_cpu_##Suffix##_inputs {                       \
    const Real* coords;                                                       \
    int64_t n_atoms;                                                          \
    int64_t coord_stride;                                                     \
    const int32_t* pairs;                                                     \
    int64_t n_pairs;                                                          \
    int64_t pair_stride;                                                      \
    const Real* params;                                                       \
    int64_t n_params;                                                         \
    int64_t param_stride;                                                     \
  } tmol_disulfide_cpu_##Suffix##_inputs;                                     \
  typedef struct tmol_disulfide_cpu_##Suffix##_forward_outputs {              \
    Real* energies;                                                           \
    int64_t energy_stride;                                                    \
    Real* coordinate_derivatives;                                             \
    int64_t derivative_pair_stride;                                           \
    int64_t derivative_atom_stride;                                           \
  } tmol_disulfide_cpu_##Suffix##_forward_outputs;                            \
  TMOL_DISULFIDE_CPU_API tmol_disulfide_cpu_workspace                         \
  tmol_disulfide_cpu_##Suffix##_forward_workspace(int64_t n_pairs);           \
  TMOL_DISULFIDE_CPU_API tmol_disulfide_cpu_workspace                         \
  tmol_disulfide_cpu_##Suffix##_recompute_vjp_workspace(int64_t n_pairs);     \
  TMOL_DISULFIDE_CPU_API int32_t tmol_disulfide_cpu_##Suffix##_forward(       \
      tmol_disulfide_cpu_executor executor,                                   \
      tmol_disulfide_cpu_##Suffix##_inputs inputs,                            \
      int32_t mode,                                                           \
      tmol_disulfide_cpu_##Suffix##_forward_outputs outputs);                 \
  TMOL_DISULFIDE_CPU_API int32_t tmol_disulfide_cpu_##Suffix##_vjp(           \
      tmol_disulfide_cpu_executor executor,                                   \
      int32_t mode,                                                           \
      const int32_t* pairs,                                                   \
      int64_t n_pairs,                                                        \
      int64_t pair_stride,                                                    \
      const Real* coordinate_derivatives,                                     \
      int64_t derivative_pair_stride,                                         \
      int64_t derivative_atom_stride,                                         \
      const Real* cotangents,                                                 \
      int64_t cotangent_stride,                                               \
      Real* coordinate_gradient,                                              \
      int64_t n_atoms,                                                        \
      int64_t gradient_stride);                                               \
  TMOL_DISULFIDE_CPU_API int32_t tmol_disulfide_cpu_##Suffix##_recompute_vjp( \
      tmol_disulfide_cpu_executor executor,                                   \
      int32_t mode,                                                           \
      tmol_disulfide_cpu_##Suffix##_inputs inputs,                            \
      const Real* cotangents,                                                 \
      int64_t cotangent_stride,                                               \
      Real* coordinate_gradient,                                              \
      int64_t gradient_stride,                                                \
      void* workspace,                                                        \
      size_t workspace_bytes)

TMOL_DISULFIDE_DECLARE_C_ABI(f32, float);
TMOL_DISULFIDE_DECLARE_C_ABI(f64, double);
#undef TMOL_DISULFIDE_DECLARE_C_ABI

TMOL_DISULFIDE_CPU_API uint32_t tmol_disulfide_cpu_abi_version(void);
TMOL_DISULFIDE_CPU_API const char* tmol_disulfide_cpu_status_message(
    int32_t status);
TMOL_DISULFIDE_CPU_API tmol_disulfide_cpu_executor
tmol_disulfide_cpu_serial_executor(void);

#ifdef __cplusplus
}
#endif
