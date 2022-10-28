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

#define CTA_REAL_REDUCE_T_TYPEDEF
#define CTA_REAL_REDUCE_T_VARIABLE

#endif
