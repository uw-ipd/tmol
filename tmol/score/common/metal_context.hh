#pragma once

// metal_context.hh
//
// Pure C++ API for the Metal context singleton.
//
// This header is included by .mps.mm (Objective-C++) and by plain C++ files
// that need to call MPS Metal dispatch helpers.  It deliberately contains no
// Objective-C types so it can be safely included from standard C++ TUs.
//
// The implementation lives in metal_context.mm (compiled as Objective-C++).

#ifdef WITH_MPS

#include <cstddef>  // size_t
#include <cstdint>

namespace tmol {
namespace mps {

// ---------------------------------------------------------------------------
// One-time initialisation — loads tmol_primitives.metallib from the same
// directory as the _C extension module.  Safe to call multiple times; only
// the first call has any effect.
// ---------------------------------------------------------------------------
void initialize();

// ---------------------------------------------------------------------------
// Synchronize all pending Metal commands to the CPU.
// Equivalent to torch::mps::synchronize().
// ---------------------------------------------------------------------------
void synchronize();

// ---------------------------------------------------------------------------
// Low-level dispatch helpers used by DeviceOperations<MPS> Phase-2 kernels.
//
// Each function encodes a Metal compute pass and commits it to the shared
// command queue.  The caller is responsible for ensuring buffers are valid
// MPS tensors (i.e. allocated on the MPS device by PyTorch).
//
// All functions block until the GPU pass is complete (synchronize is called
// internally) so that the next CPU-side operation sees up-to-date results.
// ---------------------------------------------------------------------------

// Parallel prefix scan (inclusive add) over n float elements.
// dst may equal src (in-place).
void scan_inclusive_float(float* src, float* dst, int n);

// Parallel reduction (sum) over n float elements.  Returns result on CPU.
float reduce_sum_float(const float* src, int n);

// Segmented inclusive scan (add) over n float elements.
// seg_start_inds: sorted array of n_segs segment-start indices.
// dst must be a separate buffer of size n (not aliased with src).
void segmented_scan_inclusive_float(
    const float* src,
    float* dst,
    const int* seg_start_inds,
    int n,
    int n_segs);

}  // namespace mps
}  // namespace tmol

#endif  // WITH_MPS
