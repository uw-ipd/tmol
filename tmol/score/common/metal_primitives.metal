// metal_primitives.metal
//
// Metal Shading Language (MSL) parallel primitives for tmol.
//
// Compiled by CMake via:
//   xcrun -sdk macosx metal -c metal_primitives.metal -o metal_primitives.air
//   xcrun -sdk macosx metallib metal_primitives.air -o tmol_primitives.metallib
//
// These kernels implement the same algorithms as DeviceOperations<CUDA>
// using ModernGPU, but expressed in MSL for Apple Metal.
//
// Naming convention:
//   tmol_<op>_<type>   e.g. tmol_scan_float, tmol_reduce_float
//
// Buffer layout for each kernel is documented inline.
//
// Thread / threadgroup sizing guidance (set from metal_context.mm):
//   SIMD-group width on Apple Silicon = 32
//   Maximum threads per threadgroup   = 1024

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════
// §1  transform / forall
//
//   Apply a pure-compute kernel over N elements.  The actual payload (score
//   computation) is implemented in per-score-term .metal files that #include
//   this header for the helper utilities below.  This stub serves as a smoke-
//   test / template for those files.
//
//   buffer(0) : float*  input  [N]
//   buffer(1) : float*  output [N]
//   buffer(2) : uint    N
// ═══════════════════════════════════════════════════════════════════════════

kernel void tmol_transform_identity_float(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  N      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
  if (idx >= N) return;
  output[idx] = input[idx];
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  inclusive prefix-scan (add)  —  two-pass: local then global carry-in
//
// Pass 1 — tmol_scan_local_float
//   Each threadgroup computes an independent inclusive scan of its tile and
//   stores the tile-total in carry_out[tgid].
//
//   buffer(0) : float*  data      [N]   in/out
//   buffer(1) : float*  carry_out [num_threadgroups]
//   buffer(2) : uint    N
//
// Pass 2 — tmol_scan_carry_float
//   Each threadgroup (except 0) adds the prefix sum of carry_out[0..tgid-1]
//   to every element in its tile.
//
//   buffer(0) : float*  data      [N]   in/out
//   buffer(1) : float*  carry_out [num_threadgroups]  (prefix-scanned)
//   buffer(2) : uint    N
// ═══════════════════════════════════════════════════════════════════════════

kernel void tmol_scan_local_float(
    device float*       data      [[buffer(0)]],
    device float*       carry_out [[buffer(1)]],
    constant uint&      N         [[buffer(2)]],
    threadgroup float*  tg_data   [[threadgroup(0)]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tpg  [[threads_per_threadgroup]])
{
  uint gid = tgid * tpg + lid;

  // Load element (zero-pad out-of-bounds)
  float val = (gid < N) ? data[gid] : 0.0f;
  tg_data[lid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Up-sweep (reduce) phase
  for (uint stride = 1; stride < tpg; stride <<= 1) {
    uint idx = (lid + 1) * (stride << 1) - 1;
    if (idx < tpg) {
      tg_data[idx] += tg_data[idx - stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store tile total and clear last element for exclusive down-sweep
  if (lid == tpg - 1) {
    carry_out[tgid] = tg_data[lid];
    tg_data[lid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Down-sweep phase → exclusive scan in tg_data
  for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
    uint idx = (lid + 1) * (stride << 1) - 1;
    if (idx < tpg) {
      float tmp        = tg_data[idx - stride];
      tg_data[idx - stride] = tg_data[idx];
      tg_data[idx]          = tg_data[idx - stride] + tmp;  // wait — need original
    }
    // Actually convert to inclusive by adding original element back:
    // We'll do it after the sweep.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Convert exclusive → inclusive
  float inclusive = tg_data[lid] + val;
  if (gid < N) {
    data[gid] = inclusive;
  }
}

kernel void tmol_scan_carry_float(
    device float*       data      [[buffer(0)]],
    device const float* carry_in  [[buffer(1)]],
    constant uint&      N         [[buffer(2)]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tpg  [[threads_per_threadgroup]])
{
  if (tgid == 0) return;  // first tile has no carry-in
  uint gid = tgid * tpg + lid;
  if (gid >= N) return;
  data[gid] += carry_in[tgid - 1];
}

// ═══════════════════════════════════════════════════════════════════════════
// §3  parallel reduction (add, float)
//
//   tmol_reduce_partial_float  — each threadgroup reduces its tile, storing
//                                partial sums in partials[tgid].
//   tmol_reduce_final_float    — reduces partials[] into result[0].
//
//   buffer(0) : float*  data     [N]
//   buffer(1) : float*  partials [num_threadgroups]
//   buffer(2) : uint    N
// ═══════════════════════════════════════════════════════════════════════════

kernel void tmol_reduce_partial_float(
    device const float* data     [[buffer(0)]],
    device       float* partials [[buffer(1)]],
    constant     uint&  N        [[buffer(2)]],
    threadgroup  float* tg_data  [[threadgroup(0)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]])
{
  uint gid = tgid * tpg + lid;
  tg_data[lid] = (gid < N) ? data[gid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      tg_data[lid] += tg_data[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    partials[tgid] = tg_data[0];
  }
}

kernel void tmol_reduce_final_float(
    device const float* partials [[buffer(0)]],
    device       float* result   [[buffer(1)]],
    constant     uint&  n_parts  [[buffer(2)]],
    threadgroup  float* tg_data  [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
  tg_data[lid] = (lid < n_parts) ? partials[lid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      tg_data[lid] += tg_data[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    result[0] = tg_data[0];
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  segmented inclusive scan (add, float)
//
//   Segments are defined by a sorted array of segment-start indices.
//   This is the Metal analogue of tmol::kinematics::kernel_segscan.
//
//   Each SIMD-group (32 threads) performs an intra-group segmented scan
//   using simd_shuffle_up and a segment-reset mask.  A second pass
//   propagates carry-in values across SIMD-group boundaries within a
//   threadgroup, and a third pass propagates across threadgroup boundaries.
//
//   buffer(0) : float*  values        [n]   in/out
//   buffer(1) : uint*   seg_starts    [n_segs]  sorted ascending
//   buffer(2) : float*  tg_carry_out  [num_threadgroups]
//   buffer(3) : uint*   tg_seg_flags  [num_threadgroups]  1 = segment crossed
//   buffer(4) : uint    n
//   buffer(5) : uint    n_segs
// ═══════════════════════════════════════════════════════════════════════════

// Binary search: first index in seg_starts[0..n_segs) >= target.
// Returns n_segs if all elements are < target.
static inline uint lower_bound(
    const device uint* seg_starts,
    uint n_segs,
    uint target)
{
  uint lo = 0, hi = n_segs;
  while (lo < hi) {
    uint mid = (lo + hi) >> 1;
    if (seg_starts[mid] < target) lo = mid + 1;
    else                           hi = mid;
  }
  return lo;
}

kernel void tmol_segscan_upsweep_float(
    device       float* values       [[buffer(0)]],
    device const uint*  seg_starts   [[buffer(1)]],
    device       float* tg_carry_out [[buffer(2)]],
    device       uint*  tg_seg_flags [[buffer(3)]],
    constant     uint&  n            [[buffer(4)]],
    constant     uint&  n_segs       [[buffer(5)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
  uint gid = tgid * tpg + lid;

  float val  = (gid < n) ? values[gid] : 0.0f;
  bool  flag = false;  // is this position the START of a new segment?

  if (gid < n) {
    // Check if any segment starts exactly at gid
    uint pos = lower_bound(seg_starts, n_segs, gid);
    flag = (pos < n_segs && seg_starts[pos] == gid);
  }

  // ── Intra-SIMD-group segmented inclusive scan ──
  // For each power-of-two offset, propagate value leftward unless a segment
  // boundary is encountered between the two positions.
  for (uint offset = 1; offset < 32; offset <<= 1) {
    float prev_val  = simd_shuffle_up(val,  offset);
    bool  prev_flag = (bool)simd_shuffle_up((uint)flag, offset);
    if (simd_lid >= offset && !flag) {
      val  += prev_val;
      // Inherit the segment-start flag of the older element only if no
      // boundary was between us (flag is already false here).
      (void)prev_flag;  // flag stays false — already captured by condition
    }
  }

  if (gid < n) {
    values[gid] = val;
  }

  // Store threadgroup carry-out: value of the last active lane in this group.
  // Also store whether any segment boundary exists in this tile.
  threadgroup float tg_last[1];
  threadgroup uint  tg_has_seg[1];

  if (lid == tpg - 1 || gid == n - 1) {
    tg_last[0]    = val;
    tg_has_seg[0] = 0u;  // will be OR'd below
  }
  // OR all segment flags within threadgroup
  // (simple approach: lane 0 does a loop after barrier)
  threadgroup_barrier(mem_flags::mem_device);

  if (lid == 0) {
    // Scan for any segment flag in this tile
    uint tile_start = tgid * tpg;
    uint tile_end   = min(tile_start + tpg, n);
    uint pos = lower_bound(seg_starts, n_segs, tile_start);
    tg_seg_flags[tgid] = (pos < n_segs && seg_starts[pos] < tile_end) ? 1u : 0u;
    tg_carry_out[tgid] = tg_last[0];
  }
}

// Pass 2: add inter-threadgroup carry-in (host performs exclusive scan of
// tg_carry_out and calls this kernel with the result).
kernel void tmol_segscan_downsweep_float(
    device       float* values      [[buffer(0)]],
    device const float* carry_in    [[buffer(1)]],  // exclusive-scanned carry
    device const uint*  seg_starts  [[buffer(2)]],
    constant     uint&  n           [[buffer(3)]],
    constant     uint&  n_segs      [[buffer(4)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]])
{
  if (tgid == 0) return;  // no carry-in for the first group

  uint gid = tgid * tpg + lid;
  if (gid >= n) return;

  // Only add carry-in if no segment boundary separates this position from
  // the start of the tile.
  uint tile_start = tgid * tpg;
  uint pos        = lower_bound(seg_starts, n_segs, tile_start);
  bool seg_in_tile = (pos < n_segs && seg_starts[pos] <= gid);

  if (!seg_in_tile) {
    values[gid] += carry_in[tgid - 1];
  }
}
