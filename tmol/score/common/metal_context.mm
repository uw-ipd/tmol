// metal_context.mm
//
// Objective-C++ implementation of the Metal context singleton for tmol.
//
// Responsibilities:
//   1. Load tmol_primitives.metallib from the same directory as the _C
//      extension module at first call to initialize().
//   2. Build MTLComputePipelineState objects for each primitive kernel.
//   3. Expose a C++ API (declared in metal_context.hh) for use from plain
//      C++ translation units (.mps.mm files that are compiled as ObjC++).
//
// Compile flags needed:
//   -fobjc-arc   (automatic reference counting)
//   -framework Metal -framework Foundation

#ifdef WITH_MPS

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <mutex>
#include <stdexcept>
#include <string>

#include "metal_context.hh"

// ---------------------------------------------------------------------------
// PyTorch MPS interop — get the underlying MTLBuffer for a raw pointer that
// was obtained from an MPS tensor via tensor.data_ptr().
// PyTorch >= 2.0 provides at::mps::getMTLBufferStorage().
// ---------------------------------------------------------------------------
#include <ATen/Tensor.h>
namespace at { namespace mps {
  id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor);
}}

namespace tmol {
namespace mps {

// ---------------------------------------------------------------------------
// Internal singleton
// ---------------------------------------------------------------------------
namespace {

struct MtlContext {
  id<MTLDevice>       device   = nil;
  id<MTLCommandQueue> queue    = nil;
  id<MTLLibrary>      library  = nil;

  // Primitive pipeline states
  id<MTLComputePipelineState> pso_scan_local   = nil;
  id<MTLComputePipelineState> pso_scan_carry   = nil;
  id<MTLComputePipelineState> pso_reduce_part  = nil;
  id<MTLComputePipelineState> pso_reduce_final = nil;
  id<MTLComputePipelineState> pso_segscan_up   = nil;
  id<MTLComputePipelineState> pso_segscan_down = nil;

  bool initialised = false;

  // ── helpers ──────────────────────────────────────────────────────────────

  id<MTLComputePipelineState> make_pso(NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
      throw std::runtime_error(
          std::string("tmol Metal: function not found in metallib: ")
          + [name UTF8String]);
    }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      throw std::runtime_error(
          std::string("tmol Metal: failed to create PSO for ")
          + [name UTF8String] + ": "
          + [[err localizedDescription] UTF8String]);
    }
    return pso;
  }

  void init() {
    if (initialised) return;

    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("tmol Metal: no Metal device found.");
    }
    queue = [device newCommandQueue];

    // Locate tmol_primitives.metallib alongside the _C extension module.
    // __FILE__ is metal_context.mm; the .metallib is placed next to _C.so
    // by CMake's install() rule, so we search the same directory.
    NSString* mm_path   = @(__FILE__);
    NSString* src_dir   = [mm_path stringByDeletingLastPathComponent];
    // Walk up from tmol/score/common → tmol/ (where _C.so lives)
    NSString* tmol_dir  = [[src_dir
        stringByDeletingLastPathComponent]
        stringByDeletingLastPathComponent];
    NSString* lib_path  = [tmol_dir
        stringByAppendingPathComponent:@"tmol_primitives.metallib"];

    NSError* err = nil;
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:lib_path]
                                  error:&err];
    if (!library) {
      throw std::runtime_error(
          std::string("tmol Metal: cannot load metallib at ")
          + [lib_path UTF8String] + ": "
          + [[err localizedDescription] UTF8String]);
    }

    pso_scan_local   = make_pso(@"tmol_scan_local_float");
    pso_scan_carry   = make_pso(@"tmol_scan_carry_float");
    pso_reduce_part  = make_pso(@"tmol_reduce_partial_float");
    pso_reduce_final = make_pso(@"tmol_reduce_final_float");
    pso_segscan_up   = make_pso(@"tmol_segscan_upsweep_float");
    pso_segscan_down = make_pso(@"tmol_segscan_downsweep_float");

    initialised = true;
  }
};

static MtlContext g_ctx;
static std::once_flag g_init_flag;

MtlContext& ctx() {
  std::call_once(g_init_flag, []{ g_ctx.init(); });
  return g_ctx;
}

// Wrap a raw CPU/MPS pointer in a no-copy MTLBuffer (unified memory).
id<MTLBuffer> wrap_ptr(id<MTLDevice> dev, void* ptr, size_t bytes) {
  return [dev newBufferWithBytesNoCopy:ptr
                                length:bytes
                               options:MTLResourceStorageModeShared
                           deallocator:nil];
}

void commit_and_wait(id<MTLCommandBuffer> cmd) {
  [cmd commit];
  [cmd waitUntilCompleted];
}

constexpr uint32_t kThreadsPerGroup = 256;

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public C++ API
// ---------------------------------------------------------------------------

void initialize() {
  (void)ctx();  // triggers call_once
}

void synchronize() {
  // Flush pending Metal work via an empty command buffer.
  id<MTLCommandBuffer> cmd = [ctx().queue commandBuffer];
  commit_and_wait(cmd);
}

// ── scan_inclusive_float ────────────────────────────────────────────────────

void scan_inclusive_float(float* src, float* dst, int n) {
  auto& c = ctx();
  if (n <= 0) return;

  size_t bytes   = (size_t)n * sizeof(float);
  uint32_t N     = (uint32_t)n;
  uint32_t tpg   = kThreadsPerGroup;
  uint32_t n_tg  = (N + tpg - 1) / tpg;

  id<MTLBuffer> buf_data  = wrap_ptr(c.device, src, bytes);
  id<MTLBuffer> buf_carry = [c.device newBufferWithLength:n_tg * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  id<MTLBuffer> buf_N     = wrap_ptr(c.device, &N, sizeof(uint32_t));

  // Pass 1: local scan + carry-out
  {
    id<MTLCommandBuffer>        cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso_scan_local];
    [enc setBuffer:buf_data  offset:0 atIndex:0];
    [enc setBuffer:buf_carry offset:0 atIndex:1];
    [enc setBuffer:buf_N     offset:0 atIndex:2];
    // threadgroup memory: tpg floats
    [enc setThreadgroupMemoryLength:tpg * sizeof(float) atIndex:0];
    MTLSize grid = {n_tg, 1, 1};
    MTLSize grp  = {tpg,  1, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:grp];
    [enc endEncoding];
    commit_and_wait(cmd);
  }

  // If only one threadgroup, we're done (dst == src in-place case handled
  // because buf_data wraps src directly).
  if (n_tg == 1) {
    if (dst != src) memcpy(dst, src, bytes);
    return;
  }

  // Exclusive-scan the carry array on CPU (it's small — at most n/256 elems)
  float* carry = (float*)[buf_carry contents];
  float running = 0.0f;
  for (uint32_t i = 0; i < n_tg; ++i) {
    float c_val = carry[i];
    carry[i]    = running;
    running    += c_val;
  }

  // Pass 2: add carry-in to each tile
  {
    id<MTLCommandBuffer>        cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso_scan_carry];
    [enc setBuffer:buf_data  offset:0 atIndex:0];
    [enc setBuffer:buf_carry offset:0 atIndex:1];
    [enc setBuffer:buf_N     offset:0 atIndex:2];
    MTLSize grid = {n_tg, 1, 1};
    MTLSize grp  = {tpg,  1, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:grp];
    [enc endEncoding];
    commit_and_wait(cmd);
  }

  if (dst != src) memcpy(dst, src, bytes);
}

// ── reduce_sum_float ─────────────────────────────────────────────────────────

float reduce_sum_float(const float* src, int n) {
  auto& c = ctx();
  if (n <= 0) return 0.0f;

  uint32_t N    = (uint32_t)n;
  uint32_t tpg  = kThreadsPerGroup;
  uint32_t n_tg = (N + tpg - 1) / tpg;

  id<MTLBuffer> buf_data  = wrap_ptr(c.device, (void*)src, N * sizeof(float));
  id<MTLBuffer> buf_parts = [c.device newBufferWithLength:n_tg * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  id<MTLBuffer> buf_N     = wrap_ptr(c.device, &N, sizeof(uint32_t));

  // Pass 1: partial reductions
  {
    id<MTLCommandBuffer>        cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso_reduce_part];
    [enc setBuffer:buf_data  offset:0 atIndex:0];
    [enc setBuffer:buf_parts offset:0 atIndex:1];
    [enc setBuffer:buf_N     offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:tpg * sizeof(float) atIndex:0];
    MTLSize grid = {n_tg, 1, 1};
    MTLSize grp  = {tpg,  1, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:grp];
    [enc endEncoding];
    commit_and_wait(cmd);
  }

  // Pass 2: reduce partials → result (single threadgroup, CPU-side if small)
  if (n_tg == 1) {
    return ((float*)[buf_parts contents])[0];
  }

  id<MTLBuffer> buf_result = [c.device newBufferWithLength:sizeof(float)
                                                    options:MTLResourceStorageModeShared];
  uint32_t n_parts = n_tg;
  id<MTLBuffer> buf_nparts = wrap_ptr(c.device, &n_parts, sizeof(uint32_t));

  {
    id<MTLCommandBuffer>        cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso_reduce_final];
    [enc setBuffer:buf_parts  offset:0 atIndex:0];
    [enc setBuffer:buf_result offset:0 atIndex:1];
    [enc setBuffer:buf_nparts offset:0 atIndex:2];
    [enc setThreadgroupMemoryLength:tpg * sizeof(float) atIndex:0];
    MTLSize grid = {1,   1, 1};
    MTLSize grp  = {tpg, 1, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:grp];
    [enc endEncoding];
    commit_and_wait(cmd);
  }

  return ((float*)[buf_result contents])[0];
}

// ── segmented_scan_inclusive_float ───────────────────────────────────────────

void segmented_scan_inclusive_float(
    const float* src,
    float* dst,
    const int* seg_start_inds,
    int n,
    int n_segs)
{
  auto& c = ctx();
  if (n <= 0) return;

  // Copy src → dst so the upsweep kernel can work in-place on dst.
  if (dst != src) memcpy(dst, src, (size_t)n * sizeof(float));

  uint32_t N     = (uint32_t)n;
  uint32_t NS    = (uint32_t)n_segs;
  uint32_t tpg   = kThreadsPerGroup;
  uint32_t n_tg  = (N + tpg - 1) / tpg;

  id<MTLBuffer> buf_vals   = wrap_ptr(c.device, dst, N * sizeof(float));
  id<MTLBuffer> buf_segs   = wrap_ptr(c.device, (void*)seg_start_inds,
                                      NS * sizeof(int));
  id<MTLBuffer> buf_carry  = [c.device newBufferWithLength:n_tg * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
  id<MTLBuffer> buf_flags  = [c.device newBufferWithLength:n_tg * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
  id<MTLBuffer> buf_N      = wrap_ptr(c.device, &N,  sizeof(uint32_t));
  id<MTLBuffer> buf_NS     = wrap_ptr(c.device, &NS, sizeof(uint32_t));

  // Pass 1: intra-threadgroup segmented scan (upsweep)
  {
    id<MTLCommandBuffer>        cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso_segscan_up];
    [enc setBuffer:buf_vals  offset:0 atIndex:0];
    [enc setBuffer:buf_segs  offset:0 atIndex:1];
    [enc setBuffer:buf_carry offset:0 atIndex:2];
    [enc setBuffer:buf_flags offset:0 atIndex:3];
    [enc setBuffer:buf_N     offset:0 atIndex:4];
    [enc setBuffer:buf_NS    offset:0 atIndex:5];
    MTLSize grid = {n_tg, 1, 1};
    MTLSize grp  = {tpg,  1, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:grp];
    [enc endEncoding];
    commit_and_wait(cmd);
  }

  if (n_tg == 1) return;  // single tile — no cross-tile fixup needed

  // CPU-side: compute exclusive carry-in for each threadgroup, respecting
  // segment boundaries (zero out carry when a segment starts in the tile).
  float*    carry = (float*)   [buf_carry contents];
  uint32_t* flags = (uint32_t*)[buf_flags contents];

  float running = carry[0];   // tg 0 carry is the "output" of the first tile
  carry[0]      = 0.0f;       // tg 0 has no carry-in
  for (uint32_t i = 1; i < n_tg; ++i) {
    float c_val = carry[i];
    carry[i]    = flags[i] ? 0.0f : running;  // reset at segment boundary
    running     = flags[i] ? c_val : (running + c_val);
  }

  // Pass 2: add inter-threadgroup carry-in (downsweep)
  {
    id<MTLCommandBuffer>        cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso_segscan_down];
    [enc setBuffer:buf_vals  offset:0 atIndex:0];
    [enc setBuffer:buf_carry offset:0 atIndex:1];
    [enc setBuffer:buf_segs  offset:0 atIndex:2];
    [enc setBuffer:buf_N     offset:0 atIndex:3];
    [enc setBuffer:buf_NS    offset:0 atIndex:4];
    MTLSize grid = {n_tg, 1, 1};
    MTLSize grp  = {tpg,  1, 1};
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:grp];
    [enc endEncoding];
    commit_and_wait(cmd);
  }
}

}  // namespace mps
}  // namespace tmol

#endif  // WITH_MPS
