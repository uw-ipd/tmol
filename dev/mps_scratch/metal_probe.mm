// Can we compile MSL at runtime without the offline Metal toolchain?
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

static const char *kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float *a [[buffer(0)]],
                 device const float *b [[buffer(1)]],
                 device float *out     [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
  out[i] = a[i] + b[i];
}
)";

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("no Metal device\n"); return 1; }
    std::printf("device: %s\n", [[dev name] UTF8String]);

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:nil error:&err];
    if (!lib) { std::printf("compile FAILED: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"vadd"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { std::printf("pipeline FAILED\n"); return 1; }
    std::printf("runtime MSL compile: OK  (maxThreadsPerThreadgroup=%lu, simdWidth=%lu)\n",
                (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                (unsigned long)pso.threadExecutionWidth);

    const int N = 1024;
    id<MTLBuffer> A = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> O = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
    float *a=(float*)A.contents,*b=(float*)B.contents;
    for (int i=0;i<N;++i){a[i]=i;b[i]=2*i;}
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:O offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    float *o=(float*)O.contents;
    std::printf("result check: out[10]=%.1f (expect 30.0)  %s\n", o[10],
                o[10]==30.0f ? "PASS" : "FAIL");
    return o[10]==30.0f ? 0 : 1;
  }
}
