// Per-entry-point occupancy of a compiled MSL translation unit.
//
// maxTotalThreadsPerThreadgroup is the compiler telling you how many threads
// it can fit given the registers the kernel needs. A kernel launched at 32
// threads is fine as long as this is >= 32, but the number still ranks
// kernels by register pressure -- and a low one means the compiler is close
// to, or already is, spilling.
//
// usage: msl_occupancy <file.metal> [language_version] [entry ...]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  @autoreleasepool {
    std::ifstream in(argv[1]);
    std::stringstream ss; ss << in.rdbuf();
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (argc > 2) {
      int v = atoi(argv[2]);
      opt.languageVersion = (MTLLanguageVersion)((v / 10) << 16 | (v % 10));
    }
    NSError *err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithSource:[NSString stringWithUTF8String:ss.str().c_str()]
                          options:opt error:&err];
    if (!lib) {
      std::printf("%s\n", [[err localizedDescription] UTF8String]);
      return 1;
    }
    NSArray<NSString *> *names = [lib functionNames];
    std::printf("%-46s %8s %8s %10s\n", "entry", "maxthr", "simdw", "tgmem");
    for (NSString *name in names) {
      id<MTLFunction> fn = [lib newFunctionWithName:name];
      if (!fn) continue;
      NSError *perr = nil;
      id<MTLComputePipelineState> pso =
          [dev newComputePipelineStateWithFunction:fn error:&perr];
      if (!pso) {
        std::printf("%-46s  pipeline failed: %s\n", [name UTF8String],
                    [[perr localizedDescription] UTF8String]);
        continue;
      }
      std::printf("%-46s %8lu %8lu %10lu\n", [name UTF8String],
                  (unsigned long)[pso maxTotalThreadsPerThreadgroup],
                  (unsigned long)[pso threadExecutionWidth],
                  (unsigned long)[pso staticThreadgroupMemoryLength]);
    }
    return 0;
  }
}
