// Compile a minimal MSL TU that includes tmol's portable layer, through the
// runtime shader compiler. This is the first real test of whether the shared
// headers are Metal-compilable.
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
    // Optional second argument: pin the language version, to match whatever
    // the host runtime asks for.
    if (argc > 2) {
      int v = atoi(argv[2]);
      opt.languageVersion = (MTLLanguageVersion)((v / 10) << 16 | (v % 10));
      std::printf("language version %d.%d\n", v / 10, v % 10);
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:ss.str().c_str()]
                                           options:opt error:&err];
    if (!lib) {
      std::printf("%s\n", [[err localizedDescription] UTF8String]);
      // localizedDescription truncates the linker's symbol list; userInfo
      // carries the full diagnostic.
      std::printf("--- userInfo ---\n%s\n", [[[err userInfo] description] UTF8String]);
      return 1;
    }
    std::printf("MSL COMPILED OK\n");
    return 0;
  }
}
