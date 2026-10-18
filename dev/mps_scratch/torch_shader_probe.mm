// Ask torch's stable MPS shim to compile two shaders -- one plain, one using
// a lambda -- to find out which language features its compile options allow.
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

extern "C" {
struct AOTIMetalShaderLibraryOpaque;
using Lib = AOTIMetalShaderLibraryOpaque*;
using CreateFn = int (*)(const char*, Lib*);
}

int main(int argc, char** argv) {
  void* h = dlopen(argv[1], RTLD_NOW);
  if (!h) { std::printf("dlopen failed: %s\n", dlerror()); return 1; }
  auto create = (CreateFn)dlsym(h, "aoti_torch_mps_create_shader_library");
  if (!create) { std::printf("symbol not found\n"); return 1; }

  char const* plain =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }\n";
  char const* lambda =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void k(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
      "  auto f = [&](uint j) { o[j] = 1.0f; };\n  f(i);\n}\n";

  Lib lib = nullptr;
  if (argc > 2) {
    // A file of real shader source, to find out whether torch accepts it.
    FILE* f = fopen(argv[2], "rb");
    if (!f) { std::printf("cannot open %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { std::printf("short read\n"); return 1; }
    buf[n] = 0; fclose(f);
    int rc = create(buf, &lib);
    std::printf("%s : %s (rc=%d, %ld bytes)\n", argv[2], rc == 0 ? "OK" : "FAILED", rc, n);
    return rc != 0;
  }
  std::printf("plain  : %s\n", create(plain, &lib) == 0 ? "OK" : "FAILED");
  lib = nullptr;
  std::printf("lambda : %s\n", create(lambda, &lib) == 0 ? "OK" : "FAILED");
  return 0;
}
