#include <tmol/score/common/mps_launch.hh>

#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/ops.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tmol {
namespace score {
namespace common {

namespace {

void check(AOTITorchError err, char const* what) {
  if (err != AOTI_TORCH_SUCCESS) {
    throw std::runtime_error(std::string("tmol MPS: ") + what + " failed");
  }
}

// One library per shader source and one function handle per entry point.
// Compiling a library is the expensive part -- it runs the Metal front end
// over the whole flattened kernel source -- so it happens once, lazily, on
// the first call that needs it rather than at module load.
//
// Libraries are keyed by the address of the generated source constant, which
// is a distinct object per shader and lives for the life of the process.
struct ShaderCache {
  std::mutex mutex;
  std::unordered_map<char const*, AOTIMetalShaderLibraryHandle> libraries;
  std::unordered_map<std::string, AOTIMetalKernelFunctionHandle> functions;

  AOTIMetalKernelFunctionHandle get(char const* source, char const* name) {
    std::lock_guard<std::mutex> guard(mutex);

    auto library = libraries.find(source);
    if (library == libraries.end()) {
      AOTIMetalShaderLibraryHandle handle = nullptr;
      check(
          aoti_torch_mps_create_shader_library(source, &handle),
          "shader library compilation");
      library = libraries.emplace(source, handle).first;
    }

    // Entry-point names are unique across tmol's shaders, but the key folds
    // in the library so a collision would be a lookup miss rather than a
    // silently wrong kernel.
    std::string key =
        std::to_string(reinterpret_cast<uintptr_t>(source)) + ":" + name;
    auto found = functions.find(key);
    if (found != functions.end()) {
      return found->second;
    }

    AOTIMetalKernelFunctionHandle function = nullptr;
    check(
        aoti_torch_mps_get_kernel_function(library->second, name, &function),
        "kernel function lookup");
    functions.emplace(std::move(key), function);
    return function;
  }
};

ShaderCache& shader_cache() {
  static ShaderCache cache;
  return cache;
}

// The metadata buffer, keyed on its own contents.
//
// Every launch needs the per-view shapes and the launch scalars on the device,
// but they only change when the pose stack's shape changes -- not per call. A
// relax trajectory re-launches the same kernels over the same shapes thousands
// of times, and without this each one pays a host allocation, a memcpy and a
// host-to-device copy.
//
// Keyed on contents rather than on the caller, because the scalars travel in
// the same buffer: two calls with identical shapes but a different
// output_block_pair_energies flag must not share an entry. The cache is small
// and cleared wholesale when it grows, which is correct (the next call rebuilds
// what it needs) and bounds the device memory it holds.
struct MetaCache {
  std::mutex mutex;
  std::unordered_map<std::string, tmol::native::Tensor> entries;

  // Beyond a handful of distinct shapes the hit rate is not worth the device
  // memory; the terms in one trajectory use a few each.
  static constexpr size_t max_entries = 64;

  tmol::native::Tensor get(std::vector<int64_t> const& meta) {
    std::string key(
        reinterpret_cast<char const*>(meta.data()),
        meta.size() * sizeof(int64_t));

    std::lock_guard<std::mutex> guard(mutex);
    auto found = entries.find(key);
    if (found != entries.end()) {
      return found->second;
    }

    auto meta_cpu = tmol::native::empty(
        {int64_t(meta.size())},
        tmol::native::TensorOptions()
            .dtype(tmol::native::ScalarType::Long)
            .device(tmol::native::kCPU));
    std::memcpy(
        meta_cpu.data_ptr(), meta.data(), meta.size() * sizeof(int64_t));
    tmol::native::Tensor meta_mps(torch::stable::to(
        meta_cpu.handle(), torch::stable::Device(tmol::native::kMPS)));

    if (entries.size() >= max_entries) {
      entries.clear();
    }
    entries.emplace(std::move(key), meta_mps);
    return meta_mps;
  }
};

MetaCache& meta_cache() {
  static MetaCache cache;
  return cache;
}

// Opt-in per-kernel timing, for attributing where a scoring call's GPU time
// goes. Off unless TMOL_MPS_TIME_KERNELS is set, and then it costs a device
// synchronize per launch -- which perturbs the total, since the point of the
// backend is that launches do not synchronize. Read it as a breakdown of
// relative cost, not as a wall time.
struct KernelTimer {
  std::mutex mutex;
  std::map<std::string, double> millis;
  std::map<std::string, long> calls;

  static bool enabled() {
    static bool on = std::getenv("TMOL_MPS_TIME_KERNELS") != nullptr;
    return on;
  }

  // Host-phase attribution is a separate switch. Per-kernel timing waits on
  // every launch, which drains the queue and so makes any later host-side
  // wait look free -- exactly the cost a host phase is being measured for.
  // TMOL_MPS_TIME_HOST records host phases while leaving launches async.
  static bool host_enabled() {
    static bool on = std::getenv("TMOL_MPS_TIME_HOST") != nullptr
                     || std::getenv("TMOL_MPS_TIME_KERNELS") != nullptr;
    return on;
  }

  void record(char const* name, double ms) {
    std::lock_guard<std::mutex> guard(mutex);
    millis[name] += ms;
    calls[name] += 1;
  }

  ~KernelTimer() {
    if (millis.empty()) return;
    double total = 0;
    for (auto const& entry : millis) total += entry.second;
    std::fprintf(stderr, "\ntmol MPS kernel time (synchronized per launch)\n");
    for (auto const& entry : millis) {
      std::fprintf(
          stderr,
          "  %-44s %9.1f ms  %6ld calls  %5.1f%%\n",
          entry.first.c_str(),
          entry.second,
          calls[entry.first],
          100.0 * entry.second / total);
    }
    std::fprintf(stderr, "  %-44s %9.1f ms\n", "TOTAL", total);
  }
};

KernelTimer& kernel_timer() {
  static KernelTimer timer;
  return timer;
}

}  // namespace

void mps_launch(
    char const* shader_source,
    char const* kernel_name,
    MpsArgs const& args,
    int64_t n_items,
    int64_t threadgroup_size) {
  auto const n_views = args.buffers.size();

  // The metadata has to reach the GPU as a buffer, and the only buffers this
  // API binds are tensors, so it travels as one. Cached on its contents: the
  // shapes and launch scalars are the same on every call for a given pose
  // stack. See MetaCache.
  tmol::native::Tensor meta_mps = meta_cache().get(args.meta);

  // Each view is wrapped as a byte tensor over the same allocation. The
  // dispatch layer only ever sees views -- the tensors they came from are the
  // operator layer's business -- and binding needs a tensor, so one is made
  // here. It aliases rather than copies: an MPS allocation's pointer is the
  // MTLBuffer the shim binds, and from_blob keeps it.
  std::vector<tmol::native::Tensor> wrappers;
  wrappers.reserve(n_views);
  auto const mps = torch::stable::Device(tmol::native::kMPS);
  for (size_t i = 0; i < n_views; ++i) {
    auto const& buffer = args.buffers[i];

    // A context is the union of what its family of kernels reads, so any one
    // kernel leaves some views unset -- the whole-pose forward never touches
    // dispatch_indices or dTdV, for instance. Buffer slots are positional, so
    // the slot still has to be filled for the ones after it to line up. The
    // metadata buffer stands in: it is a live allocation of known size that
    // the kernel provably does not read through this view, because the view
    // it belongs to is one the kernel never names.
    // Metal binds a buffer at a 16-byte-aligned offset and faults deep in
    // the shim otherwise. Everything tmol allocates satisfies this; a view of
    // somebody else's tensor need not. Say so here rather than crashing.
    if (buffer.data != nullptr
        && reinterpret_cast<uintptr_t>(buffer.data) % 16 != 0) {
      throw std::runtime_error(
          std::string("tmol: view ") + std::to_string(i) + " of Metal kernel "
          + kernel_name
          + " is not 16-byte aligned; copy it with "
            "tmol::native::realigned_copy before dispatching");
    }
    bool const unset = buffer.data == nullptr;
    int64_t const size = unset ? 1 : buffer.n_bytes;
    int64_t const stride = 1;
    if (unset) {
      wrappers.emplace_back(meta_mps);
      continue;
    }
    wrappers.emplace_back(torch::stable::from_blob(
        buffer.data,
        torch::headeronly::IntHeaderOnlyArrayRef(&size, 1),
        torch::headeronly::IntHeaderOnlyArrayRef(&stride, 1),
        mps,
        torch::headeronly::ScalarType::Byte));
  }

  auto function = shader_cache().get(shader_source, kernel_name);

  // Encoding is not thread-safe. start_encoding, the argument binds and the
  // dispatch all act on one command encoder owned by torch's MPS stream, so
  // two threads interleaving here corrupt it -- which is what happens as soon
  // as a backward graph has two Metal terms in it, because the autograd
  // engine runs independent nodes concurrently.
  static std::mutex encode_mutex;
  std::lock_guard<std::mutex> encode_guard(encode_mutex);

  check(aoti_torch_mps_start_encoding(function), "start encoding");

  check(
      aoti_torch_mps_set_arg_tensor(
          function, TMOL_META_BUFFER_INDEX, meta_mps.handle().get()),
      "binding metadata");
  for (size_t i = 0; i < n_views; ++i) {
    check(
        aoti_torch_mps_set_arg_tensor(
            function,
            unsigned(TMOL_FIRST_VIEW_BUFFER_INDEX + i),
            wrappers[i].handle().get()),
        "binding a view");
  }

  uint64_t const total = uint64_t(n_items);
  uint64_t const group = uint64_t(threadgroup_size);
  bool const timing = KernelTimer::enabled();
  auto const start = std::chrono::steady_clock::now();
  check(
      aoti_torch_mps_dispatch_array_with_group_size(
          function, &total, 1, &group, 1),
      "dispatch");
  if (timing) {
    // Dispatch only enqueues, so the kernel has to be waited on to be timed.
    tmol::native::Tensor probe(torch::stable::to(
        meta_mps.handle(), torch::stable::Device(tmol::native::kCPU)));
    (void)probe;
    std::chrono::duration<double, std::milli> const elapsed =
        std::chrono::steady_clock::now() - start;
    kernel_timer().record(kernel_name, elapsed.count());
  }
}

bool mps_timing_enabled() { return KernelTimer::host_enabled(); }

void mps_record_host_phase(char const* name, double ms) {
  if (!KernelTimer::host_enabled()) return;
  kernel_timer().record(name, ms);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
