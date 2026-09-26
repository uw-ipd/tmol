#include "bondsep.hh"
#include <ATen/Parallel.h>

namespace tmol::pose {

template <>
void gather_block_bondsep<Device::CPU>(
    torch::Tensor distances,
    torch::Tensor offsets,
    torch::Tensor counts,
    torch::Tensor output,
    int32_t sentinel) {
  auto const blocks = counts.size(1), ports = output.size(3);
  auto const nodes = distances.size(1);
  auto dst = output.data_ptr<int32_t>();
  auto src = distances.data_ptr<int32_t>();
  auto off = offsets.data_ptr<int64_t>();
  auto num = counts.data_ptr<int32_t>();
  if (!ports) return;
  at::parallel_for(
      0,
      counts.size(0) * blocks * blocks,
      4096,
      [=](int64_t begin, int64_t end) {
        for (auto pair = begin; pair < end; ++pair) {
          auto const a = pair / blocks;
          auto const pose = a / blocks;
          auto const b = pose * blocks + pair % blocks;
          auto target = dst + pair * ports * ports;
          std::fill_n(target, ports * ports, sentinel);
          for (int64_t i = 0; i < std::min<int64_t>(num[a], ports); ++i) {
            auto const first = off[a] + i;
            if (first < 0 || first >= nodes || off[b] < 0) continue;
            auto const length = std::min<int64_t>(
                std::min<int64_t>(num[b], ports), nodes - off[b]);
            if (length > 0)
              std::copy_n(
                  src + (pose * nodes + first) * nodes + off[b],
                  length,
                  target + i * ports);
          }
        }
      });
}

}  // namespace tmol::pose
