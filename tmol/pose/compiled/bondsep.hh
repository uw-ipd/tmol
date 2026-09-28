#pragma once

#include <torch/torch.h>
#include <c10/macros/Macros.h>
#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol::pose {

// The output is int8: separations saturate at the sentinel, which is the cap.
C10_HOST_DEVICE inline int8_t saturated_bondsep(
    int32_t distance, int32_t sentinel) {
  return int8_t(distance < sentinel ? distance : sentinel);
}

// Decode the final contiguous layout without materializing broadcast indices.
C10_HOST_DEVICE inline int8_t block_bondsep_value(
    int64_t index,
    int64_t blocks,
    int64_t ports,
    int64_t nodes,
    int32_t const* distances,
    int64_t const* offsets,
    int32_t const* counts,
    int32_t sentinel) {
  auto const second_port = index % ports;
  index /= ports;
  auto const first_port = index % ports;
  index /= ports;
  auto const second = index % blocks;
  index /= blocks;
  auto const first = index % blocks;
  auto const pose = index / blocks;
  auto const a = pose * blocks + first;
  auto const b = pose * blocks + second;
  if (first_port >= counts[a] || second_port >= counts[b])
    return int8_t(sentinel);
  auto const first_node = offsets[a] + first_port;
  auto const second_node = offsets[b] + second_port;
  if (first_node < 0 || first_node >= nodes || second_node < 0
      || second_node >= nodes)
    return int8_t(sentinel);
  return saturated_bondsep(
      distances[(pose * nodes + first_node) * nodes + second_node], sentinel);
}

template <Device D>
void gather_block_bondsep(
    torch::Tensor distances,
    torch::Tensor offsets,
    torch::Tensor counts,
    torch::Tensor output,
    int32_t sentinel);

}  // namespace tmol::pose
