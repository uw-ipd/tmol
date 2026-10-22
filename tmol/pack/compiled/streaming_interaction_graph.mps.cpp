// Instantiation of the streaming interaction-graph builder for the Metal
// backend.
//
// Only note, note_chunks and accumulate -- the passes over table entries --
// are reached with MPS tensors; they launch the Metal entry points in
// streaming_interaction_graph_mps_kernels.metal.in. The topology passes
// (initialize, finalize, finalize_chunks) work on small per-block arrays and
// run on the host, so the op layer does not dispatch them to MPS. Only float:
// Metal has no double.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
#include <tmol/pack/compiled/streaming_interaction_graph.impl.hh>

namespace tmol {
namespace pack {
namespace compiled {

template struct StreamingInteractionGraph<
    score::common::DeviceOperations,
    tmol::Device::MPS,
    float,
    int64_t>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
