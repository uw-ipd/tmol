#include <torch/extension.h>
#include <Eigen/Core>
#include <cmath>

#include <tmol/utility/datastructures/in_place_heap.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

auto create_in_place_heap(at::Tensor seq_tensor) {
  at::Tensor node_order_tensor;
  at::Tensor heap_order_tensor;
  at::Tensor values_tensor;

  auto seq = tmol::view_tensor<int32_t, 1, tmol::Device::CPU>(seq_tensor);
  int n_nodes = seq.size(0);

  auto node_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({n_nodes});
  auto node_order = node_order_t.view;

  auto heap_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({n_nodes});
  auto heap_order = heap_order_t.view;

  auto values_t = tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({n_nodes});
  auto values = values_t.view;

  tmol::InPlaceHeap<tmol::Device::CPU, int32_t> heap(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    heap.heap_insert(i, seq[i]);
  }
  for (int i = 0; i < n_nodes; ++i) {
    node_order[i] = heap.get_node_heap_ind(i);
    heap_order[i] = heap.get_heap_node_ind(i);
    values[i] = heap.get_heap_val(i);
  }

  node_order_tensor = node_order_t.tensor;
  heap_order_tensor = heap_order_t.tensor;
  values_tensor = values_t.tensor;

  return std::make_tuple(node_order_tensor, heap_order_tensor, values_tensor);
}

auto clear_heap_after_creation(at::Tensor seq_tensor1, at::Tensor seq_tensor2) {
  at::Tensor node_order_tensor;
  at::Tensor heap_order_tensor;
  at::Tensor values_tensor;

  auto seq1 = tmol::view_tensor<int32_t, 1, tmol::Device::CPU>(seq_tensor1);
  int const n_nodes1 = seq1.size(0);
  auto seq2 = tmol::view_tensor<int32_t, 1, tmol::Device::CPU>(seq_tensor2);
  int const n_nodes2 = seq2.size(0);
  assert(n_nodes2 <= n_nodes1);

  auto node_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({n_nodes2});
  auto node_order = node_order_t.view;

  auto heap_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({n_nodes2});
  auto heap_order = heap_order_t.view;

  auto values_t = tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({n_nodes2});
  auto values = values_t.view;

  tmol::InPlaceHeap<tmol::Device::CPU, int32_t> heap(n_nodes1);
  for (int i = 0; i < n_nodes1; ++i) {
    heap.heap_insert(i, seq1[i]);
  }
  heap.clear();
  for (int i = 0; i < n_nodes2; ++i) {
    heap.heap_insert(i, seq2[i]);
  }

  for (int i = 0; i < n_nodes2; ++i) {
    node_order[i] = heap.get_node_heap_ind(i);
    heap_order[i] = heap.get_heap_node_ind(i);
    values[i] = heap.get_heap_val(i);
  }

  node_order_tensor = node_order_t.tensor;
  heap_order_tensor = heap_order_t.tensor;
  values_tensor = values_t.tensor;

  return std::make_tuple(node_order_tensor, heap_order_tensor, values_tensor);
}

auto create_heap_with_gaps(int max_n_nodes, at::Tensor seq_tensor) {
  at::Tensor node_order_tensor;
  at::Tensor heap_order_tensor;
  at::Tensor values_tensor;

  auto seq = tmol::view_tensor<int32_t, 2, tmol::Device::CPU>(seq_tensor);
  int const n_insertions = seq.size(0);

  auto node_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({max_n_nodes});
  auto node_order = node_order_t.view;

  auto heap_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({max_n_nodes});
  auto heap_order = heap_order_t.view;

  auto values_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({max_n_nodes});
  auto values = values_t.view;

  tmol::InPlaceHeap<tmol::Device::CPU, int32_t> heap(max_n_nodes);
  for (int i = 0; i < n_insertions; ++i) {
    heap.heap_insert(seq[i][0], seq[i][1]);
  }

  for (int i = 0; i < max_n_nodes; ++i) {
    node_order[i] = heap.get_node_heap_ind(i);
    heap_order[i] = heap.get_heap_node_ind(i);
    values[i] = heap.get_heap_val(i);
  }

  node_order_tensor = node_order_t.tensor;
  heap_order_tensor = heap_order_t.tensor;
  values_tensor = values_t.tensor;

  return std::make_tuple(node_order_tensor, heap_order_tensor, values_tensor);
}

auto create_heap_with_gaps_clear_and_recreate(
    int max_n_nodes, at::Tensor seq_tensor1, at::Tensor seq_tensor2) {
  at::Tensor node_order_tensor;
  at::Tensor heap_order_tensor;
  at::Tensor values_tensor;

  auto seq1 = tmol::view_tensor<int32_t, 2, tmol::Device::CPU>(seq_tensor1);
  int const n_insertions1 = seq1.size(0);
  auto seq2 = tmol::view_tensor<int32_t, 2, tmol::Device::CPU>(seq_tensor2);
  int const n_insertions2 = seq2.size(0);

  auto node_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({max_n_nodes});
  auto node_order = node_order_t.view;

  auto heap_order_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({max_n_nodes});
  auto heap_order = heap_order_t.view;

  auto values_t =
      tmol::TPack<int32_t, 1, tmol::Device::CPU>::zeros({max_n_nodes});
  auto values = values_t.view;

  tmol::InPlaceHeap<tmol::Device::CPU, int32_t> heap(max_n_nodes);
  for (int i = 0; i < n_insertions1; ++i) {
    heap.heap_insert(seq1[i][0], seq1[i][1]);
  }
  heap.clear();
  for (int i = 0; i < n_insertions2; ++i) {
    heap.heap_insert(seq2[i][0], seq2[i][1]);
  }

  for (int i = 0; i < max_n_nodes; ++i) {
    node_order[i] = heap.get_node_heap_ind(i);
    heap_order[i] = heap.get_heap_node_ind(i);
    values[i] = heap.get_heap_val(i);
  }

  node_order_tensor = node_order_t.tensor;
  heap_order_tensor = heap_order_t.tensor;
  values_tensor = values_t.tensor;

  return std::make_tuple(node_order_tensor, heap_order_tensor, values_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "create_in_place_heap",
      &create_in_place_heap,
      "create heap with values for nodes given and return heap order");
  m.def(
      "clear_heap_after_creation",
      &clear_heap_after_creation,
      "create heap with values for nodes given and then clear it and fill it "
      "again");
  m.def(
      "create_heap_with_gaps",
      &create_heap_with_gaps,
      "create heap with node/value pairs given");
  m.def(
      "create_heap_with_gaps_clear_and_recreate",
      &create_heap_with_gaps_clear_and_recreate,
      "create a heap with node/value pairs, clear it, then re-populate the "
      "heap with a second set of node/value pairs");
}
