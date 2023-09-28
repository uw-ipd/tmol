#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <iostream>

namespace tmol {

template <tmol::Device D, typename Int>
class InPlaceHeap {
 public:
  AT_HOST_DEVICE
  InPlaceHeap(Int max_n_nodes)
      : capacity_(max_n_nodes),
        size_(0),
        heap_t_(TPack<Int, 2, D>::full({max_n_nodes, 2}, -1)),
        heap_(heap_t_.view),
        node_heap_ind_t_(TPack<Int, 1, D>::full({max_n_nodes}, -1)),
        node_heap_ind_(node_heap_ind_t_.view) {}

  // get the value for the smallest element of the heap
  AT_HOST_DEVICE
  Int peek_val() const {
    if (size_ > 0) {
      return heap_[0][0];
    } else {
      return -1;
    }
  }

  // get the vertex index for the smallest element of the heap
  AT_HOST_DEVICE
  Int peek_ind() const {
    if (size_ > 0) {
      return heap_[0][1];
    } else {
      return -1;
    }
  }

  // remove the smallest-valued element of the heap
  AT_HOST_DEVICE
  void pop() {
    if (size_ == 0) {
      return;
    } else if (size_ == 1) {
      node_heap_ind_[heap_[0][1]] = -1;
      size_ = 0;
      return;
    }
    --size_;
    node_heap_ind_[heap_[0][1]] = -1;

    heap_[0][0] = heap_[size_][0];
    heap_[0][1] = heap_[size_][1];

    node_heap_ind_[heap_[0][1]] = 0;
    heapify(0);
  }

  AT_HOST_DEVICE
  void heap_insert(Int node_ind, Int val) {
    assert(size_ < capacity_);
    assert(node_heap_ind_[node_ind] == -1);
    ++size_;
    Int ind = size_ - 1;
    node_heap_ind_[node_ind] = ind;
    heap_[ind][0] = val;
    heap_[ind][1] = node_ind;
    bubble_up(ind);
  }

  AT_HOST_DEVICE
  void decrease_node_val(Int node_ind, Int new_val) {
    assert(node_heap_ind_[node_ind] >= 0);
    assert(heap_[node_heap_ind_[node_ind]][0] >= new_val);
    Int heap_ind = node_heap_ind_[node_ind];
    heap_[heap_ind][0] = new_val;
    bubble_up(heap_ind);
  }

  AT_HOST_DEVICE
  Int get_node_val(Int node_ind) {
    assert(node_heap_ind_[node_ind] >= 0);
    Int heap_ind = node_heap_ind_[node_ind];
    return heap_[heap_ind][0];
  }

  AT_HOST_DEVICE
  Int get_node_heap_ind(Int node_ind) { return node_heap_ind_[node_ind]; }

  AT_HOST_DEVICE
  Int get_heap_node_ind(Int heap_ind) { return heap_[heap_ind][1]; }

  AT_HOST_DEVICE
  Int get_heap_val(Int heap_ind) { return heap_[heap_ind][0]; }

  AT_HOST_DEVICE
  void clear() {
    // O(k) expense of removing all entries from the heap
    // essential when we want to repeatedly use the same
    // allocation for many heap usages
    for (Int i = 0; i < size_; ++i) {
      Int i_node = heap_[i][1];
      node_heap_ind_[i_node] = -1;
      heap_[i][0] = -1;
      heap_[i][1] = -1;
    }
    size_ = 0;
  }

  AT_HOST_DEVICE
  bool node_in_heap(Int node_ind) const {
    return node_heap_ind_[node_ind] != -1;
  }

 private:
  AT_HOST_DEVICE
  void heapify(Int ind) {
    // assumption that everything beneath ind is a heap
    Int l = left(ind);
    Int r = right(ind);
    Int smallest = ind;
    if (l < size_ && heap_[l][0] < heap_[ind][0]) {
      smallest = l;
    }
    if (r < size_ && heap_[r][0] < heap_[smallest][0]) {
      smallest = r;
    }
    if (smallest != ind) {
      hswap(ind, smallest);
      heapify(smallest);
    }
  }

  AT_HOST_DEVICE
  void bubble_up(Int ind) {
    // restore the heap property after the given heap
    // element's value has been set to a something
    // possibly smaller than its parent's value

    Int ind_val = heap_[ind][0];
    Int ind_parent = parent(ind);
    while (ind != 0 && heap_[ind_parent][0] > ind_val) {
      hswap(ind, ind_parent);
      ind = ind_parent;
      // ind_val doesn't have to be updated because ind_val
      // does not change!
      ind_parent = parent(ind);
    }
  }

  AT_HOST_DEVICE
  Int parent(Int ind) const { return (ind - 1) / 2; }

  AT_HOST_DEVICE
  Int left(Int ind) const { return 2 * ind + 1; }

  AT_HOST_DEVICE
  Int right(Int ind) const { return 2 * ind + 2; }

  AT_HOST_DEVICE
  void hswap(Int ind1, Int ind2) {
    // swap the position of the two nodes in the heap
    Int n_ind1 = heap_[ind1][1];
    Int n_ind2 = heap_[ind2][1];
    Int v1 = heap_[ind1][0];
    Int v2 = heap_[ind2][0];
    node_heap_ind_[n_ind1] = ind2;
    node_heap_ind_[n_ind2] = ind1;
    heap_[ind1][0] = v2;
    heap_[ind1][1] = n_ind2;
    heap_[ind2][0] = v1;
    heap_[ind2][1] = n_ind1;
  }

 private:
  Int const capacity_;
  Int size_;
  TPack<Int, 2, D> heap_t_;
  TView<Int, 2, D> heap_;
  TPack<Int, 1, D> node_heap_ind_t_;
  TView<Int, 1, D> node_heap_ind_;
};

}  // namespace tmol
