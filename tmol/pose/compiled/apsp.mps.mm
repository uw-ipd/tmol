// apsp.mps.mm — MPS instantiation of All-Pairs Shortest Paths dispatch.
//
// AllPairsShortestPathsDispatch is pure C++ (Dijkstra / Floyd-Warshall) and
// runs on the CPU via unified memory.

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/datastructures/in_place_heap.h>
#include <cassert>

namespace tmol {
namespace pose {

template <tmol::Device D, typename Int>
struct AllPairsShortestPathsDispatch {
  static void f(TView<Int, 3, D> weights, int cutoff = -1) {
    int n_graphs    = weights.size(0);
    int max_n_nodes = weights.size(1);
    int const INF   = -1;

    assert(weights.size(2) == max_n_nodes);

    if (cutoff != -1) {
      InPlaceHeap<D, Int> heap(max_n_nodes);
      std::vector<std::vector<std::pair<int, int>>> edges(max_n_nodes);

      for (int gg = 0; gg < n_graphs; ++gg) {
        for (int ii = 0; ii < max_n_nodes; ++ii) {
          edges[ii].clear();
        }
        for (int ii = 0; ii < max_n_nodes; ++ii) {
          for (int jj = ii + 1; jj < max_n_nodes; ++jj) {
            Int jj_ii_weight = weights[gg][ii][jj];
            Int ii_jj_weight = weights[gg][jj][ii];
            Int weight =
                ((ii_jj_weight == INF || jj_ii_weight == INF)
                     ? std::max(ii_jj_weight, jj_ii_weight)
                     : std::min(ii_jj_weight, jj_ii_weight));
            if (weight != INF && (cutoff == -1 || weight < cutoff)) {
              edges[ii].push_back(std::make_pair(jj, weight));
              edges[jj].push_back(std::make_pair(ii, weight));
            }
          }
        }
        for (int ii = 0; ii < max_n_nodes; ++ii) {
          for (int jj = 0; jj < max_n_nodes; ++jj) {
            weights[gg][ii][jj] = cutoff;
          }
        }
        for (int ii = 0; ii < max_n_nodes; ++ii) {
          heap.clear();
          heap.heap_insert(ii, 0);
          int counter = 0;
          while (heap.size() != 0
                 && (cutoff == -1 || heap.peek_val() < cutoff)) {
            ++counter;
            if (counter > max_n_nodes) {
              std::cout << "Critical error: infinite loop in APSP" << std::endl;
              return;
            }
            int node              = heap.peek_ind();
            int path_weight       = heap.peek_val();
            weights[gg][ii][node] = path_weight;
            for (auto nw : edges[node]) {
              int neighb         = nw.first;
              int w              = nw.second;
              int new_path       = path_weight + w;
              if (heap.node_in_heap(neighb)) {
                if (new_path < heap.get_node_val(neighb)) {
                  heap.decrease_node_val(neighb, new_path);
                }
              } else if (
                  weights[gg][ii][neighb] == INF
                  || (cutoff != -1 && new_path < cutoff
                      && new_path < weights[gg][ii][neighb])
                  || (cutoff == -1 && new_path < weights[gg][ii][neighb])) {
                heap.heap_insert(neighb, new_path);
              }
            }
            heap.pop();
          }
        }
      }
    } else {
      for (int gg = 0; gg < n_graphs; ++gg) {
        for (int kk = 0; kk < max_n_nodes; ++kk) {
          for (int ii = 0; ii < max_n_nodes; ++ii) {
            for (int jj = 0; jj < max_n_nodes; ++jj) {
              if (weights[gg][ii][kk] >= 0 && weights[gg][kk][jj] >= 0) {
                int curr_weight = weights[gg][ii][jj];
                int new_path    = weights[gg][ii][kk] + weights[gg][kk][jj];
                if (curr_weight < 0 || new_path < curr_weight) {
                  weights[gg][ii][jj] = new_path;
                }
              }
            }
          }
        }
      }
    }
  }
};

template struct AllPairsShortestPathsDispatch<tmol::Device::MPS, int32_t>;
template struct AllPairsShortestPathsDispatch<tmol::Device::MPS, int64_t>;

}  // namespace pose
}  // namespace tmol
