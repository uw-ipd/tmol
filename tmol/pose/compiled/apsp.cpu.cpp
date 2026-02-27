#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/datastructures/in_place_heap.h>
#include <cassert>

namespace tmol {
namespace pose {

template <tmol::Device D, typename Int>
struct AllPairsShortestPathsDispatch {
  static void f(TView<Int, 3, D> weights, int cutoff = -1) {
    // Sentintel weight of -1 is used to indicate that
    // two nodes have an infinite path weight

    int n_graphs = weights.size(0);
    int max_n_nodes = weights.size(1);
    int const INF = -1;  // sentinel value: these nodes are not connected

    assert(weights.size(2) == max_n_nodes);

    if (cutoff != -1) {
      // Dijkstra's single-source shortest path, but where the search
      // stops once the closest next node that would be reached has
      // a distance/weight > cutoff. This O(N^2) alg is asympototically
      // faster than Floyd-Warshall, which is O(N^3). The O(N^2) cost
      // is spent in constructing the edge lists / initializing the
      // output tensor; the work of the alg itself is O(NK ln(K)) where
      // K is the expected number of neighbors beneath the given
      // cutoff; in these graphs, N >> K.

      InPlaceHeap<D, Int> heap(max_n_nodes);
      std::vector<std::vector<std::pair<int, int>>> edges(max_n_nodes);
      for (int gg = 0; gg < n_graphs; ++gg) {
        // reset from previous iteration
        for (int ii = 0; ii < max_n_nodes; ++ii) {
          edges[ii].clear();
        }
        // construct edge list representation from weight matrix
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

        // Reset the weight matrix to the max distance
        for (int ii = 0; ii < max_n_nodes; ++ii) {
          for (int jj = 0; jj < max_n_nodes; ++jj) {
            weights[gg][ii][jj] = cutoff;
          }
        }

        // ok, now we run Dijkstra's algorithm from each node ii stopping if we
        // hit the cutoff distance

        for (int ii = 0; ii < max_n_nodes; ++ii) {
          heap.clear();
          heap.heap_insert(ii, 0);
          int counter = 0;
          while (heap.size() != 0
                 && (cutoff == -1 || heap.peek_val() < cutoff)) {
            ++counter;
            if (counter > max_n_nodes) {
              std::cout << "Critical error: infinite loop detected in "
                           "all-pairs-shortest-path exiting"
                        << std::endl;
              return;
            }
            int node = heap.peek_ind();
            int path_weight_to_node = heap.peek_val();
            weights[gg][ii][node] = path_weight_to_node;
            for (auto neighb_weight_pair : edges[node]) {
              int neighb = neighb_weight_pair.first;
              int weight = neighb_weight_pair.second;
              int new_path_weight_to_neighb = path_weight_to_node + weight;
              if (heap.node_in_heap(neighb)) {
                if (new_path_weight_to_neighb < heap.get_node_val(neighb)) {
                  heap.decrease_node_val(neighb, new_path_weight_to_neighb);
                }
              } else if (
                  weights[gg][ii][neighb] == INF
                  || (cutoff != -1 && new_path_weight_to_neighb < cutoff
                      && new_path_weight_to_neighb < weights[gg][ii][neighb])
                  || (cutoff == -1
                      && new_path_weight_to_neighb < weights[gg][ii][neighb])) {
                // Asks: have we already visited neighb and therefore we should
                // skip adding it to the heap?
                heap.heap_insert(neighb, new_path_weight_to_neighb);
              }
            }
            heap.pop();
          }
        }
      }
    } else {
      // Floyd Warshall algorithm
      // More efficient than Dijkstra's when no threshold given
      for (int gg = 0; gg < n_graphs; ++gg) {
        // for each intermediate node, kk
        for (int kk = 0; kk < max_n_nodes; ++kk) {
          // for each node ii
          for (int ii = 0; ii < max_n_nodes; ++ii) {
            // for each node jj
            for (int jj = 0; jj < max_n_nodes; ++jj) {
              // ask: is there a shorter path from ii to jj by passing
              // first through intermediate node kk?

              if (weights[gg][ii][kk] >= 0 && weights[gg][kk][jj] >= 0) {
                int curr_weight = weights[gg][ii][jj];
                int new_path_weight = weights[gg][ii][kk] + weights[gg][kk][jj];
                if (curr_weight < 0 || new_path_weight < curr_weight) {
                  weights[gg][ii][jj] = new_path_weight;
                }
              }

            }  // for node jj
          }  // for node ii
        }  // for intermediate node kk
      }  // for graph gg
    }
  }
};

template struct AllPairsShortestPathsDispatch<tmol::Device::CPU, int32_t>;
template struct AllPairsShortestPathsDispatch<tmol::Device::CPU, int64_t>;

}  // namespace pose
}  // namespace tmol
