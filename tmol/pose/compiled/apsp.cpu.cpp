#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/datastructures/in_place_heap.h>
#include <cassert>

namespace tmol {
namespace pose {

template <tmol::Device D, typename Int>
struct AllPairsShortestPathsDispatch {
  static void f(TView<Int, 3, D> weights) {
    // Warshall's Algorithm for each graph.
    // Sentintel weights below 0 are used to indicate that
    // two nodes have an infinite path weight

    int n_graphs = weights.size(0);
    int max_n_nodes = weights.size(1);
    int const INF = 6;

    assert(weights.size(2) == max_n_nodes);

    InPlaceHeap<D, Int> heap(max_n_nodes);
    auto weights2_tp =
        TPack<Int, 3, D>::full({n_graphs, max_n_nodes, max_n_nodes}, INF);
    auto weights2 = weights2_tp.view;

    for (int gg = 0; gg < n_graphs; ++gg) {
      std::vector<std::list<std::pair<int, int>>> edges;
      for (int ii = 0; ii < max_n_nodes; ++ii) {
        for (int jj = ii + 1; jj < max_n_nodes; ++jj) {
          if (weights[gg][ii][jj] < INF) {
            edges[ii].push_back(std::make_pair(jj, weights[gg][ii][jj]));
            edges[jj].push_back(std::make_pair(ii, weights[gg][ii][jj]));
          }
        }
      }
      // ok, now we run Dijkstra's algorithm from each node ii
      for (int ii = 0; ii < max_n_nodes; ++ii) {
        heap.clear();
        heap.heap_insert(ii, 0);
        while (heap.peek_val() < INF) {
          int node = heap.peek_ind();
          int path_weight_to_node = heap.peek_val();
          weights2[gg][ii][node] = path_weight_to_node;
          for (auto neighb_weight_pair : edges[node]) {
            int neighb = neighb_weight_pair.first;
            int weight = neighb_weight_pair.second;
            if (heap.node_in_heap(neighb)) {
              int new_path_weight_to_neighb = path_weight_to_node + weight;
              if (new_path_weight_to_neighb < heap.get_node_val(neighb)) {
                heap.decrease_node_val(neighb, new_path_weight_to_neighb);
              }
            } else if (weights2[gg][ii][neighb] == INF && weight < INF) {
              heap.heap_insert(neighb, weight);
            }
          }
          heap.pop();
        }
      }
    }

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
        }    // for node ii
      }      // for intermediate node kk
    }        // for graph gg

    // Check our Dijkstra's calculation
    for (int gg = 0; gg < n_graphs; ++gg) {
      for (int ii = 0; ii < max_n_nodes; ++ii) {
        for (int jj = 0; jj < max_n_nodes; ++jj) {
          if (weights[gg][ii][jj] != weights2[gg][ii][jj]) {
            std::cout << "Discrepancy in " << gg << " " << ii << " " << jj
                      << ": ";
            std::cout << weights[gg][ii][jj] << " vs " << weights2[gg][ii][jj]
                      << std::endl;
          }
        }
      }
    }
  }
};

// template <class T>
// class InPlaceHeap
// {
// private:
//   std::vector<T> values_;
// };

// template <tmol::Device D, typename Int>
// struct LimitedSparseAllPairsShortestPathsDispatch {
//   static
//   TPack<Int, 3, D>
//   f(
//     TView<Int, 2, D> n_conn_for_nodes,
//     TView<Int, 2, D> conn_offset_for_nodes,
//     TView<Int, 3, D> connections_for_nodes,
//     int limit
//   ) {
//
//     // Dijkstra's algorithm starting from each node
//     // heading toward every other node, but stopping
//     // once the limit is reached
//     int n_graphs = n_conn_for_nodes.size(0);
//     int max_n_nodes = n_conn_for_nodes.size(1);
//
//     assert(conn_offset_for_nodes.size(0) == n_graphs);
//     assert(conn_offset_for_nodes.size(1) == max_n_nodes);
//     assert(connections_for_nodes.size(0) == n_graphs);
//     assert(connections_for_nodes.size(1) == max_n_ndoes);
//
//     auto weights_tp = TPack<Int, 3, D>::full({n_graphs, max_n_nodes,
//     max_n_nodes}, limit); auto weights = weights_tp.view;
//
//
//     for (int gg = 0; gg < n_graphs; ++gg) {
//       // for each intermediate node, kk
//       for (int kk = 0; kk < max_n_nodes; ++kk) {
//         // for each node ii
//         for (int ii = 0; ii < max_n_nodes; ++ii) {
//           // for each node jj
//           for (int jj = 0; jj < max_n_nodes; ++jj) {
//             // ask: is there a shorter path from ii to jj by passing
//             // first through intermediate node kk?
//
//             if (weights[gg][ii][kk] >= 0 && weights[gg][kk][jj] >= 0) {
//               int curr_weight = weights[gg][ii][jj];
//               int new_path_weight = weights[gg][ii][kk] +
//               weights[gg][kk][jj]; if (curr_weight < 0 || new_path_weight <
//               curr_weight) {
//                 weights[gg][ii][jj] = new_path_weight;
//               }
//             }
//
//           }  // for node jj
//         }    // for node ii
//       }      // for intermediate node kk
//     }        // for graph gg
//   }
// };

template struct AllPairsShortestPathsDispatch<tmol::Device::CPU, int32_t>;
template struct AllPairsShortestPathsDispatch<tmol::Device::CPU, int64_t>;

}  // namespace pose
}  // namespace tmol
