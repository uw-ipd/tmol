#include <tmol/utility/tensor/TensorAccessor.h>
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

    assert(weights.size(2) == max_n_nodes);

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
