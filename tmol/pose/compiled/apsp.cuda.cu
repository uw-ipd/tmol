#include <tmol/utility/tensor/TensorAccessor.h>
#include <cassert>

#define TILE_SIZE 32

namespace tmol {
namespace pose {

template <class F>
__global__ void launch(F f, int i) {
  f(i);
};

template <tmol::Device D, typename Int>
struct AllPairsShortestPathsDispatch {
  static void f(TView<Int, 3, D> weights) {
    // Warshall's Algorithm for each graph.
    // Sentintel weights below 0 are used to indicate that
    // two nodes have an infinite path weight

    // From:
    // Katz, G. J., & Kider, J. T. (2008). All-Pairs Shortest-Paths for Large
    // Graphs on the GPU. Proceedings of the 23rd ACM SIGGRAPH/EUROGRAPHICS
    // Symposium on Graphics Hardware (GH '08), 47-55.
    // https://repository.upenn.edu/cgi/viewcontent.cgi?article=1213&context=hms

    int n_graphs = weights.size(0);
    int max_n_nodes = weights.size(1);

    assert(weights.size(2) == max_n_nodes);

    auto phase1 = ([=] __device__(int block) {
      int graph_ind = blockIdx.x;
      int x = threadIdx.y;
      int y = threadIdx.x;
      int i = TILE_SIZE * block + x;
      int j = TILE_SIZE * block + y;

      __shared__ Int shared_weights[TILE_SIZE * TILE_SIZE];

      Int x_y_curr;
      if (i < max_n_nodes && j < max_n_nodes) {
        x_y_curr = weights[graph_ind][i][j];
        shared_weights[TILE_SIZE * x + y] = x_y_curr;
      } else {
        x_y_curr = -1;
        shared_weights[TILE_SIZE * x + y] = -1;
      }

      __syncthreads();

      for (int kk = 0; kk < TILE_SIZE; ++kk) {
        Int xkk_weight = shared_weights[TILE_SIZE * x + kk];
        Int kky_weight = shared_weights[TILE_SIZE * kk + y];

        __syncthreads();  // wait for read

        if (xkk_weight >= 0 && kky_weight >= 0) {
          if (x_y_curr < 0 || xkk_weight + kky_weight < x_y_curr) {
            x_y_curr = xkk_weight + kky_weight;
            shared_weights[TILE_SIZE * x + y] = x_y_curr;
          }
        }
        __syncthreads();  // wait for write
      }

      if (i < max_n_nodes && j < max_n_nodes) {
        weights[graph_ind][i][j] = x_y_curr;
      }
    });

    auto phase2 = ([=] __device__(int primary_block) {
      __shared__ struct {
        Int primary_block_weights[TILE_SIZE * TILE_SIZE];
        Int target_block_weights[TILE_SIZE * TILE_SIZE];
      } shared;

      int const graph_ind = blockIdx.x;
      int const off_diag_block = blockIdx.y;
      int const z = blockIdx.z;
      int const x = threadIdx.y;
      int const y = threadIdx.x;
      int const is_after_primary = off_diag_block >= primary_block ? 1 : 0;

      // Figure out which node pair i,j we are examining.
      // If Z is 0, then we are targetting a block in the same column as the
      // primary block. If Z is 1, then we are targetting a block in the same
      // row.
      int const i =
          TILE_SIZE
              * (z == 0 ? off_diag_block + is_after_primary : primary_block)
          + x;
      int const j =
          TILE_SIZE
              * (z == 1 ? off_diag_block + is_after_primary : primary_block)
          + y;

      int const primary_i = TILE_SIZE * primary_block + x;
      int const primary_j = TILE_SIZE * primary_block + y;

      // Now we'll load the primary- and target weights from global memory.
      // Mark off-the-end nodes w/ "infinite" distance
      // 1. primary.
      if (primary_i < max_n_nodes && primary_j < max_n_nodes) {
        shared.primary_block_weights[TILE_SIZE * x + y] =
            weights[graph_ind][primary_i][primary_j];
      } else {
        shared.primary_block_weights[TILE_SIZE * x + y] = -1;
      }

      Int x_y_curr;
      if (i < max_n_nodes && j < max_n_nodes) {
        x_y_curr = weights[graph_ind][i][j];
        shared.target_block_weights[TILE_SIZE * x + y] = x_y_curr;
      } else {
        // sentinel value: off the end
        x_y_curr = -1;
        shared.target_block_weights[TILE_SIZE * x + y] = -1;
      }

      // Wait until everyone is done before we proceed
      __syncthreads();

      // Now iterate across all k in the primary row (column if z = 1) and see
      // if the
      if (z == 0) {
        for (int k = 0; k < TILE_SIZE; ++k) {
          Int ik_weight = shared.target_block_weights[TILE_SIZE * x + k];
          Int kj_weight = shared.primary_block_weights[TILE_SIZE * k + y];

          __syncthreads();  // wait for read

          if (ik_weight >= 0 && kj_weight >= 0) {
            Int ikj_weight = ik_weight + kj_weight;
            if (ikj_weight < x_y_curr || x_y_curr < 0) {
              x_y_curr = ikj_weight;
              shared.target_block_weights[TILE_SIZE * x + y] = x_y_curr;
            }
          }
          __syncthreads();  // wait for write
        }
      } else {
        for (int k = 0; k < TILE_SIZE; ++k) {
          Int ik_weight = shared.primary_block_weights[TILE_SIZE * x + k];
          Int kj_weight = shared.target_block_weights[TILE_SIZE * k + y];

          __syncthreads();  // wait for read

          if (ik_weight >= 0 && kj_weight >= 0) {
            Int ikj_weight = ik_weight + kj_weight;
            if (ikj_weight < x_y_curr || x_y_curr < 0) {
              x_y_curr = ikj_weight;
              shared.target_block_weights[TILE_SIZE * x + y] = x_y_curr;
            }
          }
          __syncthreads();  // wait for write
        }
      }
      if (i < max_n_nodes && j < max_n_nodes) {
        weights[graph_ind][i][j] = x_y_curr;
      }
    });

    auto phase3 = ([=] __device__(int primary_block) {
      __shared__ struct {
        Int ph2_column_block[TILE_SIZE * TILE_SIZE];
        Int ph2_row_block[TILE_SIZE * TILE_SIZE];
      } shared;

      int const graph_ind = blockIdx.x;
      int const target_block_row =
          blockIdx.y + (blockIdx.y >= primary_block ? 1 : 0);
      int const target_block_column =
          blockIdx.z + (blockIdx.z >= primary_block ? 1 : 0);
      int const x = threadIdx.y;
      int const y = threadIdx.x;

      int const primary_i = TILE_SIZE * primary_block + x;
      int const primary_j = TILE_SIZE * primary_block + y;
      int const i = TILE_SIZE * target_block_row + x;
      int const j = TILE_SIZE * target_block_column + y;

      // Now let's load data into shared memory
      // 1. The column block: same row index as the target block
      if (i < max_n_nodes && primary_j < max_n_nodes) {
        shared.ph2_column_block[TILE_SIZE * x + y] =
            weights[graph_ind][i][primary_j];
      } else {
        shared.ph2_column_block[TILE_SIZE * x + y] = -1;
      };  // 2. The row block: same column index as the target block
      if (primary_i < max_n_nodes && j < max_n_nodes) {
        shared.ph2_row_block[TILE_SIZE * x + y] =
            weights[graph_ind][primary_i][j];
      } else {
        shared.ph2_row_block[TILE_SIZE * x + y] = -1;
      }

      __syncthreads();  // wait until everyone has written to shared memory

      Int x_y_curr = weights[graph_ind][i][j];

      // Now
      for (int k = 0; k < TILE_SIZE; ++k) {
        Int const ik_weight = shared.ph2_column_block[TILE_SIZE * x + k];
        Int const kj_weight = shared.ph2_row_block[TILE_SIZE * k + y];
        if (ik_weight >= 0 && kj_weight >= 0) {
          Int const ikj_weight = ik_weight + kj_weight;
          if (ikj_weight < x_y_curr || x_y_curr < 0) {
            x_y_curr = ikj_weight;
          }
        }
      }

      // No need to wait!
      if (i < max_n_nodes && j < max_n_nodes) {
        weights[graph_ind][i][j] = x_y_curr;
      }
    });

    // OK! Now we need to iterate across the blocks and launch the three phases
    // per block

    // auto p1_kernel = ([=] __global__ (int primary_block)
    // {phase1(primary_block);}); auto p2_kernel = ([=] __global__ (int
    // primary_block) {phase2(primary_block);}); auto p3_kernel = ([=]
    // __global__ (int primary_block) {phase3(primary_block);});

    int const n_blocks = (max_n_nodes - 1) / TILE_SIZE + 1;
    // int const n_threads = TILE_SIZE * TILE_SIZE;
    dim3 n_threads(TILE_SIZE, TILE_SIZE, 1);

    for (int i = 0; i < n_blocks; ++i) {
      launch<<<n_graphs, n_threads>>>(phase1, i);
      if (n_blocks > 1) {
        dim3 p2_dim(n_graphs, n_blocks - 1, 2);
        launch<<<p2_dim, n_threads>>>(phase2, i);

        dim3 p3_dim(n_graphs, n_blocks - 1, n_blocks - 1);
        launch<<<p3_dim, n_threads>>>(phase3, i);
      }
    }
  }
};

template struct AllPairsShortestPathsDispatch<tmol::Device::CUDA, int32_t>;
template struct AllPairsShortestPathsDispatch<tmol::Device::CUDA, int64_t>;

}  // namespace pose
}  // namespace tmol
