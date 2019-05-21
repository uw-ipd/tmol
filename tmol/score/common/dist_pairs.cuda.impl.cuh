#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

namespace tmol {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// Intra-system: only compute upper-diagonal set of I/J
// pairs
template <
    typename Real,
    typename Int,
    int block_size = 8,
    int nthreads_per_block = block_size* block_size>
struct TriuDistanceCutoff {
  static auto f(
      TView<Vec<Real, 3>, 1, tmol::Device::CUDA> coords,
      TView<Int, 1, tmol::Device::CUDA> nearby,
      TView<Int, 1, tmol::Device::CUDA> nearby_scan,
      TView<Int, 1, tmol::Device::CUDA> nearby_i,
      TView<Int, 1, tmol::Device::CUDA> nearby_j,
      Real cutoff) -> Int {
    NVTXRange _function(__FUNCTION__);

    int const natoms = coords.size(0);
    int npairs = (natoms * (natoms - 1)) / 2;
    printf("natoms %d; npairs: %d\n", natoms, npairs);

    auto total_t = TPack<Int, 1, tmol::Device::CUDA>::empty({1});
    auto total = total_t.view;
    // auto nearby_pair_ind_t = TPack<Int, 1, D>::empty({npairs});
    // auto nearby_pair_i_t = TPack<Int, 1, D>::empty({npairs});
    // auto nearby_pair_j_t = TPack<Int, 1, D>::empty({npairs});
    // auto nearby_pair_scores_t = TPack<Real, 1, D>::empty({npairs});
    //
    // auto nearby_pair_ind = nearby_pair_ind_t.view;
    // auto nearby_pair_i = nearby_pair_ind_t.view;
    // auto nearby_pair_j = nearby_pair_ind_t.view;
    // auto nearby_pair_scores = nearby_pair_scores_t.view;

    // Real const threshold_distance = 6.0;
    Real const cutoff_squared = cutoff * cutoff;

    auto func_neighbors = ([=] MGPU_DEVICE(int thread_ind, int block_ind) {
      __shared__ Real i_coords[3 * block_size];
      __shared__ Real j_coords[3 * block_size];
      int nblocks =
          ((natoms - block_size) > 0 ? (natoms - block_size) : 0) / block_size
          + 1;

      int const i_block =
          nblocks
          - floorf(
                (sqrtf(8 * (nblocks * (nblocks + 1) / 2 - block_ind - 1) + 1)
                 - 1)
                / 2)
          - 1;
      int const j_block =
          nblocks - (nblocks * (nblocks + 1) / 2 - block_ind - 1)
          + (nblocks - 1 - i_block) * (nblocks - i_block) / 2 - 1;

      // int const i_block = nblocks - 2 - floorf(sqrtf(-8*block_ind +
      // 4*nblocks*(nblocks-1)-7)/2.0 - 0.5); int const j_block = block_ind +
      // i_block + 1 - nblocks*(nblocks-1)/2 +
      // (nblocks-i_block)*((nblocks-i_block)-1)/2;

      int const i_begin = i_block * block_size;
      int const j_begin = j_block * block_size;

      int const n_iterations = block_size * block_size / nthreads_per_block;
      // if ( thread_ind == 0 ) {
      //  printf("thread_ind %d block_ind %d nblocks %d i_block %d j_block %d
      //  i_begin %d j_begin %d\n",
      //    thread_ind, block_ind, nblocks, i_block, j_block, i_begin, j_begin);
      //  printf("sqrtf(8*(nblocks*(nblocks+1)/2 - block_ind - 1) + 1) %f\n",
      //    sqrtf(8*(nblocks*(nblocks+1)/2 - block_ind - 1) + 1));
      //  printf("floorf((sqrtf(8*(nblocks*(nblocks+1)/2 - block_ind - 1) + 1) -
      //  1)/2) %f\n",
      //    floorf((sqrtf(8*(nblocks*(nblocks+1)/2 - block_ind - 1) + 1) -
      //    1)/2));
      //}

      // ALT IDEA: all 2*block_size threads read from coords at the same time
      // by computing an offset to use based on thread index

      if (thread_ind < block_size) {
        if (i_begin + thread_ind < natoms) {
          i_coords[3 * thread_ind + 0] = coords[i_begin + thread_ind][0];
          i_coords[3 * thread_ind + 1] = coords[i_begin + thread_ind][1];
          i_coords[3 * thread_ind + 2] = coords[i_begin + thread_ind][2];
          // printf("load i %d + %d + %5.3f\n", i_begin, thread_ind,
          // i_coords[3*thread_ind]);
        }
      } else if (thread_ind < 2 * block_size) {
        if (j_begin + thread_ind - block_size < natoms) {
          j_coords[3 * (thread_ind - block_size) + 0] =
              coords[j_begin + thread_ind - block_size][0];
          j_coords[3 * (thread_ind - block_size) + 1] =
              coords[j_begin + thread_ind - block_size][1];
          j_coords[3 * (thread_ind - block_size) + 2] =
              coords[j_begin + thread_ind - block_size][2];
          // printf("load j %d + %d + %5.3f\n", j_begin, thread_ind,
          // j_coords[3*(thread_ind-block_size)]);
        }
      }
      __syncthreads();

      for (int count = 0; count < n_iterations; ++count) {
        int const local_i =
            (count * nthreads_per_block + thread_ind) / block_size;
        int const local_j =
            (count * nthreads_per_block + thread_ind) % block_size;
        int const i = i_begin + local_i;
        int const j = j_begin + local_j;
        if (i < j && j < natoms) {
          int global_ind = (natoms * (natoms - 1)) / 2
                           - (natoms - i) * (natoms - i - 1) / 2 + j - i - 1;
          if (global_ind < natoms * (natoms - 1) / 2) {
            Eigen::Matrix<Real, 3, 1> my_i_coord;
            Eigen::Matrix<Real, 3, 1> my_j_coord;
            my_i_coord[0] = i_coords[3 * local_i + 0];
            my_i_coord[1] = i_coords[3 * local_i + 1];
            my_i_coord[2] = i_coords[3 * local_i + 2];
            my_j_coord[0] = j_coords[3 * local_j + 0];
            my_j_coord[1] = j_coords[3 * local_j + 1];
            my_j_coord[2] = j_coords[3 * local_j + 2];
            Real dis2 = (my_i_coord - my_j_coord).squaredNorm();
            nearby[global_ind] =
                dis2 < cutoff_squared;  // NaN coordinates --> NaN distances -->
                                        // compare false

            // printf("thread_ind %d block_ind %d nblocks %d i_block %d j_block
            // %d i_begin %d j_begin %d local_i %d local_j %d i %d j %d
            // global_ind %d, my_i_coord[0], %5.3f my_j_coord[0], %5.3f dis2
            // %5.3f\n",
            //  thread_ind, block_ind, nblocks, i_block, j_block, i_begin,
            //  j_begin, local_i, local_j, i, j, global_ind, my_i_coord[0],
            //  my_j_coord[0], dis2);
          }
        }
      }
    });

    clock_t start = clock();

    mgpu::standard_context_t context;

    int nblocks =
        ((natoms - block_size) > 0 ? (natoms - block_size) : 0) / block_size
        + 1;
    int nblock_pairs = nblocks * (nblocks + 1) / 2;
    // printf("\n");
    // printf("Nblocks: %d, nblock_pairs: %d, natoms: %d\n", nblocks,
    // nblock_pairs, natoms);
    nvtx_range_push("st1");
    // int constexpr npairs_in_block = block_size * block_size;
    mgpu::cta_launch<nthreads_per_block, 1>(
        func_neighbors, nblock_pairs, context);
    nvtx_range_pop();

    int nearby0;
    cudaMemcpy(&nearby0, &nearby[0], sizeof(int), cudaMemcpyDeviceToHost);
    printf("nearby 0: %d\n", nearby0);

    clock_t stop = clock();
    printf("distpair eval took: %f\n", ((double)stop - start) / CLOCKS_PER_SEC);

    // Scan.

    start = clock();
    nvtx_range_push("st2");
    mgpu::scan<mgpu::scan_type_exc>(
        nearby.data(),
        npairs,
        nearby_scan.data(),
        mgpu::plus_t<Real>(),
        total.data(),
        context);
    nvtx_range_pop();
    int scan0;
    cudaMemcpy(&scan0, &nearby_scan[0], sizeof(int), cudaMemcpyDeviceToHost);
    printf("scan 0: %d\n", scan0);
    stop = clock();
    printf("scan took: %f\n", ((double)stop - start) / CLOCKS_PER_SEC);

    // Return the number of nearby atom pairs as the last element of the scanned
    // nearby_pair tensor.

    auto write_nearby_pair_inds = ([=] MGPU_DEVICE(int idx) {
      if (idx < npairs) {
        int const i =
            natoms - 2
            - floorf(
                  sqrtf(-8 * idx + 4 * natoms * (natoms - 1) - 7) / 2.0 - 0.5);
        int const j = idx + i + 1 - natoms * (natoms - 1) / 2
                      + (natoms - i) * ((natoms - i) - 1) / 2;
        if (nearby[idx]) {
          int out_idx = nearby_scan[idx];
          nearby_i[out_idx] = i;
          nearby_j[out_idx] = j;
        }
      }
    });
    nvtx_range_push("st3");
    start = clock();
    mgpu::transform(write_nearby_pair_inds, npairs, context);
    nvtx_range_pop();

    int total_cpu;
    cudaMemcpy(&total_cpu, &total[0], sizeof(int), cudaMemcpyDeviceToHost);
    // printf( "Total n nearby: %d\n", total_cpu);
    stop = clock();
    printf("stage 3 took: %f\n", ((double)stop - start) / CLOCKS_PER_SEC);
    return total_cpu;
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
