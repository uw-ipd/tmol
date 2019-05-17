#include <tmol/score/common/simple_dispatch.cuda.impl.cuh>

// TEMPT !! #include "lk_isotropic.dispatch.impl.hh"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "lk_isotropic.dispatch.hh"
#include "lk_isotropic.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

// TEMP !! #define declare_dispatch(Real, Int)    \
// TEMP !!                                        \
// TEMP !!   template struct LKIsotropicDispatch< \
// TEMP !!       AABBDispatch,                    \
// TEMP !!       tmol::Device::CUDA,              \
// TEMP !!       Real,                            \
// TEMP !!       Int>;                            \
// TEMP !!   template struct LKIsotropicDispatch< \
// TEMP !!       AABBTriuDispatch,                \
// TEMP !!       tmol::Device::CUDA,              \
// TEMP !!       Real,                            \
// TEMP !!       Int>;
// TEMP !!
// TEMP !! declare_dispatch(float, int64_t);
// TEMP !! declare_dispatch(double, int64_t);
// TEMP !!
// TEMP !! #undef declare_dispatch

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LKIsotropicDispatch<Dispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 1, D> coords_i,
    TView<Int, 1, D> atom_type_i,

    TView<Vec<Real, 3>, 1, D> coords_j,
    TView<Int, 1, D> atom_type_j,

    TView<Real, 2, D> bonded_path_lengths,

    TView<LKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params)
    -> std::tuple<
        TPack<Real, 1, D>,
        TPack<Vec<Real, 3>, 1, D>,
        TPack<Vec<Real, 3>, 1, D>> {
  NVTXRange _function(__FUNCTION__);

  nvtx_range_push("alloc");
  auto V_t = TPack<Real, 1, D>::zeros({1});
  auto dV_dI_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_i.size(0)});
  auto dV_dJ_t = TPack<Vec<Real, 3>, 1, D>::zeros({coords_j.size(0)});

  auto V = V_t.view;
  auto dV_dI = dV_dI_t.view;
  auto dV_dJ = dV_dJ_t.view;

  // int const natoms = coords_i.size(0);
  // int npairs = (natoms*(natoms-1))/2;
  //
  // auto nearby_pair_ind_t = TPack<Int, 1, D>::empty({npairs});
  // auto nearby_pair_i_t = TPack<Int, 1, D>::empty({npairs});
  // auto nearby_pair_j_t = TPack<Int, 1, D>::empty({npairs});
  // auto nearby_pair_scores_t = TPack<Real, 1, D>::empty({npairs});
  //
  // auto nearby_pair_ind = nearby_pair_ind_t.view;
  // auto nearby_pair_i = nearby_pair_ind_t.view;
  // auto nearby_pair_j = nearby_pair_ind_t.view;
  // auto nearby_pair_scores = nearby_pair_scores_t.view;
  nvtx_range_pop();

  // Four parts.
  // 1. decide what atom pairs are near each other
  // 2. scan the array of nearby atom pairs to get offset inds
  // 3. write down which pairs need their energies evaluated
  // 4. evaluate energies for nearby atom pairs.

  /// ASSUMPTION! This is intra-system only. Only upper-diagonal
  /// of I vs J evaluation performed.

  // int constexpr block_size = 8;
  // Real const threshold_distance = 6.0;
  // Real const threshold_squared = threshold_distance * threshold_distance;
  //
  // auto func_neighbors = ([=] MGPU_DEVICE(int thread_ind, int block_ind) {
  //    __shared__ Real i_coords[3 * block_size];
  //    __shared__ Real j_coords[3 * block_size];
  //    int nblocks = ((natoms - block_size) > 0 ? (natoms - block_size) : 0 ) /
  //    block_size + 1;
  //
  //    int const i_block = nblocks - floorf(sqrtf(-8*(nblocks*(nblocks+1)/2 -
  //    block_ind - 1) + 1)/2) - 1; int const j_block = nblocks -
  //    (nblocks*(nblocks+1)/2 - block_ind - 1 ) + (nblocks - 1 -
  //    i_block)*(nblocks - i_block) / 2 - 1;
  //
  //    //int const i_block = nblocks - 2 - floorf(sqrtf(-8*block_ind +
  //    4*nblocks*(nblocks-1)-7)/2.0 - 0.5);
  //    //int const j_block = block_ind + i_block + 1 - nblocks*(nblocks-1)/2 +
  //    (nblocks-i_block)*((nblocks-i_block)-1)/2;
  //
  //    int const i_begin = i_block * block_size;
  //    int const j_begin = j_block * block_size;
  //
  //    // if ( thread_ind == 0 ) {
  //    // 	printf("thread_ind %d block_ind %d nblocks %d i_block %d j_block
  //    %d i_begin %d j_begin %d\n",
  //    // 	thread_ind, block_ind, nblocks, i_block, j_block, i_begin,
  //    j_begin);
  //    // }
  //
  //    if ( thread_ind < block_size ) {
  //	if (i_begin + thread_ind < natoms) {
  //	  i_coords[3 * thread_ind + 0] = coords_i[i_begin + thread_ind][0];
  //	  i_coords[3 * thread_ind + 1] = coords_i[i_begin + thread_ind][1];
  //	  i_coords[3 * thread_ind + 2] = coords_i[i_begin + thread_ind][2];
  //	}
  //    } else if ( thread_ind < 2*block_size) {
  //	if (j_begin + thread_ind < natoms) {
  //	  j_coords[3 * (thread_ind-block_size) + 0] = coords_j[j_begin +
  // thread_ind][0]; 	  j_coords[3 * (thread_ind-block_size) + 1] =
  // coords_j[j_begin
  //+ thread_ind][1]; 	  j_coords[3 * (thread_ind-block_size) + 2] =
  // coords_j[j_begin + thread_ind][2];
  //	}
  //    }
  //    __syncthreads;
  //
  //    int const local_i = thread_ind / block_size;
  //    int const local_j = thread_ind % block_size;
  //    int const i = i_begin + local_i;
  //    int const j = j_begin + local_j;
  //    if ( i < j ) {
  //	int global_ind = (natoms*(natoms-1))/2 - (natoms-i)*(natoms-i-1)/2 + j;
  //	if (global_ind < natoms*(natoms-1)/2) {
  //
  //	  //printf("thread_ind %d block_ind %d nblocks %d i_block %d j_block %d
  // i_begin %d j_begin %d global_ind %d\n",
  //	  //  thread_ind, block_ind, nblocks, i_block, j_block, i_begin,
  // j_begin, global_ind);
  //
  //	  Eigen::Matrix<Real, 3, 1> my_i_coord;
  //	  Eigen::Matrix<Real, 3, 1> my_j_coord;
  //	  my_i_coord[0] = i_coords[3*local_i + 0];
  //	  my_i_coord[1] = i_coords[3*local_i + 1];
  //	  my_i_coord[2] = i_coords[3*local_i + 2];
  //	  my_j_coord[0] = j_coords[3*local_j + 0];
  //	  my_j_coord[1] = j_coords[3*local_j + 1];
  //	  my_j_coord[2] = j_coords[3*local_j + 2];
  //	  Real dis2 = (my_i_coord - my_j_coord).squaredNorm();
  //	  nearby_pair_ind[global_ind] = dis2 < threshold_squared;
  //	}
  //    }
  //
  //  });
  //
  // mgpu::standard_context_t context;
  //
  // int nblocks = ((natoms - block_size) > 0 ? (natoms - block_size) : 0 ) /
  // block_size + 1; int nblock_pairs = nblocks * (nblocks + 1) / 2;
  // printf("\n");
  // printf("Nblocks: %d, nblock_pairs: %d, natoms: %d\n", nblocks,
  // nblock_pairs, natoms); mgpu::cta_launch<64,1>(func_neighbors, nblock_pairs,
  // context);

  nvtx_range_push("dispatch::score");
  Real threshold_distance = 6.0;
  Dispatch<D>::forall_pairs(
      threshold_distance,
      coords_i,
      coords_j,
      [=] EIGEN_DEVICE_FUNC(int i, int j) {
        Int ati = atom_type_i[i];
        Int atj = atom_type_j[j];

        auto dist_r = distance<Real>::V_dV(coords_i[i], coords_j[j]);
        auto& dist = dist_r.V;
        auto& ddist_dI = dist_r.dV_dA;
        auto& ddist_dJ = dist_r.dV_dB;

        auto lk = lk_isotropic_score<Real>::V_dV(
            dist,
            bonded_path_lengths[i][j],
            type_params[ati],
            type_params[atj],
            global_params[0]);

        accumulate<D, Real>::add(V[0], lk.V);
        accumulate<D, Vec<Real, 3>>::add(dV_dI[i], lk.dV_ddist * ddist_dI);
        accumulate<D, Vec<Real, 3>>::add(dV_dJ[j], lk.dV_ddist * ddist_dJ);
      });
  nvtx_range_pop();

  nvtx_range_pop();

  return {V_t, dV_dI_t, dV_dJ_t};
};

#define declare_dispatch(Real, Int)    \
                                       \
  template struct LKIsotropicDispatch< \
      AABBDispatch,                    \
      tmol::Device::CUDA,              \
      Real,                            \
      Int>;                            \
  template struct LKIsotropicDispatch< \
      AABBTriuDispatch,                \
      tmol::Device::CUDA,              \
      Real,                            \
      Int>;

declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

#undef declare_dispatch

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
