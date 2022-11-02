#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/coordinate_load.cuh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/debug.cuh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>
// #include <tmol/score/ljlk/potentials/sphere_overlap.cuda.cuh>

// #include <tmol/score/common/sphere_overlap.impl.hh>

#include <chrono>

//#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <tmol/score/common/device_operations.cuda.impl.cuh>

#include <moderngpu/operators.hxx>
#include <moderngpu/cta_reduce.hxx>
#include <moderngpu/transform.hxx>

// This file moves in more recent versions of Torch
#include <c10/cuda/CUDAStream.h>

#include <tmol/score/ljlk/potentials/ljlk_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKPoseScoreDispatch<
    // ForallDispatch,
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct LJLKPoseScoreDispatch<
    // ForallDispatch,
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
