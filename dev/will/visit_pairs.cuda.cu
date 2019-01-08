#include <ATen/ATen.h>
#include <ATen/ScalarTypeUtils.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "lj.hh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tmol {
namespace willtest {

template<typename Real>
int visit_pairs_cpu_naive(at::Tensor pts_t, Real dis){
  typedef Eigen::Matrix<Real, 3, 1> V3;

  auto pts = tmol::view_tensor<V3, 2, RestrictPtrTraits>(pts_t);

  int count = 0;
  for(int i = 0; i < pts.size(0); ++i){
    V3 a = pts[i][0];
    for(int j = 0; j < i; j++){
      V3 b = pts[j][0];
      Real dis2 = (a-b).squaredNorm();
      if(dis2 <= dis*dis) count++;
    }
  }
  return count;
}

template int visit_pairs_cpu_naive<float>(at::Tensor, float);

}
}
