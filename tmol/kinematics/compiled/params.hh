#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace kinematics {

template <typename Int>
struct KinTreeParams {
  Int id;
  Int doftype;
  Int parent;
  Int frame_x;
  Int frame_y;
  Int frame_z;
};

template <typename Int>
struct KinTreeGenData {
  Int node_start;
  Int scan_start;
};

template <typename Real>
struct JumpDofTypes {
  Real RBtrans[3];
  Real RBtrans_del[3];
  Real RBangle[3];
}

template <typename Real>
struct BondDofTypes {
  Real phi_p;
  Real theta;
  Real d;
  Real phi_c;
}

template <typename Real>
struct DofTypes {
  union {
    struct JumpDofTypes<Real> asjump;
    struct BondDofTypes<Real> asbond;
  } data;
}  // namespace kinematics
}  // namespace kinematics

namespace tmol {

template <typename Real>
struct enable_tensor_view<kinematics::KinTreeParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::KinTreeParams<Int>) / sizeof(Int) : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<kinematics::KinTreeGenData<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::KinTreeGenData<Int>) / sizeof(Int) : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<kinematics::DofTypes<Real>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::DofTypes<Real>) / sizeof(Real) : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
