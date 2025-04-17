#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace kinematics {

enum BondDOFTypes {
  // Indices of bond dof types within KinDOF.raw

  bond_dof_phi_p = 0,
  bond_dof_theta,
  bond_dof_d,
  bond_dof_phi_c,
  bond_dof_n_movable_dofs
};

enum JumpDOFTypes {
  // Indices of jump dof types within KinDOF.raw

  jump_dof_RBx = 0,
  jump_dof_RBy,
  jump_dof_RBz,
  jump_dof_RBdel_alpha,
  jump_dof_RBdel_beta,
  jump_dof_RBdel_gamma,
  jump_dof_RBalpha,
  jump_dof_RBbeta,
  jump_dof_RBgamma,
};

enum EdgeTypes {
  ff_polymer_edge = 0,
  ff_jump_edge,
  ff_root_jump_edge,
};

template <typename Int>
struct KinForestParams {
  Int id;
  Int doftype;
  Int parent;
  Int frame_x;
  Int frame_y;
  Int frame_z;
};

template <typename Int>
struct KinForestGenData {
  Int node_start;
  Int scan_start;
};

}  // namespace kinematics
}  // namespace tmol

namespace tmol {

template <typename Int>
struct enable_tensor_view<kinematics::KinForestParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::KinForestParams<Int>) / sizeof(Int)
                    : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Int>
struct enable_tensor_view<kinematics::KinForestGenData<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(kinematics::KinForestGenData<Int>) / sizeof(Int)
                    : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
