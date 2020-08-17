// Device-agnostic common numeric routines for kinematics.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

#include <pybind11/pybind11.h>

namespace tmol {
namespace kinematics {

#define Dofs Eigen::Matrix<Real, 9, 1>
#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define QuatTranslation Eigen::Matrix<Real, 7, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

enum DOFtype { ROOT = 0, JUMP, BOND };

enum JumpDOFidx {
  DOF_RBx = 0,
  DOF_RBy,
  DOF_RBz,
  DOF_rotdelalpha,
  DOF_rotdelbeta,
  DOF_rotdelgamma,
  DOF_rotalpha,
  DOF_rotbeta,
  DOF_rotgamma
};
enum BondDOFidx { DOF_phip = 0, DOF_theta, DOF_D, DOF_phic };

template <tmol::Device D, typename Real, typename Int>
struct common {
  // compute JUMP derivatives from f1/f2
  //
  // Translational derivatives are straightforward dot products of f2s
  //     (the downstream derivative sum)
  //
  // Rotational derivatives use the Abe and Go "trick" that allows us to easily
  // compute derivatives with respect to rotation about an axis.
  //
  // In this case, there are three axes to compute derivatives of::
  //     1) the Z axis (alpha rotation)
  //     2) the Y axis after applying the alpha rotation (beta rotation)
  //     3) the X axis after applying the alpha & beta rot (gamma rotation)
  //
  // Derivatives are ONLY assigned to the RBdel DOFs
  static auto EIGEN_DEVICE_FUNC jumpDerivatives(
      Dofs dof,
      HomogeneousTransform M,
      HomogeneousTransform Mparent,
      Coord f1,
      Coord f2) -> Dofs {
    Dofs dsc_ddof = Dofs::Constant(0.0);

    Coord xaxis = Mparent.block(0, 0, 1, 3).transpose();
    Coord yaxis = Mparent.block(1, 0, 1, 3).transpose();
    Coord zaxis = Mparent.block(2, 0, 1, 3).transpose();

    // dE/drb_i
    dsc_ddof[DOF_RBx] = xaxis.dot(f2);
    dsc_ddof[DOF_RBy] = yaxis.dot(f2);
    dsc_ddof[DOF_RBz] = zaxis.dot(f2);

    Coord end_pos = M.block(3, 0, 1, 3).transpose();
    Coord rotdof3_axes = -zaxis;

    HomogeneousTransform zrot = HomogeneousTransform::Identity();
    zrot(0, 0) = std::cos(dof[DOF_rotdelgamma]);
    zrot(0, 1) = std::sin(dof[DOF_rotdelgamma]);
    zrot(1, 0) = -std::sin(dof[DOF_rotdelgamma]);
    zrot(1, 1) = std::cos(dof[DOF_rotdelgamma]);
    Coord rotdof2_axes = -(zrot * Mparent).block(1, 0, 1, 3).transpose();

    HomogeneousTransform yrot = HomogeneousTransform::Identity();
    yrot(0, 0) = std::cos(-dof[DOF_rotdelbeta]);
    yrot(0, 2) = std::sin(-dof[DOF_rotdelbeta]);
    yrot(2, 0) = -std::sin(-dof[DOF_rotdelbeta]);
    yrot(2, 2) = std::cos(-dof[DOF_rotdelbeta]);
    Coord rotdof1_axes =
        -(yrot * (zrot * Mparent)).block(0, 0, 1, 3).transpose();

    // dE/drot_i
    dsc_ddof[DOF_rotdelalpha] =
        rotdof1_axes.dot(f1) + f2.dot(rotdof1_axes.cross(end_pos));
    dsc_ddof[DOF_rotdelbeta] =
        rotdof2_axes.dot(f1) + f2.dot(rotdof2_axes.cross(end_pos));
    dsc_ddof[DOF_rotdelgamma] =
        rotdof3_axes.dot(f1) + f2.dot(rotdof3_axes.cross(end_pos));

    return dsc_ddof;
  }

  // Compute BOND derivatives from f1/f2
  //
  // The d derivatives are straightforward dot products of f2s
  //      (the downstream derivative sum)
  //
  // Other DOF derivatives use the Abe and Go "trick" that allows us to
  //    easily compute derivatives with respect to rotation about an axis.
  //
  // The phi_p and phi_c derivs are simply rotation about the X axis of the
  //    parent and child coordinate frame, respectively
  //
  // The theta derivs are more complex. Similar to jump derivs, we need to
  //    UNDO the phi_c rotation, and then take the Z axis of the child HT
  static auto EIGEN_DEVICE_FUNC bondDerivatives(
      Dofs dof,
      HomogeneousTransform M,
      HomogeneousTransform Mparent,
      Coord f1,
      Coord f2) -> Dofs {
    Dofs dsc_ddof =
        Dofs::Constant(0);  //(NAN); <- chnage to 0 so deriv checks work wo mask

    Coord end_ppos = Mparent.block(3, 0, 1, 3).transpose();
    Coord end_cpos = M.block(3, 0, 1, 3).transpose();
    Coord phi_paxis = Mparent.block(0, 0, 1, 3).transpose();
    Coord phi_caxis = M.block(0, 0, 1, 3).transpose();

    // to get the theta axis, we need to undo the phi_c rotation (about x)
    HomogeneousTransform unrot_phic = HomogeneousTransform::Identity();
    unrot_phic(1, 1) = std::cos(-dof[DOF_phic]);
    unrot_phic(1, 2) = std::sin(-dof[DOF_phic]);
    unrot_phic(2, 1) = -std::sin(-dof[DOF_phic]);
    unrot_phic(2, 2) = std::cos(-dof[DOF_phic]);
    Coord theta_axis = (unrot_phic * M).block(2, 0, 1, 3).transpose();

    dsc_ddof[DOF_D] = phi_caxis.dot(f2);
    dsc_ddof[DOF_theta] =
        -theta_axis.dot(f1) - f2.dot(theta_axis.cross(end_ppos));
    dsc_ddof[DOF_phip] = -phi_paxis.dot(f1) - f2.dot(phi_paxis.cross(end_ppos));
    dsc_ddof[DOF_phic] = -phi_caxis.dot(f1) - f2.dot(phi_caxis.cross(end_cpos));

    return dsc_ddof;
  }

  // convert BOND dofs to HT
  //   * each bond has four dofs: [phi_p, theta, d, phi_c]
  //   * in the local frame:
  //      - phi_p and phi_c are a rotation about x
  //      - theta is a rotation about z
  //      - d is a translation along x
  //   * the matrix 'HT' is a composition:
  //        M = (
  //            rot(phi_p, [1,0,0])
  //            @ rot(theta, [0,0,1]
  //            @ trans(d, [1,0,0])
  //            @ rot(phi_c, [1,0,0])
  //        )
  static auto EIGEN_DEVICE_FUNC bondTransform(Dofs dof) {
    Real cpp = std::cos(dof[DOF_phip]);
    Real spp = std::sin(dof[DOF_phip]);
    Real cpc = std::cos(dof[DOF_phic]);
    Real spc = std::sin(dof[DOF_phic]);
    Real cth = std::cos(dof[DOF_theta]);
    Real sth = std::sin(dof[DOF_theta]);
    Real d = dof[DOF_D];

    HomogeneousTransform HT;
    HT(0, 0) = cth;
    HT(1, 0) = -cpc * sth;
    HT(2, 0) = spc * sth;
    HT(3, 0) = d * cth;
    HT(0, 1) = cpp * sth;
    HT(1, 1) = cpc * cpp * cth - spc * spp;
    HT(2, 1) = -cpp * cth * spc - cpc * spp;
    HT(3, 1) = d * cpp * sth;
    HT(0, 2) = spp * sth;
    HT(1, 2) = cpp * spc + cpc * cth * spp;
    HT(2, 2) = cpc * cpp - cth * spc * spp;
    HT(3, 2) = d * spp * sth;
    HT(0, 3) = 0;
    HT(1, 3) = 0;
    HT(2, 3) = 0;
    HT(3, 3) = 1;

    return HT;
  }

  // HTs -> BOND dofs
  //
  // Given the matrix definition in BondTransforms, we calculate the dofs
  //   that give rise to this HT.
  // A special case below handles a "singularity," that is, a configuration
  //   where there are multiple parameterizations that give the same HT
  // Specifically, when theta==0, the rx rotation can be put into
  //   phi_c or phi_p (we use phi_c)
  static auto EIGEN_DEVICE_FUNC invBondTransform(HomogeneousTransform M) {
    Dofs dof =
        Dofs::Constant(0);  //(NAN); <- chnage to 0 so deriv checks work wo mask

    if (std::fabs(M(0, 0) - 1) < 1e-6) {
      dof[DOF_phip] = 0.0;
      dof[DOF_phic] = std::atan2(M(1, 2), M(1, 1));
      dof[DOF_theta] = 0.0;
    } else {
      dof[DOF_phip] = std::atan2(M(0, 2), M(0, 1));
      dof[DOF_phic] = std::atan2(-M(2, 0), -M(1, 0));
      dof[DOF_theta] =
          std::atan2(std::sqrt(M(1, 0) * M(1, 0) + M(2, 0) * M(2, 0)), M(0, 0));
    }
    dof[DOF_D] = M.bottomLeftCorner(1, 3).norm();

    return dof;
  }

  // convert JUMP dofs to HT
  //   * each jump has _9_ parameters:
  //     - 3 translational  [0-2]
  //     - 3 rotational deltas [3-5]
  //     - 3 rotational [6-8]
  //   * only the rotational deltas are exposed to minimization
  //     - RBdel_* is meant to be reset to zero at the beginning of a
  //       minimization trajectory, as when parameters are near 0,
  //       minimization is well-behaved.
  //   * translations are represented as an offset in X,Y,Z
  //   * rotations and rotational deltas are ZYX Euler angles.
  //      - that is, a rotation about Z, then Y, then X.
  //   * the matrix HT is a composition:
  //      M = trans( RBx, RBy, RBz)
  //          @ roteuler( RBdel_alpha, RBdel_alpha, RBdel_alpha)
  //          @ roteuler( RBalpha, RBalpha, RBalpha)
  static auto EIGEN_DEVICE_FUNC jumpTransform(Dofs dof) {
    Real si = std::sin(dof[DOF_rotdelalpha]);
    Real sj = std::sin(dof[DOF_rotdelbeta]);
    Real sk = std::sin(dof[DOF_rotdelgamma]);
    Real ci = std::cos(dof[DOF_rotdelalpha]);
    Real cj = std::cos(dof[DOF_rotdelbeta]);
    Real ck = std::cos(dof[DOF_rotdelgamma]);
    Real cc = ci * ck;
    Real cs = ci * sk;
    Real sc = si * ck;
    Real ss = si * sk;

    HomogeneousTransform HTdelta;
    HTdelta(0, 0) = cj * ck;
    HTdelta(1, 0) = sj * sc - cs;
    HTdelta(2, 0) = sj * cc + ss;
    HTdelta(0, 1) = cj * sk;
    HTdelta(1, 1) = sj * ss + cc;
    HTdelta(2, 1) = sj * cs - sc;
    HTdelta(0, 2) = -sj;
    HTdelta(1, 2) = cj * si;
    HTdelta(2, 2) = cj * ci;

    HTdelta(3, 0) = dof[DOF_RBx];
    HTdelta(3, 1) = dof[DOF_RBy];
    HTdelta(3, 2) = dof[DOF_RBz];
    HTdelta(0, 3) = HTdelta(1, 3) = HTdelta(2, 3) = 0;
    HTdelta(3, 3) = 1;

    si = std::sin(dof[DOF_rotalpha]);
    sj = std::sin(dof[DOF_rotbeta]);
    sk = std::sin(dof[DOF_rotgamma]);
    ci = std::cos(dof[DOF_rotalpha]);
    cj = std::cos(dof[DOF_rotbeta]);
    ck = std::cos(dof[DOF_rotgamma]);
    cc = ci * ck;
    cs = ci * sk;
    sc = si * ck;
    ss = si * sk;

    HomogeneousTransform HTglobal;
    HTglobal(0, 0) = cj * ck;
    HTglobal(1, 0) = sj * sc - cs;
    HTglobal(2, 0) = sj * cc + ss;
    HTglobal(0, 1) = cj * sk;
    HTglobal(1, 1) = sj * ss + cc;
    HTglobal(2, 1) = sj * cs - sc;
    HTglobal(0, 2) = -sj;
    HTglobal(1, 2) = cj * si;
    HTglobal(2, 2) = cj * ci;

    HTglobal(3, 0) = HTglobal(3, 1) = HTglobal(3, 2) = 0;
    HTglobal(0, 3) = HTglobal(1, 3) = HTglobal(2, 3) = 0;
    HTglobal(3, 3) = 1;

    HomogeneousTransform HT = HTglobal * HTdelta;

    return HT;
  }

  // HTs -> JUMP dofs
  //
  // Given the matrix definition in JumpTransforms, we calculate the dofs
  //   that give rise to this HT.
  // A special case handles the problematic region where cos(beta)=0.
  // In this case, the alpha and gamma rotation are coincident so
  //    we assign all rotation to alpha.
  //
  // Since RB and RBdel are redundant, this function always returns its
  //    non-zero components into RB, and RBdel is always 0
  static auto EIGEN_DEVICE_FUNC invJumpTransform(HomogeneousTransform M) {
    Dofs dof;

    dof[DOF_RBx] = M(3, 0);
    dof[DOF_RBy] = M(3, 1);
    dof[DOF_RBz] = M(3, 2);
    dof[DOF_rotdelalpha] = dof[DOF_rotdelbeta] = dof[DOF_rotdelgamma] = 0.0;

    Real cy = std::sqrt(M(0, 0) * M(0, 0) + M(0, 1) * M(0, 1));
    if (cy < 1e-6) {
      dof[DOF_rotalpha] = std::atan2(-M(2, 1), M(1, 1));
      dof[DOF_rotbeta] = std::atan2(-M(0, 2), cy);
      dof[DOF_rotgamma] = 0.0;
    } else {
      dof[DOF_rotalpha] = std::atan2(M(1, 2), M(2, 2));
      dof[DOF_rotbeta] = std::atan2(-M(0, 2), cy);
      dof[DOF_rotgamma] = std::atan2(M(0, 1), M(0, 0));
    }

    return dof;
  }

  static auto EIGEN_DEVICE_FUNC
  hts_from_frames(Coord ori, Coord a, Coord b, Coord c) {
    HomogeneousTransform ht;

    Coord xaxis = (a - b).normalized();
    Coord zaxis = xaxis.cross(c - a).normalized();
    Coord yaxis = zaxis.cross(xaxis).normalized();

    ht.block(0, 0, 1, 3) = xaxis.transpose();
    ht.block(1, 0, 1, 3) = yaxis.transpose();
    ht.block(2, 0, 1, 3) = zaxis.transpose();
    ht.block(3, 0, 1, 3) = ori.transpose();
    ht.col(3) = Eigen::Matrix<Real, 1, 4>(0, 0, 0, 1);

    return ht;
  }

  // note: not a proper inversion; assumes the R component is a
  //   proper rotation
  static auto EIGEN_DEVICE_FUNC ht_inv(HomogeneousTransform ht) {
    HomogeneousTransform htinv;
    htinv.topLeftCorner(3, 3) = ht.topLeftCorner(3, 3).transpose();
    htinv.bottomLeftCorner(1, 3) =
        -ht.bottomLeftCorner(1, 3) * htinv.topLeftCorner(3, 3);
    htinv.col(3) = Eigen::Matrix<Real, 1, 4>(0, 0, 0, 1);
    return htinv;
  }
};

#undef Dofs
#undef HomogeneousTransform
#undef QuatTranslation
#undef Coord

}  // namespace kinematics
}  // namespace tmol
