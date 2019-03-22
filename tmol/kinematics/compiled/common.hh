// Device-agnostic common numeric routines for kinematics.

#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

#include <pybind11/pybind11.h>

namespace tmol {
namespace kinematics {

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define QuatTranslation Eigen::Matrix<Real, 7, 1>

enum DOFtype { ROOT = 0, JUMP, BOND };

template <tmol::Device D, typename Real, typename Int>
struct common {
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
  static auto EIGEN_DEVICE_FUNC bondTransform(Eigen::Matrix<Real, 9, 1> dof) {
    Real cpp = std::cos(dof[0]);
    Real spp = std::sin(dof[0]);
    Real cpc = std::cos(dof[3]);
    Real spc = std::sin(dof[3]);
    Real cth = std::cos(dof[1]);
    Real sth = std::sin(dof[1]);
    Real d = dof[2];

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
  static auto EIGEN_DEVICE_FUNC jumpTransform(Eigen::Matrix<Real, 9, 1> dof) {
    Real si = std::sin(dof[3]);
    Real sj = std::sin(dof[4]);
    Real sk = std::sin(dof[5]);
    Real ci = std::cos(dof[3]);
    Real cj = std::cos(dof[4]);
    Real ck = std::cos(dof[5]);
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

    HTdelta(3, 0) = dof[0];
    HTdelta(3, 1) = dof[1];
    HTdelta(3, 2) = dof[2];
    HTdelta(0, 3) = HTdelta(1, 3) = HTdelta(2, 3) = 0;
    HTdelta(3, 3) = 1;

    si = std::sin(dof[6]);
    sj = std::sin(dof[7]);
    sk = std::sin(dof[8]);
    ci = std::cos(dof[6]);
    cj = std::cos(dof[7]);
    ck = std::cos(dof[8]);
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

  // convert a HT (16 elements) to a quaternion+translation (7 elts)
  static auto EIGEN_DEVICE_FUNC ht2quat_trans(HomogeneousTransform HT) {
    QuatTranslation retval;

    // rotation
    Real S = 0.0;
    if (HT(0, 0) == 1.0 && HT(1, 1) == 1.0 && HT(2, 2) == 1.0) {
      retval[0] = 0;
      retval[1] = 0;
      retval[2] = 0;
      retval[3] = 1;
    } else if (HT(0, 0) > HT(1, 1) && HT(0, 0) > HT(2, 2)) {
      S = std::sqrt(1.0 + HT(0, 0) - HT(1, 1) - HT(2, 2)) * 2;
      retval[0] = 0.25 * S;
      retval[1] = (HT(1, 0) + HT(0, 1)) / S;
      retval[2] = (HT(2, 0) + HT(0, 2)) / S;
      retval[3] = (HT(2, 1) - HT(1, 2)) / S;
    } else if (HT(1, 1) > HT(2, 2)) {
      S = std::sqrt(1.0 + HT(1, 1) - HT(0, 0) - HT(2, 2)) * 2;
      retval[0] = (HT(1, 0) + HT(0, 1)) / S;
      retval[1] = 0.25 * S;
      retval[2] = (HT(2, 1) + HT(1, 2)) / S;
      retval[3] = (HT(0, 2) - HT(2, 0)) / S;
    } else {
      S = std::sqrt(1.0 + HT(2, 2) - HT(0, 0) - HT(1, 1)) * 2;
      retval[0] = (HT(2, 0) + HT(0, 2)) / S;
      retval[1] = (HT(2, 1) + HT(1, 2)) / S;
      retval[2] = 0.25 * S;
      retval[3] = (HT(1, 0) - HT(0, 1)) / S;
    }

    // translation
    retval[4] = HT(3, 0);
    retval[5] = HT(3, 1);
    retval[6] = HT(3, 2);

    return retval;
  }

  // convert a quaternion+translation (7 elts) to a HT (16 elements)
  static auto EIGEN_DEVICE_FUNC quat_trans2ht(QuatTranslation qt) {
    HomogeneousTransform retval = HomogeneousTransform::Identity();

    retval(0, 0) = 1 - 2 * (qt[1] * qt[1] + qt[2] * qt[2]);
    retval(0, 1) = 2 * (qt[0] * qt[1] - qt[2] * qt[3]);
    retval(0, 2) = 2 * (qt[0] * qt[2] + qt[1] * qt[3]);
    retval(1, 0) = 2 * (qt[0] * qt[1] + qt[2] * qt[3]);
    retval(1, 1) = 1 - 2 * (qt[0] * qt[0] + qt[2] * qt[2]);
    retval(1, 2) = 2 * (qt[1] * qt[2] - qt[0] * qt[3]);
    retval(2, 0) = 2 * (qt[0] * qt[2] - qt[1] * qt[3]);
    retval(2, 1) = 2 * (qt[1] * qt[2] + qt[0] * qt[3]);
    retval(2, 2) = 1 - 2 * (qt[0] * qt[0] + qt[1] * qt[1]);

    // translation
    retval(3, 0) = qt[4];
    retval(3, 1) = qt[5];
    retval(3, 2) = qt[6];

    return retval;
  }

  static void EIGEN_DEVICE_FUNC quat_trans_norm(QuatTranslation qt) {
    Real S = std::sqrt(
        qt[0] * qt[0] + qt[1] * qt[1] + qt[2] * qt[2] + qt[3] * qt[3]);
    qt[0] /= S;
    qt[1] /= S;
    qt[2] /= S;
    qt[3] /= S;
  }

  // compose two quat/trans pairs
  static auto EIGEN_DEVICE_FUNC
  quat_trans_compose(QuatTranslation qt1, QuatTranslation qt2) {
    HomogeneousTransform ht1, ht2;
    ht1 = quat_trans2ht(qt1);
    ht2 = quat_trans2ht(qt2);

    ht1 = ht1 * ht2;
    qt1 = ht2quat_trans(ht1);
    quat_trans_norm(qt1);

    return (qt1);
  }
};

#undef HomogeneousTransform
#undef QuatTranslation

}  // namespace kinematics
}  // namespace tmol
