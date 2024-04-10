#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

#define Real3 Vec<Real, 3>

template <typename Real>
struct distance {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB); }
  };

  static def V(Real3 A, Real3 B) -> Real { return (A - B).norm(); }

  static def V_dV(Real3 A, Real3 B) -> V_dV_T {
    Real3 delta = (A - B);
    Real V = delta.norm();

    if (V != 0) {
      return {V, delta / V, -delta / V};
    } else {
      // Correct for nan, gradient is discontinuous across dist = 0
      return {V, Real3({0.0, 0.0, 0.0}), Real3({0.0, 0.0, 0.0})};
    }
  }
};

template <typename Real>
struct interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB); }
  };

  static def V(Real3 A, Real3 B) -> Real {
    auto CR = A.cross(B);
    auto z_unit = CR.normalized();

    auto A_norm = A.norm();
    auto B_norm = B.norm();

    return 2 * std::atan2(CR.dot(z_unit), A_norm * B_norm + A.dot(B));
  }

  static def V_dV(Real3 A, Real3 B) -> V_dV_T {
    auto CR = A.cross(B);
    auto z_unit = CR.normalized();

    auto A_norm = A.norm();
    auto B_norm = B.norm();

    return {
        2 * std::atan2(CR.dot(z_unit), A_norm * B_norm + A.dot(B)),
        (A / A_norm).cross(z_unit) / A_norm,
        -(B / B_norm).cross(z_unit) / B_norm};
  }
};

template <typename Real>
struct pt_interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;
    Vec<Real, 3> dV_dC;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB, dV_dC); }
  };
  static def V(Real3 A, Real3 B, Real3 C) -> Real {
    Real3 BA = A - B;
    Real3 BC = C - B;

    return interior_angle<Real>::V(BA, BC);
  }

  static def V_dV(Real3 A, Real3 B, Real3 C) -> V_dV_T {
    Real3 BA = A - B;
    Real3 BC = C - B;

    auto angle = interior_angle<Real>::V_dV(BA, BC);
    return {angle.V, angle.dV_dA, -(angle.dV_dA + angle.dV_dB), angle.dV_dB};
  }
};

template <typename Real>
struct cos_interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB); }
  };

  static def V(Real3 A, Real3 B) -> Real {
    return A.dot(B) / (A.norm() * B.norm());
  }

  static def V_dV(Real3 A, Real3 B) -> V_dV_T {
    auto A_norm = A.norm();
    auto B_norm = B.norm();
    auto AB_norm = A.norm() * B.norm();

    auto cosAB = (A.dot(B) / AB_norm);

    return {
        cosAB,
        -cosAB * A / (A_norm * A_norm) + B / AB_norm,
        -cosAB * B / (B_norm * B_norm) + A / AB_norm};
  }
};

template <typename Real>
struct pt_cos_interior_angle {
  struct V_dV_T {
    Real V;
    Vec<Real, 3> dV_dA;
    Vec<Real, 3> dV_dB;
    Vec<Real, 3> dV_dC;

    def astuple() -> auto { return make_tuple(V, dV_dA, dV_dB, dV_dC); }
  };

  static def V(Real3 A, Real3 B, Real3 C) -> Real {
    Real3 BA = A - B;
    Real3 BC = C - B;

    return cos_interior_angle<Real>::V(BA, BC);
  }

  static def V_dV(Real3 A, Real3 B, Real3 C) -> V_dV_T {
    Real3 BA = A - B;
    Real3 BC = C - B;
    Real V;
    Real3 dV_dBA, dV_dBC;

    auto angle = cos_interior_angle<Real>::V_dV(BA, BC);

    return {angle.V, angle.dV_dA, -(angle.dV_dA + angle.dV_dB), angle.dV_dB};
  }
};

template <typename Real>
struct dihedral_angle {
  struct V_dV_T {
    Real V;
    Real3 dV_dI;
    Real3 dV_dJ;
    Real3 dV_dK;
    Real3 dV_dL;

    def astuple() -> auto { return make_tuple(V, dV_dI, dV_dJ, dV_dK, dV_dL); }
  };

  static def V(Real3 I, Real3 J, Real3 K, Real3 L) -> Real {
    // Blondel A, Karplus M. New formulation for derivatives of torsion angles
    // and improper torsion angles in molecular mechanics: Elimination of
    // singularities. J Comput Chem. 1996;17: 1132–1141.
    auto F = I - J;
    auto G = J - K;
    auto H = L - K;

    auto A = F.cross(G);
    auto B = H.cross(G);

    Real sign = G.dot(A.cross(B)) >= 0 ? -1.0 : 1.0;

    return sign
           * std::acos(std::fmax(
               (Real)-1.0,
               std::fmin(A.dot(B) / (A.norm() * B.norm()), (Real)1.0)));
  }

  static def V_dV(Real3 I, Real3 J, Real3 K, Real3 L) -> V_dV_T {
    // Blondel A, Karplus M. New formulation for derivatives of torsion angles
    // and improper torsion angles in molecular mechanics: Elimination of
    // singularities. J Comput Chem. 1996;17: 1132–1141.
    auto F = I - J;
    auto G = J - K;
    auto H = L - K;

    auto A = F.cross(G);
    auto B = H.cross(G);

    Real sign = G.dot(A.cross(B)) >= 0 ? -1.0 : 1.0;
    auto V = sign
             * std::acos(std::fmax(
                 (Real)-1.0,
                 std::fmin(A.dot(B) / (A.norm() * B.norm()), (Real)1.0)));

    return {
        V,
        -(G.norm() / A.dot(A)) * A,
        G.norm() / A.dot(A) * A + F.dot(G) / (A.dot(A) * G.norm()) * A
            - (H.dot(G) / (B.dot(B) * G.norm())) * B,
        -G.norm() / B.dot(B) * B - F.dot(G) / (A.dot(A) * G.norm()) * A
            + (H.dot(G) / (B.dot(B) * G.norm())) * B,
        G.norm() / B.dot(B) * B};
  }
};

#undef Real3
#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
