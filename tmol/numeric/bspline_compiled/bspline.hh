// N-dimensional B-spline interpolation with periodic boundary conditions.
// From:
//    Thévenaz, Philippe, Thierry Blu, and Michael Unser.
//    "Interpolation revisited [medical images application]."
//    IEEE Transactions on medical imaging 19.7 (2000): 739-758.
//    http://bigwww.epfl.ch/publications/thevenaz0002.pdf

// Dimension- and degree-templated, header-only implementation.

#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

#include <pybind11/pybind11.h>

#include <tuple>

namespace tmol {
namespace numeric {
namespace bspline {

template <std::size_t NDIM, std::size_t DEGREE, tmol::Device D, typename Real, typename Int>
struct ndspline {
  static auto square(Real v) -> Real { return v * v; }
  static auto cube(Real v) -> Real { return v * v * v; }

  // 1D stripe setter
  static auto EIGEN_DEVICE_FUNC _put_line(
      TView<Real, NDIM, D> coeffs,
      TView<Real, 1, D> line,
      Int start,
      Int step) {
    Real* vptr = &coeffs.data()[start];
    for (int i = 0; i < line.size(0); i++) {
      *vptr = line[i];
      vptr = &vptr[step];
    }
  }

  // 1D stripe getter
  static auto EIGEN_DEVICE_FUNC _get_line(
      TView<Real, NDIM, D> coeffs,
      TView<Real, 1, D> line,
      Int start,
      Int step) {
    Real* vptr = &coeffs.data()[start];
    for (int i = 0; i < line.size(0); i++) {
      line[i] = *vptr;
      vptr = &vptr[step];
    }
  }

  static auto EIGEN_DEVICE_FUNC
  _init_causal_coeff(TView<Real, 1, D> line, Real pole) {
    // inplace calculation of "cell 0" coefficient
    // ** currently, initialization corresponds to periodic boundaries
    //    (if one were to add alternate boundary conditions, this would be the
    //    place)
    Int N = line.size(0);
    Real tol = 4 * std::numeric_limits<Real>::epsilon();
    Int horiz = (Int)std::ceil(std::log(tol) / std::log(std::fabs(pole)));
    Real zn = pole;

    if (horiz < N) {
      for (int i = 1; i < horiz; ++i) {
        line[0] += zn * line[N - i];
        zn *= pole;
      }
    } else {
      for (int i = 1; i < N; ++i) {
        line[0] += zn * line[N - i];
        zn *= pole;
      }
      line[0] /= (1 - zn);
    }
  }

  static auto EIGEN_DEVICE_FUNC
  _init_anticausal_coeff(TView<Real, 1, D> line, Real pole) {
    // inplace calculation of "cell (N-1)" coefficient
    // ** currently, initialization corresponds to periodic boundaries
    Int N = line.size(0);
    Real tol = 4 * std::numeric_limits<Real>::epsilon();
    Int horiz = (Int)std::ceil(std::log(tol) / std::log(std::fabs(pole)));
    Real zn = pole;

    if (horiz < N) {
      for (int i = 0; i < horiz; ++i) {
        line[N - 1] += zn * line[i];
        zn *= pole;
      }
      line[N - 1] *= -pole;
    } else {
      for (int i = 0; i < N; ++i) {
        line[N - 1] += zn * line[i];
        zn *= pole;
      }
      line[N - 1] *= -pole / (1.0 - zn);
    }
  }

  static auto EIGEN_DEVICE_FUNC
  _convert_interp_coeffs(TView<Real, 1, D> line, TView<Real, 1, D> poles) {
    // interpolation coefficients along one line (in place)
    Real lambda = 1.0;
    Int N = line.size(0);
    Int npoles = poles.size(0);

    if (N == 1) return;

    for (int i = 0; i < npoles; i++) {
      lambda *= (1.0 - poles[i]) * (1.0 - 1.0 / poles[i]);
    }

    for (int j = 0; j < N; j++) {
      line[j] *= lambda;
    }

    for (int i = 0; i < npoles; i++) {
      // set line[0]
      _init_causal_coeff(line, poles[i]);
      for (int j = 1; j < N; j++) {
        line[j] += poles[i] * line[j - 1];
      }

      // set line[N-1]
      _init_anticausal_coeff(line, poles[i]);
      for (int j = N - 2; j >= 0; j--) {
        line[j] = poles[i] * (line[j + 1] - line[j]);
      }
    }
  }

  static auto EIGEN_DEVICE_FUNC _get_poles() -> TPack<Real, 1, D> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    TPack<Real, 1, D> poles_t;
    if (DEGREE == 2) {
      poles_t = TPack<Real, 1, D>::empty(1);
      auto poles = poles_t.view;
      poles[0] = std::sqrt(8.0) - 3.0;
      return poles_t;
    } else if (DEGREE == 3) {
      poles_t = TPack<Real, 1, D>::empty(1);
      auto poles = poles_t.view;
      poles[0] = std::sqrt(3.0) - 2.0;
      return poles_t;
    } else if (DEGREE == 4) {
      poles_t = TPack<Real, 1, D>::empty(2);
      auto poles = poles_t.view;
      poles[0] =
          std::sqrt(664.0 - std::sqrt(438976.0)) + std::sqrt(304.0) - 19.0;
      poles[1] =
          std::sqrt(664.0 + std::sqrt(438976.0)) - std::sqrt(304.0) - 19.0;
      return poles_t;
    } else {
      poles_t = TPack<Real, 1, D>::empty(2);
      auto poles = poles_t.view;
      poles[0] = std::sqrt(135.0 / 2.0 - std::sqrt(17745.0 / 4.0))
                 + std::sqrt(105.0 / 4.0) - 13.0 / 2.0;
      poles[1] = std::sqrt(135.0 / 2.0 + std::sqrt(17745.0 / 4.0))
                 - std::sqrt(105.0 / 4.0) - 13.0 / 2.0;
    }
    return poles_t;
  }

  // compute coefficients main
  static auto EIGEN_DEVICE_FUNC computeCoeffs(TView<Real, NDIM, D> values)
      -> TPack<Real, NDIM, D> {
    // 'values' must be C-contiguous.
    //   -> this is currently guaranteed by TView
    auto coeffs_t = TPack<Real, NDIM, D>::zeros_like(values);
    auto coeffs = coeffs_t.view;

    auto poles_t = _get_poles();
    auto poles = poles_t.view;

    Int npoints = 1;
    for (int i = 0; i < NDIM; ++i) {
      npoints *= coeffs.size(i);
    }

    // copy values -> coeffs
    for (int i = 0; i < npoints; ++i) {
      coeffs.data()[i] = values.data()[i];
    }

    // loop over all dimensions
    for (int dim = 0; dim < NDIM; ++dim) {
      // allocate memory for one slice in this dimension
      auto line_i_t = TPack<Real, 1, D>::zeros(coeffs.size(dim));
      auto line_i = line_i_t.view;

      // loop over all dimensions _except_ this one,
      //   process 1D stripes along this dimension
      Int step_inner = 1, step_outer = 1;
      for (int i = dim + 1; i < NDIM; ++i) {
        step_inner *= coeffs.size(i);
      }
      for (int i = dim - 1; i >= 0; --i) {
        step_outer *= coeffs.size(i);
      }
      Int start = 0;
      for (int outer = 0; outer < step_outer; ++outer) {
        for (int inner = 0; inner < step_inner; ++inner) {
          // pybind11::print (start+inner, step_inner);
          _get_line(coeffs, line_i, start + inner, step_inner);
          _convert_interp_coeffs(line_i, poles);
          _put_line(coeffs, line_i, start + inner, step_inner);
        }
        start += step_inner * coeffs.size(dim);
      }
    }

    return (coeffs_t);
  }

  static auto EIGEN_DEVICE_FUNC _get_weights(Eigen::Matrix<Real, NDIM, 1> frac)
      -> TPack<Real, 2, D> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    TPack<Real, 2, D> wts_t = TPack<Real, 2, D>::empty({NDIM, DEGREE + 1});
    TView<Real, 2, D> wts = wts_t.view;
    if (DEGREE == 2) {
      for (int dim = 0; dim < NDIM; ++dim) {
        wts[dim][1] = 3.0 / 4.0 - square(frac[dim]);
        wts[dim][2] = (1.0 / 2.0) * (frac[dim] - wts[dim][1] + 1.0);
        wts[dim][0] = 1.0 - wts[dim][1] - wts[dim][2];
      }
    } else if (DEGREE == 3) {
      for (int dim = 0; dim < NDIM; ++dim) {
        wts[dim][3] = (1.0 / 6.0) * cube(frac[dim]);
        wts[dim][0] = (1.0 / 6.0) + (1.0 / 2.0) * frac[dim] * (frac[dim] - 1.0)
                      - wts[dim][3];
        wts[dim][2] = frac[dim] + wts[dim][0] - 2.0 * wts[dim][3];
        wts[dim][1] = 1.0 - wts[dim][0] - wts[dim][2] - wts[dim][3];
      }
    } else if (DEGREE == 4) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real t = (1.0 / 6.0) * square(frac[dim]);
        Real t0 = frac[dim] * (t - 11.0 / 24.0);
        Real t1 = 19.0 / 96.0 + square(frac[dim]) * (1.0 / 4.0 - t);
        wts[dim][0] = (1.0 / 24.0) * cube(1.0 / 2.0 - frac[dim]);
        wts[dim][1] = t1 + t0;
        wts[dim][3] = t1 - t0;
        wts[dim][4] = wts[dim][0] + t0 + (1.0 / 2.0) * frac[dim];
        wts[dim][2] =
            1.0 - wts[dim][0] - wts[dim][1] - wts[dim][3] - wts[dim][4];
      }
    } else if (DEGREE == 5) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real w2 = frac[dim] * frac[dim];
        wts[dim][5] = (1.0 / 120.0) * cube(frac[dim]) * square(frac[dim]);
        Real w4 = square(square(frac[dim]) - frac[dim]);
        Real w = frac[dim] - 0.5;
        Real t = w2 * (w2 - 3.0);
        wts[dim][0] = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - wts[dim][5];
        Real t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
        Real t1 = (-1.0 / 12.0) * w * (t + 4.0);
        wts[dim][2] = t0 + t1;
        wts[dim][3] = t0 - t1;
        t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
        t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
        wts[dim][1] = t0 + t1;
        wts[dim][4] = t0 - t1;
      }
    }
    return wts_t;
  }

  static auto EIGEN_DEVICE_FUNC _get_dweights(Eigen::Matrix<Real, NDIM, 1> frac)
      -> TPack<Real, 2, D> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    TPack<Real, 2, D> dwts_t = TPack<Real, 2, D>::empty({NDIM, DEGREE + 1});
    TView<Real, 2, D> dwts = dwts_t.view;
    if (DEGREE == 2) {
      for (int dim = 0; dim < NDIM; ++dim) {
        dwts[dim][1] = -2.0 * frac[dim];
        dwts[dim][2] = (1.0 / 2.0) * (1.0 - dwts[dim][1]);
        dwts[dim][0] = -dwts[dim][1] - dwts[dim][2];
      }
    } else if (DEGREE == 3) {
      for (int dim = 0; dim < NDIM; ++dim) {
        dwts[dim][3] = (1.0 / 2.0) * square(frac[dim]);
        dwts[dim][0] = (frac[dim] - 0.5) - dwts[dim][3];
        dwts[dim][2] = 1.0 + dwts[dim][0] - 2.0 * dwts[dim][3];
        dwts[dim][1] = -dwts[dim][0] - dwts[dim][2] - dwts[dim][3];
      }
    } else if (DEGREE == 4) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real t = (1.0 / 6.0) * square(frac[dim]);
        Real dt = (1.0 / 3.0) * frac[dim];
        Real dt0 = (t - 11.0 / 24.0) + frac[dim] * dt;
        Real dt1 = 2.0 * frac[dim] * (1.0 / 4.0 - t) - square(frac[dim]) * dt;
        dwts[dim][0] = -(1.0 / 8.0) * square(1.0 / 2.0 - frac[dim]);
        dwts[dim][1] = dt1 + dt0;
        dwts[dim][3] = dt1 - dt0;
        dwts[dim][4] = dwts[dim][0] + dt0 + (1.0 / 2.0);
        dwts[dim][2] =
            -dwts[dim][0] - dwts[dim][1] - dwts[dim][3] - dwts[dim][4];
      }
    } else if (DEGREE == 5) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real w2 = frac[dim] * frac[dim];
        dwts[dim][5] = (1.0 / 24.0) * square(square(frac[dim]));
        Real w4 = square(square(frac[dim]) - frac[dim]);
        Real dw4 =
            2.0 * (square(frac[dim]) - frac[dim]) * (2.0 * frac[dim] - 1.0);
        Real w = frac[dim] - 0.5;
        Real t = w2 * (w2 - 3.0);
        Real dt = 4.0 * cube(frac[dim]) - 6.0 * frac[dim];
        dwts[dim][0] = (1.0 / 24.0) * (2.0 * frac[dim] + dw4) - dwts[dim][5];
        Real dt0 = (1.0 / 24.0) * (4.0 * cube(frac[dim]) - 10.0 * frac[dim]);
        Real dt1 = (-1.0 / 12.0) * (w * dt + (t + 4.0));
        dwts[dim][2] = dt0 + dt1;
        dwts[dim][3] = dt0 - dt1;
        dt0 = -(1.0 / 16.0) * dt;
        dt1 = (1.0 / 24.0) * ((w4 - w2 - 5.0) + w * (dw4 - 2 * frac[dim]));
        dwts[dim][1] = dt0 + dt1;
        dwts[dim][4] = dt0 - dt1;
      }
    }
    return dwts_t;
  }

  static auto EIGEN_DEVICE_FUNC
  interpolate(TView<Real, NDIM, D> coeffs, TView<Real, 1, D> X)
      -> std::tuple<Real, TPack<Real, 1, D> > {
    typedef Eigen::Matrix<Real, NDIM, 1> RealN;
    typedef Eigen::Matrix<Int, NDIM, 1> IntN;

    Real interp = 0;

    // get indices to loop over
    Int nprods = 1;
    IntN idx;
    RealN frac;
    for (int dim = 0; dim < NDIM; ++dim) {
      idx[dim] = (Int)std::floor(X[dim]);
      frac[dim] = X[dim] - idx[dim];
      idx[dim] -= (DEGREE / 2);
      nprods *= (DEGREE + 1);
    }

    // get weights in each dimension
    TPack<Real, 2, D> wts_t = _get_weights(frac);
    TView<Real, 2, D> wts = wts_t.view;
    TPack<Real, 2, D> dwts_t = _get_dweights(frac);
    TView<Real, 2, D> dwts = dwts_t.view;

    TPack<Real, 1, D> dinterp_dX_t = TPack<Real, 1, D>::zeros(NDIM);
    TView<Real, 1, D> dinterp_dX = dinterp_dX_t.view;

    // do the dot product
    for (int pt = 0; pt < nprods; ++pt) {
      Real weight = 1;
      RealN dweight = RealN::Constant(1.0);

      Int stride = 1;
      Int pt_indexer = pt, idx_ij = 0;
      for (int dim = NDIM - 1; dim >= 0; --dim) {
        Int idx_box_i = pt_indexer % (DEGREE + 1);
        Int idx_i = (idx[dim] + idx_box_i) % coeffs.size(dim);
        if (idx_i < 0) idx_i += coeffs.size(dim);
        idx_ij += stride * idx_i;
        stride *= coeffs.size(dim);
        weight *= wts[dim][idx_box_i];

        for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
          dweight[d_dim] *=
              (d_dim == dim ? dwts[dim][idx_box_i] : wts[dim][idx_box_i]);
        }

        pt_indexer /= (DEGREE + 1);
      }
      Real coeff_i = coeffs.data()[idx_ij];
      interp += weight * coeff_i;
      for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
        dinterp_dX[d_dim] += dweight[d_dim] * coeff_i;
      }
    }
    return {interp, dinterp_dX_t};
  }
};

}  // namespace bspline
}  // namespace numeric
}  // namespace tmol
