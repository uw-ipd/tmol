#pragma once

// B-spline coefficient construction.
//
// Host-side setup: filters a parameter table into spline coefficients using a
// dynamically sized scratch buffer per row stripe, which is why it still uses
// Eigen. Kept out of bspline.hh -- which kernels include for interpolate() --
// so that the interpolation half stays free of Eigen and of dynamic
// allocation. Coefficients are computed once here and uploaded; nothing on a
// device calls this (it is only ever instantiated at Device::CPU, from
// bspline.pybind.cpp).

#include <Eigen/Core>

#include <tmol/numeric/bspline_compiled/bspline.hh>

namespace tmol {
namespace numeric {
namespace bspline {

template <
    std::size_t NDIM,
    std::size_t DEGREE,
    tmol::Device D,
    typename Real,
    typename Int>
struct ndspline_coeffs {
  // 1D stripe setter
  static auto TMOL_HD _put_line(
      TensorAccessor<Real, NDIM, D> coeffs,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> line,
      Int start,
      Int step) {
    Real* vptr = &coeffs.data()[start];
    for (int i = 0; i < line.size(); i++) {
      *vptr = line[i];
      vptr = &vptr[step];
    }
  }

  // 1D stripe getter
  static auto TMOL_HD _get_line(
      TensorAccessor<Real, NDIM, D> coeffs,
      Eigen::Matrix<Real, Eigen::Dynamic, 1>& line,
      Int start,
      Int step) {
    Real* vptr = &coeffs.data()[start];
    for (int i = 0; i < line.size(); i++) {
      line[i] = *vptr;
      vptr = &vptr[step];
    }
  }

  static auto TMOL_HD
  _init_causal_coeff(Eigen::Matrix<Real, Eigen::Dynamic, 1>& line, Real pole) {
    // inplace calculation of "cell 0" coefficient
    // ** currently, initialization corresponds to periodic boundaries
    //    (if one were to add alternate boundary conditions, this would be the
    //    place)
    Int N = line.size();
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

  static auto TMOL_HD _init_anticausal_coeff(
      Eigen::Matrix<Real, Eigen::Dynamic, 1>& line, Real pole) {
    // inplace calculation of "cell (N-1)" coefficient
    // ** currently, initialization corresponds to periodic boundaries
    Int N = line.size();
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
      // Stop before the last entry: it is the accumulator and has already
      // been included once. After N-1 terms, zn is pole^N.
      for (int i = 0; i < N - 1; ++i) {
        line[N - 1] += zn * line[i];
        zn *= pole;
      }
      line[N - 1] *= -pole / (1.0 - zn);
    }
  }

  static auto TMOL_HD _convert_interp_coeffs(
      Eigen::Matrix<Real, Eigen::Dynamic, 1>& line,
      Eigen::Matrix<Real, DEGREE / 2, 1> poles) {
    // interpolation coefficients along one line (in place)
    Real lambda = 1.0;
    Int N = line.size();
    Int npoles = poles.size();

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

  static auto TMOL_HD _get_poles() -> Eigen::Matrix<Real, DEGREE / 2, 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    Eigen::Matrix<Real, DEGREE / 2, 1> poles;
    if (DEGREE == 2) {
      poles[0] = std::sqrt(8.0) - 3.0;
    } else if (DEGREE == 3) {
      poles[0] = std::sqrt(3.0) - 2.0;
    } else if (DEGREE == 4) {
      poles[0] =
          std::sqrt(664.0 - std::sqrt(438976.0)) + std::sqrt(304.0) - 19.0;
      poles[1] =
          std::sqrt(664.0 + std::sqrt(438976.0)) - std::sqrt(304.0) - 19.0;
    } else {
      poles[0] = std::sqrt(135.0 / 2.0 - std::sqrt(17745.0 / 4.0))
                 + std::sqrt(105.0 / 4.0) - 13.0 / 2.0;
      poles[1] = std::sqrt(135.0 / 2.0 + std::sqrt(17745.0 / 4.0))
                 - std::sqrt(105.0 / 4.0) - 13.0 / 2.0;
    }
    return poles;
  }

  // compute coefficients (in-place)
  static auto TMOL_HD computeCoeffs(TView<Real, NDIM, D> values) {
    // 'values' must be C-contiguous.
    //   -> this is currently guaranteed by TensorAccessor
    auto poles = _get_poles();

    Int npoints = 1;
    for (int i = 0; i < NDIM; ++i) {
      npoints *= values.size(i);
    }

    typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RowStripe;
    RowStripe line_i(values.size(0));

    // loop over all dimensions
    for (int dim = 0; dim < NDIM; ++dim) {
      typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> RowStripe;

      // allocate memory for one slice in this dimension
      RowStripe line_i(values.size(dim));

      // loop over all dimensions _except_ this one,
      //   process 1D stripes along this dimension
      Int step_inner = 1, step_outer = 1;
      for (int i = dim + 1; i < NDIM; ++i) {
        step_inner *= values.size(i);
      }
      for (int i = dim - 1; i >= 0; --i) {
        step_outer *= values.size(i);
      }
      Int start = 0;
      for (int outer = 0; outer < step_outer; ++outer) {
        for (int inner = 0; inner < step_inner; ++inner) {
          // pybind11::print (start+inner, step_inner);
          _get_line(values, line_i, start + inner, step_inner);
          _convert_interp_coeffs(line_i, poles);
          _put_line(values, line_i, start + inner, step_inner);
        }
        start += step_inner * values.size(dim);
      }
    }

    return;
  }
};

}  // namespace bspline
}  // namespace numeric
}  // namespace tmol
