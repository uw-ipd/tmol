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

namespace tmol {
namespace numeric {
namespace bspline {

template <std::size_t NDIM, std::size_t DEGREE, tmol::Device D, typename Real, typename Int>
struct ndspline {
  // 1D stripe setter
  static auto EIGEN_DEVICE_FUNC _put_line(
      TView<Real, NDIM, D> coeffs,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> line,
      Int start,
      Int step) {
    Real *vptr = &coeffs.data()[start];
    for (int i = 0; i < line.size(); i++) {
      *vptr = line[i];
      vptr = &vptr[step];
    }
  }

  // 1D stripe getter
  static auto EIGEN_DEVICE_FUNC _get_line(
      TView<Real, NDIM, D> coeffs,
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &line,
      Int start,
      Int step) {
    Real *vptr = &coeffs.data()[start];
    for (int i = 0; i < line.size(); i++) {
      line[i] = *vptr;
      vptr = &vptr[step];
    }
  }

  static auto EIGEN_DEVICE_FUNC
  _init_causal_coeff(Eigen::Matrix<Real, Eigen::Dynamic, 1> &line, Real pole) {
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

  static auto EIGEN_DEVICE_FUNC _init_anticausal_coeff(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &line, Real pole) {
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
      for (int i = 0; i < N; ++i) {
        line[N - 1] += zn * line[i];
        zn *= pole;
      }
      line[N - 1] *= -pole / (1.0 - zn);
    }
  }

  static auto EIGEN_DEVICE_FUNC _convert_interp_coeffs(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &line,
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

  static auto EIGEN_DEVICE_FUNC _get_poles()
      -> Eigen::Matrix<Real, DEGREE / 2, 1> {
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
  static auto EIGEN_DEVICE_FUNC computeCoeffs(TView<Real, NDIM, D> values) {
    // 'values' must be C-contiguous.
    //   -> this is currently guaranteed by TView
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

  static auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE square(Real v) -> Real {
    return v * v;
  }
  static auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE cube(Real v) -> Real {
    return v * v * v;
  }

  static auto EIGEN_DEVICE_FUNC _get_weights(Eigen::Matrix<Real, NDIM, 1> frac)
      -> Eigen::Matrix<Real, NDIM, DEGREE + 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    Eigen::Matrix<Real, NDIM, DEGREE + 1> wts =
        Eigen::Matrix<Real, NDIM, DEGREE + 1>::Constant(0.0);

    if (DEGREE == 2) {
      for (int dim = 0; dim < NDIM; ++dim) {
        wts(dim, 1) = 3.0 / 4.0 - square(frac[dim]);
        wts(dim, 2) = (1.0 / 2.0) * (frac[dim] - wts(dim, 1) + 1.0);
        wts(dim, 0) = 1.0 - wts(dim, 1) - wts(dim, 2);
      }
    } else if (DEGREE == 3) {
      for (int dim = 0; dim < NDIM; ++dim) {
        wts(dim, 3) = (1.0 / 6.0) * cube(frac[dim]);
        wts(dim, 0) = (1.0 / 6.0) + (1.0 / 2.0) * frac[dim] * (frac[dim] - 1.0)
                      - wts(dim, 3);
        wts(dim, 2) = frac[dim] + wts(dim, 0) - 2.0 * wts(dim, 3);
        wts(dim, 1) = 1.0 - wts(dim, 0) - wts(dim, 2) - wts(dim, 3);
      }
    } else if (DEGREE == 4) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real t = (1.0 / 6.0) * square(frac[dim]);
        Real t0 = frac[dim] * (t - 11.0 / 24.0);
        Real t1 = 19.0 / 96.0 + square(frac[dim]) * (1.0 / 4.0 - t);
        wts(dim, 0) = (1.0 / 24.0) * cube(1.0 / 2.0 - frac[dim]);
        wts(dim, 1) = t1 + t0;
        wts(dim, 3) = t1 - t0;
        wts(dim, 4) = wts(dim, 0) + t0 + (1.0 / 2.0) * frac[dim];
        wts(dim, 2) =
            1.0 - wts(dim, 0) - wts(dim, 1) - wts(dim, 3) - wts(dim, 4);
      }
    } else if (DEGREE == 5) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real w2 = frac[dim] * frac[dim];
        wts(dim, 5) = (1.0 / 120.0) * cube(frac[dim]) * square(frac[dim]);
        Real w4 = square(square(frac[dim]) - frac[dim]);
        Real w = frac[dim] - 0.5;
        Real t = w2 * (w2 - 3.0);
        wts(dim, 0) = (1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) - wts(dim, 5);
        Real t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0);
        Real t1 = (-1.0 / 12.0) * w * (t + 4.0);
        wts(dim, 2) = t0 + t1;
        wts(dim, 3) = t0 - t1;
        t0 = (1.0 / 16.0) * (9.0 / 5.0 - t);
        t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0);
        wts(dim, 1) = t0 + t1;
        wts(dim, 4) = t0 - t1;
      }
    }

    return wts;
  }

  static auto EIGEN_DEVICE_FUNC _get_dweights(Eigen::Matrix<Real, NDIM, 1> frac)
      -> Eigen::Matrix<Real, NDIM, DEGREE + 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    Eigen::Matrix<Real, NDIM, DEGREE + 1> dwts =
        Eigen::Matrix<Real, NDIM, DEGREE + 1>::Constant(0.0);

    if (DEGREE == 2) {
      for (int dim = 0; dim < NDIM; ++dim) {
        dwts(dim, 1) = -2.0 * frac[dim];
        dwts(dim, 2) = (1.0 / 2.0) * (1.0 - dwts(dim, 1));
        dwts(dim, 0) = -dwts(dim, 1) - dwts(dim, 2);
      }
    } else if (DEGREE == 3) {
      for (int dim = 0; dim < NDIM; ++dim) {
        dwts(dim, 3) = (1.0 / 2.0) * square(frac[dim]);
        dwts(dim, 0) = (frac[dim] - 0.5) - dwts(dim, 3);
        dwts(dim, 2) = 1.0 + dwts(dim, 0) - 2.0 * dwts(dim, 3);
        dwts(dim, 1) = -dwts(dim, 0) - dwts(dim, 2) - dwts(dim, 3);
      }
    } else if (DEGREE == 4) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real t = (1.0 / 6.0) * square(frac[dim]);
        Real dt = (1.0 / 3.0) * frac[dim];
        Real dt0 = (t - 11.0 / 24.0) + frac[dim] * dt;
        Real dt1 = 2.0 * frac[dim] * (1.0 / 4.0 - t) - square(frac[dim]) * dt;
        dwts(dim, 0) = -(1.0 / 8.0) * square(1.0 / 2.0 - frac[dim]);
        dwts(dim, 1) = dt1 + dt0;
        dwts(dim, 3) = dt1 - dt0;
        dwts(dim, 4) = dwts(dim, 0) + dt0 + (1.0 / 2.0);
        dwts(dim, 2) =
            -dwts(dim, 0) - dwts(dim, 1) - dwts(dim, 3) - dwts(dim, 4);
      }
    } else if (DEGREE == 5) {
      for (int dim = 0; dim < NDIM; ++dim) {
        Real w2 = frac[dim] * frac[dim];
        dwts(dim, 5) = (1.0 / 24.0) * square(square(frac[dim]));
        Real w4 = square(square(frac[dim]) - frac[dim]);
        Real dw4 =
            2.0 * (square(frac[dim]) - frac[dim]) * (2.0 * frac[dim] - 1.0);
        Real w = frac[dim] - 0.5;
        Real t = w2 * (w2 - 3.0);
        Real dt = 4.0 * cube(frac[dim]) - 6.0 * frac[dim];
        dwts(dim, 0) = (1.0 / 24.0) * (2.0 * frac[dim] + dw4) - dwts(dim, 5);
        Real dt0 = (1.0 / 24.0) * (4.0 * cube(frac[dim]) - 10.0 * frac[dim]);
        Real dt1 = (-1.0 / 12.0) * (w * dt + (t + 4.0));
        dwts(dim, 2) = dt0 + dt1;
        dwts(dim, 3) = dt0 - dt1;
        dt0 = -(1.0 / 16.0) * dt;
        dt1 = (1.0 / 24.0) * ((w4 - w2 - 5.0) + w * (dw4 - 2 * frac[dim]));
        dwts(dim, 1) = dt0 + dt1;
        dwts(dim, 4) = dt0 - dt1;
      }
    }
    return dwts;
  }

  static auto EIGEN_DEVICE_FUNC interpolate(
      TView<Real, NDIM, D> coeffs, TView<Eigen::Matrix<Real, NDIM, 1>, 1, D> Xs)
      -> tmol::score::common::
          tuple<TPack<Real, 1, D>, TPack<Eigen::Matrix<Real, NDIM, 1>, 1, D> > {
    auto num_Vs = Xs.size(0);
    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto Vs = Vs_t.view;
    auto dV_dIs_t = TPack<Eigen::Matrix<Real, NDIM, 1>, 1, D>::empty(num_Vs);
    auto dV_dIs = dV_dIs_t.view;

    // fd - cuda specific code could be added here
    //  (current there are only CPU python bindings)
    for (int i = 0; i < num_Vs; ++i) {
      tmol::score::common::tie(Vs[i], dV_dIs[i]) = interpolate(coeffs, Xs[i]);
    }
    return {Vs_t, dV_dIs_t};
  }

  static auto EIGEN_DEVICE_FUNC
  interpolate(TView<Real, NDIM, D> coeffs, Eigen::Matrix<Real, NDIM, 1> X)
      -> tmol::score::common::tuple<Real, Eigen::Matrix<Real, NDIM, 1> > {
    typedef Eigen::Matrix<Real, NDIM, 1> RealN;
    typedef Eigen::Matrix<Int, NDIM, 1> IntN;

    Real interp = 0;
    RealN dinterp_dX = RealN::Constant(0.0);

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
    Eigen::Matrix<Real, NDIM, DEGREE + 1> wts = _get_weights(frac);
    Eigen::Matrix<Real, NDIM, DEGREE + 1> dwts = _get_dweights(frac);

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
        weight *= wts(dim, idx_box_i);

        for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
          dweight[d_dim] *=
              (d_dim == dim ? dwts(dim, idx_box_i) : wts(dim, idx_box_i));
        }

        pt_indexer /= (DEGREE + 1);
      }
      Real coeff_i = coeffs.data()[idx_ij];
      interp += weight * coeff_i;
      for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
        dinterp_dX[d_dim] += dweight[d_dim] * coeff_i;
      }
    }

    return {interp, dinterp_dX};
  }
};

}  // namespace bspline
}  // namespace numeric
}  // namespace tmol
