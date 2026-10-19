// N-dimensional B-spline interpolation with periodic boundary conditions.
// From:
//    Thévenaz, Philippe, Thierry Blu, and Michael Unser.
//    "Interpolation revisited [medical images application]."
//    IEEE Transactions on medical imaging 19.7 (2000): 739-758.
//    http://bigwww.epfl.ch/publications/thevenaz0002.pdf

// Dimension- and degree-templated, header-only implementation.

#pragma once

#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace numeric {
namespace bspline {

template <
    std::size_t NDIM,
    std::size_t DEGREE,
    tmol::Device D,
    typename Real,
    typename Int>
struct ndspline {
  static auto TMOL_HD square(Real v) -> Real { return v * v; }
  static auto TMOL_HD cube(Real v) -> Real { return v * v * v; }

  static auto TMOL_HD _get_weights(tmol::portable::Vec<Real, NDIM> frac)
      -> tmol::portable::Mat<Real, NDIM, DEGREE + 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> wts =
        tmol::portable::Mat<Real, NDIM, DEGREE + 1>::Constant(0.0);

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

  static auto TMOL_HD _get_dweights(tmol::portable::Vec<Real, NDIM> frac)
      -> tmol::portable::Mat<Real, NDIM, DEGREE + 1> {
    static_assert(DEGREE >= 2 && DEGREE <= 5, "Invalid spline degree.");
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> dwts =
        tmol::portable::Mat<Real, NDIM, DEGREE + 1>::Constant(0.0);

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

  static auto TMOL_HD interpolate_tv(
      TView<Real, NDIM, D> coeffs,
      TView<tmol::portable::Vec<Real, NDIM>, 1, D> Xs) -> tmol::score::common::
      tuple<TPack<Real, 1, D>, TPack<tmol::portable::Vec<Real, NDIM>, 1, D> > {
    return interpolate(
        TensorAccessor<Real, NDIM, D>(coeffs),
        TensorAccessor<tmol::portable::Vec<Real, NDIM>, 1, D>(Xs));
  }

  static auto TMOL_HD interpolate(
      TensorAccessor<Real, NDIM, D> coeffs,
      TensorAccessor<tmol::portable::Vec<Real, NDIM>, 1, D> Xs)
      -> tmol::score::common::tuple<
          TPack<Real, 1, D>,
          TPack<tmol::portable::Vec<Real, NDIM>, 1, D> > {
    auto num_Vs = Xs.size(0);
    auto Vs_t = TPack<Real, 1, D>::empty({num_Vs});
    auto Vs = Vs_t.view;
    auto dV_dIs_t = TPack<tmol::portable::Vec<Real, NDIM>, 1, D>::empty(num_Vs);
    auto dV_dIs = dV_dIs_t.view;

    // fd - cuda specific code could be added here
    //  (current there are only CPU python bindings)
    for (int i = 0; i < num_Vs; ++i) {
      tmol::score::common::tie(Vs[i], dV_dIs[i]) = interpolate(coeffs, Xs[i]);
    }
    return {Vs_t, dV_dIs_t};
  }

  static auto TMOL_HD interpolate_V(
      TensorAccessor<Real, NDIM, D> coeffs, tmol::portable::Vec<Real, NDIM> X)
      -> Real {
    typedef tmol::portable::Vec<Real, NDIM> RealN;
    typedef tmol::portable::Vec<Int, NDIM> IntN;

    Int nprods = 1;
    IntN idx;
    RealN frac;

    for (int dim = 0; dim < NDIM; ++dim) {
      idx[dim] = (Int)std::floor(X[dim]);
      frac[dim] = X[dim] - idx[dim];
      idx[dim] -= (DEGREE / 2);
      nprods *= (DEGREE + 1);
    }

    tmol::portable::Mat<Real, NDIM, DEGREE + 1> wts = _get_weights(frac);
    Real interp = 0;
    for (int pt = 0; pt < nprods; ++pt) {
      Real weight = 1;
      Int stride = 1;
      Int pt_indexer = pt;
      Int idx_ij = 0;

      for (int dim = NDIM - 1; dim >= 0; --dim) {
        Int idx_box_i = pt_indexer % (DEGREE + 1);
        Int idx_i = (idx[dim] + idx_box_i) % coeffs.size(dim);
        if (idx_i < 0) idx_i += coeffs.size(dim);
        idx_ij += stride * idx_i;
        stride *= coeffs.size(dim);
        weight *= wts(dim, idx_box_i);
        pt_indexer /= (DEGREE + 1);
      }
      interp += weight * coeffs.data()[idx_ij];
    }
    return interp;
  }

  static auto TMOL_HD interpolate(
      TensorAccessor<Real, NDIM, D> coeffs, tmol::portable::Vec<Real, NDIM> X)
      -> tmol::score::common::tuple<Real, tmol::portable::Vec<Real, NDIM> > {
    typedef tmol::portable::Vec<Real, NDIM> RealN;
    typedef tmol::portable::Vec<Int, NDIM> IntN;

    Real interp = 0;
    RealN dinterp_dX = RealN::Constant(0.0);

    Int nprods = 1;
    Int pt_indexer, idx_box_i;
    int64_t idx_ij;
    IntN idx;
    RealN frac;

    Real weight = 1;
    RealN dweight;

    for (int dim = 0; dim < NDIM; ++dim) {
      idx[dim] = (Int)std::floor(X[dim]);
      frac[dim] = X[dim] - idx[dim];
      idx[dim] -= (DEGREE / 2);
      nprods *= (DEGREE + 1);
    }

    // get weights in each dimension
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> wts = _get_weights(frac);
    tmol::portable::Mat<Real, NDIM, DEGREE + 1> dwts = _get_dweights(frac);

    // Cache offsets on CPU and for small CUDA stencils. Larger CUDA stencils
    // keep only wrapped base indices, avoiding a per-thread offset array.
    constexpr bool cache_offsets = D == Device::CPU || NDIM <= 2;
    // A zero-row matrix is not a legal device variable, so the unused
    // instantiation keeps one dummy row rather than none.
    tmol::portable::Mat<int64_t, cache_offsets ? NDIM : 1, DEGREE + 1>
        offsets;
    if constexpr (cache_offsets) {
      for (int dim = 0; dim < NDIM; ++dim) {
        for (int tap = 0; tap <= DEGREE; ++tap) {
          Int wrapped = (idx[dim] + tap) % coeffs.size(dim);
          if (wrapped < 0) wrapped += coeffs.size(dim);
          offsets(dim, tap) = coeffs.stride(dim) * wrapped;
        }
      }
    } else {
      for (int dim = 0; dim < NDIM; ++dim) {
        idx[dim] %= coeffs.size(dim);
        if (idx[dim] < 0) idx[dim] += coeffs.size(dim);
      }
    }

    // do the dot product
    for (int pt = 0; pt < nprods; ++pt) {
      weight = 1;
      dweight = RealN::Constant(1.0);

      pt_indexer = pt;
      idx_ij = 0;

      for (int dim = NDIM - 1; dim >= 0; --dim) {
        idx_box_i = pt_indexer % (DEGREE + 1);
        if constexpr (cache_offsets) {
          idx_ij += offsets(dim, idx_box_i);
        } else {
          Int idx_i = idx[dim] + idx_box_i;
          while (idx_i >= coeffs.size(dim)) idx_i -= coeffs.size(dim);
          idx_ij += coeffs.stride(dim) * idx_i;
        }
        weight *= wts(dim, idx_box_i);

        for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
          dweight[d_dim] *=
              (d_dim == dim ? dwts(dim, idx_box_i) : wts(dim, idx_box_i));
        }

        pt_indexer /= (DEGREE + 1);
      }
      interp += weight * coeffs.data()[idx_ij];
      for (int d_dim = 0; d_dim < NDIM; ++d_dim) {
        dinterp_dX[d_dim] += dweight[d_dim] * coeffs.data()[idx_ij];
      }
    }

    return {interp, dinterp_dX};
  }
};

}  // namespace bspline
}  // namespace numeric
}  // namespace tmol
