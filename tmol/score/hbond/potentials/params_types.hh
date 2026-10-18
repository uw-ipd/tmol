#pragma once

// Plain parameter structs, free of the torch tensor layer.
//
// Split out of params.hh so that device-side scalar math can include the
// struct definitions without dragging in TensorUtil.h (and through it Eigen
// and torch), which exist only for the enable_tensor_view specializations.
//
// The polynomial tables are float64 on the host and CUDA, and MPS has no
// float64 at either the tensor or the shader level. Rounding the coefficients
// to plain float32 is not an option -- these are degree-10 fits that cancel
// hard, and float32 coefficients cost a *relative* error of 1.96, enough to
// flip the sign of a near-zero value. Metal carries them as an unevaluated
// hi + lo float32 pair instead, which measures identical to float64
// coefficients. See MPS_PORT_NOTES.md.

#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/qualifiers.hh>
#include <tmol/utility/portable/double_float.hh>
#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real>
struct HBondPoly {
  tmol::portable::Vec<Real, 11> coeffs;
  Real zero_pad;  // Eigen wants to align to memory boundaries
  tmol::portable::Vec<Real, 2> range;
  tmol::portable::Vec<Real, 2> bound;
};

template <typename Real>
struct HBondPolynomials {
  HBondPoly<Real> AHdist_poly;
  HBondPoly<Real> cosBAH_poly;
  HBondPoly<Real> cosAHD_poly;
};

// What a backend stores a polynomial coefficient in. float64 everywhere it
// exists; a pair of float32 on Metal, which is the same eight bytes, so the
// parameter structs keep their layout footprint either way.
template <tmol::Device D>
struct hbond_poly_scalar {
#if TMOL_PORTABLE_METAL
  // Metal has no float64 at all, and it rejects the type even in a template
  // body it will never instantiate. A shader only ever asks for the MPS
  // specialization below, so the primary template names the same type there.
  typedef tmol::portable::DoubleFloat<float> type;
#else
  typedef double type;
#endif
};

template <>
struct hbond_poly_scalar<tmol::Device::MPS> {
  typedef tmol::portable::DoubleFloat<float> type;
};

template <tmol::Device D>
using HBondPolysFor = HBondPolynomials<typename hbond_poly_scalar<D>::type>;

template <typename Real>
struct HBondGlobalParams {
  Real hb_sp2_range_span;
  Real hb_sp2_BAH180_rise;
  Real hb_sp2_outer_width;
  Real hb_sp3_softmax_fade;
  Real threshold_distance;
  Real max_ha_dis;
};

template <typename Real>
struct HBondPairParams {
  Real acceptor_hybridization;  // integer, stored as float
  Real acceptor_weight;
  Real donor_weight;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
