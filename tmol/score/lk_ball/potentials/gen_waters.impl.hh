#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>

#include <tmol/score/hbond/identification.hh>
#include <tmol/score/ljlk/potentials/params.hh>

#include "water.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    int MAX_WATER>
struct GenerateWaters {
  static def forward(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Int, 1, D> atom_types,
      TView<Vec<Int, 2>, 1, D> indexed_bond_bonds,
      TView<Vec<Int, 2>, 1, D> indexed_bond_spans,
      TView<LKBallWaterGenTypeParams<Int>, 1, D> type_params,
      TView<LKBallWaterGenGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> sp2_water_tors,
      TView<Real, 1, D> sp3_water_tors,
      TView<Real, 1, D> ring_water_tors)
      ->TPack<Vec<Real, 3>, 2, D> {
    NVTXRange _function(__FUNCTION__);

    using tmol::score::hbond::AcceptorBases;
    using tmol::score::hbond::AcceptorHybridization;

    int num_Vs = coords.size(0);

    nvtx_range_push("watergen::setup");
    auto waters_t =
        TPack<Vec<Real, 3>, 2, D>::empty({coords.size(0), MAX_WATER});
    auto waters = waters_t.view;

    tmol::score::bonded_atom::IndexedBonds<Int, D> indexed_bonds;
    indexed_bonds.bonds = indexed_bond_bonds;
    indexed_bonds.bond_spans = indexed_bond_spans;

    int nsp2wats = sp2_water_tors.size(0);
    int nsp3wats = sp3_water_tors.size(0);
    int nringwats = ring_water_tors.size(0);
    nvtx_range_pop();

    nvtx_range_push("watergen::gen");
    auto is_hydrogen = ([=] EIGEN_DEVICE_FUNC(int j) {
      return (bool)type_params[atom_types[j]].is_hydrogen;
    });

    auto f_watergen = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_types[i];
      int wi = 0;

      if (type_params[ati].is_acceptor) {
        Int hyb = type_params[ati].acceptor_hybridization;

        auto bases = AcceptorBases<Int>::for_acceptor(
            i, hyb, indexed_bonds, is_hydrogen);

        Vec<Real, 3> XA = coords[bases.A];
        Vec<Real, 3> XB = coords[bases.B];
        Vec<Real, 3> XB0 = coords[bases.B0];

        Real dist;
        Real angle;
        Real *tors;
        Int ntors;

        if (hyb == AcceptorHybridization::sp2) {
          dist = global_params[0].lkb_water_dist;
          angle = global_params[0].lkb_water_angle_sp2;
          ntors = nsp2wats;
          tors = sp2_water_tors.data();
        } else if (hyb == AcceptorHybridization::sp3) {
          dist = global_params[0].lkb_water_dist;
          angle = global_params[0].lkb_water_angle_sp3;
          ntors = nsp3wats;
          tors = sp3_water_tors.data();
        } else if (hyb == AcceptorHybridization::ring) {
          dist = global_params[0].lkb_water_dist;
          angle = global_params[0].lkb_water_angle_ring;
          ntors = nringwats;
          tors = ring_water_tors.data();
          XB = 0.5 * (XB + XB0);
        }

        for (int ti = 0; ti < ntors; ti++) {
          waters[i][wi] =
              build_acc_water<Real>::V(XA, XB, XB0, dist, angle, tors[ti]);
          wi++;
        }
      }

      if (type_params[ati].is_donor) {
        for (int other_atom : indexed_bonds.bound_to(i)) {
          if (is_hydrogen(other_atom)) {
            waters[i][wi] = build_don_water<Real>::V(
                coords[i], coords[other_atom], global_params[0].lkb_water_dist);
            wi++;
          };
        }
      }

      for (; wi < MAX_WATER; wi++) {
        waters[i][wi] = Vec<Real, 3>::Constant(NAN);
      }
    });

    Dispatch<D>::forall(num_Vs, f_watergen);
    nvtx_range_pop();

    return waters_t;
  };

  static def backward(
      TView<Vec<Real, 3>, 2, D> dE_dW,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Int, 1, D> atom_types,
      TView<Vec<Int, 2>, 1, D> indexed_bond_bonds,
      TView<Vec<Int, 2>, 1, D> indexed_bond_spans,
      TView<LKBallWaterGenTypeParams<Int>, 1, D> type_params,
      TView<LKBallWaterGenGlobalParams<Real>, 1, D> global_params,
      TView<Real, 1, D> sp2_water_tors,
      TView<Real, 1, D> sp3_water_tors,
      TView<Real, 1, D> ring_water_tors)
      ->TPack<Vec<Real, 3>, 1, D> {
    NVTXRange _function(__FUNCTION__);

    nvtx_range_push("watergen::dsetup");

    using tmol::score::hbond::AcceptorBases;
    using tmol::score::hbond::AcceptorHybridization;

    int num_Vs = coords.size(0);

    auto dE_d_coord_t = TPack<Vec<Real, 3>, 1, D>::empty({coords.size(0)});
    auto dE_d_coord = dE_d_coord_t.view;
    auto zero = [=] EIGEN_DEVICE_FUNC(int i) {
      for (int j = 0; j < 3; ++j) {
        dE_d_coord[i](j) = 0;
      }
    };
    Dispatch<D>::forall(coords.size(0), zero);

    tmol::score::bonded_atom::IndexedBonds<Int, D> indexed_bonds;
    indexed_bonds.bonds = indexed_bond_bonds;
    indexed_bonds.bond_spans = indexed_bond_spans;

    int nsp2wats = sp2_water_tors.size(0);
    int nsp3wats = sp3_water_tors.size(0);
    int nringwats = ring_water_tors.size(0);
    nvtx_range_pop();

    nvtx_range_push("watergen::dgen");
    auto is_hydrogen = ([=] EIGEN_DEVICE_FUNC(int j) {
      return (bool)type_params[atom_types[j]].is_hydrogen;
    });

    auto df_watergen = ([=] EIGEN_DEVICE_FUNC(int i) {
      Int ati = atom_types[i];
      int wi = 0;

      if (type_params[ati].is_acceptor) {
        Int hyb = type_params[ati].acceptor_hybridization;

        auto bases = AcceptorBases<Int>::for_acceptor(
            i, hyb, indexed_bonds, is_hydrogen);

        Vec<Real, 3> XA = coords[bases.A];
        Vec<Real, 3> XB = coords[bases.B];
        Vec<Real, 3> XB0 = coords[bases.B0];

        Vec<Real, 3> dE_dXA = Vec<Real, 3>::Zero();
        Vec<Real, 3> dE_dXB = Vec<Real, 3>::Zero();
        Vec<Real, 3> dE_dXB0 = Vec<Real, 3>::Zero();

        Real dist;
        Real angle;
        Real *tors;
        Int ntors;

        if (hyb == AcceptorHybridization::sp2) {
          dist = global_params[0].lkb_water_dist;
          angle = global_params[0].lkb_water_angle_sp2;
          ntors = nsp2wats;
          tors = sp2_water_tors.data();
        } else if (hyb == AcceptorHybridization::sp3) {
          dist = global_params[0].lkb_water_dist;
          angle = global_params[0].lkb_water_angle_sp3;
          ntors = nsp3wats;
          tors = sp3_water_tors.data();
        } else if (hyb == AcceptorHybridization::ring) {
          dist = global_params[0].lkb_water_dist;
          angle = global_params[0].lkb_water_angle_ring;
          ntors = nringwats;
          tors = ring_water_tors.data();
          XB = 0.5 * (XB + XB0);
        }

        for (int ti = 0; ti < ntors; ti++) {
          auto dW =
              build_acc_water<Real>::dV(XA, XB, XB0, dist, angle, tors[ti]);
          auto dE_dWi = dE_dW[i][wi];

          dE_dXA += dW.dA * dE_dWi;
          dE_dXB += dW.dB * dE_dWi;
          dE_dXB0 += dW.dB0 * dE_dWi;

          wi++;
        }

        common::accumulate<D, Vec<Real, 3>>::add(dE_d_coord[bases.A], dE_dXA);

        if (hyb == AcceptorHybridization::ring) {
          common::accumulate<D, Vec<Real, 3>>::add(
              dE_d_coord[bases.B], dE_dXB / 2.0);
          common::accumulate<D, Vec<Real, 3>>::add(
              dE_d_coord[bases.B0], dE_dXB / 2.0);
        } else {
          common::accumulate<D, Vec<Real, 3>>::add(dE_d_coord[bases.B], dE_dXB);
        }

        common::accumulate<D, Vec<Real, 3>>::add(dE_d_coord[bases.B0], dE_dXB0);
      }

      if (type_params[ati].is_donor) {
        for (int other_atom : indexed_bonds.bound_to(i)) {
          if (is_hydrogen(other_atom)) {
            auto dE_dWi = dE_dW[i][wi];
            auto dW = build_don_water<Real>::dV(
                coords[i], coords[other_atom], global_params[0].lkb_water_dist);

            common::accumulate<D, Vec<Real, 3>>::add(
                dE_d_coord[i], dW.dD * dE_dWi);
            common::accumulate<D, Vec<Real, 3>>::add(
                dE_d_coord[other_atom], dW.dH * dE_dWi);

            wi++;
          };
        }
      }
    });

    Dispatch<D>::forall(num_Vs, df_watergen);
    nvtx_range_pop();

    return dE_d_coord_t;
  };
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
