#include <tmol/score/common/dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"
#include "water.hh"

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/transform.hxx>
#include <tmol/score/hbond/identification.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <tmol::Device D, typename Real, typename Int, int MAX_WATER>
struct attached_waters {
  static def forward(
      TView<Vec<Real, 3>, 1, D> coords,
      tmol::score::bonded_atom::IndexedBonds<Int, D> indexed_bonds,
      AtomTypes<D> atom_types,
      LKBallGlobalParameters<Real, D> global_params)
      ->TPack<Vec<Real, 3>, 2, D> {
    using tmol::score::hbond::AcceptorBases;
    using tmol::score::hbond::AcceptorHybridization;

    int num_Vs = coords.size(0);

    auto waters_t =
        TPack<Vec<Real, 3>, 2, D>::empty({coords.size(0), MAX_WATER});
    auto waters = waters_t.view;

    auto f_watergen = ([=] EIGEN_DEVICE_FUNC(int i) {
      int wi = 0;
      if (atom_types.is_acceptor[i]) {
        Int hyb = atom_types.acceptor_hybridization[i];
        auto bases = AcceptorBases<Int>::for_acceptor(
            i,
            atom_types.acceptor_hybridization[i],
            indexed_bonds,
            atom_types.is_hydrogen);
        Vec<Real, 3> XA = coords[bases.A];
        Vec<Real, 3> XB = coords[bases.B];
        Vec<Real, 3> XB0 = coords[bases.B0];

        Real dist;
        Real angle;
        Real *tors;
        Int ntors;

        if (hyb == AcceptorHybridization::sp2) {
          dist = global_params.lkb_water_dist;
          angle = global_params.lkb_water_angle_sp2;
          ntors = global_params.lkb_water_tors_sp2.size(0);
          tors = global_params.lkb_water_tors_sp2.data();
        } else if (hyb == AcceptorHybridization::sp3) {
          dist = global_params.lkb_water_dist;
          angle = global_params.lkb_water_angle_sp3;
          ntors = global_params.lkb_water_tors_sp3.size(0);
          tors = global_params.lkb_water_tors_sp3.data();
        } else if (hyb == AcceptorHybridization::ring) {
          dist = global_params.lkb_water_dist;
          angle = global_params.lkb_water_angle_ring;
          ntors = global_params.lkb_water_tors_ring.size(0);
          tors = global_params.lkb_water_tors_ring.data();
          XB = 0.5 * (XB + XB0);
        }

        for (int ti = 0; ti < ntors; ti++) {
          waters[i][wi] =
              build_acc_water<Real>::V(XA, XB, XB0, dist, angle, tors[ti]);
          wi++;
        }
      }

      if (atom_types.is_donor[i]) {
        for (int other_atom : indexed_bonds.bound_to(i)) {
          if (atom_types.is_hydrogen[other_atom]) {
            waters[i][wi] = build_don_water<Real>::V(
                coords[i], coords[other_atom], global_params.lkb_water_dist);
            wi++;
          };
        }
      }

      for (; wi < MAX_WATER; wi++) {
        waters[i][wi] = Vec<Real, 3>::Constant(NAN);
      }
    });

    mgpu::standard_context_t context;
    mgpu::transform(f_watergen, num_Vs, context);

    return waters_t;
  };

  static def backward(
      TView<Vec<Real, 3>, 2, D> dE_dW,
      TView<Vec<Real, 3>, 1, D> coords,
      tmol::score::bonded_atom::IndexedBonds<Int, D> indexed_bonds,
      AtomTypes<D> atom_types,
      LKBallGlobalParameters<Real, D> global_params)
      ->TPack<Vec<Real, 3>, 1, D> {
    using tmol::score::hbond::AcceptorBases;
    using tmol::score::hbond::AcceptorHybridization;

    int num_Vs = coords.size(0);

    auto dE_d_coord_t =
        TPack<Vec<Real, 3>, 1, D>::zeros({coords.size(0), MAX_WATER});
    auto dE_d_coord = dE_d_coord_t.view;

    auto f_watergen = ([=] EIGEN_DEVICE_FUNC(int i) {
      int wi = 0;

      if (atom_types.is_acceptor[i]) {
        Int hyb = atom_types.acceptor_hybridization[i];
        auto bases = AcceptorBases<Int>::for_acceptor(
            i,
            atom_types.acceptor_hybridization[i],
            indexed_bonds,
            atom_types.is_hydrogen);
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
          dist = global_params.lkb_water_dist;
          angle = global_params.lkb_water_angle_sp2;
          ntors = global_params.lkb_water_tors_sp2.size(0);
          tors = global_params.lkb_water_tors_sp2.data();
        } else if (hyb == AcceptorHybridization::sp3) {
          dist = global_params.lkb_water_dist;
          angle = global_params.lkb_water_angle_sp3;
          ntors = global_params.lkb_water_tors_sp3.size(0);
          tors = global_params.lkb_water_tors_sp3.data();
        } else if (hyb == AcceptorHybridization::ring) {
          dist = global_params.lkb_water_dist;
          angle = global_params.lkb_water_angle_ring;
          ntors = global_params.lkb_water_tors_ring.size(0);
          tors = global_params.lkb_water_tors_ring.data();
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

        dE_d_coord[bases.A] += dE_dXA;

        if (hyb == AcceptorHybridization::ring) {
          dE_d_coord[bases.B] += dE_dXB / 2;
          dE_d_coord[bases.B0] += dE_dXB / 2;
        } else {
          dE_d_coord[bases.B] += dE_dXB;
        }

        dE_d_coord[bases.B0] += dE_dXB0;
      }

      if (atom_types.is_donor[i]) {
        for (int other_atom : indexed_bonds.bound_to(i)) {
          if (atom_types.is_hydrogen[other_atom]) {
            auto dE_dWi = dE_dW[i][wi];
            auto dW = build_don_water<Real>::dV(
                coords[i], coords[other_atom], global_params.lkb_water_dist);

            dE_d_coord[i] += dW.dD * dE_dWi;
            dE_d_coord[other_atom] += dW.dH * dE_dWi;

            wi++;
          };
        }
      }
    });

    mgpu::standard_context_t context;
    mgpu::transform(f_watergen, num_Vs, context);

    return dE_d_coord_t;
  };
};

template struct LKBallDispatch<
    common::NaiveDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct LKBallDispatch<
    common::NaiveDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;
template struct attached_waters<tmol::Device::CUDA, float, int64_t, 4>;
template struct attached_waters<tmol::Device::CUDA, double, int64_t, 4>;

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
