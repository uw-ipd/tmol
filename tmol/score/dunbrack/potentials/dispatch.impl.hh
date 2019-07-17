#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/utility/nvtx.hh>

#include <ATen/Tensor.h>

#include <tuple>

#include <pybind11/pybind11.h>

#include "params.hh"
#include "potentials.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Coord Vec<Real, 3>
#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    int MAXBB,
    int MAXCHI>
struct DunbrackDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, MAXBB + 1, D> rotameric_tables,
      TView<RotamericTableParams<Real, MAXBB>, 1, D> rotameric_table_params,
      TView<Real, MAXBB + 2, D> semirotameric_tables,
      TView<SemirotamericTableParams<Real, MAXBB>, 1, D>
          semirotameric_table_params,
      TView<DunResParameters<Int, MAXBB, MAXCHI>, 1, D> residue_params,
      TView<DunTableLookupParams<Int, MAXCHI>, 1, D> residue_lookup_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Coord, 2, D>> {
    NVTXRange _function(__FUNCTION__);

    auto Vs_t = TPack<Real, 1, D>::zeros({3});
    auto dVs_dx_t = TPack<Coord, 2, D>::zeros({coords.size(0), 3});

    auto Vs = Vs_t.view;
    auto dVs_dx = dVs_dx_t.view;

    auto f_dunbrack = ([=] EIGEN_DEVICE_FUNC(int res) {
      int aa_idx = residue_params[res].aa_index;

      // 0. pre-allocate
      Real rotE = 0, semirotE = 0, rotdevE = 0;
      Vec<Real, MAXBB> drotEdphipsi = Vec<Real, MAXBB>::Zero();
      Vec<Real, MAXBB> dsemirotEdphipsi = Vec<Real, MAXBB>::Zero();
      Vec<Real, MAXBB> drotdevEdphipsi = Vec<Real, MAXBB>::Zero();
      Vec<Real, MAXCHI> dsemirotEdchi = Vec<Real, MAXCHI>::Zero();
      Vec<Real, MAXCHI> drotdevEdchi = Vec<Real, MAXCHI>::Zero();

      Vec<Real, MAXCHI> all_chis;
      Vec<Real, MAXBB> all_phis;
      Vec<Real, MAXBB> bb_idx;
      Eigen::Matrix<Real, MAXCHI * 4, 3> all_dchi_dxs;
      Eigen::Matrix<Real, MAXBB * 4, 3> all_dphi_dxs;

      // 1A. compute chi dihedrals
      int nchi = residue_lookup_params[aa_idx].nrotchi;
      if (residue_lookup_params[aa_idx].semirotchi >= 0) nchi++;
      for (int i = 0; i < nchi; ++i) {
        auto dihe = dihedral_angle<Real>::V_dV(
            coords[residue_params[res].chi_indices[i][0]],
            coords[residue_params[res].chi_indices[i][1]],
            coords[residue_params[res].chi_indices[i][2]],
            coords[residue_params[res].chi_indices[i][3]]);
        all_chis[i] = dihe.V;
        all_dchi_dxs.row(4 * i + 0) = dihe.dV_dI;
        all_dchi_dxs.row(4 * i + 1) = dihe.dV_dJ;
        all_dchi_dxs.row(4 * i + 2) = dihe.dV_dK;
        all_dchi_dxs.row(4 * i + 3) = dihe.dV_dL;
      }

      // 1B. compute bb dihedrals
      for (int i = 0; i < MAXBB; i++) {
        typename dihedral_angle<Real>::V_dV_T dihe;
        if (residue_params[res].bb_indices[i][0] >= 0) {
          dihe = dihedral_angle<Real>::V_dV(
              coords[residue_params[res].bb_indices[i][0]],
              coords[residue_params[res].bb_indices[i][1]],
              coords[residue_params[res].bb_indices[i][2]],
              coords[residue_params[res].bb_indices[i][3]]);
        } else {
          // "neutral" phi = -60deg;  neutral psi = 60deg
          dihe = {(i == 0 ? -1 : 1) * 60.0 * M_PI / 180,
                  Vec<Real, 3>(0, 0, 0),
                  Vec<Real, 3>(0, 0, 0),
                  Vec<Real, 3>(0, 0, 0),
                  Vec<Real, 3>(0, 0, 0)};
        }
        all_phis[i] = dihe.V;
        all_dphi_dxs.row(4 * i + 0) = dihe.dV_dI;
        all_dphi_dxs.row(4 * i + 1) = dihe.dV_dJ;
        all_dphi_dxs.row(4 * i + 2) = dihe.dV_dK;
        all_dphi_dxs.row(4 * i + 3) = dihe.dV_dL;
      }

      // 2. compute the rotameric bin and corresponding table indices
      int rotidx =
          classify_rotamer(all_chis, residue_lookup_params[aa_idx].nrotchi);
      int rotprobtableidx =
          residue_lookup_params[aa_idx].rotidx2probtableidx[rotidx];
      int semiprobtableidx =
          residue_lookup_params[aa_idx].semirotidx2probtableidx[rotidx];
      int rotmeantableidx =
          residue_lookup_params[aa_idx].rotidx2meantableidx[rotidx];
      int semirotchi = residue_lookup_params[aa_idx].nrotchi;

      // 3. ln(P)
      bool is_semirotameric = (semiprobtableidx >= 0);
      if (!is_semirotameric) {
        // 3A. compute -ln(P) for rotameric
        Vec<Real, MAXBB> bb_steps;
        Real bbstart, bbstep, bbperiod;
        for (int i = 0; i < MAXBB; i++) {
          bbstart = rotameric_table_params[rotprobtableidx].bbstarts[i];
          bbstep = rotameric_table_params[rotprobtableidx].bbsteps[i];
          bbperiod = (Real)rotameric_tables[rotprobtableidx].size(i);
          bb_idx[i] = pos_fmod((all_phis[i] - bbstart) / bbstep, bbperiod);
        }

        tie(rotE, drotEdphipsi) =
            tmol::numeric::bspline::ndspline<MAXBB, 3, D, Real, Int>::
                interpolate(rotameric_tables[rotprobtableidx + 1], bb_idx);

        for (int i = 0; i < MAXBB; i++) {
          bbstep = rotameric_table_params[rotprobtableidx].bbsteps[i];
          drotEdphipsi[i] = drotEdphipsi[i] / bbstep;
        }
      } else {
        // 3B. compute -ln(P) for semirotameric
        Vec<Real, MAXBB + 1> bbchi_idx;
        Vec<Real, MAXBB + 1> dVdphipsichi;
        Real bbstart, bbstep, bbperiod;
        for (int i = 0; i < MAXBB; i++) {
          bbstart = semirotameric_table_params[semiprobtableidx].bbstarts[i];
          bbstep = semirotameric_table_params[semiprobtableidx].bbsteps[i];
          bbperiod = (Real)semirotameric_tables[semiprobtableidx].size(i);
          bbchi_idx[i] = pos_fmod((all_phis[i] - bbstart) / bbstep, bbperiod);
        }
        bbstart = semirotameric_table_params[semiprobtableidx].bbstarts[MAXBB];
        bbstep = semirotameric_table_params[semiprobtableidx].bbsteps[MAXBB];
        bbperiod = (Real)semirotameric_tables[semiprobtableidx].size(MAXBB);
        bbchi_idx[MAXBB] =
            pos_fmod((all_chis[semirotchi] - bbstart) / bbstep, bbperiod);

        tie(semirotE, dVdphipsichi) = tmol::numeric::bspline::
            ndspline<MAXBB + 1, 3, D, Real, Int>::interpolate(
                semirotameric_tables[semiprobtableidx + 1], bbchi_idx);

        for (int i = 0; i < MAXBB; i++) {
          bbstep = semirotameric_table_params[semiprobtableidx].bbsteps[i];
          dsemirotEdphipsi[i] = dVdphipsichi[i] / bbstep;
        }
        bbstep = semirotameric_table_params[semiprobtableidx].bbsteps[MAXBB];
        dsemirotEdchi[semirotchi] = dVdphipsichi[MAXBB] / bbstep;

        // save for dev
        bb_idx = bbchi_idx.topRows(MAXBB);
      }

      // 4. compute the chi-deviation penalty
      for (int i = 0; i < residue_lookup_params[aa_idx].nrotchi; ++i) {
        Real Xmean, Xdev;
        Vec<Real, MAXBB> dXmean, dXdev;

        tie(Xmean, dXmean) = tmol::numeric::bspline::
            ndspline<MAXBB, 3, D, Real, Int>::interpolate(
                rotameric_tables[rotmeantableidx + 2 * i + 0], bb_idx);

        tie(Xdev, dXdev) = tmol::numeric::bspline::
            ndspline<MAXBB, 3, D, Real, Int>::interpolate(
                rotameric_tables[rotmeantableidx + 2 * i + 1], bb_idx);

        Real rotdelta_i =
            (all_chis[i] < -120.0_2rad ? all_chis[i] + Real(2 * M_PI) - Xmean
                                       : all_chis[i] - Xmean);

        rotdevE += rotdelta_i * rotdelta_i / (2 * Xdev * Xdev);
        drotdevEdchi[i] = rotdelta_i / (Xdev * Xdev);

        for (int k = 0; k < MAXBB; ++k) {
          Real bbstep =
              is_semirotameric
                  ? semirotameric_table_params[semiprobtableidx].bbsteps[k]
                  : rotameric_table_params[rotprobtableidx].bbsteps[k];
          drotdevEdphipsi[k] += -rotdelta_i / bbstep / (Xdev * Xdev * Xdev)
                                * (Xdev * dXmean[k] + rotdelta_i * dXdev[k]);
        }
      }

      // 5. accumulate all
      accumulate<D, Real>::add(Vs[0], rotE);
      accumulate<D, Real>::add(Vs[1], rotdevE);
      accumulate<D, Real>::add(Vs[2], semirotE);

      for (int k = 0; k < MAXBB; ++k) {
        for (int j = 0; j < 4; ++j) {
          Int phi_j = residue_params[res].bb_indices[k][j];
          if (phi_j >= 0) {
            accumulate<D, Vec<Real, 3>>::add(
                dVs_dx[phi_j][0],
                drotEdphipsi[k] * all_dphi_dxs.row(4 * k + j));
            accumulate<D, Vec<Real, 3>>::add(
                dVs_dx[phi_j][1],
                dsemirotEdphipsi[k] * all_dphi_dxs.row(4 * k + j));
            accumulate<D, Vec<Real, 3>>::add(
                dVs_dx[phi_j][2],
                drotdevEdphipsi[k] * all_dphi_dxs.row(4 * k + j));
          }
        }
      }

      for (int k = 0; k < nchi; ++k) {
        for (int j = 0; j < 4; ++j) {
          Int chi_j = residue_params[res].chi_indices[k][j];
          accumulate<D, Vec<Real, 3>>::add(
              dVs_dx[chi_j][1], dsemirotEdchi[k] * all_dchi_dxs.row(4 * k + j));
          accumulate<D, Vec<Real, 3>>::add(
              dVs_dx[chi_j][2], drotdevEdchi[k] * all_dchi_dxs.row(4 * k + j));
        }
      }
    });

    int n_res = residue_params.size(0);
    Dispatch<D>::forall(n_res, f_dunbrack);

    return {Vs_t, dVs_dx_t};
  }
};

#undef Coord
#undef CoordQuad

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
