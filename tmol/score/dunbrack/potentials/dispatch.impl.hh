#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>

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
    auto Vs_t = TPack<Real, 1, D>::zeros({3});
    auto dVs_dx_t = TPack<Coord, 2, D>::zeros({coords.size(0), 3});

    auto Vs = Vs_t.view;
    auto dVs_dx = dVs_dx_t.view;

    // fd: lets try everything (for 1 res) in a single kernel
    auto f_dunbrack = ([=] EIGEN_DEVICE_FUNC(int res) {
      int aa_idx = residue_params[res].aa_index;

      // 1. compute the chi dihedrals
      Eigen::Matrix<Real, MAXCHI * 4, 3> all_dchi_dxs;
      Eigen::Matrix<Real, MAXCHI, 1> all_chis;
      for (int i = 0; i < residue_lookup_params[aa_idx].nrotchi; ++i) {
        CoordQuad chi_i;
        for (int j = 0; j < 4; ++j) {
          chi_i.row(j) = coords[residue_params[res].chi_indices[i][j]];
        }

        auto dihe = dihedral_angle<Real>::V_dV(
            chi_i.row(0), chi_i.row(1), chi_i.row(2), chi_i.row(3));
        all_chis[i] = dihe.V;
        all_dchi_dxs.row(4 * i + 0) = dihe.dV_dI;
        all_dchi_dxs.row(4 * i + 1) = dihe.dV_dJ;
        all_dchi_dxs.row(4 * i + 2) = dihe.dV_dK;
        all_dchi_dxs.row(4 * i + 3) = dihe.dV_dL;
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

      // 3. compute -ln(P) energy
      // use the same data structure to hold rot & semirot indices
      //   -> for rot indices, the final elements are unused
      Eigen::Matrix<Real, MAXBB + 1, 1> all_rottable_idxs;
      Eigen::Matrix<Real, (MAXBB + 1) * 4, 3> all_drottable_idx_dxs;
      typename dihedral_angle<Real>::V_dV_T dihe;

      // 3A - compute bb (and semi chi) indices into table
      bool is_semirotameric = (semiprobtableidx >= 0);
      int ntabledims = is_semirotameric ? MAXBB + 1 : MAXBB;
      for (int ii = 0; ii < ntabledims; ii++) {
        if (ii < MAXBB) {
          if (residue_params[res].bb_indices[ii][0] >= 0) {
            CoordQuad tor_i;
            for (int j = 0; j < 4; ++j) {
              tor_i.row(j) = coords[residue_params[res].bb_indices[ii][j]];
            }
            dihe = dihedral_angle<Real>::V_dV(
                tor_i.row(0), tor_i.row(1), tor_i.row(2), tor_i.row(3));
          } else {
            // "neutral" phi = -60deg;  neutral psi = 60deg
            dihe = {(ii == 0 ? -1 : 1) * 60.0 * M_PI / 180,
                    Vec<Real, 3>(0, 0, 0),
                    Vec<Real, 3>(0, 0, 0),
                    Vec<Real, 3>(0, 0, 0),
                    Vec<Real, 3>(0, 0, 0)};
          }
        } else {
          CoordQuad tor_i;
          for (int j = 0; j < 4; ++j) {
            tor_i.row(j) =
                coords[residue_params[res].chi_indices[semirotchi][j]];
          }
          dihe = dihedral_angle<Real>::V_dV(
              tor_i.row(0), tor_i.row(1), tor_i.row(2), tor_i.row(3));
        }

        Real bbstart, bbstep, bbperiod;
        if (is_semirotameric) {
          bbstart = semirotameric_table_params[semiprobtableidx].bbstarts[ii];
          bbstep = semirotameric_table_params[semiprobtableidx].bbsteps[ii];
          bbperiod = (Real)semirotameric_tables[semiprobtableidx].size(ii);
        } else {
          bbstart = rotameric_table_params[rotprobtableidx].bbstarts[ii];
          bbstep = rotameric_table_params[rotprobtableidx].bbsteps[ii];
          bbperiod = (Real)rotameric_tables[rotprobtableidx].size(ii);
        }

        all_rottable_idxs[ii] = pos_fmod((dihe.V - bbstart) / bbstep, bbperiod);
        all_drottable_idx_dxs.row(4 * ii + 0) = (dihe.dV_dI) / bbstep;
        all_drottable_idx_dxs.row(4 * ii + 1) = (dihe.dV_dJ) / bbstep;
        all_drottable_idx_dxs.row(4 * ii + 2) = (dihe.dV_dK) / bbstep;
        all_drottable_idx_dxs.row(4 * ii + 3) = (dihe.dV_dL) / bbstep;
      }

      // 3B - lookup -ln(P) in table
      if (!is_semirotameric) {
        Real V;
        Vec<Real, MAXBB> dVdphipsi;

        // compute the -ln(P) energy for rotameric residues
        //    rotprobtableidx+0 is prob
        //    rotprobtableidx+1 is -ln(prob)
        tie(V, dVdphipsi) = tmol::numeric::bspline::
            ndspline<MAXBB, 3, D, Real, Int>::interpolate(
                rotameric_tables[rotprobtableidx + 1],
                all_rottable_idxs.topRows(MAXBB));
        accumulate<D, Real>::add(Vs[0], V);

        for (int i = 0; i < MAXBB; ++i) {
          for (int j = 0; j < 4; ++j) {
            Int phi_j = residue_params[res].bb_indices[i][j];
            accumulate<D, Vec<Real, 3>>::add(
                dVs_dx[phi_j][0],
                dVdphipsi[i] * all_drottable_idx_dxs.row(4 * i + j));
          }
        }
      } else {
        Real V;
        Vec<Real, MAXBB + 1> dVdphipsichi;

        // compute the -ln(P) energy for semirotameric residues
        //    semiprobtableidx+0 is prob
        //    semiprobtableidx+1 is -ln(prob)
        tie(V, dVdphipsichi) = tmol::numeric::bspline::
            ndspline<MAXBB + 1, 3, D, Real, Int>::interpolate(
                semirotameric_tables[semiprobtableidx + 1], all_rottable_idxs);
        accumulate<D, Real>::add(Vs[2], V);

        for (int i = 0; i <= MAXBB; ++i) {
          for (int j = 0; j < 4; ++j) {
            Int phi_j;
            if (i < MAXBB) {
              phi_j = residue_params[res].bb_indices[i][j];
            } else {
              phi_j = residue_params[res].chi_indices[semirotchi][j];
            }
            accumulate<D, Vec<Real, 3>>::add(
                dVs_dx[phi_j][2],
                dVdphipsichi[i] * all_drottable_idx_dxs.row(4 * i + j));
          }
        }
      }

      // 4. compute the chi-deviation penalty
      Real rotdevE = 0;
      for (int i = 0; i < residue_lookup_params[aa_idx].nrotchi; ++i) {
        Real Xmean, Xdev;
        Vec<Real, MAXBB> dXmean, dXdev;

        tie(Xmean, dXmean) = tmol::numeric::bspline::
            ndspline<MAXBB, 3, D, Real, Int>::interpolate(
                rotameric_tables[rotmeantableidx + 2 * i + 0],
                all_rottable_idxs.topRows(MAXBB));
        tie(Xdev, dXdev) = tmol::numeric::bspline::
            ndspline<MAXBB, 3, D, Real, Int>::interpolate(
                rotameric_tables[rotmeantableidx + 2 * i + 1],
                all_rottable_idxs.topRows(MAXBB));

        Real rotdelta_i =
            (all_chis[i] < -120.0_2rad ? all_chis[i] + Real(2 * M_PI) - Xmean
                                       : all_chis[i] - Xmean);

        rotdevE += rotdelta_i * rotdelta_i / (2 * Xdev * Xdev);

        // chi derivs
        for (int j = 0; j < 4; ++j) {
          Int chi_j = residue_params[res].chi_indices[i][j];
          Real drotdevEdchi = rotdelta_i / (Xdev * Xdev);
          accumulate<D, Vec<Real, 3>>::add(
              dVs_dx[chi_j][1], drotdevEdchi * all_dchi_dxs.row(4 * i + j));
        }

        // bb derivs
        for (int k = 0; k < MAXBB; ++k) {
          Real drotdevEdphi = -rotdelta_i / (Xdev * Xdev * Xdev)
                              * (Xdev * dXmean[k] + rotdelta_i * dXdev[k]);
          for (int j = 0; j < 4; ++j) {
            Int phi_j = residue_params[res].bb_indices[k][j];
            accumulate<D, Vec<Real, 3>>::add(
                dVs_dx[phi_j][1],
                drotdevEdphi * all_drottable_idx_dxs.row(4 * k + j));
          }
        }
      }
      accumulate<D, Real>::add(Vs[1], (rotdevE));
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
