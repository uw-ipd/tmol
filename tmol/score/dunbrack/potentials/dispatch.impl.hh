#include <Eigen/Core>

#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>

#include <ATen/Tensor.h>

#include <tuple>

#include <pybind11/pybind11.h>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>
#define Real2 Vec<Real, 2>
#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <template <tmol::Device> class Dispatch, tmol::Device D, typename Real, typename Int>
struct DunbrackDispatch {
  static auto forward(
      TView<Vec<Real, 3>, 2, D> coords,

      //TView<Real, 3, D> rotameric_prob_tables,
      TView<Real, 3, D> rotameric_neglnprob_tables,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
      TView<Real, 3, D> rotameric_mean_tables,
      TView<Real, 3, D> rotameric_sdev_tables,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
      TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries
      TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
      TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
      TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset
      TView<Int, 1, D> rotameric_rotind2tableind,
      TView<Int, 1, D> semirotameric_rotind2tableind,

      TView<Int, 2, D> ndihe_for_res,               // nres x 1
      TView<Int, 2, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 2, D> dihedral_atom_inds,  // ndihe x 4

      TView<Int, 2, D> rottable_set_for_res,              // nres x 1
      TView<Int, 2, D> nchi_for_res,                      // nres x 1
      TView<Int, 2, D> nrotameric_chi_for_res,            // nres x 1
      TView<Int, 2, D> rotres2resid,                      // nres x 1
      TView<Int, 2, D> prob_table_offset_for_rotresidue,  // n-rotameric-res x 1
      TView<Int, 2, D> rotind2tableind_offset_for_res,    // n-res x 1

      TView<Int, 2, D> rotmean_table_offset_for_residue,  // n-res x 1

      TView<Int, 3, D> rotameric_chi_desc,  // n-rotameric-chi x 2
      // rotchi_desc[:,:,0] == residue index for this chi
      // rotchi_desc[:,:,1] == chi_dihedral_index for res

      TView<Int, 3, D> semirotameric_chi_desc,  // n-semirotameric-residues x 4
      // semirotchi_desc[:,:,0] == residue index
      // semirotchi_desc[:,:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,:,2] == semirot_table_offset
      // semirotchi_desc[:,:,3] == semirot_table_set (e.g. 0-7)

      // scratch space, perhaps does not belong as an input parameter?
      TView<Real, 2, D> dihedrals,                        // ndihe x 1
      TView<Eigen::Matrix<Real, 4, 3>, 2, D> ddihe_dxyz,  // ndihe x 3
      // TView<Real, 1, D> rotchi_devpen,                    // n-rotameric-chi
      // x 1 TView<Real, 2, D> ddevpen_dbb,  // Where d chimean/d dbbdihe is
      //                                // stored, nscdihe x 2
      TView<Int, 2, D> rotameric_rottable_assignment,     // nres x 1
      TView<Int, 2, D> semirotameric_rottable_assignment  // nres x 1

      )
      -> std::tuple<
          TPack<Real, 2, D>,       // sum (energies) [rot, dev, semi]
          TPack<CoordQuad, 3, D>,  // d(-ln(prob_rotameric)) / dbb atoms
          TPack<CoordQuad, 3, D>,  // ddevpen_dtor_xyz -- nrotchi x (nbb+1)
          TPack<CoordQuad, 3, D>>  // d(-ln(prob_nonrotameric)) / dtor --
                                   // nsemirot-res x 3
  {
    Int const nstacks(nrotameric_chi_for_res.size(0));
    Int const nres(nrotameric_chi_for_res.size(1));
    Int const n_rotameric_res(prob_table_offset_for_rotresidue.size(1));
    Int const n_rotameric_chi(rotameric_chi_desc.size(1));
    Int const n_semirotameric_res(semirotameric_chi_desc.size(1));
    Int const n_dihedrals(dihedral_atom_inds.size(1));

    auto V_tpack = TPack<Real, 2, D>::zeros({nstacks, 3});
    auto V = V_tpack.view;

    // auto neglnprob_rot_tpack = TPack<Real, 2, D>::zeros(nstacks,
    // n_rotameric_res);
    auto dneglnprob_rot_dbb_xyz_tpack =
        TPack<CoordQuad, 3, D>::zeros({nstacks, n_rotameric_res, 2});

    // auto rotchi_devpen_tpack = TPack<Real, 2, D>::zeros(nstacks,
    // n_rotameric_chi);
    auto drotchi_devpen_dtor_xyz_tpack =
        TPack<CoordQuad, 3, D>::zeros({nstacks, n_rotameric_chi, 3});

    // auto neglnprob_nonrot_tpack = TPack<Real, 1,
    // D>::zeros(n_semirotameric_res);
    auto dneglnprob_nonrot_dtor_xyz_tpack =
        TPack<CoordQuad, 3, D>::zeros({nstacks, n_semirotameric_res, 3});

    // auto neglnprob_rot = neglnprob_rot_tpack.view;
    auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_tpack.view;

    // auto rotchi_devpen = rotchi_devpen_tpack.view;
    auto drotchi_devpen_dtor_xyz = drotchi_devpen_dtor_xyz_tpack.view;

    // auto neglnprob_nonrot = neglnprob_nonrot_tpack.view;
    auto dneglnprob_nonrot_dtor_xyz = dneglnprob_nonrot_dtor_xyz_tpack.view;

    // auto rotameric_neglnprob_tables_view = rotameric_neglnprob_tables.view;
    // auto rotameric_mean_tables_view = rotameric_mean_tables.view;
    // auto rotameric_sdev_tables_view = rotameric_sdev_tables.view;
    // auto semirotameric_tables_view = semirotameric_tables.view;

    // Five steps to this calculation
    // 1. compute the dihedrals and put them into the dihedrals array
    // 2. compute the rotameric bin for each residue
    // 3. compute the -ln(P) energy for rotameric residues
    // 4. compute the chi-deviation penalty for all rotameric chi
    // 5. compute the -ln(P) energy for the semi-rotameric residues

    // 1.
    auto func_dihe = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      measure_dihedrals_V_dV(
          coords[stack],
          i,
          dihedral_atom_inds[stack],
          dihedrals[stack],
          ddihe_dxyz[stack]);
    });
    Dispatch<D>::forall_stacks(nstacks, n_dihedrals, func_dihe);

    // 2.
    auto func_rot = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      // Templated on there being 2 backbone dihedrals for canonical aas.
      if (nrotameric_chi_for_res[stack][i] >= 0) {
        classify_rotamer_for_res<2>(
            dihedrals[stack],
            dihedral_offset_for_res[stack],
            nrotameric_chi_for_res[stack],
            rotind2tableind_offset_for_res[stack],
            rotameric_rotind2tableind,
            semirotameric_rotind2tableind,
            rotameric_rottable_assignment[stack],
            semirotameric_rottable_assignment[stack],
            i);
      }
    });
    Dispatch<D>::forall_stacks(nstacks, nres, func_rot);

    // 3.
    auto func_rotameric_prob = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      if (rotres2resid[stack][i] >= 0) {
        auto Erot = rotameric_chi_probability_for_res(
            rotameric_neglnprob_tables,
            rotprob_table_sizes,
            rotprob_table_strides,
            rotameric_bb_start,
            rotameric_bb_step,
            rotameric_bb_periodicity,
            prob_table_offset_for_rotresidue[stack],
            dihedrals[stack],
            dihedral_offset_for_res[stack],
            rottable_set_for_res[stack],
            rotameric_rottable_assignment[stack],
            rotres2resid[stack],
            dneglnprob_rot_dbb_xyz[stack],
            ddihe_dxyz[stack],
            i);
        common::accumulate<D, Real>::add(V[stack][0], Erot);
      }
    });
    Dispatch<D>::forall_stacks(nstacks, n_rotameric_res, func_rotameric_prob);

    // 4.
    auto func_chidevpen = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      if (rotameric_chi_desc[stack][i][0] >= 0) {
        auto Erotdev = deviation_penalty_for_chi(
            rotameric_mean_tables,
            rotameric_sdev_tables,
            rotmean_table_sizes,
            rotmean_table_strides,
            rotameric_bb_start,
            rotameric_bb_step,
            rotameric_bb_periodicity,
            dihedrals[stack],
            dihedral_offset_for_res[stack],
            rottable_set_for_res[stack],
            rotmean_table_offset_for_residue[stack],
            rotameric_rottable_assignment[stack],
            rotameric_chi_desc[stack],
            nchi_for_res[stack],
            drotchi_devpen_dtor_xyz[stack],
            ddihe_dxyz[stack],
            i);
        common::accumulate<D, Real>::add(V[stack][1], Erotdev);
      }
    });
    Dispatch<D>::forall_stacks(nstacks, n_rotameric_chi, func_chidevpen);

    // 5.
    auto func_semirot = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      if (semirotameric_chi_desc[stack][i][0] >= 0) {
        auto Esemi = semirotameric_energy(
            semirotameric_tables,
            semirot_table_sizes,
            semirot_table_strides,
            semirot_start,
            semirot_step,
            semirot_periodicity,
            dihedral_offset_for_res[stack],
            dihedrals[stack],
            semirotameric_rottable_assignment[stack],
            semirotameric_chi_desc[stack],
            i,
            dneglnprob_nonrot_dtor_xyz[stack],
            ddihe_dxyz[stack]);
        common::accumulate<D, Real>::add(V[stack][2], Esemi);
      }
    });
    Dispatch<D>::forall_stacks(nstacks, n_semirotameric_res, func_semirot);

    return {V_tpack,
            dneglnprob_rot_dbb_xyz_tpack,
            drotchi_devpen_dtor_xyz_tpack,
            dneglnprob_nonrot_dtor_xyz_tpack};
  }

  static auto backward(
      TView<Real, 2, D> dTdV,
      TView<Vec<Real, 3>, 2, D> coords,
      TView<CoordQuad, 3, D> drot_nlp_dbb_xyz,  // nstacks x n-rotameric-res x 2
      TView<CoordQuad, 3, D> ddevpen_dtor_xyz,  // nstacks x n-rotameric-chi x 3
      TView<CoordQuad, 3, D> dnonrot_nlp_dtor_xyz,
      TView<Int, 2, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 2, D> dihedral_atom_inds,  // ndihe x 4
      TView<Int, 2, D> rotres2resid,                // nres x 1
      TView<Int, 3, D> rotameric_chi_desc,          // n-rotameric-chi x 2
      TView<Int, 3, D> semirotameric_chi_desc  // n-semirotameric-residues x 4
      ) -> TPack<Real3, 2, D> {
    int nstacks = coords.size(0);
    int natoms = coords.size(1);
    int n_rotameric_res = rotres2resid.size(1);
    int n_rotameric_chi = rotameric_chi_desc.size(1);
    int n_semirotameric_res = semirotameric_chi_desc.size(1);

    auto dE_dxyz_tpack = TPack<Real3, 2, D>::zeros({nstacks, natoms});
    auto dE_dxyz = dE_dxyz_tpack.view;

    auto func_accum_rotnlp = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      int ires = rotres2resid[stack][i];
      if (ires >= 0) {
        int ires_dihe_offset = dihedral_offset_for_res[stack][ires];
        for (int ii = 0; ii < 2; ++ii) {
          for (int jj = 0; jj < 4; ++jj) {
            int const jj_at =
                dihedral_atom_inds[stack][ires_dihe_offset + ii](jj);
            if (jj_at >= 0) {
              accumulate<D, Vec<Real, 3>>::add(
                  dE_dxyz[stack][jj_at],
                  dTdV[stack][0] * drot_nlp_dbb_xyz[stack][i][ii].row(jj));
            }
          }
        }
      }
    });
    Dispatch<D>::forall_stacks(nstacks, n_rotameric_res, func_accum_rotnlp);

    auto func_accum_chidev = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      int ires = rotameric_chi_desc[stack][i][0];
      if (ires >= 0) {
        int ires_dihe_offset = dihedral_offset_for_res[stack][ires];
        int ichi_ind = rotameric_chi_desc[stack][i][1];

        for (int ii = 0; ii < 3; ++ii) {
          int tor_ind = ires_dihe_offset + (ii == 2 ? (2 + ichi_ind) : ii);
          for (int jj = 0; jj < 4; ++jj) {
            if (dihedral_atom_inds[stack][tor_ind](0) >= 0) {
              accumulate<D, Vec<Real, 3>>::add(
                  dE_dxyz[stack][dihedral_atom_inds[stack][tor_ind](jj)],
                  dTdV[stack][1] * ddevpen_dtor_xyz[stack][i][ii].row(jj));
            }
          }
        }
      }
    });
    Dispatch<D>::forall_stacks(nstacks, n_rotameric_chi, func_accum_chidev);

    auto func_accum_nonrotnlp = ([=] EIGEN_DEVICE_FUNC(int stack, int i) {
      int ires = semirotameric_chi_desc[stack][i][0];
      if (ires >= 0) {
        int ires_dihe_offset = dihedral_offset_for_res[stack][ires];
        int ichi_ind = semirotameric_chi_desc[stack][i][1];
        for (int ii = 0; ii < 3; ++ii) {
          int tor_ind = ii == 2 ? ichi_ind : (ires_dihe_offset + ii);
          for (int jj = 0; jj < 4; ++jj) {
            if (dihedral_atom_inds[stack][tor_ind](0) >= 0) {
              accumulate<D, Vec<Real, 3>>::add(
                  dE_dxyz[stack][dihedral_atom_inds[stack][tor_ind](jj)],
                  dTdV[stack][2] * dnonrot_nlp_dtor_xyz[stack][i][ii].row(jj));
            }
          }
        }
      }
    });
    Dispatch<D>::forall_stacks(
        nstacks, n_semirotameric_res, func_accum_nonrotnlp);

    return dE_dxyz_tpack;
  }
};

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
