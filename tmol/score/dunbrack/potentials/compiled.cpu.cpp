#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/score/common/geom.hh>
#include <tmol/numeric/bspline_compiled/bspline.hh>

#include <ATen/Tensor.h>

#include <tuple>

#include <pybind11/pybind11.h>

//#include "potentials.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>
#define Real2 Vec<Real, 2>

template <tmol::Device D, typename Real, typename Int>
struct DunbrackDispatch {
  static auto f(
      TCollection<Real, 2, D> rotameric_prob_tables,
      TCollection<Real, 2, D> rotameric_mean_tables,
      TCollection<Real, 2, D> rotameric_sdev_tables,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,  // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,  // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries
      TCollection<Real, 3, D> semirotameric_tables, // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_start, // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_step, // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_periodicity, // n-semirot-tabset
      TCollection<Int, 1, D> rotind2tableind,

      TView<Vec<Real, 3>, 1, D> coords,

      TView<Int, 1, D> ndihe_for_res,            // nres x 1
      TView<Int, 1, D> dihedral_offset_for_res,  // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,       // ndihe x 4

      TView<Int, 1, D> rottable_set_for_res, // nres x 1
      TView<Int, 1, D> nchi_for_res, // nres x 1
      TView<Int, 1, D> nrotameric_chi_for_res, // nres x 1
      TView<Int, 1, D> rotres2resid, // nres x 1
      TView<Int, 1, D> prob_table_offset_for_rotresidue, // n-rotameric-res x 1
      TView<Int, 1, D> rotind2tableind_offset_for_res, // n-res x 1

      TView<Int, 1, D> rotmean_table_offset_for_residue, // n-res x 1
      
      TView<Int, 2, D> rotameric_chi_desc, // n-rotameric-chi x 2
      // rotchi_desc[:,0] == residue index for this chi
      // rotchi_desc[:,1] == chi_dihedral_index for res

      TView<Int, 2, D> semirotameric_chi_desc, // n-semirotameric-residues x 4
      // semirotchi_desc[:,0] == residue index
      // semirotchi_desc[:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,2] == semirot_table_offset
      // semirotchi_desc[:,3] == semirot_table_set (e.g. 0-7)
      
      // scratch space, perhaps does not belong as an input parameter?
      TView<Real, 1, D> dihedrals,  // ndihe x 1
      TView<Real, 2, D> ddihe_dxyz,  // ndihe x 3
      TView<Real, 1, D> dihedral_dE_ddihe,  // ndihe x 1
      TView<Real, 1, D> rotchi_devpen, // n-rotameric-chi x 1
      TView<Real, 2, D> ddevpen_dbb,  // Where d chimean/d dbbdihe is
                                      // stored, nscdihe x 2
      TView<Int, 1, D> rottable_assignment,  // nres x 1
      ) -> TPack<Real, 1, D>
  {
    Int const nres(nrotameric_chi_for_res.size(0));
    Int const n_rotameric_res(prob_table_offset_for_residue.size(0));
    Int const n_rotameric_chi(rotameric_chi_desc.size(0));
    Int const n_semirotameric_res(semirotameric_chi_desc.size(0));
    Int const n_dihedrals(dihedral_atom_inds.size(0));

    auto Vs_t = TPack<Real, 1, D>::empty(nres);
    auto Vs = Vs_t.view;
    
    // Five steps to this calculation
    // 0. (Initialization)
    // 1. compute the dihedrals and put them into the dihedrals array
    // 2. compute the rotameric bin for each residue
    // 3. compute the -ln(P) energy for rotameric residues
    // 4. compute the chi-deviation penalty for all rotameric chi
    // 5. compute the -ln(P) energy for the semi-rotameric residues

    // 0.
    for (int ii = 0; ii < nres; ++ii) {
      Vs[ii] = 0;
    }
    for (int ii = 0; ii < n_dihedrals; ++ii) {
      for (int jj = 0; jj < 3; ++jj ) {
	ddihe_dxyz[ii][jj] = 0;
      }
      dihedral_dE_ddihe[ii] = 0;
    }

    
    // 1.
    auto func_dihe = ([=] EIGEN_DEVICE_FUNC(int i) {
	measure_dihedrals_V_dV(
	  coords, i, dihedral_atom_inds,
	  dihedrals, ddihe_dxyz);
      });
    for (Int ii = 0; ii < n_dihedrals; ++ii) {
      func_dihe(i);
    }

    // 2.
    auto func_rot = ([=] EIGEN_DEVICE_FUNC(int i) {
	// Add in 2 backbone dihedrals for this residue
	Int dihe_offset = dihedral_offset_for_res[i] + 2;
	Int rot_ind = classify_rotamer( dihedrals,
	  nrotameric_chi_for_res[i],
	  dihe_offset);
	Int table_ind = rotind2tableind[
	  rotind2tableind_offset_for_res[i] + rot_ind];
	rottable_assignment[i] = table_ind;
      });
    for (Int ii = 0; ii < nres; ++ii) {
      func_rot(i);
    }

    // 3.
    auto func_rotameric_prob = ([=] EIGEN_DEVICE_FUNC(Int i){
	Real neglnprobE;
	Eigen::Matrix<Real,2,1> dneglnprob_ddihe;
	Int ires = rotres2resid[i];
	std::tie(neglnprobE, dneglnprob_ddihe) = rotameric_chi_probability(
	  rotameric_prob_tables,
	  prob_table_offset_for_rotresidue,
	  resi,
	  i,
	  dihedrals,
	  dihedral_offset_for_res,
	  rottable_assignment);
	
	Vs[resi] = neglnprobE;
	Int const ires_dihedral_offset = dihedral_offset_for_res[residue_ind];
	for (Int ii = 0; ii < 2; ++ii ) {
	  dihedral_dE_ddihe[ires_dihedral_offset + ii] += dneglnprob_ddihe[ii];
	}
      });
    for (Int ii = 0; ii < n_rotameric_res; ++ii) {
      func_rotameric_prob(ii);
    }
	  

    // 4.
    auto func_chidevpen = ([=] EIGEN_DEVICE_FUNC(int i) {
	int ires = rotameric_chi_desc[i][0];
	int ichi_ind = rotameric_chi_desc[i][1];
	int inchi = nchi_for_res[ires];
	Real devpen, dpen_dchi;
	Eigen::Matrix<Real, 2, 1> dpen_dbb;
	
	std::tie(devpen, dpen_dchi, dpen_dbb) = chi_deviation_penalty(
	  rotameric_mean_tables,
	  rotameric_sdev_tables,
	  rotameric_bb_start,
	  rotameric_bb_step,
	  ires,
	  inchi,
	  ichi_ind,
	  dihedrals,
	  dihedral_offset_for_res,
	  rottable_set_for_res,
	  rotmean_table_set_offset,
	  rottable_assignment);
	rotchi_devpen[i] = devpen;
	Int const ires_dihe_offset = dihedral_offset_for_res[ires];
	dihedral_dE_ddihe[ires_dihe_offset+2+ichi_ind] = dpen_dchi;
	// either need to accumulate dpen_dbb into a scratch space
	// or use atomic increments into dihedral_dE_ddihe
	for (Int ii = 0; ii < 2; ++ii) {
	  ddevpen_dbb[ires_dihe_offset + ii] = dpen_dbb[ii];
	}
      });
    for (Int ii = 0; ii < n_rotameric_chi; ++ii ) {
      func_chidevpen(ii);
    }

    // 5.
    auto func_semirot = ([=] EIGEN_DEVICE_FUNC(int i) {
	Real neglnprob;
	Eigen::Matrix<Real, 3, 1> dnlp_ddihe;

	Int const resid = semirotchi_desc[i][0];
	Int const semirot_dihedral_index = semirotchi_desc[i][1];
	Int const semirot_table_offset = semirotchi_desc[i][2];
	Int const semirot_table_set = semirotchi_desc[i][3];
	
	Int const res_dihe_offset = dihedral_offset_for_res[resid];
      
	tie(neglnprob, dnlp_ddihe) = semirotameric_energy(
	  semirotameric_tables,
	  semirot_start,
	  semirot_stop,
	  semirot_periodicity,
	  semirot_table_offset,
	  dihedral_offset_for_res,
	  rottable_assignment,
	  resid,
	  semirot_dihedral_index,
	  semirot_table_offset,
	  semitor_table_set);

	Vs[resid] = neglnprob;
	for ( int ii = 0; ii < 3; ++ii ) {
	  int ii_dihe_ind = res_dihe_offset + ( ii == 2 ? semirot_dihedral_index : ii );
	  dihedral_dE_ddihe[ii_dihe_ind] += dnlp_ddihe[ii];
	}
      });

    for ( int ii = 0; ii < n_semirotameric_res; ++ii ) {
      func_semirot(ii);
    }

    // OK! now we just do some bookkeeping to accumulate the energies we've just computed
    // on a per residue / per dihedral basis, and convert these into per-atom derivatives.

    for (int ii = 0; ii < n_rotameric_chi; ++ii) {
      int iiresid = rotameric_chi_desc[ii][0];
      int ii_dihe_offset = dihedral_offset_for_res[iiresid];
      Vs[iiresid] += rotchi_devpen[ii];
      for (int jj = 0; jj < 2; ++jj) {
	dihedral_dE_ddihe[ii_dihe_offset + jj] +=
	  ddevpen_dbb[ii][jj];
      }
    }

    for (int ii = 0; ii <= n_dihedrals; ++ii) {
      for (int jj = 0; jj < 4; ++jj) {
	int jjat = dihedral_atom_inds[ii][jj];
	dVs_dxyz[jjat] += ddihe_dxyz[ii][jj] * dihedral_dE_ddihe[ii];
      }
    }
    
  }
};



template struct DunbrackDispatch<tmol::Device::CPU,float,int32_t>;
template struct DunbrackDispatch<tmol::Device::CPU,double,int32_t>;
template struct DunbrackDispatch<tmol::Device::CPU,float,int64_t>;
template struct DunbrackDispatch<tmol::Device::CPU,double,int64_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol