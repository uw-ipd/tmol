#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/score/common/geom.hh>
#include <tmol/numeric/bspline_compiled/bspline.hh>

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


template <tmol::Device D, typename Real, typename Int>
struct DunbrackDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,

      TCollection<Real, 2, D> rotameric_prob_tables,
      TCollection<Real, 2, D> rotameric_neglnprob_tables,
      TCollection<Real, 2, D> rotameric_mean_tables,
      TCollection<Real, 2, D> rotameric_sdev_tables,
      TView<Vec<Real, 2>, 1, D> rotameric_bb_start,  // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_step,  // ntable-set entries
      TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries
      TCollection<Real, 3, D> semirotameric_tables, // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_start, // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_step, // n-semirot-tabset
      TView<Vec<Real, 3>, 1, D> semirot_periodicity, // n-semirot-tabset
      TView<Int, 1, D> rotameric_rotind2tableind,
      TView<Int, 1, D> semirotameric_rotind2tableind,

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
      TView<Eigen::Matrix<Real, 4, 3>, 1, D> ddihe_dxyz,  // ndihe x 4 x 3
      TView<Real, 1, D> dihedral_dE_ddihe,  // ndihe x 1
      // ! ! TView<Real, 1, D> rotchi_devpen, // n-rotameric-chi x 1
      // ! ! TView<Real, 2, D> ddevpen_dbb,  // Where d chimean/d dbbdihe is
                                      // stored, n-rotameric-chi x 2
      TView<Int, 1, D> rotameric_rottable_assignment,  // nres x 1
      TView<Int, 1, D> semirotameric_rottable_assignment  // nres x 1
  ) -> std::tuple<
    TPack<Real, 1, D>,       // -ln(prob_rotameric)
    TPack<CoordQuad, 2, D>,  // d(-ln(prob_rotameric)) / dbb atoms
    TPack<Real, 1, D>,       // Erotameric_chi_devpen
    TPack<CoordQuad, 1, D>,  // ddevpen_dphi
    TPack<CoordQuad, 1, D>,  // ddevpen_dphi
    TPack<CoordQuad, 1, D>,  // ddevpen_dchi
    TPack<Real, 1, D>,       // -ln(prob_nonrotameric)
    TPack<CoordQuad, 1, D>,  // d(-ln(prob_nonrotameric)) / dphi atoms
    TPack<CoordQuad, 1, D>,  // d(-ln(prob_nonrotameric)) / dpsi atoms
    TPack<CoordQuad, 1, D> > // d(-ln(prob_nonrotameric)) / dchi atoms
  {
    Int const nres(nrotameric_chi_for_res.size(0));
    Int const n_rotameric_res(prob_table_offset_for_rotresidue.size(0));
    Int const n_rotameric_chi(rotameric_chi_desc.size(0));
    Int const n_semirotameric_res(semirotameric_chi_desc.size(0));
    Int const n_dihedrals(dihedral_atom_inds.size(0));

    //std::cout << "nres " <<  nres << std::endl;
    //std::cout << "n_rotameric_res " << n_rotameric_res << std::endl;
    //std::cout << "n_rotameric_chi " << n_rotameric_chi << std::endl;
    //std::cout << "n_semirotameric_res " << n_semirotameric_res << std::endl;
    //std::cout << "n_dihedrals " << n_dihedrals << std::endl;

    auto neglnprob_rot_tpack = TPack<Real, 1, D>::zeros(n_rotameric_res);
    auto dneglnprob_rot_dbb_xyz_tpack = TPack<CoordQuad, 2, D>::zeros({n_rotameric_res, 2});

    auto rotchi_devpen_tpack = TPack<Real, 1, D>::zeros(n_rotameric_chi);
    auto drotchi_devpen_dphi_xyz_tpack = TPack<CoordQuad, 1, D>::zeros(n_rotameric_chi);
    auto drotchi_devpen_dpsi_xyz_tpack = TPack<CoordQuad, 1, D>::zeros(n_rotameric_chi);
    auto drotchi_devpen_dchi_xyz_tpack = TPack<CoordQuad, 1, D>::zeros(n_rotameric_chi);

    auto neglnprob_nonrot_tpack = TPack<Real, 1, D>::zeros(n_semirotameric_res);
    auto dneglnprob_nonrot_dphi_xyz_tpack = TPack<CoordQuad, 1, D>::zeros(n_semirotameric_res);
    auto dneglnprob_nonrot_dpsi_xyz_tpack = TPack<CoordQuad, 1, D>::zeros(n_semirotameric_res);
    auto dneglnprob_nonrot_dchi_xyz_tpack = TPack<CoordQuad, 1, D>::zeros(n_semirotameric_res);

    auto neglnprob_rot = neglnprob_rot_tpack.view;
    auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_tpack.view;

    auto rotchi_devpen = rotchi_devpen_tpack.view;
    auto drotchi_devpen_dphi_xyz =drotchi_devpen_dphi_xyz_tpack.view;
    auto drotchi_devpen_dpsi_xyz =drotchi_devpen_dpsi_xyz_tpack.view;
    auto drotchi_devpen_dchi_xyz =drotchi_devpen_dchi_xyz_tpack.view;

    auto neglnprob_nonrot = neglnprob_nonrot_tpack.view;
    auto dneglnprob_nonrot_dphi_xyz = dneglnprob_nonrot_dphi_xyz_tpack.view;
    auto dneglnprob_nonrot_dpsi_xyz = dneglnprob_nonrot_dpsi_xyz_tpack.view;
    auto dneglnprob_nonrot_dchi_xyz = dneglnprob_nonrot_dchi_xyz_tpack.view;
    
    
    //auto Vs_t = TPack<Real, 1, D>::zeros(nres);
    //auto Vs = Vs_t.view;
    //
    //auto dVs_dxyz_t = TPack<Real3, 1, D>::empty(coords.size(0));
    //auto dVs_dxyz = dVs_dxyz_t.view;

    // Five steps to this calculation
    // 0. (Initialization)
    // 1. compute the dihedrals and put them into the dihedrals array
    // 2. compute the rotameric bin for each residue
    // 3. compute the -ln(P) energy for rotameric residues
    // 4. compute the chi-deviation penalty for all rotameric chi
    // 5. compute the -ln(P) energy for the semi-rotameric residues

    //std::cout << "step 0" << std::endl;
    // 0.
    //for (int ii = 0; ii < nres; ++ii) {
    //  Vs[ii] = 0;
    //}

    for (int ii = 0; ii < n_dihedrals; ++ii) {
      ddihe_dxyz[ii].fill(0);
      dihedral_dE_ddihe[ii] = 0.;
    }
    //for (int ii=0; ii < coords.size(0); ++ii) {
    //  dVs_dxyz[ii].fill(0);
    //}


    // 1.
    //std::cout << "step 1" << std::endl;
    auto func_dihe = ([=] EIGEN_DEVICE_FUNC(int i) {
	measure_dihedrals_V_dV(
	  coords, i, dihedral_atom_inds,
	  dihedrals, ddihe_dxyz);
      });
    for (Int ii = 0; ii < n_dihedrals; ++ii) {
      func_dihe(ii);
    }

    // 2.
    //std::cout << "step 2" << std::endl;
    auto func_rot = ([=] EIGEN_DEVICE_FUNC(int i) {
	// Add in 2 backbone dihedrals for this residue
	Int dihe_offset = dihedral_offset_for_res[i] + 2;
	Int rot_ind = classify_rotamer( dihedrals,
	  nrotameric_chi_for_res[i],
	  dihe_offset);
	Int ri2ti_offset = rotind2tableind_offset_for_res[i];
	Int rotameric_table_ind = rotameric_rotind2tableind[
	  ri2ti_offset + rot_ind];
	Int semirotameric_table_ind = semirotameric_rotind2tableind[
	  ri2ti_offset + rot_ind];
	//std::cout << "residue " << i << " rotamer " << rot_ind << " rot table ind " << rotameric_table_ind << " semirot table ind " << semirotameric_table_ind << std::endl;
	rotameric_rottable_assignment[i] = rotameric_table_ind;
	semirotameric_rottable_assignment[i] = semirotameric_table_ind;

      });
    for (Int ii = 0; ii < nres; ++ii) {
      func_rot(ii);
    }

    // 3.
    //std::cout << "step 3" << std::endl;
    auto func_rotameric_prob = ([=] EIGEN_DEVICE_FUNC(Int i){
	Real neglnprobE;
	Eigen::Matrix<Real,2,1> dneglnprob_ddihe;
	Int ires = rotres2resid[i];
	std::tie(neglnprobE, dneglnprob_ddihe) = rotameric_chi_probability(
	  rotameric_neglnprob_tables,
	  rotameric_bb_start,
	  rotameric_bb_step,
	  rotameric_bb_periodicity,
	  prob_table_offset_for_rotresidue,
	  ires,
	  i,
	  dihedrals,
	  dihedral_offset_for_res,
	  rottable_set_for_res,
	  rotameric_rottable_assignment);

	//Vs[ires] = neglnprobE;
	neglnprob_rot[i] = neglnprobE;
	int ires_dihe_offset = dihedral_offset_for_res[ires];
	for ( int ii = 0; ii < 2; ++ii ) {
  	  for ( int jj = 0; jj < 4; ++jj ) {
	    dneglnprob_rot_dbb_xyz[i][ii].row(jj) = dneglnprob_ddihe(ii)*ddihe_dxyz[ires_dihe_offset+ii].row(jj);
	  }
	}

      });
    for (Int ii = 0; ii < n_rotameric_res; ++ii) {
      func_rotameric_prob(ii); // working properly!
    }


    // 4.
    //std::cout << "step 4" << std::endl;
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
	  rotameric_bb_periodicity,
	  ires,
	  inchi,
	  ichi_ind,
	  dihedrals,
	  dihedral_offset_for_res,
	  rottable_set_for_res,
	  rotmean_table_offset_for_residue,
	  rotameric_rottable_assignment);
	rotchi_devpen[i] = devpen;

	int phi_ind = dihedral_offset_for_res[ires];
	int psi_ind = dihedral_offset_for_res[ires] + 1;
	int chi_ind = dihedral_offset_for_res[ires] + 2 + ichi_ind;
	for ( int ii = 0; ii < 4; ++ii ) {
	  for ( int jj = 0; jj < 3; ++jj ) {
	    drotchi_devpen_dphi_xyz[i](ii,jj) = dpen_dbb(0)*ddihe_dxyz[phi_ind](ii,jj);
	    drotchi_devpen_dpsi_xyz[i](ii,jj) = dpen_dbb(1)*ddihe_dxyz[psi_ind](ii,jj);
	    drotchi_devpen_dchi_xyz[i](ii,jj) = dpen_dchi  *ddihe_dxyz[chi_ind](ii,jj);
	  }
	}

	//int const ires_dihe_offset = dihedral_offset_for_res[ires];
	//// dihedral_dE_ddihe[ires_dihe_offset+2+ichi_ind] = dpen_dchi;
	//// either need to accumulate dpen_dbb into a scratch space
	//// or use atomic increments into dihedral_dE_ddihe
	//for (Int ii = 0; ii < 2; ++ii) {
	//  ddevpen_dbb[i][ii] = dpen_dbb[ii];
	//}
      });
    for (Int ii = 0; ii < n_rotameric_chi; ++ii ) {
      func_chidevpen(ii); // working!
    }

    // 5.
    //std::cout << "step 5" << std::endl;
    auto func_semirot = ([=] EIGEN_DEVICE_FUNC(int i) {
	Real neglnprob;
	Eigen::Matrix<Real, 3, 1> dnlp_ddihe;

	Int const resid = semirotameric_chi_desc[i][0];
	Int const semirot_dihedral_index = semirotameric_chi_desc[i][1];
	Int const semirot_table_offset = semirotameric_chi_desc[i][2];
	Int const semirot_table_set = semirotameric_chi_desc[i][3];

	Int const res_dihe_offset = dihedral_offset_for_res[resid];

	tie(neglnprob, dnlp_ddihe) = semirotameric_energy(
	  semirotameric_tables,
	  semirot_start,
	  semirot_step,
	  semirot_periodicity,
	  dihedral_offset_for_res,
	  dihedrals,
	  semirotameric_rottable_assignment,
	  resid,
	  semirot_dihedral_index,
	  semirot_table_offset,
	  semirot_table_set);

	neglnprob_nonrot[i] = neglnprob;
	//Vs[resid] = neglnprob;

	int phi_ind = res_dihe_offset;
	int psi_ind = res_dihe_offset+1;
	int chi_ind = semirot_dihedral_index;
	for ( int ii = 0; ii < 4; ++ii ) {
	  for ( int jj = 0; jj < 3; ++jj ) {
	    dneglnprob_nonrot_dphi_xyz[i](ii,jj) = dnlp_ddihe(0)*ddihe_dxyz[phi_ind](ii,jj);
	    dneglnprob_nonrot_dpsi_xyz[i](ii,jj) = dnlp_ddihe(1)*ddihe_dxyz[psi_ind](ii,jj);
	    dneglnprob_nonrot_dchi_xyz[i](ii,jj) = dnlp_ddihe(2)*ddihe_dxyz[chi_ind](ii,jj);
	  }
	}
      });

    for ( int ii = 0; ii < n_semirotameric_res; ++ii ) {
      func_semirot(ii);
    }

    // // OK! now we just do some bookkeeping to accumulate the energies we've just computed
    // // on a per residue / per dihedral basis, and convert these into per-atom derivatives.
    // 
    //std::cout << "step 6" << std::endl;
    //for (int ii = 0; ii < n_rotameric_chi; ++ii) {
    //  int iiresid = rotameric_chi_desc[ii][0];
    //  int ii_dihe_offset = dihedral_offset_for_res[iiresid];
    //  Vs[iiresid] += rotchi_devpen[ii];
    //  //std::cout << ii << " " << iiresid << " " << ii_dihe_offset << " " << Vs[iiresid] << std::endl;
    //  // for (int jj = 0; jj < 2; ++jj) {
    //  // 	dihedral_dE_ddihe[ii_dihe_offset + jj] +=
    //  // 	  ddevpen_dbb[ii][jj];
    //  // 	//std::cout << "dihedral_dE_ddihe[" << ii_dihe_offset << " + " << jj << "] = " << dihedral_dE_ddihe[ii_dihe_offset + jj] << std::endl;
    //  // }
    //}
    
    // for (int ii = 0; ii < n_dihedrals; ++ii) {
    //   for (int jj = 0; jj < 4; ++jj) {
    // 	int jjat = dihedral_atom_inds[ii][jj];
    // 	//std::cout << "dihedral " << ii << " jjat " << jjat << std::endl;
    // 	if (jjat < 0) {
    // 	  continue;
    // 	}
    // 	for ( int kk = 0; kk < 3; ++kk ) {
    // 	  dVs_dxyz[jjat](kk) += ddihe_dxyz[ii](jj,kk) * dihedral_dE_ddihe[ii];
    // 	  //std::cout << "dVs_dxyz[" << jjat << "](" << kk << "): " << dVs_dxyz[jjat](kk) << " += " << ddihe_dxyz[ii](jj,kk) << " * " << dihedral_dE_ddihe[ii] << std::endl;
    // 	}
    //   }
    // }
    //std::cout << "done!" << std::endl;
     
    //for (int ii = 0; ii < nres; ++ii ) {
    //  std::cout << "Vs[ " << ii << "] = " << Vs[ii] << std::endl;
    //}
    // return {Vs_t, dVs_dxyz_t};
    return {
      neglnprob_rot_tpack,
      dneglnprob_rot_dbb_xyz_tpack,
      
      rotchi_devpen_tpack,
      drotchi_devpen_dphi_xyz_tpack,
      drotchi_devpen_dpsi_xyz_tpack,
      drotchi_devpen_dchi_xyz_tpack,
      
      neglnprob_nonrot_tpack,
      dneglnprob_nonrot_dphi_xyz_tpack,
      dneglnprob_nonrot_dpsi_xyz_tpack,
      dneglnprob_nonrot_dchi_xyz_tpack};
      
  }

  auto df(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> dE_drotnlp,
      TView<CoordQuad, 2, D> drot_nlp_dbb_xyz,
      TView<Real, 1, D> dE_ddevpen,
      TView<CoordQuad, 1, D> ddevpen_dphi_xyz, 
      TView<CoordQuad, 1, D> ddevpen_dpsi_xyz,
      TView<CoordQuad, 1, D> ddevpen_dchi_xyz,
      TView<Real, 1, D> dE_dnonrotnlp,
      TView<CoordQuad, 1, D> dnonrot_nlp_dphi_xyz,
      TView<CoordQuad, 1, D> dnonrot_nlp_dpsi_xyz,
      TView<CoordQuad, 1, D> dnonrot_nlp_dchi_xyz,
      TView<Int, 1, D> dihedral_offset_for_res,     // nres x 1
      TView<Vec<Int, 4>, 1, D> dihedral_atom_inds,  // ndihe x 4
      TView<Int, 1, D> rotres2resid,                      // nres x 1
      TView<Int, 2, D> rotameric_chi_desc,  // n-rotameric-chi x 2
      // rotchi_desc[:,0] == residue index for this chi
      // rotchi_desc[:,1] == chi_dihedral_index for res

      TView<Int, 2, D> semirotameric_chi_desc  // n-semirotameric-residues x 4
      // semirotchi_desc[:,0] == residue index
      // semirotchi_desc[:,1] == semirotchi_dihedral_index res
      // semirotchi_desc[:,2] == semirot_table_offset
      // semirotchi_desc[:,3] == semirot_table_set (e.g. 0-7)

  ) -> TPack<Real3, 1, D>
  {
    int natoms = coords.size(0);
    int n_rotameric_res = rotres2resid.size(0);
    int n_rotameric_chi = rotameric_chi_desc.size(0);
    int n_semirotameric_res = semirotameric_chi_desc.size(0);

    auto dE_dxyz_tpack = TPack<Real3, 1, D>::zeros(natoms);
    auto dE_dxyz = dE_dxyz_tpack.view;

    for ( int i = 0; i < n_rotameric_res; ++i ) {
      int ires = rotres2resid[i];
      int ires_dihe_offset = dihedral_offset_for_res[ires];
      for ( int ii = 0; ii < 2; ++ii ) {
	for ( int jj = 0; jj < 4; ++jj ) {
	  if ( dihedral_atom_inds[ires_dihe_offset+ii](0) >= 0 ) {
	    dE_dxyz[dihedral_atom_inds[ires_dihe_offset+ii](jj)] += dE_drotnlp[i] * drot_nlp_dbb_xyz[i][ii].row(jj);
	  }
	}
      }
    }
    for ( int i = 0; i < n_rotameric_chi; ++i ) {
      int ires = rotameric_chi_desc[i][0];
      int phi_ind = dihedral_offset_for_res[ires];
      int psi_ind = phi_ind + 1;
      int chi_ind = phi_ind + 2 + rotameric_chi_desc[i][1];
      for ( int ii = 0; ii < 4; ++ii ) {
	if ( dihedral_atom_inds[phi_ind](0) >= 0 ) {
	  dE_dxyz[dihedral_atom_inds[phi_ind](ii)] += dE_ddevpen[i] * ddevpen_dphi_xyz[i].row(ii);
	}
	if ( dihedral_atom_inds[psi_ind](0) >= 0 ) {
	  dE_dxyz[dihedral_atom_inds[psi_ind](ii)] += dE_ddevpen[i] * ddevpen_dpsi_xyz[i].row(ii);
	}
	dE_dxyz[dihedral_atom_inds[chi_ind](ii)] += dE_ddevpen[i] * ddevpen_dchi_xyz[i].row(ii);
      }
    }

    for ( int i = 0; i < n_semirotameric_res; ++i ) {
      int ires = semirotameric_chi_desc[i][0];
      int phi_ind = dihedral_offset_for_res[ires];
      int psi_ind = dihedral_offset_for_res[ires]+1;
      int chi_ind = semirotameric_chi_desc[i][1];
      for ( int ii = 0; ii < 4; ++ii ) {
	if ( dihedral_atom_inds[phi_ind](0) >= 0 ) {
	  dE_dxyz[dihedral_atom_inds[phi_ind](ii)] += dE_dnonrotnlp[i] * dnonrot_nlp_dphi_xyz[i].row(ii);
	}
	if ( dihedral_atom_inds[psi_ind](0) >= 0 ) {
	  dE_dxyz[dihedral_atom_inds[psi_ind](ii)] += dE_dnonrotnlp[i] * dnonrot_nlp_dpsi_xyz[i].row(ii);
	}
	dE_dxyz[dihedral_atom_inds[chi_ind](ii)] += dE_dnonrotnlp[i] * dnonrot_nlp_dchi_xyz[i].row(ii);
      }
    }

    return dE_dxyz_tpack;
  }
  

  
};

template struct DunbrackDispatch<tmol::Device::CPU,float,int32_t>;
template struct DunbrackDispatch<tmol::Device::CPU,double,int32_t>;
template struct DunbrackDispatch<tmol::Device::CPU,float,int64_t>;
template struct DunbrackDispatch<tmol::Device::CPU,double,int64_t>;

//TEMP!!!
template struct DunbrackDispatch<tmol::Device::CUDA,float,int32_t>;
template struct DunbrackDispatch<tmol::Device::CUDA,double,int32_t>;
template struct DunbrackDispatch<tmol::Device::CUDA,float,int64_t>;
template struct DunbrackDispatch<tmol::Device::CUDA,double,int64_t>;


}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
