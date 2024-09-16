// Device-agnostic common numeric routines for kinematics.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <moderngpu/scan_types.hxx>
#include <moderngpu/operators.hxx>

#include <pybind11/pybind11.h>

namespace tmol {
namespace kinematics {

#define Dofs Eigen::Matrix<Real, 9, 1>
#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define QuatTranslation Eigen::Matrix<Real, 7, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

enum DOFtype { ROOT = 0, JUMP, BOND };

enum JumpDOFidx {
  DOF_RBx = 0,
  DOF_RBy,
  DOF_RBz,
  DOF_rotdelalpha,
  DOF_rotdelbeta,
  DOF_rotdelgamma,
  DOF_rotalpha,
  DOF_rotbeta,
  DOF_rotgamma
};
enum BondDOFidx { DOF_phip = 0, DOF_theta, DOF_D, DOF_phic };

template <tmol::Device D, typename Real, typename Int>
struct common {
  // compute JUMP derivatives from f1/f2
  //
  // Translational derivatives are straightforward dot products of f2s
  //     (the downstream derivative sum)
  //
  // Rotational derivatives use the Abe and Go "trick" that allows us to easily
  // compute derivatives with respect to rotation about an axis.
  //
  // In this case, there are three axes to compute derivatives of::
  //     1) the Z axis (alpha rotation)
  //     2) the Y axis after applying the alpha rotation (beta rotation)
  //     3) the X axis after applying the alpha & beta rot (gamma rotation)
  //
  // Derivatives are ONLY assigned to the RBdel DOFs
  static auto EIGEN_DEVICE_FUNC jumpDerivatives(
      Dofs dof,
      HomogeneousTransform M,
      HomogeneousTransform Mparent,
      Coord f1,
      Coord f2) -> Dofs {
    Dofs dsc_ddof = Dofs::Constant(0.0);

    Coord xaxis = Mparent.block(0, 0, 1, 3).transpose();
    Coord yaxis = Mparent.block(1, 0, 1, 3).transpose();
    Coord zaxis = Mparent.block(2, 0, 1, 3).transpose();

    // dE/drb_i
    dsc_ddof[DOF_RBx] = xaxis.dot(f2);
    dsc_ddof[DOF_RBy] = yaxis.dot(f2);
    dsc_ddof[DOF_RBz] = zaxis.dot(f2);

    Coord end_pos = M.block(3, 0, 1, 3).transpose();
    Coord rotdof3_axes = -zaxis;

    HomogeneousTransform zrot = HomogeneousTransform::Identity();
    zrot(0, 0) = std::cos(dof[DOF_rotdelgamma]);
    zrot(0, 1) = std::sin(dof[DOF_rotdelgamma]);
    zrot(1, 0) = -std::sin(dof[DOF_rotdelgamma]);
    zrot(1, 1) = std::cos(dof[DOF_rotdelgamma]);
    Coord rotdof2_axes = -(zrot * Mparent).block(1, 0, 1, 3).transpose();

    HomogeneousTransform yrot = HomogeneousTransform::Identity();
    yrot(0, 0) = std::cos(-dof[DOF_rotdelbeta]);
    yrot(0, 2) = std::sin(-dof[DOF_rotdelbeta]);
    yrot(2, 0) = -std::sin(-dof[DOF_rotdelbeta]);
    yrot(2, 2) = std::cos(-dof[DOF_rotdelbeta]);
    Coord rotdof1_axes =
        -(yrot * (zrot * Mparent)).block(0, 0, 1, 3).transpose();

    // dE/drot_i
    dsc_ddof[DOF_rotdelalpha] =
        rotdof1_axes.dot(f1) + f2.dot(rotdof1_axes.cross(end_pos));
    dsc_ddof[DOF_rotdelbeta] =
        rotdof2_axes.dot(f1) + f2.dot(rotdof2_axes.cross(end_pos));
    dsc_ddof[DOF_rotdelgamma] =
        rotdof3_axes.dot(f1) + f2.dot(rotdof3_axes.cross(end_pos));

    return dsc_ddof;
  }

  // Compute BOND derivatives from f1/f2
  //
  // The d derivatives are straightforward dot products of f2s
  //      (the downstream derivative sum)
  //
  // Other DOF derivatives use the Abe and Go "trick" that allows us to
  //    easily compute derivatives with respect to rotation about an axis.
  //
  // The phi_p and phi_c derivs are simply rotation about the X axis of the
  //    parent and child coordinate frame, respectively
  //
  // The theta derivs are more complex. Similar to jump derivs, we need to
  //    UNDO the phi_c rotation, and then take the Z axis of the child HT
  static auto EIGEN_DEVICE_FUNC bondDerivatives(
      Dofs dof,
      HomogeneousTransform M,
      HomogeneousTransform Mparent,
      Coord f1,
      Coord f2) -> Dofs {
    Dofs dsc_ddof =
        Dofs::Constant(0);  //(NAN); <- chnage to 0 so deriv checks work wo mask

    Coord end_ppos = Mparent.block(3, 0, 1, 3).transpose();
    Coord end_cpos = M.block(3, 0, 1, 3).transpose();
    Coord phi_paxis = Mparent.block(0, 0, 1, 3).transpose();
    Coord phi_caxis = M.block(0, 0, 1, 3).transpose();

    // to get the theta axis, we need to undo the phi_c rotation (about x)
    HomogeneousTransform unrot_phic = HomogeneousTransform::Identity();
    unrot_phic(1, 1) = std::cos(-dof[DOF_phic]);
    unrot_phic(1, 2) = std::sin(-dof[DOF_phic]);
    unrot_phic(2, 1) = -std::sin(-dof[DOF_phic]);
    unrot_phic(2, 2) = std::cos(-dof[DOF_phic]);
    Coord theta_axis = (unrot_phic * M).block(2, 0, 1, 3).transpose();

    dsc_ddof[DOF_D] = phi_caxis.dot(f2);
    dsc_ddof[DOF_theta] =
        -theta_axis.dot(f1) - f2.dot(theta_axis.cross(end_ppos));
    dsc_ddof[DOF_phip] = -phi_paxis.dot(f1) - f2.dot(phi_paxis.cross(end_ppos));
    dsc_ddof[DOF_phic] = -phi_caxis.dot(f1) - f2.dot(phi_caxis.cross(end_cpos));

    return dsc_ddof;
  }

  // convert BOND dofs to HT
  //   * each bond has four dofs: [phi_p, theta, d, phi_c]
  //   * in the local frame:
  //      - phi_p and phi_c are a rotation about x
  //      - theta is a rotation about z
  //      - d is a translation along x
  //   * the matrix 'HT' is a composition:
  //        M = (
  //            rot(phi_p, [1,0,0])
  //            @ rot(theta, [0,0,1]
  //            @ trans(d, [1,0,0])
  //            @ rot(phi_c, [1,0,0])
  //        )
  static auto EIGEN_DEVICE_FUNC bondTransform(Dofs dof) {
    Real cpp = std::cos(dof[DOF_phip]);
    Real spp = std::sin(dof[DOF_phip]);
    Real cpc = std::cos(dof[DOF_phic]);
    Real spc = std::sin(dof[DOF_phic]);
    Real cth = std::cos(dof[DOF_theta]);
    Real sth = std::sin(dof[DOF_theta]);
    Real d = dof[DOF_D];

    HomogeneousTransform HT;
    HT(0, 0) = cth;
    HT(1, 0) = -cpc * sth;
    HT(2, 0) = spc * sth;
    HT(3, 0) = d * cth;
    HT(0, 1) = cpp * sth;
    HT(1, 1) = cpc * cpp * cth - spc * spp;
    HT(2, 1) = -cpp * cth * spc - cpc * spp;
    HT(3, 1) = d * cpp * sth;
    HT(0, 2) = spp * sth;
    HT(1, 2) = cpp * spc + cpc * cth * spp;
    HT(2, 2) = cpc * cpp - cth * spc * spp;
    HT(3, 2) = d * spp * sth;
    HT(0, 3) = 0;
    HT(1, 3) = 0;
    HT(2, 3) = 0;
    HT(3, 3) = 1;

    return HT;
  }

  // HTs -> BOND dofs
  //
  // Given the matrix definition in BondTransforms, we calculate the dofs
  //   that give rise to this HT.
  // A special case below handles a "singularity," that is, a configuration
  //   where there are multiple parameterizations that give the same HT
  // Specifically, when theta==0, the rx rotation can be put into
  //   phi_c or phi_p (we use phi_c)
  static auto EIGEN_DEVICE_FUNC invBondTransform(HomogeneousTransform M) {
    Dofs dof =
        Dofs::Constant(0);  //(NAN); <- chnage to 0 so deriv checks work wo mask

    if (std::fabs(M(0, 0) - 1) < 1e-6) {
      dof[DOF_phip] = 0.0;
      dof[DOF_phic] = std::atan2(M(1, 2), M(1, 1));
      dof[DOF_theta] = 0.0;
    } else {
      dof[DOF_phip] = std::atan2(M(0, 2), M(0, 1));
      dof[DOF_phic] = std::atan2(-M(2, 0), -M(1, 0));
      dof[DOF_theta] =
          std::atan2(std::sqrt(M(1, 0) * M(1, 0) + M(2, 0) * M(2, 0)), M(0, 0));
    }
    dof[DOF_D] = M.bottomLeftCorner(1, 3).norm();

    return dof;
  }

  // convert JUMP dofs to HT
  //   * each jump has _9_ parameters:
  //     - 3 translational  [0-2]
  //     - 3 rotational deltas [3-5]
  //     - 3 rotational [6-8]
  //   * only the rotational deltas are exposed to minimization
  //     - RBdel_* is meant to be reset to zero at the beginning of a
  //       minimization trajectory, as when parameters are near 0,
  //       minimization is well-behaved.
  //   * translations are represented as an offset in X,Y,Z
  //   * rotations and rotational deltas are ZYX Euler angles.
  //      - that is, a rotation about Z, then Y, then X.
  //   * the matrix HT is a composition:
  //      M = trans( RBx, RBy, RBz)
  //          @ roteuler( RBdel_alpha, RBdel_alpha, RBdel_alpha)
  //          @ roteuler( RBalpha, RBalpha, RBalpha)
  static auto EIGEN_DEVICE_FUNC jumpTransform(Dofs dof) {
    Real si = std::sin(dof[DOF_rotdelalpha]);
    Real sj = std::sin(dof[DOF_rotdelbeta]);
    Real sk = std::sin(dof[DOF_rotdelgamma]);
    Real ci = std::cos(dof[DOF_rotdelalpha]);
    Real cj = std::cos(dof[DOF_rotdelbeta]);
    Real ck = std::cos(dof[DOF_rotdelgamma]);
    Real cc = ci * ck;
    Real cs = ci * sk;
    Real sc = si * ck;
    Real ss = si * sk;

    HomogeneousTransform HTdelta;
    HTdelta(0, 0) = cj * ck;
    HTdelta(1, 0) = sj * sc - cs;
    HTdelta(2, 0) = sj * cc + ss;
    HTdelta(0, 1) = cj * sk;
    HTdelta(1, 1) = sj * ss + cc;
    HTdelta(2, 1) = sj * cs - sc;
    HTdelta(0, 2) = -sj;
    HTdelta(1, 2) = cj * si;
    HTdelta(2, 2) = cj * ci;

    HTdelta(3, 0) = dof[DOF_RBx];
    HTdelta(3, 1) = dof[DOF_RBy];
    HTdelta(3, 2) = dof[DOF_RBz];
    HTdelta(0, 3) = HTdelta(1, 3) = HTdelta(2, 3) = 0;
    HTdelta(3, 3) = 1;

    si = std::sin(dof[DOF_rotalpha]);
    sj = std::sin(dof[DOF_rotbeta]);
    sk = std::sin(dof[DOF_rotgamma]);
    ci = std::cos(dof[DOF_rotalpha]);
    cj = std::cos(dof[DOF_rotbeta]);
    ck = std::cos(dof[DOF_rotgamma]);
    cc = ci * ck;
    cs = ci * sk;
    sc = si * ck;
    ss = si * sk;

    HomogeneousTransform HTglobal;
    HTglobal(0, 0) = cj * ck;
    HTglobal(1, 0) = sj * sc - cs;
    HTglobal(2, 0) = sj * cc + ss;
    HTglobal(0, 1) = cj * sk;
    HTglobal(1, 1) = sj * ss + cc;
    HTglobal(2, 1) = sj * cs - sc;
    HTglobal(0, 2) = -sj;
    HTglobal(1, 2) = cj * si;
    HTglobal(2, 2) = cj * ci;

    HTglobal(3, 0) = HTglobal(3, 1) = HTglobal(3, 2) = 0;
    HTglobal(0, 3) = HTglobal(1, 3) = HTglobal(2, 3) = 0;
    HTglobal(3, 3) = 1;

    HomogeneousTransform HT = HTglobal * HTdelta;

    return HT;
  }

  // HTs -> JUMP dofs
  //
  // Given the matrix definition in JumpTransforms, we calculate the dofs
  //   that give rise to this HT.
  // A special case handles the problematic region where cos(beta)=0.
  // In this case, the alpha and gamma rotation are coincident so
  //    we assign all rotation to alpha.
  //
  // Since RB and RBdel are redundant, this function always returns its
  //    non-zero components into RB, and RBdel is always 0
  static auto EIGEN_DEVICE_FUNC invJumpTransform(HomogeneousTransform M) {
    Dofs dof;

    dof[DOF_RBx] = M(3, 0);
    dof[DOF_RBy] = M(3, 1);
    dof[DOF_RBz] = M(3, 2);
    dof[DOF_rotdelalpha] = dof[DOF_rotdelbeta] = dof[DOF_rotdelgamma] = 0.0;

    Real cy = std::sqrt(M(0, 0) * M(0, 0) + M(0, 1) * M(0, 1));
    if (cy < 1e-6) {
      dof[DOF_rotalpha] = std::atan2(-M(2, 1), M(1, 1));
      dof[DOF_rotbeta] = std::atan2(-M(0, 2), cy);
      dof[DOF_rotgamma] = 0.0;
    } else {
      dof[DOF_rotalpha] = std::atan2(M(1, 2), M(2, 2));
      dof[DOF_rotbeta] = std::atan2(-M(0, 2), cy);
      dof[DOF_rotgamma] = std::atan2(M(0, 1), M(0, 0));
    }

    return dof;
  }

  static auto EIGEN_DEVICE_FUNC
  hts_from_frames(Coord ori, Coord a, Coord b, Coord c) {
    HomogeneousTransform ht;

    Coord xaxis = (a - b).normalized();
    Coord zaxis = xaxis.cross(c - a).normalized();
    Coord yaxis = zaxis.cross(xaxis).normalized();

    ht.block(0, 0, 1, 3) = xaxis.transpose();
    ht.block(1, 0, 1, 3) = yaxis.transpose();
    ht.block(2, 0, 1, 3) = zaxis.transpose();
    ht.block(3, 0, 1, 3) = ori.transpose();
    ht.col(3) = Eigen::Matrix<Real, 1, 4>(0, 0, 0, 1);

    return ht;
  }

  // note: not a proper inversion; assumes the R component is a
  //   proper rotation
  static auto EIGEN_DEVICE_FUNC ht_inv(HomogeneousTransform ht) {
    HomogeneousTransform htinv;
    htinv.topLeftCorner(3, 3) = ht.topLeftCorner(3, 3).transpose();
    htinv.bottomLeftCorner(1, 3) =
        -ht.bottomLeftCorner(1, 3) * htinv.topLeftCorner(3, 3);
    htinv.col(3) = Eigen::Matrix<Real, 1, 4>(0, 0, 0, 1);
    return htinv;
  }
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
struct KinForestFromStencil {
  static auto get_kfo_indices_for_atoms(
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,
      TView<Int, 1, D> block_type_n_atoms,
      TView<bool, 2, D> block_type_atom_is_real)
      -> std::tuple<TPack<Int, 2, D>, TPack<Int, 2, D>, TPack<Int, 3, D>>;

  static auto get_kfo_atom_parents(
      TView<Int, 2, D> pose_stack_block_type,                 // P x L
      TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
      TView<Int, 2, D> pose_stack_ff_parent,                  // P x L
      TView<Int, 2, D> pose_stack_ff_conn_to_parent,          // P x L
      TView<Int, 3, D> pose_stack_block_in_and_first_out,     // P x L x 2
      TView<Int, 3, D> block_type_parents,                    // T x O x A
      TView<Int, 2, D> kfo_2_orig_mapping,                    // K x 3
      TView<Int, 3, D> atom_kfo_index,                        // P x L x A
      TView<Int, 1, D> block_type_jump_atom,                  // T
      TView<Int, 1, D> block_type_n_conn,                     // T
      TView<Int, 2, D> block_type_conn_atom) -> TPack<Int, 1, D>;

  //   static auto get_parent_atoms(
  //     TView<Int, 2, D> ff_block_parent, // Which block is the parent? -1 for
  //     root TView<Int, 2, D> ff_conn_to_parent, // What kind of connection:
  //     1=lower connect, 2=upper connect, 3=jump TView<Int, 3, D>
  //     block_in_and_first_out, // Which connection is the input connection,
  //     which the output connection? TView<Int, 2, D>
  //     pose_stack_block_coord_offset, TView<Int, 2, D> pose_stack_block_type,

  //     TView<Int, 2, D> kfo_block_offset,
  //     TView<Int, 2, D> real_bt_ind_for_bt,

  //     // For determining which atoms to retrieve from neighboring
  //     // residues we have to know how the blocks in the Pose
  //     // are connected
  //     TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,

  //     //////////////////////
  //     // Chemical properties
  //     // how many atoms for a given block
  //     // Dimsize n_block_types
  //     TView<Int, 1, D> block_type_n_atoms,
  //     // TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

  //     // n-bt x max-n-ats x 3 x 3
  //     // TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,

  //     // n-bt x max-n-ats x 3 [phi, theta, D]
  //     // TView<Real, 3, Dev> block_type_atom_icoors,

  //     // TEMP! Handle the case when an atom's coordinate depends on
  //     // an un-resolvable atom, e.g., "down" for an N-terminal atom
  //     // n-bt x max-n-ats x 3 x 3
  //     // TView<UnresolvedAtomID<Int>, 3, Dev>
  //     block_type_atom_ancestors_backup,
  //     // n-bt x max-n-ats x 3 [phi, theta, D]
  //     // TView<Real, 3, Dev> block_type_atom_icoors_backup

  //     // the maximum number of atoms in a Pose
  //     int const max_n_atoms
  //   ) -> TPack<Vec<Real, 3>, 2, Dev>
  //   {
  //     int const n_poses = ff_block_parent.size(0);
  //     TPack<Int, 2, D> parent_atoms = TPack<Int, 2, Dev>::zeros({n_poses,
  //     max_n_atoms});

  //     auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int ind) {

  //         return lj_atom_energy(
  //             atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
  //     });
  //   }

  // static auto EIGEN_DEVICE_FUNC get_parent(
  // ) -> Int {
  //   return 0;
  // }

  // static auto EIGEN_DEVICE_FUNC get_c1_and_c2_atoms(
  //     int jump_atom,
  //     TView<Int, 1, D> atom_is_jump,
  //     TView<Int, 2, D> child_list_span,
  //     TView<Int, 1, D> child_list,
  //     TView<Int, 1, D> parents) -> tuple {
  //   int first_nonjump_child = -1;
  //   int second_nonjump_child = -1;
  //   for (int child_ind = child_list_span[jump_atom][0];
  //        child_ind < child_list_span[jump_atom][1]; ++child_ind) {
  //     int child_atom = child_list[child_ind];
  //     if (atom_is_jump[child_atom]) {
  //       continue;
  //     }
  //     if (first_nonjump_child == -1) {
  //       first_nonjump_child = child_atom;
  //     } else {
  //       second_nonjump_child = child_atom;
  //       break;
  //     }
  //   }
  //   if (first_nonjump_child == -1) {
  //     int jump_parent = parents[jump_atom];
  //     assert(jump_parent != jump_atom);
  //     return get_c1_and_c2_atoms(jump_parent, atom_is_jump, child_list_span,
  //                                child_list, parents);
  //   }
  //   for (int grandchild_ind = child_list_span[first_nonjump_child][0];
  //        grandchild_ind < child_list_span[first_nonjump_child][1];
  //        ++grandchild_ind) {
  //     int grandchild_atom = child_list[grandchild_ind];
  //     if (!atom_is_jump[grandchild_atom]) {
  //       return std::make_tuple(first_nonjump_child, grandchild_atom);
  //     }
  //   }
  //   if (second_nonjump_child == -1) {
  //     int jump_parent = parents[jump_atom];
  //     assert(jump_parent != jump_atom);
  //     return get_c1_and_c2_atoms(jump_parent, atom_is_jump, child_list_span,
  //                                child_list, parents);
  //   }
  //   return std::make_tuple(first_nonjump_child, second_nonjump_child);
  // }
};

// @numba.jit(nopython=True)
// def get_c1_and_c2_atoms(
//     jump_atom: int,
//     atom_is_jump: NDArray[int][:],
//     child_list_span: NDArray[int][:],
//     child_list: NDArray[int][:],
//     parents: NDArray[int][:],
// ) -> tuple:
//     """Preferably a jump should steal DOFs from its first (nonjump) child
//     and its first (nonjump) grandchild, but if the first child does not
//     have any children, then it can steal a DOF from its second (nonjump)
//     child. If a jump does not have a sufficient number of descendants, then
//     we must recurse to its parent.
//     """

//     first_nonjump_child = -1
//     second_nonjump_child = -1
//     for child_ind in range(
//         child_list_span[jump_atom, 0], child_list_span[jump_atom, 1]
//     ):
//         child_atom = child_list[child_ind]
//         if atom_is_jump[child_atom]:
//             continue
//         if first_nonjump_child == -1:
//             first_nonjump_child = child_atom
//         else:
//             second_nonjump_child = child_atom
//             break

//     if first_nonjump_child == -1:
//         jump_parent = parents[jump_atom]
//         assert jump_parent != jump_atom
//         return get_c1_and_c2_atoms(
//             jump_parent, atom_is_jump, child_list_span, child_list, parents
//         )

//     for grandchild_ind in range(
//         child_list_span[first_nonjump_child, 0],
//         child_list_span[first_nonjump_child, 1]
//     ):
//         grandchild_atom = child_list[grandchild_ind]
//         if not atom_is_jump[grandchild_atom]:
//             return first_nonjump_child, grandchild_atom

//     if second_nonjump_child == -1:
//         jump_parent = parents[jump_atom]
//         assert jump_parent != jump_atom
//         return get_c1_and_c2_atoms(
//             jump_parent, atom_is_jump, child_list_span, child_list, parents
//         )

//     return first_nonjump_child, second_nonjump_child

// @numba.jit(nopython=True)
// def fix_jump_nodes(
//     parents: NDArray[int][:],
//     frame_x: NDArray[int][:],
//     frame_y: NDArray[int][:],
//     frame_z: NDArray[int][:],
//     roots: NDArray[int][:],
//     jumps: NDArray[int][:],
// ):
//     # nelts = parents.shape[0]
//     n_children, child_list_span, child_list = get_children(parents)

//     atom_is_jump = numpy.full(parents.shape, 0, dtype=numpy.int32)
//     atom_is_jump[roots] = 1
//     atom_is_jump[jumps] = 1

//     for root in roots:
//         assert stub_defined_for_jump_atom(
//             root, atom_is_jump, child_list_span, child_list
//         )

//         root_c1, second_descendent = get_c1_and_c2_atoms(
//             root, atom_is_jump, child_list_span, child_list, parents
//         )

//         # set the frame_x, _y, and _z to the same values for both the root
//         # and the root's first child

//         frame_x[root] = root_c1
//         frame_y[root] = root
//         frame_z[root] = second_descendent

//         frame_x[root_c1] = root_c1
//         frame_y[root_c1] = root
//         frame_z[root_c1] = second_descendent

//         # all the other children of the root need an updated kinematic
//         description for child_ind in range(child_list_span[root, 0] + 1,
//         child_list_span[root, 1]):
//             child = child_list[child_ind]
//             if atom_is_jump[child]:
//                 continue
//             if child == root_c1:
//                 continue
//             frame_x[child] = child
//             frame_y[child] = root
//             frame_z[child] = root_c1

//     for jump in jumps:
//         if stub_defined_for_jump_atom(jump, atom_is_jump, child_list_span,
//         child_list):
//             jump_c1, jump_c2 = get_c1_and_c2_atoms(
//                 jump, atom_is_jump, child_list_span, child_list, parents
//             )

//             # set the frame_x, _y, and _z to the same values for both the
//             jump # and the jump's first child

//             frame_x[jump] = jump_c1
//             frame_y[jump] = jump
//             frame_z[jump] = jump_c2

//             frame_x[jump_c1] = jump_c1
//             frame_y[jump_c1] = jump
//             frame_z[jump_c1] = jump_c2

//             # all the other children of the jump need an updated kinematic
//             description for child_ind in range(
//                 child_list_span[jump, 0] + 1, child_list_span[jump, 1]
//             ):
//                 child = child_list[child_ind]
//                 if atom_is_jump[child]:
//                     continue
//                 if child == jump_c1:
//                     continue
//                 frame_x[child] = child
//                 frame_y[child] = jump
//                 frame_z[child] = jump_c1
//         else:
//             # ok, so... I don't understand the atom tree well enough to
//             understand this # situation. If the jump has no non-jump
//             children, then certainly none # of them need their frame
//             definitions updated c1, c2 = get_c1_and_c2_atoms(
//                 parents[jump], atom_is_jump, child_list_span, child_list,
//                 parents
//             )

//             frame_x[jump] = c1
//             frame_y[jump] = jump
//             frame_z[jump] = c2

//             # the jump may have one child; it's not entirely clear to me
//             # what frame the child should have!
//             # TO DO: figure this out
//             for child_ind in range(
//                 child_list_span[jump, 0] + 1, child_list_span[jump, 1]
//             ):
//                 child = child_list[child_ind]
//                 if atom_is_jump[child]:
//                     continue
//                 frame_x[child] = c1
//                 frame_y[child] = jump
//                 frame_z[child] = c2

template <tmol::Device D, typename Int>
void get_c1_and_c2_atoms(
    int jump_atom,
    TView<Int, 1, D> atom_is_jump,
    TView<Int, 1, D> child_list_span,
    TView<Int, 1, D> child_list,
    TView<Int, 1, D> parents) {
  // Preferably a jump should steal DOFs from its first (nonjump) child
  // and its first (nonjump) grandchild, but if the first child does not
  // have any children, then it can steal a DOF from its second (nonjump)
  // child. If a jump does not have a sufficient number of descendants, then
  // we must recurse to its parent.

  // TO DO!
}

#undef Dofs
#undef HomogeneousTransform
#undef QuatTranslation
#undef Coord

}  // namespace kinematics
}  // namespace tmol
