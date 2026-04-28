import math
import numpy
import torch
import attr

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from tmol.types.functional import validate_args

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import KinForest
from tmol.pack.rotamer.conformer_sampler import ConformerSampler
from tmol.numeric.dihedrals import coord_dihedrals
from tmol.utility.tensor.common_operations import exclusive_cumsum1d

# Residue categories that get NHQ flip treatment (requires fix_NHQ=True)
_NQ_FLIP_BASES = frozenset(("ASN", "GLN"))
_HIS_FLIP_BASES = frozenset(("HIS", "HIS_D"))


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OptHSamplerRTCache:
    """Per-residue-type annotation for OptHSampler.

    Covers two orthogonal features:
    1. Proton chi sampling (SER/THR/TYR/CYS): samples the terminal (proton)
       chi angle using values from restype definition.
    2. NHQ flip (ASN/GLN/HIS/HIS_D): generates the input conformation plus a
       180-degree rotation about the last chi angle.
       HIS additionally generates both protonation states.
    """

    # proton chis (S/T/Y/C)
    has_proton_chi: bool
    n_chi_total: int
    chi_defining_atom: NDArray[numpy.int32][:]
    n_proton_samples: int
    expanded_samples: NDArray[numpy.float32][:, :]
    n_samples_per_chi: NDArray[numpy.int32][:]

    # N/H/Q flips
    nhq_chi_col: int  # chi index or -1
    nhq_chi_atom: int
    nhq_chi_4atoms: NDArray[numpy.int32][:]
    is_his: bool


def _build_empty_proton_cache(nhq_chi_col, nhq_chi_atom, nhq_chi_4atoms, is_his):
    return OptHSamplerRTCache(
        has_proton_chi=False,
        n_chi_total=0,
        chi_defining_atom=numpy.zeros(0, dtype=numpy.int32),
        n_proton_samples=0,
        expanded_samples=numpy.zeros((0, 0), dtype=numpy.float32),
        n_samples_per_chi=numpy.zeros(0, dtype=numpy.int32),
        nhq_chi_col=nhq_chi_col,
        nhq_chi_atom=nhq_chi_atom,
        nhq_chi_4atoms=nhq_chi_4atoms,
        is_his=is_his,
    )


def _downstream_kfo_atoms(rt) -> list:
    """Return KFO indices of all atoms downstream of rt's NHQ chi-defining atom.

    Traverses the per-residue-type kinforest parent array starting from the
    children of nhq_chi_atom.  Returns an empty list for all non-NHQ residues.
    """
    nhq_rto = int(rt.opth_sampler_cache.nhq_chi_atom)
    if nhq_rto < 0:
        return []
    kfidx = rt.rotamer_kinforest.kinforest_idx  # (n_atoms,) numpy, TO -> KFO
    parents = rt.rotamer_kinforest.parent  # (n_atoms,) numpy, KFO -> parent KFO
    kfo_nhq = int(kfidx[nhq_rto])
    n_at = len(parents)
    downstream = []
    queue = [k for k in range(n_at) if int(parents[k]) == kfo_nhq]
    while queue:
        k = queue.pop()
        downstream.append(k)
        queue.extend(ch for ch in range(n_at) if int(parents[ch]) == k)
    return downstream


def _opth_fill_dofs(
    pose_stack,
    task,
    gbt_for_conformer,
    block_type_ind_for_conformer,
    n_dof_atoms_offset_for_conformer,
    conf_inds_for_sampler,
    orig_dofs_kto,
    chi_atoms,
    chi_vals,
    conf_dofs_kto,
    fix_NHQ,
):
    """Fill conf_dofs_kto for all OptHSampler rotamers

    1. Copy DOFs from pose into conf_dofs_kto.
    2. For HNQ only: atoms that are kinematic children of the chi-defining atom
       of the flip are reset to their ideal DOF values
    3. Write the corrected chi torsion into DOF column 3 for
       the chi-defining atom of each flip / proton-chi rotamer.
    """
    from tmol.pack.rotamer.build_rotamers import _build_chi_phi_c_corrections

    pbt = pose_stack.packed_block_types
    dev = conf_dofs_kto.device

    # Per-sampler-rotamer lookup vectors (torch, length n_rots)
    bt_inds = block_type_ind_for_conformer[conf_inds_for_sampler]
    at_offs = n_dof_atoms_offset_for_conformer[conf_inds_for_sampler]

    # Source offsets in orig_dofs_kto: mirrors create_full_dof_inds_to_copy_...
    orig_bt_ind = (
        pose_stack.block_type_ind[pose_stack.block_type_ind != -1]
        .view(-1)
        .to(torch.int64)
    )
    orig_at_off_base = exclusive_cumsum1d(pbt.n_atoms[orig_bt_ind]).to(torch.int64)
    max_n_blocks = pose_stack.block_coord_offset.shape[1]
    flat_block_for_gbt = torch.tensor(
        [
            i * max_n_blocks + j
            for i, one_pose_blts in enumerate(task.blts)
            for j, blt in enumerate(one_pose_blts)
            for _ in blt.considered_block_types
        ],
        dtype=torch.int64,
        device=dev,
    )
    real_idx_for_flat = torch.full(
        (pose_stack.block_type_ind.shape[0] * pose_stack.block_type_ind.shape[1],),
        -1,
        dtype=torch.int64,
        device=dev,
    )
    real_idx_for_flat[pose_stack.block_type_ind.view(-1) != -1] = torch.arange(
        orig_bt_ind.shape[0], dtype=torch.int64, device=dev
    )
    orig_at_offs = orig_at_off_base[
        real_idx_for_flat[flat_block_for_gbt[gbt_for_conformer[conf_inds_for_sampler]]]
    ]

    # Step 1: vectorised all-atom copy
    n_rots = conf_inds_for_sampler.shape[0]
    n_atoms_per_rot = pbt.n_atoms[bt_inds]
    dummy = (
        torch.arange(pbt.max_n_atoms, dtype=torch.int64, device=dev)
        .view(1, -1)
        .expand(n_rots, -1)
    )
    real_mask = dummy < n_atoms_per_rot.unsqueeze(1)
    dst = (at_offs.unsqueeze(1).expand(-1, pbt.max_n_atoms) + dummy)[real_mask]
    src = (orig_at_offs.unsqueeze(1).expand(-1, pbt.max_n_atoms) + dummy)[real_mask]
    conf_dofs_kto[dst + 1, :] = orig_dofs_kto[src + 1, :]

    # Steps 2 & 3: only for rotamers that have a chi override
    if chi_atoms.shape[0] == 0 or not (chi_atoms >= 0).any():
        return

    kfidx = pbt.rotamer_kinforest.kinforest_idx  # (n_types, max_n_atoms) numpy
    dofs_ideal_t = torch.as_tensor(
        pbt.rotamer_kinforest.dofs_ideal, dtype=torch.float32, device=dev
    )
    corrections = _build_chi_phi_c_corrections(pbt)  # (n_types, max_n_chi) numpy

    downstream_cache = {}
    reset_kto, reset_bt, reset_k = [], [], []
    chi_kto, chi_val_list = [], []

    flip_sis = (chi_atoms >= 0).any(dim=1).nonzero(as_tuple=True)[0].tolist()
    for si in flip_sis:
        bt_idx = bt_inds[si].item()
        at_off = at_offs[si].item()

        # Step 2: reset downstream atoms to ideal for NHQ flip rotamers
        if fix_NHQ:
            if bt_idx not in downstream_cache:
                downstream_cache[bt_idx] = _downstream_kfo_atoms(
                    pbt.active_block_types[bt_idx]
                )
            for k in downstream_cache[bt_idx]:
                reset_kto.append(k + at_off + 1)
                reset_bt.append(bt_idx)
                reset_k.append(k)

        # Step 3: override chi torsion (DOF column 3) with corrected value
        for chi_col in range(chi_atoms.shape[1]):
            chi_rto = chi_atoms[si, chi_col].item()
            if chi_rto < 0:
                continue
            kfo = int(kfidx[bt_idx, chi_rto])
            corr = float(corrections[bt_idx, chi_col])
            chi_kto.append(kfo + at_off + 1)
            chi_val_list.append(chi_vals[si, chi_col].item() - corr)

    if reset_kto:
        conf_dofs_kto[torch.tensor(reset_kto, dtype=torch.int64, device=dev)] = (
            dofs_ideal_t[
                torch.tensor(reset_bt, dtype=torch.int64, device=dev),
                torch.tensor(reset_k, dtype=torch.int64, device=dev),
            ]
        )

    if chi_kto:
        conf_dofs_kto[torch.tensor(chi_kto, dtype=torch.int64, device=dev), 3] = (
            torch.tensor(chi_val_list, dtype=torch.float32, device=dev)
        )


def _n_rots_for_gbt(sampler, blt, orig, orig_cache, bt, bt_cache):
    """Return the number of rotamers OptHSampler generates for one GBT entry."""
    # Proton chi: only for the original block type
    if bt_cache.has_proton_chi and bt is blt.original_block_type:
        return bt_cache.n_proton_samples

    # NHQ flip
    if sampler.fix_NHQ and orig_cache.nhq_chi_col >= 0:
        if orig_cache.is_his:
            # HIS/HIS_D: 2 rotamers for EVERY HIS/HIS_D considered block type
            if bt_cache.is_his:
                return 2
        else:
            # ASN/GLN: 2 rotamers only for the original block type
            if bt is blt.original_block_type:
                return 2

    return 0


def _chi_cols_needed(bt_cache, orig_cache, fix_NHQ):
    """Return the minimum chi tensor width needed for a GBT with non-zero rots."""
    if bt_cache.has_proton_chi:
        return bt_cache.n_chi_total
    if fix_NHQ and orig_cache.nhq_chi_col >= 0:
        return orig_cache.nhq_chi_col + 1
    return 1


@attr.s(auto_attribs=True, frozen=True)
class OptHSampler(ConformerSampler):
    """Build rotamers by sampling proton chi angles only, keeping all heavy
    atoms at their input-conformation positions.

    When fix_NHQ is True (default), also builds flip rotamers for:
    - ASN/GLN: current conformation + 180-degree rotation of the last chi.
    - HIS/HIS_D: {HIS, HIS_D} x {current chi2, chi2+180} = 4 rotamers.
      All atoms through CG are taken from the input; ring atoms are rebuilt
      from ideal geometry for three non-input variants.

    NOTE: DunbrackChiSampler and OptHSampler must not be assigned to the
    same block (Dunbrack already samples proton chis, so both on one block
    oversamples). Assigning them to different blocks in the same task is fine.
    """

    fix_NHQ: bool = True

    @classmethod
    def sampler_name(cls):
        return "OptHSampler"

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        if hasattr(rt, "opth_sampler_cache"):
            return

        base = rt.base_name

        # NHQ flip annotation
        nhq_chi_col = -1
        nhq_chi_atom = -1
        nhq_chi_4atoms = numpy.zeros(4, dtype=numpy.int32)
        is_his = base in _HIS_FLIP_BASES

        if base in _NQ_FLIP_BASES or is_his:
            chi_names = sorted(k for k in rt.torsion_to_uaids if k.startswith("chi"))
            last_chi = chi_names[-1]
            nhq_chi_col = len(chi_names) - 1  # 0-based index of the last chi
            uaids = rt.torsion_to_uaids[last_chi]
            nhq_chi_atom = int(uaids[2][0])  # 3rd atom = defining atom
            nhq_chi_4atoms = numpy.array(
                [int(uaids[k][0]) for k in range(4)], dtype=numpy.int32
            )

        # proton chi annotation
        if not rt.chi_samples:
            setattr(
                rt,
                "opth_sampler_cache",
                _build_empty_proton_cache(
                    nhq_chi_col, nhq_chi_atom, nhq_chi_4atoms, is_his
                ),
            )
            return

        deg_to_rad = math.pi / 180

        chi_inds = [int(cs.chi_dihedral[3:]) - 1 for cs in rt.chi_samples]
        n_chi_total = max(chi_inds) + 1

        chi_defining_atom = numpy.full(n_chi_total, -1, dtype=numpy.int32)
        n_samples_per_chi = numpy.zeros(n_chi_total, dtype=numpy.int32)

        max_n_expanded = max(
            len(cs.samples) * (1 + 2 * len(cs.expansions)) for cs in rt.chi_samples
        )
        expanded_samples = numpy.zeros(
            (n_chi_total, max_n_expanded), dtype=numpy.float32
        )

        for cs in rt.chi_samples:
            ci = int(cs.chi_dihedral[3:]) - 1
            chi_defining_atom[ci] = rt.torsion_to_uaids[cs.chi_dihedral][2][0]

            n_samp = len(cs.samples)
            n_exp_per_samp = 1 + 2 * len(cs.expansions)
            n_samples_per_chi[ci] = n_samp * n_exp_per_samp

            for i in range(n_samp):
                for j in range(n_exp_per_samp):
                    if j == 0:
                        expanded_samples[ci, n_exp_per_samp * i] = (
                            deg_to_rad * cs.samples[i]
                        )
                    else:
                        exp_idx = (j - 1) // 2
                        factor = -1 if (j - 1) % 2 == 0 else 1
                        expanded_samples[ci, n_exp_per_samp * i + j] = deg_to_rad * (
                            cs.samples[i] + factor * cs.expansions[exp_idx]
                        )

        n_proton_samples_total = 1
        for n in n_samples_per_chi:
            if n > 0:
                n_proton_samples_total *= int(n)

        setattr(
            rt,
            "opth_sampler_cache",
            OptHSamplerRTCache(
                has_proton_chi=True,
                n_chi_total=n_chi_total,
                chi_defining_atom=chi_defining_atom,
                n_proton_samples=n_proton_samples_total,
                expanded_samples=expanded_samples,
                n_samples_per_chi=n_samples_per_chi,
                nhq_chi_col=nhq_chi_col,
                nhq_chi_atom=nhq_chi_atom,
                nhq_chi_4atoms=nhq_chi_4atoms,
                is_his=is_his,
            ),
        )

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        # as this depends on input chi1/2 nothing to do here
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        if rt.chi_samples:  # has a proton chi
            return True
        if self.fix_NHQ:  # is NHQ if flipNHQ is enabled
            return rt.base_name in _NQ_FLIP_BASES or rt.base_name in _HIS_FLIP_BASES
        return False

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return ("CB",)

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: F821
    ) -> Tuple[Tensor[torch.int32][:], Tensor[torch.int32][:], dict,]:
        # ensure dunbrack and optH sampler are not _both_ specified for the same block
        for one_pose_blts in task.blts:
            for blt in one_pose_blts:
                sampler_names = {s.sampler_name() for s in blt.conformer_samplers}
                if (
                    "DunbrackChiSampler" in sampler_names
                    and "OptHSampler" in sampler_names
                ):
                    raise RuntimeError(
                        "OptHSampler and DunbrackChiSampler cannot both be assigned "
                        "to the same block. DunbrackChiSampler already samples proton "
                        "chi angles as part of each library rotamer."
                    )

        # 1) compute:
        #      n_rots per GBT
        #      max chi tensor width
        #      current last-chi angle
        # for each NHQ position in the input
        pi = math.pi
        coords = pose_stack.coords.double()  # coord_dihedrals needs float64

        # pos_flip_chi[(pose_i, block_j)] = current last-chi in radians
        pos_flip_chi = {}

        n_rots_for_gbt_list = []
        max_n_chi_cols = 1

        for pose_i, one_pose_blts in enumerate(task.blts):
            for block_j, blt in enumerate(one_pose_blts):
                orig = blt.original_block_type
                orig_cache = orig.opth_sampler_cache
                opth_assigned = self in blt.conformer_samplers

                # Measure chi-to-flip for NHQ
                if opth_assigned and self.fix_NHQ and orig_cache.nhq_chi_col >= 0:
                    off = int(pose_stack.block_coord_offset[pose_i, block_j].item())
                    a4 = orig_cache.nhq_chi_4atoms
                    c = coords[pose_i][
                        torch.tensor(
                            [off + int(a4[k]) for k in range(4)],
                            dtype=torch.int64,
                            device=pose_stack.device,
                        )
                    ]  # (4, 3)
                    pos_flip_chi[(pose_i, block_j)] = float(
                        coord_dihedrals(c[0:1], c[1:2], c[2:3], c[3:4])[0].item()
                    )

                for bt in blt.considered_block_types:
                    bt_cache = bt.opth_sampler_cache
                    if not opth_assigned:
                        n_rots_for_gbt_list.append(0)
                        continue
                    n_rots = _n_rots_for_gbt(self, blt, orig, orig_cache, bt, bt_cache)
                    n_rots_for_gbt_list.append(n_rots)
                    if n_rots > 0:
                        max_n_chi_cols = max(
                            max_n_chi_cols,
                            _chi_cols_needed(bt_cache, orig_cache, self.fix_NHQ),
                        )

        # Allocate output tensors
        n_rots_for_gbt = torch.tensor(
            n_rots_for_gbt_list, dtype=torch.int32, device=pose_stack.device
        )
        n_rots_total = int(n_rots_for_gbt.sum().item())

        empty_chi = torch.zeros(
            (0, max_n_chi_cols), dtype=torch.int32, device=pose_stack.device
        )
        if n_rots_total == 0:
            return (
                n_rots_for_gbt,
                torch.zeros(0, dtype=torch.int32, device=pose_stack.device),
                dict(
                    chi_defining_atom_for_rotamer=empty_chi,
                    chi_for_rotamers=empty_chi.float(),
                ),
            )
        gbt_for_rotamer = torch.repeat_interleave(
            torch.arange(
                len(n_rots_for_gbt_list), dtype=torch.int32, device=pose_stack.device
            ),
            n_rots_for_gbt.to(torch.int64),
        )
        chi_defining_atom_for_rotamer = torch.full(
            (n_rots_total, max_n_chi_cols),
            -1,
            dtype=torch.int32,
            device=pose_stack.device,
        )
        chi_for_rotamers = torch.zeros(
            (n_rots_total, max_n_chi_cols),
            dtype=torch.float32,
            device=pose_stack.device,
        )

        # 2) fill chi tensors
        rot_offset = 0
        gbt_idx = 0
        for pose_i, one_pose_blts in enumerate(task.blts):
            for block_j, blt in enumerate(one_pose_blts):
                orig = blt.original_block_type
                orig_cache = orig.opth_sampler_cache
                flip_chi = pos_flip_chi.get((pose_i, block_j), None)
                opth_assigned = self in blt.conformer_samplers

                for bt in blt.considered_block_types:
                    bt_cache = bt.opth_sampler_cache
                    if not opth_assigned:
                        gbt_idx += 1
                        continue
                    n_rots = n_rots_for_gbt_list[gbt_idx]
                    gbt_idx += 1

                    if n_rots == 0:
                        continue

                    # Proton chi (SER/THR/TYR/CYS) ... just flip chi
                    if bt_cache.has_proton_chi and bt is blt.original_block_type:
                        ci = next(
                            i
                            for i in range(bt_cache.n_chi_total)
                            if bt_cache.chi_defining_atom[i] >= 0
                        )
                        n_samp = int(bt_cache.n_samples_per_chi[ci])
                        chi_defining_atom_for_rotamer[
                            rot_offset : rot_offset + n_samp, ci
                        ] = int(bt_cache.chi_defining_atom[ci])
                        chi_for_rotamers[rot_offset : rot_offset + n_samp, ci] = (
                            torch.tensor(
                                bt_cache.expanded_samples[ci, :n_samp],
                                dtype=torch.float32,
                                device=pose_stack.device,
                            )
                        )

                    # NHQ: flip and (for flips/tautamer change) idealize downstream
                    elif self.fix_NHQ and orig_cache.nhq_chi_col >= 0:
                        # rotamer offset 0 = current chi
                        # rotamer offset 1 = current chi + pi
                        chi_col = orig_cache.nhq_chi_col

                        # Only rebuild rotamer offset 0 for HIS flips
                        if orig_cache.is_his and bt is not blt.original_block_type:
                            chi_defining_atom_for_rotamer[rot_offset, chi_col] = (
                                bt_cache.nhq_chi_atom
                            )
                            chi_for_rotamers[rot_offset, chi_col] = float(flip_chi)
                        # always rebuild rotamer offset 1
                        chi_defining_atom_for_rotamer[rot_offset + 1, chi_col] = (
                            bt_cache.nhq_chi_atom
                        )
                        chi_for_rotamers[rot_offset + 1, chi_col] = float(flip_chi) + pi

                    rot_offset += n_rots

        return (
            n_rots_for_gbt,
            gbt_for_rotamer,
            dict(
                chi_defining_atom_for_rotamer=chi_defining_atom_for_rotamer,
                chi_for_rotamers=chi_for_rotamers,
            ),
        )

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: F821
        orig_kinforest: KinForest,
        orig_dofs_kto: Tensor[torch.float32][:, 9],
        gbt_for_conformer: Tensor[torch.int64][:],
        block_type_ind_for_conformer: Tensor[torch.int64][:],
        n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
        conformer_built_by_sampler: Tensor[torch.bool][:],
        conf_inds_for_sampler: Tensor[torch.int64][:],
        sampler_n_rots_for_gbt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict,
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ):
        if sampler_gbt_for_rotamer.shape[0] == 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        _opth_fill_dofs(
            pose_stack,
            task,
            gbt_for_conformer,
            block_type_ind_for_conformer,
            n_dof_atoms_offset_for_conformer,
            conf_inds_for_sampler,
            orig_dofs_kto,
            sample_dict["chi_defining_atom_for_rotamer"],
            sample_dict["chi_for_rotamers"],
            conf_dofs_kto,
            self.fix_NHQ,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
