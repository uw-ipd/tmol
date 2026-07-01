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
from tmol.pack.rotamer.single_residue_kinforest import (
    construct_single_residue_kinforest,
)
from tmol.numeric.dihedrals import coord_dihedrals
from tmol.utility.tensor.common_operations import exclusive_cumsum1d

# Residue categories that get NHQ flip treatment (requires flip_NHQ=True)
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
    nhq_downstream_kfo: NDArray[numpy.int32][:]
    is_his: bool


@attr.s(auto_attribs=True, slots=True, frozen=True)
class OptHSamplerPackedBlockTypeCache:
    opth_sample_for_bt: Tensor[torch.bool][:]
    has_proton_chi: Tensor[torch.bool][:]
    n_chi_total: Tensor[torch.int32][:]
    chi_defining_atom: Tensor[torch.int32][:, :]
    n_proton_samples: Tensor[torch.int32][:]
    expanded_samples: Tensor[torch.float32][:, :, :]
    n_samples_per_chi: Tensor[torch.int32][:, :]

    nhq_chi_col: Tensor[torch.int32][:]
    nhq_chi_atom: Tensor[torch.int32][:]
    nhq_chi_4atoms: Tensor[torch.int32][:, 4]
    nhq_downstream_kfo: Tensor[torch.int32][:, :]
    is_his: Tensor[torch.bool][:]

    # These two tensors use 0 or 1 as an index to dim=0 to represent
    # either flip_NHQ=False or flip_NHQ=True
    n_samples_for_bt_by_orig_bt: Tensor[torch.int32][2, :, :]
    n_chi_needed_for_bt: Tensor[torch.int32][2, :]


def _build_empty_proton_cache(
    nhq_chi_col, nhq_chi_atom, nhq_chi_4atoms, nhq_downstream_kfo, is_his
):
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
        nhq_downstream_kfo=nhq_downstream_kfo,
        is_his=is_his,
    )


def _compute_nhq_downstream_kfo(rt, nhq_rto: int) -> NDArray[numpy.int32]:
    """Return KFO indices of all atoms downstream of rt's NHQ chi-defining atom.

    Traverses the per-residue-type kinforest parent array starting from the
    children of nhq_chi_atom.  Computed once during _annotate_residue_type and
    cached on the OptHSamplerRTCache; returns an empty array for non-NHQ rts.
    """
    if nhq_rto < 0:
        return numpy.zeros(0, dtype=numpy.int32)
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
    return numpy.array(downstream, dtype=numpy.int32)


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
    flip_NHQ,
):
    """Fill conf_dofs_kto for all OptHSampler rotamers

    1. Copy DOFs from pose into conf_dofs_kto.
    2. For NHQ only: atoms that are kinematic children of the chi-defining atom
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
    # max_n_blocks = pose_stack.block_coord_offset.shape[1]
    # OLD flat_block_for_gbt = torch.tensor(
    # OLD     [
    # OLD         i * max_n_blocks + j
    # OLD         for i, one_pose_blts in enumerate(task.blts)
    # OLD         for j, blt in enumerate(one_pose_blts)
    # OLD         for _ in blt.considered_block_types
    # OLD     ],
    # OLD     dtype=torch.int64,
    # OLD     device=dev,
    # OLD )
    flat_block_for_gbt = task.global_block_ind_for_considered_block_types
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

    reset_kto, reset_bt, reset_k = [], [], []
    chi_kto, chi_val_list = [], []

    # chi_atoms is set >= 0 only for rotamers needing a chi override:
    # NHQ flips, HIS tautomer swaps, and proton-chi samples.  Step 2 and
    # step 3 below run only over those rotamers.
    flip_sis = (chi_atoms >= 0).any(dim=1).nonzero(as_tuple=True)[0].tolist()
    for si in flip_sis:
        bt_idx = bt_inds[si].item()
        at_off = at_offs[si].item()

        # Step 2: reset downstream atoms to ideal.
        if flip_NHQ:
            downstream = pbt.active_block_types[
                bt_idx
            ].opth_sampler_cache.nhq_downstream_kfo
            for k in downstream:
                k = int(k)
                reset_kto.append(k + at_off + 1)
                reset_bt.append(bt_idx)
                reset_k.append(k)

        # Step 3: write the chi torsion(s) into DOF column 3.
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
    if sampler.flip_NHQ and orig_cache.nhq_chi_col >= 0:
        if orig_cache.is_his:
            # HIS/HIS_D: 2 rotamers for EVERY HIS/HIS_D considered block type
            if bt_cache.is_his:
                return 2
        else:
            # ASN/GLN: 2 rotamers only for the original block type
            if bt is blt.original_block_type:
                return 2

    return 0


def _chi_cols_needed(bt_cache, orig_cache, flip_NHQ):
    """Return the minimum chi tensor width needed for a GBT with non-zero rots."""
    if bt_cache.has_proton_chi:
        return bt_cache.n_chi_total
    if flip_NHQ and orig_cache.nhq_chi_col >= 0:
        return orig_cache.nhq_chi_col + 1
    return 1


@attr.s(auto_attribs=True, frozen=True)
class OptHSampler(ConformerSampler):
    """Build rotamers by sampling proton chi angles only, keeping all heavy
    atoms at their input-conformation positions.

    When flip_NHQ is True (default), also builds flip rotamers for:
    - ASN/GLN: current conformation + 180-degree rotation of the last chi.
    - HIS/HIS_D: {HIS, HIS_D} x {current chi2, chi2+180} = 4 rotamers.
      All atoms through CG are taken from the input; ring atoms are rebuilt
      from ideal geometry for three non-input variants.

    NOTE: DunbrackChiSampler and OptHSampler must not be assigned to the
    same block (Dunbrack already samples proton chis, so both on one block
    oversamples). Assigning them to different blocks in the same task is fine.
    """

    flip_NHQ: bool = True

    @classmethod
    def sampler_name(cls):
        return "OptHSampler"

    @validate_args
    def _annotate_residue_type(self, rt: RefinedResidueType):
        if hasattr(rt, "opth_sampler_cache"):
            return

        base = rt.base_name

        # NHQ flip annotation
        nhq_chi_col = -1
        nhq_chi_atom = -1
        nhq_chi_4atoms = numpy.zeros(4, dtype=numpy.int32)
        nhq_downstream_kfo = numpy.zeros(0, dtype=numpy.int32)
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
            # rotamer_kinforest is required to walk downstream atoms; build it
            # here (idempotent) since the OptHSampler annotation pass runs
            # before build_rotamers.annotate_restype.
            construct_single_residue_kinforest(rt)
            nhq_downstream_kfo = _compute_nhq_downstream_kfo(rt, nhq_chi_atom)

        # proton chi annotation
        if not rt.chi_samples:
            setattr(
                rt,
                "opth_sampler_cache",
                _build_empty_proton_cache(
                    nhq_chi_col,
                    nhq_chi_atom,
                    nhq_chi_4atoms,
                    nhq_downstream_kfo,
                    is_his,
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
                nhq_downstream_kfo=nhq_downstream_kfo,
                is_his=is_his,
            ),
        )

    @validate_args
    def _annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        if hasattr(packed_block_types, "opth_sample_cache"):
            return
        for bt in packed_block_types.active_block_types:
            self._annotate_residue_type(bt)

        opth_sample_for_bt = [
            self.defines_rotamers_for_rt(bt)
            for bt in packed_block_types.active_block_types
        ]
        opth_sample_for_bt = torch.tensor(
            opth_sample_for_bt, dtype=torch.bool, device=packed_block_types.device
        )

        has_proton_chi = torch.zeros(
            (packed_block_types.n_types,),
            dtype=torch.bool,
            device=packed_block_types.device,
        )
        n_chi_total = torch.zeros(
            (packed_block_types.n_types,),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        max_n_chi = 0
        for i, orig_bt in enumerate(packed_block_types.active_block_types):
            max_n_chi = max(max_n_chi, orig_bt.opth_sampler_cache.n_chi_total)
        print("max_n_chi", max_n_chi)

        chi_defining_atom = torch.full(
            (packed_block_types.n_types, max_n_chi),
            fill_value=-1,
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        n_proton_samples = torch.zeros(
            (packed_block_types.n_types,),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        max_n_expanded = max(
            bt.opth_sampler_cache.expanded_samples.shape[1]
            for bt in packed_block_types.active_block_types
        )
        expanded_samples = torch.zeros(
            (packed_block_types.n_types, max_n_chi, max_n_expanded),
            dtype=torch.float32,
            device=packed_block_types.device,
        )
        n_samples_per_chi = torch.zeros(
            (packed_block_types.n_types, max_n_chi),
            dtype=torch.int32,
            device=packed_block_types.device,
        )

        nhq_chi_col = torch.zeros(
            (packed_block_types.n_types,),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        nhq_chi_atom = torch.zeros(
            (packed_block_types.n_types,),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        nhq_chi_4atoms = torch.zeros(
            (packed_block_types.n_types, 4),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        nhq_downstream_kfo = torch.zeros(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        is_his = torch.zeros(
            (packed_block_types.n_types,),
            dtype=torch.bool,
            device=packed_block_types.device,
        )
        n_samples_for_bt_by_orig_bt = torch.zeros(
            (2, packed_block_types.n_types, packed_block_types.n_types),
            dtype=torch.int32,
            device=packed_block_types.device,
        )
        n_chi_needed_for_bt = torch.ones(
            (2, packed_block_types.n_types),
            dtype=torch.int32,
            device=packed_block_types.device,
        )  # the minimum chi tensor width needed for a GBT with non-zero rots

        for i, orig_bt in enumerate(packed_block_types.active_block_types):
            if orig_bt.opth_sampler_cache.has_proton_chi:
                # use n_proton_samples wether or not we're in flip_NHQ mode
                n_samples_for_bt_by_orig_bt[:, i, i] = (
                    orig_bt.opth_sampler_cache.n_proton_samples
                )
                n_chi_needed_for_bt[:, i] = orig_bt.opth_sampler_cache.n_chi_total
            elif orig_bt.opth_sampler_cache.nhq_chi_col >= 0:
                n_chi_needed_for_bt[1, i] = orig_bt.opth_sampler_cache.nhq_chi_col + 1
                if orig_bt.opth_sampler_cache.is_his:
                    n_samples_for_bt_by_orig_bt[1, i, i] = 2
                    for j, alt_bt in enumerate(packed_block_types.active_block_types):
                        if (
                            alt_bt.opth_sampler_cache.nhq_chi_col >= 0
                            and alt_bt.opth_sampler_cache.is_his
                        ):
                            # HIS/HIS_D: 2 rotamers for EVERY HIS/HIS_D considered block type
                            n_samples_for_bt_by_orig_bt[1, i, j] = 2
                else:
                    n_samples_for_bt_by_orig_bt[1, i, i] = 2
                    # ASN/GLN: 2 rotamers only for the original block type; no need to fill w_flipNHQ since it's the same
            has_proton_chi[i] = orig_bt.opth_sampler_cache.has_proton_chi
            n_chi_total[i] = orig_bt.opth_sampler_cache.n_chi_total
            chi_defining_atom[i, : orig_bt.opth_sampler_cache.n_chi_total] = (
                torch.tensor(
                    orig_bt.opth_sampler_cache.chi_defining_atom,
                    dtype=torch.int32,
                    device=packed_block_types.device,
                )
            )
            n_proton_samples[i] = orig_bt.opth_sampler_cache.n_proton_samples
            expanded_samples[
                i,
                : orig_bt.opth_sampler_cache.n_chi_total,
                : orig_bt.opth_sampler_cache.expanded_samples.shape[1],
            ] = torch.tensor(
                orig_bt.opth_sampler_cache.expanded_samples,
                dtype=torch.float32,
                device=packed_block_types.device,
            )
            n_samples_per_chi[i, : orig_bt.opth_sampler_cache.n_chi_total] = (
                torch.tensor(
                    orig_bt.opth_sampler_cache.n_samples_per_chi,
                    dtype=torch.int32,
                    device=packed_block_types.device,
                )
            )
            nhq_chi_col[i] = orig_bt.opth_sampler_cache.nhq_chi_col
            nhq_chi_atom[i] = orig_bt.opth_sampler_cache.nhq_chi_atom
            nhq_chi_4atoms[i, :] = torch.tensor(
                orig_bt.opth_sampler_cache.nhq_chi_4atoms,
                dtype=torch.int32,
                device=packed_block_types.device,
            )
            nhq_downstream_kfo[
                i, : len(orig_bt.opth_sampler_cache.nhq_downstream_kfo)
            ] = torch.tensor(
                orig_bt.opth_sampler_cache.nhq_downstream_kfo,
                dtype=torch.int32,
                device=packed_block_types.device,
            )
            is_his[i] = orig_bt.opth_sampler_cache.is_his

        cache = OptHSamplerPackedBlockTypeCache(
            opth_sample_for_bt=opth_sample_for_bt,
            has_proton_chi=has_proton_chi,
            n_chi_total=n_chi_total,
            chi_defining_atom=chi_defining_atom,
            n_proton_samples=n_proton_samples,
            expanded_samples=expanded_samples,
            n_samples_per_chi=n_samples_per_chi,
            nhq_chi_col=nhq_chi_col,
            nhq_chi_atom=nhq_chi_atom,
            nhq_chi_4atoms=nhq_chi_4atoms,
            nhq_downstream_kfo=nhq_downstream_kfo,
            is_his=is_his,
            n_samples_for_bt_by_orig_bt=n_samples_for_bt_by_orig_bt,
            n_chi_needed_for_bt=n_chi_needed_for_bt,
        )

        setattr(packed_block_types, "opth_sample_cache", cache)

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        if rt.chi_samples:  # has a proton chi
            return True
        if self.flip_NHQ:  # is NHQ if flipNHQ is enabled
            return rt.base_name in _NQ_FLIP_BASES or rt.base_name in _HIS_FLIP_BASES
        return False

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        self._annotate_packed_block_types(pbt)
        return pbt.opth_sample_cache.opth_sample_for_bt[bt_inds]

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return ("CB",)

    def _assert_no_dun_opth_conflict(self, task: "SetPackerTask"):  # noqa: F821
        # for one_pose_blts in task.blts:
        #     for blt in one_pose_blts:
        #         sampler_names = {s.sampler_name() for s in blt.conformer_samplers}
        #         if (
        #             "DunbrackChiSampler" in sampler_names
        #             and "OptHSampler" in sampler_names
        #         ):
        #             raise RuntimeError(
        #                 "OptHSampler and DunbrackChiSampler cannot both be assigned "
        #                 "to the same block. DunbrackChiSampler already samples proton "
        #                 "chi angles as part of each library rotamer."
        #             )
        self_index_in_task = task.conformer_sampler_index[id(self)]
        optH_sampler_allowed = task.per_block_conformer_sampler_allowed[
            :, :, self_index_in_task
        ]
        for i, sampler in enumerate(task.conformer_samplers):
            if sampler is self:
                continue
            if sampler.sampler_name() == "DunbrackChiSampler":
                i_allowed = task.per_block_conformer_sampler_allowed[:, :, i]
                if torch.any(torch.logical_and(optH_sampler_allowed, i_allowed)):
                    raise RuntimeError(
                        "OptHSampler and DunbrackChiSampler cannot both be assigned "
                        "to the same block. DunbrackChiSampler already samples proton "
                        "chi angles as part of each library rotamer."
                    )

    def _measure_all_nhq_flip_chis(self, pose_stack, pose_inds, block_inds):
        offsets = pose_stack.block_coord_offset64[pose_inds, block_inds]
        block_types = pose_stack.block_type_ind64[pose_inds, block_inds]
        a4 = pose_stack.packed_block_types.opth_sample_cache.nhq_chi_4atoms[block_types]

        pose_inds_expanded = pose_inds.repeat_interleave(4)
        offsets_expanded = offsets.repeat_interleave(4)

        c = pose_stack.coords[
            pose_inds_expanded, offsets_expanded + a4.flatten()
        ]  # (4 * n, 3)
        c = c.view(-1, 4, 3)  # (n, 4, 3)
        c = c.to(dtype=torch.float64)
        return coord_dihedrals(c[:, 0], c[:, 1], c[:, 2], c[:, 3])  # (n,)

    def _measure_nhq_flip_chi(self, pose_stack, coords, pose_i, block_j, orig_cache):
        off = int(pose_stack.block_coord_offset[pose_i, block_j].item())
        a4 = orig_cache.nhq_chi_4atoms
        c = coords[pose_i][
            torch.tensor(
                [off + int(a4[k]) for k in range(4)],
                dtype=torch.int64,
                device=pose_stack.device,
            )
        ]  # (4, 3)
        return float(coord_dihedrals(c[0:1], c[1:2], c[2:3], c[3:4])[0].item())

    def _count_rots_and_measure_all_flips(self, pose_stack, task, coords):
        # First we have to get the list of all the blocks where we are using
        # this sampler. Next we will identify the subset of the NHQ blocks
        # where we will measure the chi dihedrals for the flip. Then we will
        # count the number of rotamers for each block as well as the number
        # of chi columns.
        # We will return:
        # n_rots_for_gbt, max_n_chi_cols, pos_flip_chi
        # n_rots_for_gbt: tensor[n_gbt]
        # max_n_chi_cols: int
        # pos_flip_chi: tensor[n_poses, max_n_blocks]

        pbt = pose_stack.packed_block_types
        optH_cache = pbt.opth_sample_cache

        n_gbt = task.cons_bt_pose.shape[0]
        n_rots_for_gbt = torch.zeros(n_gbt, dtype=torch.int32, device=pose_stack.device)
        self_index_in_task = task.conformer_sampler_index[id(self)]

        allowed_bt_is_optH_buildable = task.per_block_conformer_sampler_allowed[
            task.allowed_bt_pose, task.allowed_bt_block, self_index_in_task
        ]  # size (n_allowed_bt,)
        nz_allowed_bt_is_optH_buildable = torch.nonzero(
            allowed_bt_is_optH_buildable, as_tuple=True
        )[
            0
        ]  # size (n_allowed_and_buildable_bt,)
        allowed_and_buildable_pose = task.allowed_bt_pose[
            nz_allowed_bt_is_optH_buildable
        ]
        allowed_and_buildable_block = task.allowed_bt_block[
            nz_allowed_bt_is_optH_buildable
        ]
        allowed_and_buildable_bt = task.allowed_bt_block_type[
            nz_allowed_bt_is_optH_buildable
        ]
        orig_bt_for_allowed_and_buildable = pose_stack.block_type_ind64[
            allowed_and_buildable_pose, allowed_and_buildable_block
        ]

        # print("allowed_and_buildable_bt", allowed_and_buildable_bt.shape)

        n_rots_for_allowed_and_buildable = optH_cache.n_samples_for_bt_by_orig_bt[
            1 if self.flip_NHQ else 0,
            orig_bt_for_allowed_and_buildable,
            allowed_and_buildable_bt,
        ]
        # print("n_rots_for_allowed_and_buildable", n_rots_for_allowed_and_buildable.shape)
        # print("task.allowed_cons_bt[nz_allowed_bt_is_optH_buildable]", task.allowed_cons_bt[nz_allowed_bt_is_optH_buildable].shape)
        n_rots_for_gbt[task.allowed_cons_bt[nz_allowed_bt_is_optH_buildable]] = (
            n_rots_for_allowed_and_buildable
        )

        # max_n_chi_cols = torch.max(
        #     optH_cache.n_chi_needed_for_bt[
        #         1 if self.flip_NHQ else 0, allowed_and_buildable_bt
        #     ]
        # ).item()
        max_n_chi_cols = optH_cache.expanded_samples.shape[1]

        # now we need to figure out which residues are NHQ and measure their chi dihedrals
        if not self.flip_NHQ:
            pos_flip_chi = torch.zeros(
                (pose_stack.n_poses, pose_stack.max_n_blocks),
                dtype=torch.float32,
                device=pose_stack.device,
            )
        else:
            is_allowed_and_buildable_bt_nhq = (
                optH_cache.nhq_chi_col[allowed_and_buildable_bt] >= 0
            )
            nz_nhq = torch.nonzero(is_allowed_and_buildable_bt_nhq, as_tuple=True)[0]
            if len(nz_nhq) == 0:
                pos_flip_chi = torch.zeros(
                    (pose_stack.n_poses, pose_stack.max_n_blocks),
                    dtype=torch.float32,
                    device=pose_stack.device,
                )
            else:
                pose_inds_nhq = allowed_and_buildable_pose[nz_nhq]
                block_inds_nhq = allowed_and_buildable_block[nz_nhq]
                pos_flip_chi_nhq = self._measure_all_nhq_flip_chis(
                    pose_stack, pose_inds_nhq, block_inds_nhq
                )
                pos_flip_chi = torch.zeros(
                    (pose_stack.n_poses, pose_stack.max_n_blocks),
                    dtype=torch.float32,
                    device=pose_stack.device,
                )
                pos_flip_chi[pose_inds_nhq, block_inds_nhq] = pos_flip_chi_nhq
        # TO DO: consider returning the indices of the NHQ blocks
        return n_rots_for_gbt, max_n_chi_cols, pos_flip_chi

    def _count_rots_and_measure_flips(self, pose_stack, task, coords):
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
                if opth_assigned and self.flip_NHQ and orig_cache.nhq_chi_col >= 0:
                    pos_flip_chi[(pose_i, block_j)] = self._measure_nhq_flip_chi(
                        pose_stack, coords, pose_i, block_j, orig_cache
                    )

                for bt in blt.considered_block_types:
                    if not opth_assigned:
                        n_rots_for_gbt_list.append(0)
                        continue
                    bt_cache = bt.opth_sampler_cache
                    n_rots = _n_rots_for_gbt(self, blt, orig, orig_cache, bt, bt_cache)
                    n_rots_for_gbt_list.append(n_rots)
                    if n_rots > 0:
                        max_n_chi_cols = max(
                            max_n_chi_cols,
                            _chi_cols_needed(bt_cache, orig_cache, self.flip_NHQ),
                        )
        return n_rots_for_gbt_list, max_n_chi_cols, pos_flip_chi

    def _fill_proton_chi_for_all_blocks(
        self,
        pose_stack,
        task,
        rot_offset_for_gbt,
        gbt_for_rotamer,
        chi_for_rotamers,
    ):
        pbt = pose_stack.packed_block_types
        opth_cache = pbt.opth_sample_cache

        bt_for_rotamer = task.cons_bt_block_type[gbt_for_rotamer]
        n_samples_for_rotamer = opth_cache.n_proton_samples[bt_for_rotamer].to(
            torch.int64
        )

        rotamers_w_proton_chi_samples = torch.nonzero(
            n_samples_for_rotamer > 1, as_tuple=True
        )[0]
        bt_for_proton_rotamer = bt_for_rotamer[rotamers_w_proton_chi_samples]
        n_rotamers = gbt_for_rotamer.shape[0]
        sample_ind_for_rotamer = (
            torch.arange(n_rotamers, dtype=torch.int64, device=pose_stack.device)
            - rot_offset_for_gbt[gbt_for_rotamer]
        )
        sample_ind_for_proton_rotamer = sample_ind_for_rotamer[
            rotamers_w_proton_chi_samples
        ]

        # print("gbt_for_rotamer", gbt_for_rotamer.shape)
        # print("rotamers_w_proton_chi_samples", rotamers_w_proton_chi_samples.shape)
        # print("rotamers_w_proton_chi_samples[:10]", rotamers_w_proton_chi_samples[:10])
        print("chi_for_rotamers", chi_for_rotamers.shape)
        print("opth_cache.expanded_samples", opth_cache.expanded_samples.shape)
        print(
            "opth_cache.expanded_samples[bt_for_proton_rotamer, :, sample_ind_for_proton_rotamer]",
            opth_cache.expanded_samples[
                bt_for_proton_rotamer, :, sample_ind_for_proton_rotamer
            ].shape,
        )
        # print("bt_for_proton_rotamer", bt_for_proton_rotamer.shape)
        # print("sample_ind_for_proton_rotamer", sample_ind_for_proton_rotamer.shape)

        max_n_expanded_chi = opth_cache.expanded_samples.shape[1]
        chi_for_rotamers[rotamers_w_proton_chi_samples, :max_n_expanded_chi] = (
            opth_cache.expanded_samples[
                bt_for_proton_rotamer, :, sample_ind_for_proton_rotamer
            ]
        )

    def _fill_proton_chi_block(
        self,
        pose_stack,
        bt_cache,
        rot_offset,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ):
        ci = next(
            i for i in range(bt_cache.n_chi_total) if bt_cache.chi_defining_atom[i] >= 0
        )
        n_samp = int(bt_cache.n_samples_per_chi[ci])
        chi_defining_atom_for_rotamer[rot_offset : rot_offset + n_samp, ci] = int(
            bt_cache.chi_defining_atom[ci]
        )
        chi_for_rotamers[rot_offset : rot_offset + n_samp, ci] = torch.tensor(
            bt_cache.expanded_samples[ci, :n_samp],
            dtype=torch.float32,
            device=pose_stack.device,
        )

    def _fill_all_nhq_blocks(
        self,
        pose_stack,
        task,
        gbt_for_rotamer,
        pos_flip_chi,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ):
        pbt = pose_stack.packed_block_types
        opth_cache = pbt.opth_sample_cache

        bt_for_rotamer = task.cons_bt_block_type[gbt_for_rotamer]
        rotamer_nhq_chi_col = opth_cache.nhq_chi_col[bt_for_rotamer]
        rotamer_is_flippable = rotamer_nhq_chi_col >= 0
        flippable_rotamers = torch.nonzero(rotamer_is_flippable, as_tuple=True)[0]
        n_flippable_rotamers = flippable_rotamers.shape[0]

        flipped_flippable_rotamers = (
            2
            * torch.arange(
                n_flippable_rotamers / 2, dtype=torch.int64, device=pose_stack.device
            )
            + 1
        )
        flipped_rotamers = flippable_rotamers[flipped_flippable_rotamers]
        is_flipped_flippable_rotamer = torch.zeros(
            (n_flippable_rotamers,), dtype=torch.bool, device=pose_stack.device
        )
        is_flipped_flippable_rotamer[flipped_flippable_rotamers] = True
        is_unflipped_flippable_rotamer = torch.ones(
            (n_flippable_rotamers,), dtype=torch.bool, device=pose_stack.device
        )
        is_unflipped_flippable_rotamer[flipped_flippable_rotamers] = False
        unflipped_flippable_rotamers = flippable_rotamers[
            is_unflipped_flippable_rotamer
        ]

        flipped_rot_chi = rotamer_nhq_chi_col[flipped_rotamers]
        gbt_for_flipped_rotamer = gbt_for_rotamer[flipped_rotamers]
        pose_for_flipped_rotamer = task.cons_bt_pose[gbt_for_flipped_rotamer]
        block_for_flipped_rotamer = task.cons_bt_block[gbt_for_flipped_rotamer]
        bt_for_flipped_rotamer = bt_for_rotamer[flipped_rotamers]

        is_his_rotamer = opth_cache.is_his[bt_for_rotamer]
        is_orig_bt_rotamer = bt_for_rotamer == (
            pose_stack.block_type_ind64[
                task.cons_bt_pose[gbt_for_rotamer], task.cons_bt_block[gbt_for_rotamer]
            ]
        )
        is_his_taut_rotamer = torch.logical_and(is_his_rotamer, ~is_orig_bt_rotamer)
        is_unflipped_rotamer = torch.zeros(
            gbt_for_rotamer.shape[0], dtype=torch.bool, device=pose_stack.device
        )
        is_unflipped_rotamer[unflipped_flippable_rotamers] = True
        is_his_taut_not_flipped_rotamer = torch.logical_and(
            is_his_taut_rotamer, is_unflipped_rotamer
        )
        his_taut_not_flipped_rotamers = torch.nonzero(
            is_his_taut_not_flipped_rotamer, as_tuple=True
        )[0]
        his_taut_not_flipped_rot_chi = rotamer_nhq_chi_col[
            his_taut_not_flipped_rotamers
        ]

        gbt_for_his_taut_not_flipped_rotamer = gbt_for_rotamer[
            his_taut_not_flipped_rotamers
        ]
        pose_for_his_taut_not_flipped_rotamer = task.cons_bt_pose[
            gbt_for_his_taut_not_flipped_rotamer
        ]
        block_for_his_taut_not_flipped_rotamer = task.cons_bt_block[
            gbt_for_his_taut_not_flipped_rotamer
        ]
        bt_for_his_taut_not_flipped_rotamer = bt_for_rotamer[
            his_taut_not_flipped_rotamers
        ]

        # mark the chi for the HIS tautomer in its non-flipped state
        # because we have to rebuild the ring from ideal geometry.
        chi_for_rotamers[
            his_taut_not_flipped_rotamers, his_taut_not_flipped_rot_chi
        ] = pos_flip_chi[
            pose_for_his_taut_not_flipped_rotamer,
            block_for_his_taut_not_flipped_rotamer,
        ]
        chi_defining_atom_for_rotamer[
            his_taut_not_flipped_rotamers, his_taut_not_flipped_rot_chi
        ] = opth_cache.nhq_chi_atom[bt_for_his_taut_not_flipped_rotamer]

        # mark the flipped chi for all NHQ rotamers
        chi_for_rotamers[flipped_rotamers, flipped_rot_chi] = (
            pos_flip_chi[
                pose_for_flipped_rotamer,
                block_for_flipped_rotamer,
            ]
            + math.pi
        )
        chi_defining_atom_for_rotamer[flipped_rotamers, flipped_rot_chi] = (
            opth_cache.nhq_chi_atom[bt_for_flipped_rotamer]
        )

    def _fill_nhq_block(
        self,
        bt,
        blt,
        bt_cache,
        orig_cache,
        flip_chi,
        rot_offset,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ):
        # Two NHQ rotamers per (blt, considered-bt):
        #   offset 0 = input chi  (no rotation)
        #   offset 1 = input chi + 180 deg
        # Setting chi_defining_atom_for_rotamer >= 0 marks a rotamer for
        # chi-override + downstream-rebuild in _opth_fill_dofs; rotamers
        # left at -1 inherit input DOFs unchanged.  So we only mark offsets
        # whose chi actually differs from input.
        chi_col = orig_cache.nhq_chi_col

        # Offset 0: chi == input, so only mark on HIS<->HIS_D tautomer swap
        # (atom layout differs, ring must be rebuilt from ideal).
        if orig_cache.is_his and bt is not blt.original_block_type:
            chi_defining_atom_for_rotamer[rot_offset, chi_col] = bt_cache.nhq_chi_atom
            chi_for_rotamers[rot_offset, chi_col] = float(flip_chi)

        # Offset 1: always the +180 deg flip, always marked.
        chi_defining_atom_for_rotamer[rot_offset + 1, chi_col] = bt_cache.nhq_chi_atom
        chi_for_rotamers[rot_offset + 1, chi_col] = float(flip_chi) + math.pi

    def _fill_all_chi_tensors(
        self,
        pose_stack,
        task,
        rot_offset_for_gbt,
        gbt_for_rotamer,
        pos_flip_chi,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ):
        self._fill_proton_chi_for_all_blocks(
            pose_stack,
            task,
            rot_offset_for_gbt,
            gbt_for_rotamer,
            chi_for_rotamers,
        )
        self._fill_all_nhq_blocks(
            pose_stack,
            task,
            gbt_for_rotamer,
            pos_flip_chi,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )

    def _fill_chi_tensors(
        self,
        pose_stack,
        task,
        n_rots_for_gbt_list,
        pos_flip_chi,
        chi_defining_atom_for_rotamer,
        chi_for_rotamers,
    ):
        rot_offset = 0
        gbt_idx = 0
        for pose_i, one_pose_blts in enumerate(task.blts):
            for block_j, blt in enumerate(one_pose_blts):
                orig_cache = blt.original_block_type.opth_sampler_cache
                flip_chi = pos_flip_chi.get((pose_i, block_j), None)
                opth_assigned = self in blt.conformer_samplers

                for bt in blt.considered_block_types:
                    if not opth_assigned:
                        gbt_idx += 1
                        continue
                    n_rots = n_rots_for_gbt_list[gbt_idx]
                    gbt_idx += 1
                    if n_rots == 0:
                        continue

                    bt_cache = bt.opth_sampler_cache
                    # Proton chi (SER/THR/TYR/CYS) ... just flip chi
                    if bt_cache.has_proton_chi and bt is blt.original_block_type:
                        self._fill_proton_chi_block(
                            pose_stack,
                            bt_cache,
                            rot_offset,
                            chi_defining_atom_for_rotamer,
                            chi_for_rotamers,
                        )
                    # NHQ: flip and (for flips/tautamer change) idealize downstream
                    elif self.flip_NHQ and orig_cache.nhq_chi_col >= 0:
                        self._fill_nhq_block(
                            bt,
                            blt,
                            bt_cache,
                            orig_cache,
                            flip_chi,
                            rot_offset,
                            chi_defining_atom_for_rotamer,
                            chi_for_rotamers,
                        )

                    rot_offset += n_rots

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
    ) -> Tuple[Tensor[torch.int32][:], Tensor[torch.int32][:], dict,]:
        self._annotate_packed_block_types(pose_stack.packed_block_types)

        # ensure dunbrack and optH sampler are not _both_ specified for the same block
        self._assert_no_dun_opth_conflict(task)

        # 1) compute:
        #      n_rots per GBT
        #      max chi tensor width
        #      current last-chi angle
        # for each NHQ position in the input
        coords = pose_stack.coords.double()  # coord_dihedrals needs float64
        # TO DO: remove this call.
        # n_rots_for_gbt_list, max_n_chi_cols, pos_flip_chi = (
        #     self._count_rots_and_measure_flips(pose_stack, task, coords)
        # )
        n_rots_for_gbt, max_n_chi_cols, pos_flip_chi = (
            self._count_rots_and_measure_all_flips(pose_stack, task, coords)
        )

        # n_rots_for_gbt = torch.tensor(
        #     n_rots_for_gbt_list, dtype=torch.int32, device=pose_stack.device
        # )
        # torch.testing.assert_close(
        #     n_rots_for_gbt, n_rots_for_gbt2, rtol=1e-5, atol=1e-5
        # )
        n_rots_total = int(n_rots_for_gbt.sum().item())

        if n_rots_total == 0:
            empty_chi = torch.zeros(
                (0, max_n_chi_cols), dtype=torch.int32, device=pose_stack.device
            )
            return (
                n_rots_for_gbt,
                torch.zeros(0, dtype=torch.int32, device=pose_stack.device),
                dict(
                    chi_defining_atom_for_rotamer=empty_chi,
                    chi_for_rotamers=empty_chi.float(),
                ),
            )
        # gbt_for_rotamer = torch.repeat_interleave(
        #     torch.arange(
        #         len(n_rots_for_gbt_list), dtype=torch.int32, device=pose_stack.device
        #     ),
        #     n_rots_for_gbt.to(torch.int64),
        # )
        rot_offset_for_gbt = exclusive_cumsum1d(n_rots_for_gbt)
        gbt_for_rotamer = torch.repeat_interleave(
            torch.arange(
                n_rots_for_gbt.shape[0], dtype=torch.int32, device=pose_stack.device
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
        # chi_defining_atom_for_rotamer2 = torch.full_like(
        #     chi_defining_atom_for_rotamer, -1
        # )
        # chi_for_rotamers2 = torch.zeros_like(chi_for_rotamers)

        # # 2) fill chi tensors
        # self._fill_chi_tensors(
        #     pose_stack,
        #     task,
        #     n_rots_for_gbt_list,
        #     pos_flip_chi,
        #     chi_defining_atom_for_rotamer,
        #     chi_for_rotamers,
        # )

        self._fill_all_chi_tensors(
            pose_stack,
            task,
            rot_offset_for_gbt,
            gbt_for_rotamer,
            pos_flip_chi,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )
        # diff = chi_for_rotamers - chi_for_rotamers2
        # is_sig_diff = torch.abs(diff) > 1e-5
        # pos_w_diffs = torch.nonzero(is_sig_diff, as_tuple=True)

        # torch.testing.assert_close(
        #     chi_for_rotamers, chi_for_rotamers2, rtol=1e-5, atol=1e-5
        # )

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
            self.flip_NHQ,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
