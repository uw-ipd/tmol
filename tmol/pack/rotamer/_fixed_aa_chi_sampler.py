from __future__ import annotations

import numpy
import torch
import attr

from typing import Tuple

from tmol.types._torch import Tensor
from tmol.chemical import RefinedResidueType
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack

from tmol.pack.rotamer._chi_sampler import ChiSampler


def _is_fixed_aa(rt: RefinedResidueType) -> bool:
    """Alpha amino acid whose only chi, if any, are proton chi: one conformer."""
    polymer = rt.properties.polymer
    if not polymer.is_polymer or polymer.polymer_type != "amino_acid":
        return False
    if polymer.backbone_type != "alpha_aa":
        return False
    proton_chis = {cs.chi_dihedral for cs in rt.chi_samples if cs.is_proton}
    return all(t.name in proton_chis for t in rt.torsions if t.name.startswith("chi"))


@attr.s(auto_attribs=True, frozen=True)
class FixedAAChiSampler(ChiSampler):
    """Generate one ideal side-chain conformer for fixed amino acids."""

    @classmethod
    def sampler_name(cls):
        return "FixedAAChiSampler"

    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return _is_fixed_aa(rt)

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return pbt.fixed_aa_chi_sampler_builds_bt[bt_inds]

    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        """CA's sidechain branches: the substituents that carry further atoms.

        With none (glycine), every non-mainchain CA substituent is rebuilt:
        glycine's alpha hydrogens exchange under reflection, so rebuilding only
        one can place it on top of the retained one, and a glycine-to-amino-acid
        design must not inherit an arbitrary one.
        """
        ca, mainchain = rt.atom_to_idx["CA"], rt.properties.polymer.mainchain_atoms
        n_bonds = numpy.bincount(rt.bond_indices[:, 0], minlength=len(rt.atoms))
        substituents = [
            j
            for i, j in rt.bond_indices
            if i == ca and rt.atoms[j].name not in mainchain
        ]
        branches = [j for j in substituents if n_bonds[j] > 1]
        return tuple(rt.atoms[j].name for j in (branches or substituents))

    def annotate_residue_type(self, block_type):
        if hasattr(block_type, "fixed_aa_chi_sampler_builds_bt"):
            return
        setattr(block_type, "fixed_aa_chi_sampler_builds_bt", _is_fixed_aa(block_type))

    def annotate_packed_block_types(self, packed_block_types):
        if hasattr(packed_block_types, "fixed_aa_chi_sampler_builds_bt"):
            return
        builds_bt = torch.tensor(
            [
                bt.fixed_aa_chi_sampler_builds_bt
                for bt in packed_block_types.active_block_types
            ],
            dtype=torch.bool,
            device=packed_block_types.device,
        )
        setattr(packed_block_types, "fixed_aa_chi_sampler_builds_bt", builds_bt)

    def sample_chi_for_poses(
        self, poses: PoseStack, task
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        pbt = poses.packed_block_types
        self_ind_in_task = task.conformer_sampler_index[id(self)]
        faas_allowed = task.per_block_conformer_sampler_allowed[:, :, self_ind_in_task]
        faas_allowed_for_cons_bt = faas_allowed[task.cons_bt_pose, task.cons_bt_block]
        faa_builds_bt = pbt.fixed_aa_chi_sampler_builds_bt
        faa_builds_bt_for_cons_bt = faa_builds_bt[task.cons_bt_block_type]
        cons_bt_is_allowed = task.per_block_is_block_type_allowed[
            task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
        ]

        is_bt_faas_allowed_and_built_by = torch.logical_and(
            faas_allowed_for_cons_bt, faa_builds_bt_for_cons_bt
        )
        n_rots_for_gbt = torch.logical_and(
            is_bt_faas_allowed_and_built_by, cons_bt_is_allowed
        ).to(torch.int32)

        gbt_for_rotamer = n_rots_for_gbt.nonzero(as_tuple=True)[0].to(torch.int32)
        chi_for_rotamers = torch.zeros(
            (gbt_for_rotamer.shape[0], 1), dtype=torch.float32, device=poses.device
        )
        chi_defining_atom_for_rotamer = torch.full_like(
            chi_for_rotamers, -1, dtype=torch.int32
        )

        return (
            n_rots_for_gbt,
            gbt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )
