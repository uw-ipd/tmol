import torch

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.bonded_atom import stacked_bonded_atoms_for_system
from tmol.score.modules.elec import ElecScore
from tmol.score.elec.params import ElecParamResolver
from tmol.score.bonded_atom import bonded_path_length, bonded_path_length_stacked


def test_resolve_elec_params(default_database, ubq_system, torch_device):
    param_resolver = ElecParamResolver.from_database(
        default_database.scoring.elec, torch_device
    )

    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    res_names = ubq_system.atom_metadata["residue_name"].copy()
    res_indices = ubq_system.atom_metadata["residue_index"].copy()
    atom_pair_bpl = bonded_path_length(ubq_system.bonds, ubq_system.coords.shape[0], 6)

    param_resolver.remap_bonded_path_lengths(
        atom_pair_bpl[None, :],
        res_names[None, :],
        res_indices[None, :],
        atom_names[None, :],
    )
