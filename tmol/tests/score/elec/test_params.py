import torch

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.system.score_support import score_method_to_even_weights_dict
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


def test_jagged_parameter_resolution_rbpl(ubq_res, default_database, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    param_resolver = ElecParamResolver.from_database(
        default_database.scoring.elec, torch_device
    )

    def rbpl(system):
        atom_names = system.atom_metadata["atom_name"].copy()
        res_names = system.atom_metadata["residue_name"].copy()
        res_indices = system.atom_metadata["residue_index"].copy()
        atom_pair_bpl = bonded_path_length(system.bonds, system.coords.shape[0], 6)

        return param_resolver.remap_bonded_path_lengths(
            atom_pair_bpl[None, :],
            res_names[None, :],
            res_indices[None, :],
            atom_names[None, :],
        )

    ub40_rbpl = rbpl(ubq40)
    ub60_rbpl = rbpl(ubq60)

    score_system = ScoreSystem.build_for(
        twoubq, {ElecScore}, score_method_to_even_weights_dict(ElecScore)
    )
    twoubq_dict = stacked_bonded_atoms_for_system(twoubq, score_system)
    twoubq_bonds = bonded_path_length_stacked(
        twoubq_dict.bonds, 2, ubq60.system_size, 6
    )

    tubq_rbpl = param_resolver.remap_bonded_path_lengths(
        twoubq_bonds,
        twoubq_dict.res_names,
        twoubq_dict.res_indices,
        twoubq_dict.atom_names,
    )

    torch.testing.assert_allclose(
        ub40_rbpl, tubq_rbpl[0:1, : ub40_rbpl.shape[1], : ub40_rbpl.shape[1]]
    )
    torch.testing.assert_allclose(ub60_rbpl, tubq_rbpl[1:2, :])


def test_jagged_parameter_resolution_part_charges(
    ubq_res, default_database, torch_device
):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    twoubq = PackedResidueSystemStack((ubq40, ubq60))

    param_resolver = ElecParamResolver.from_database(
        default_database.scoring.elec, torch_device
    )

    def part_char(system):
        atom_names = system.atom_metadata["atom_name"].copy()
        res_names = system.atom_metadata["residue_name"].copy()

        return param_resolver.resolve_partial_charge(
            res_names[None, :], atom_names[None, :]
        )

    ub40_pcs = part_char(ubq40)
    ub60_pcs = part_char(ubq60)

    score_system = ScoreSystem.build_for(
        twoubq, {ElecScore}, score_method_to_even_weights_dict(ElecScore)
    )
    twoubq_dict = stacked_bonded_atoms_for_system(twoubq, score_system)
    tubq_pcs = param_resolver.resolve_partial_charge(
        twoubq_dict.res_names, twoubq_dict.atom_names
    )

    torch.testing.assert_allclose(ub40_pcs, tubq_pcs[0:1, : ub40_pcs.shape[1]])
    torch.testing.assert_allclose(ub60_pcs, tubq_pcs[1:2, :])
