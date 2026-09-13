"""Proton sampling must not displace a nucleotide's covalent scaffold."""

import pytest
import torch

from tmol.io import pose_stack_from_cif
from tmol.pack import PackerPalette, PackerTask, SetPackerTask, pack_rotamers
from tmol.pack.rotamer import OptHSampler, build_rotamers
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path


@pytest.mark.parametrize(
    "fixture",
    [
        "na_dna_5mc_1d17",
        "na_dna_8og_183d",
        "na_rna_psu_1bzt",
        "na_rna_2ome_310d",
        "na_dna_ttd_1ttd",
    ],
)
def test_proton_rotamers_and_packing_preserve_heavy_atoms(fixture, torch_device):
    pose, context = pose_stack_from_cif(
        data_path("ncaa_fixtures") / f"{fixture}.cif",
        torch_device,
        prepare_ligands=True,
        ligand_seed=20250828,
        no_optH=True,
        return_context=True,
    )
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(OptHSampler(flip_NHQ=False))
    _, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), context.parameter_database.chemical
    )
    elements = {
        a.name: a.element for a in context.parameter_database.chemical.atom_types
    }
    heavy_indices = []
    for block, bt_index in enumerate(pose.block_type_ind[0].tolist()):
        bt = pose.packed_block_types.active_block_types[bt_index]
        heavy = torch.tensor(
            [i for i, a in enumerate(bt.atoms) if elements[a.atom_type] != "H"],
            device=torch_device,
        )
        offset = int(pose.block_coord_offset[0, block])
        heavy_indices.append(heavy + offset)
        expected = pose.coords[0, heavy + offset]
        first = int(rotamers.rot_offset_for_block[0, block])
        count = int(rotamers.n_rots_for_block[0, block])
        assert count > 0
        for rot in range(first, first + count):
            start = int(rotamers.coord_offset_for_rot[rot])
            torch.testing.assert_close(
                rotamers.coords[start + heavy],
                expected,
                atol=2e-4,
                rtol=0,
                msg=f"{fixture}, {bt.name}, block {block}, rotamer {rot - first}",
            )
    score = beta2016_score_function(torch_device, param_db=context.parameter_database)
    packed = pack_rotamers(pose, score, task, verbose=False)
    indices = torch.cat(heavy_indices)
    torch.testing.assert_close(
        packed.coords[0, indices], pose.coords[0, indices], atol=2e-4, rtol=0
    )
