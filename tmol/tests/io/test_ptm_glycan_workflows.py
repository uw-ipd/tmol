import gzip

from biotite.structure.io.pdbx import CIFFile, get_structure
import pytest
import torch

from tmol import beta2016_score_function, run_cart_min
from tmol.io import biotite_from_pose_stack, pose_stack_from_biotite
from tmol.pack import PackerPalette, PackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.tests.data import data_path


def _read_fixture(name):
    with gzip.open(data_path("cif", name), "rt") as handle:
        return get_structure(CIFFile.read(handle), model=1, include_bonds=True)


def _score_function(pose, context):
    return beta2016_score_function(pose.device, param_db=context.parameter_database)


def _score(pose, score_function):
    scorer = score_function.render_whole_pose_scoring_module(pose)
    return scorer(pose.coords).sum()


def _pack(pose, context, score_function, disabled=None):
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(
        create_dunbrack_sampler_from_database(context.parameter_database, pose.device)
    )
    task.add_conformer_sampler(FixedAAChiSampler())
    if disabled is not None:
        task.disable_packing_by_block_mask(disabled)
    return pack_rotamers(pose, score_function, task, verbose=False)


def _block_types(pose):
    return [
        pose.packed_block_types.active_block_types[int(index)]
        for index in pose.block_type_ind64[0]
        if index >= 0
    ]


def _assert_score_and_gradient(pose, score_function):
    coords = pose.coords.detach().clone().requires_grad_(True)
    score = score_function.render_whole_pose_scoring_module(pose)(coords).sum()
    score.backward()
    assert torch.isfinite(score)
    assert torch.isfinite(coords.grad).all()


@pytest.mark.parametrize(
    ("chain", "expected_parents"),
    [
        ("A", {"SEP": "SER", "PTR": "TYR"}),
        ("B", {"TPO": "THR"}),
        ("C", {"MLZ": "LYS"}),
        ("D", {"MLY": "LYS"}),
        ("E", {"M3L": "LYS"}),
    ],
)
def test_each_pdb_ptm_builds_as_a_modified_polymer(
    torch_device, chain, expected_parents
):
    structure = _read_fixture("pdb_ptm_peptides.cif.gz")
    structure = structure[structure.chain_id == chain]
    pose = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
    )
    parents = {
        block_type.name3: block_type.base_name
        for block_type in _block_types(pose)
        if block_type.name3 in expected_parents
    }
    assert parents == expected_parents


def test_pdb_ptms_score_pack_and_minimize(torch_device):
    pose, context = pose_stack_from_biotite(
        _read_fixture("pdb_ptm_peptides.cif.gz"),
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    ptm_parents = {
        block_type.name3: block_type.base_name
        for block_type in _block_types(pose)
        if block_type.name3 in {"SEP", "TPO", "PTR", "MLZ", "MLY", "M3L"}
    }
    assert ptm_parents == {
        "SEP": "SER",
        "TPO": "THR",
        "PTR": "TYR",
        "MLZ": "LYS",
        "MLY": "LYS",
        "M3L": "LYS",
    }

    score_function = _score_function(pose, context)
    _assert_score_and_gradient(pose, score_function)
    packed = _pack(pose, context, score_function)
    minimized = run_cart_min(
        packed.clone(), score_function, optimizer_kwargs={"max_iter": 20}
    )
    assert torch.isfinite(_score(packed, score_function))
    assert _score(minimized, score_function) <= _score(packed, score_function) + 1e-3


def _has_ser_nag_link(pose):
    block_types = _block_types(pose)
    for block1, block_type1 in enumerate(block_types):
        for connection1, (block2, connection2) in enumerate(
            pose.inter_residue_connections64[0, block1].tolist()
        ):
            if block2 < 0:
                continue
            block_type2 = block_types[block2]
            atoms = {
                block_type1.connections[connection1].atom,
                block_type2.connections[connection2].atom,
            }
            if atoms == {"OG", "C1"}:
                return True
    return False


def test_pdb_o_glcnac_score_pack_minimize_and_round_trip(torch_device):
    structure = _read_fixture("5VVU_o_glcnac_peptide.cif.gz")
    pose, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    assert _has_ser_nag_link(pose)
    assert "SER:covalent_OG" in {bt.name for bt in _block_types(pose)}
    nag = next(bt for bt in _block_types(pose) if bt.name3 == "NAG")
    assert nag.properties.polymer.polymer_type == "carbohydrate"
    assert "down" in nag.connection_to_cidx

    score_function = _score_function(pose, context)
    _assert_score_and_gradient(pose, score_function)
    disable_glycan = torch.tensor(
        [
            [
                bt.properties.polymer.polymer_type == "carbohydrate"
                for bt in _block_types(pose)
            ]
        ],
        device=pose.device,
    )
    packed = _pack(pose, context, score_function, disable_glycan)
    minimized = run_cart_min(
        packed.clone(), score_function, optimizer_kwargs={"max_iter": 20}
    )
    assert torch.isfinite(_score(packed, score_function))
    assert _score(minimized, score_function) <= _score(packed, score_function) + 1e-3
    assert _has_ser_nag_link(minimized)

    exported = biotite_from_pose_stack(minimized, co=context.canonical_ordering)
    rebuilt = pose_stack_from_biotite(
        exported, torch_device, context=context, no_optH=True
    )
    assert _has_ser_nag_link(rebuilt)
