from tmol.io import (
    pose_stack_from_file,
    default_canonical_ordering,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)
from tmol.io._write_pose_stack_pdb import (
    _atom_records_from_pose_stack,
)
from tmol.io._pdb_parsing import _to_pdb
from tmol.io._canonical_ordering import _canonical_form_from_pdb

# from tmol.chemical import find_simple_polymeric_connections
from tmol.pose import PoseStackBuilder


def test_atom_records_from_pose_stack_1(ubq_pdb, torch_device):
    p = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )

    records = _atom_records_from_pose_stack(p)
    pdb_lines = _to_pdb(records)

    pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    starting_ubq_pdb_atom_lines = [x for x in ubq_pdb.split("\n") if x[:6] == "ATOM  "]

    assert len(pdb_atom_lines) == len(starting_ubq_pdb_atom_lines)


def test_atom_records_from_pose_stack_2(ubq_pdb, torch_device):
    p1 = pose_stack_from_file(
        ubq_pdb,
        torch_device,
        residue_end=5,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    p2 = pose_stack_from_file(
        ubq_pdb,
        torch_device,
        residue_end=7,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    records = _atom_records_from_pose_stack(poses)
    pdb_lines = _to_pdb(records)

    assert pdb_lines[: len("MODEL 1\n")] == "MODEL 1\n"
    # I cannot for the life of me calculate the "correct" size of the pdb_lines string
    # My calculation producing a size of 14376 matches what "wc" report for the file
    # when it is written to disk, but the len of the string that produces that file
    # is bigger by almost 300 characters??
    # target_len = 67 * 214 + len("TER\n") * 2 + len("MODEL 1\n") * 2 + len("ENDMDL\n") * 2

    assert len(pdb_lines) == 14778


def test_atom_records_for_multi_chain_pdb(pertuzumab_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = _canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)
    pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)

    records = _atom_records_from_pose_stack(pose_stack)
    pdb_lines = _to_pdb(records)
    pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    pertuzumab_atom_lines = [x for x in pertuzumab_pdb.split("\n") if x[:6] == "ATOM  "]
    assert len(pdb_atom_lines) > len(pertuzumab_atom_lines)
