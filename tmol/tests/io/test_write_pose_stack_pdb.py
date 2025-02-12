import numpy
import torch
import os

from tmol.io import pose_stack_from_pdb
from tmol.io.write_pose_stack_pdb import (
    write_pose_stack_pdb,
    atom_records_from_pose_stack,
)

# from tmol.chemical.restypes import find_simple_polymeric_connections
from tmol.io.pdb_parsing import to_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


def test_atom_records_from_pose_stack_1(ubq_pdb, torch_device):
    p = pose_stack_from_pdb(ubq_pdb, torch_device)

    records = atom_records_from_pose_stack(p)
    pdb_lines = to_pdb(records)

    pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    starting_ubq_pdb_atom_lines = [x for x in ubq_pdb.split("\n") if x[:6] == "ATOM  "]

    assert len(pdb_atom_lines) == len(starting_ubq_pdb_atom_lines)


def test_atom_records_from_pose_stack_2(ubq_pdb, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    records = atom_records_from_pose_stack(poses)
    pdb_lines = to_pdb(records)

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
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    records = atom_records_from_pose_stack(
        pose_stack, None, numpy.array([x for x in "LH"], dtype=str)
    )
    pdb_lines = to_pdb(records)
    pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    pertuzumab_atom_lines = [x for x in pertuzumab_pdb.split("\n") if x[:6] == "ATOM  "]
    assert len(pdb_atom_lines) > len(pertuzumab_atom_lines)


def test_write_pose_stack_pdb(ubq_pdb):
    device = torch.device("cpu")
    ps = pose_stack_from_pdb(ubq_pdb, device)
    output_fname = "tmol/tests/io/write_pose_stack_pdb.pdb"
    assert not os.path.isfile(output_fname)
    write_pose_stack_pdb(ps, output_fname)
    assert os.path.isfile(output_fname)

    # incidentally: test the call path that reads a PDB from disk
    # instead of from the contents of file
    ps2 = pose_stack_from_pdb(output_fname, device)

    torch.testing.assert_close(ps.coords, ps2.coords)

    os.remove(output_fname)
