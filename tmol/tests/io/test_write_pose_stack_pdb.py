import torch
import numpy

from tmol.io.write_pose_stack_pdb import (
    atom_records_from_pose_stack,
)
from tmol.chemical.restypes import find_simple_polymeric_connections
from tmol.io.pdb_parsing import to_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.basic_resolution import pose_stack_from_canonical_form


def test_atom_records_from_pose_stack_1(ubq_pdb, ubq_res, torch_device):
    connections = find_simple_polymeric_connections(ubq_res)
    p = PoseStackBuilder.one_structure_from_residues_and_connections(
        ubq_res, connections, torch_device
    )

    records = atom_records_from_pose_stack(p)
    pdb_lines = to_pdb(records)

    pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    starting_ubq_pdb_atom_lines = [x for x in ubq_pdb.split("\n") if x[:6] == "ATOM  "]

    assert len(pdb_atom_lines) == len(starting_ubq_pdb_atom_lines)


def test_atom_records_from_pose_stack_2(ubq_pdb, ubq_res, torch_device):
    connections5 = find_simple_polymeric_connections(ubq_res[:5])
    p1 = PoseStackBuilder.one_structure_from_residues_and_connections(
        ubq_res[:5], connections5, torch_device
    )
    connections7 = find_simple_polymeric_connections(ubq_res[:7])
    p2 = PoseStackBuilder.one_structure_from_residues_and_connections(
        ubq_res[:7], connections7, torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    records = atom_records_from_pose_stack(poses)
    pdb_lines = to_pdb(records)

    out_fname = (
        "test_write_multi_model_pose_stack_antibody_cpu.pdb"
        if torch_device == torch.device("cpu")
        else "test_write_multi_model_pose_stack_antibody_cuda.pdb"
    )
    with open(out_fname, "w") as fid:
        fid.write(pdb_lines)

    # pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    # starting_ubq_pdb_atom_lines = [x for x in ubq_pdb.split("\n") if x[:6] == "ATOM  "]
    #
    # assert len(pdb_atom_lines) == len(starting_ubq_pdb_atom_lines)


def test_atom_records_for_multi_chain_pdb(pertuzumab_lines, torch_device):
    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(
        pertuzumab_lines
    )

    ch_beg = torch.tensor(ch_beg, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    pose_stack = pose_stack_from_canonical_form(ch_beg, can_rts, coords, at_is_pres)

    records = atom_records_from_pose_stack(
        pose_stack, numpy.array([x for x in "LH"], dtype=str)
    )
    pdb_lines = to_pdb(records)
    pdb_atom_lines = [x for x in pdb_lines.split("\n") if x[:6] == "ATOM  "]
    pertuzumab_atom_lines = [
        x for x in pertuzumab_lines.split("\n") if x[:6] == "ATOM  "
    ]
    assert len(pdb_atom_lines) > len(pertuzumab_atom_lines)

    # out_fname = (
    #     "test_write_pose_stack_antibody_cpu.pdb"
    #     if torch_device == torch.device("cpu") else
    #     "test_write_pose_stack_antibody_cuda.pdb"
    # )
    # with open(out_fname, "w") as fid:
    #     fid.write(pdb_lines)
