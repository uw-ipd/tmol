import os
import torch
import biotite.structure
from biotite.structure.io.pdb import PDBFile


from tmol.io.pose_stack_from_biotite import (
    biotite_from_canonical_form,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)
from tmol.io.canonical_form import CanonicalForm


def test_canonical_form_from_biotite(biotite_1r21, torch_device):
    pdb = canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


def test_pose_stack_from_and_to_biotite_1ubq(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)
    # print(pose_stack.coords[0,0:30])
    biotite_atom_array = biotite_from_pose_stack(pose_stack)

    file = PDBFile()
    file.set_structure(biotite_atom_array)
    file.write("test_out.pdb")
    print(biotite_atom_array)


def test_pose_stack_from_biotite_1ubq_slice(biotite_1ubq, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1ubq)
    bt = biotite_1ubq[0 : starts[30]]
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_n_term(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][0 : starts[3]]  # subscript 0 to get the first structure
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_his_d(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][
        starts[52] : starts[55]
    ]  # subscript 0 to get the first structure
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_missing_sidechain(biotite_1bl8, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1bl8)
    bt = biotite_1bl8[starts[0] : starts[6]]
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)

    biotite_atom_array = biotite_from_pose_stack(pose_stack)
    file = PDBFile()
    file.set_structure(biotite_atom_array)
    file.write("test_out.pdb")
    print(biotite_atom_array)
