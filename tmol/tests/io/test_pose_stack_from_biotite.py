import biotite.structure
from biotite.structure.io.pdb import PDBFile

import pathlib


from tmol.io.pose_stack_from_biotite import (
    biotite_from_canonical_form,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)


def test_load_bulk_cif_from_biotite(torch_device):
    dir_path = pathlib.Path("/home/jflat06/pdbs/")

    for file_path in dir_path.iterdir():
        exclude = [
            # pathlib.PosixPath('/home/jflat06/pdbs/7k2e__1__1.A__1.C.cif'),
            pathlib.PosixPath(
                "/home/jflat06/pdbs/4tlm__1__1.A_1.B__1.I.cif"
            ),  # 581 missing sidechains
            # pathlib.PosixPath('/home/jflat06/pdbs/6h9v__1__1.A_1.B__1.C.cif'), # OXT
            # pathlib.PosixPath('/home/jflat06/pdbs/3n0i__1__1.A_1.B_1.C__1.D.cif'), # OXT
            # pathlib.PosixPath('/home/jflat06/pdbs/6c4c__2__1.E_1.F__1.X.cif'), # H
            # pathlib.PosixPath('/home/jflat06/pdbs/4krm__2__1.C_1.D__1.O.cif'), # H1 H2 H3
        ]
        if file_path.is_file() and file_path.suffix == ".cif" and file_path in exclude:
            print(file_path, end=" ")
            biotite_structure = biotite.structure.io.load_structure(
                file_path, extra_fields=["occupancy", "b_factor"]
            )
            print("atom_array")
            pdb = pose_stack_from_biotite(biotite_structure, torch_device)
            print("pose_stack")
            try:
                pdb = pose_stack_from_biotite(biotite_structure, torch_device)
                print("pose_stack")
            except:
                print("CRASH")


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
    # print(biotite_atom_array)


def test_pose_stack_from_and_to_biotite_multiple_poses(biotite_1r21, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1r21, torch_device=torch_device)
    # print(pose_stack.coords[0,0:30])
    biotite_atom_array = biotite_from_pose_stack(pose_stack)

    file = PDBFile()
    file.set_structure(biotite_atom_array)
    file.write("test_out.pdb")


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
    bt = biotite_1bl8
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)

    biotite_atom_array = biotite_from_pose_stack(pose_stack)
    file = PDBFile()
    file.set_structure(biotite_atom_array)
    # file.write("test_out.pdb")
    # print(biotite_atom_array)


def test_pose_stack_from_biotite_missing_single_sidechain(biotite_1bl8, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1bl8)
    bt = biotite_1bl8[starts[0] : starts[6]]
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)

    biotite_atom_array = biotite_from_pose_stack(pose_stack)
    file = PDBFile()
    file.set_structure(biotite_atom_array)
    # file.write("test_out.pdb")
    # print(biotite_atom_array)
