import biotite.structure
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, set_structure

import pandas as pd
import numpy

import pathlib
import torch


from tmol.io.pose_stack_from_biotite import (
    biotite_from_canonical_form,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)


def test_load_save_bulk_cif_from_biotite(torch_device):
    dir_path = pathlib.Path("/home/jflat06/pdbs/in")
    out_path = pathlib.Path("/home/jflat06/pdbs/out")
    failures_file = pathlib.Path("/home/jflat06/pdbs/out/failures.txt")
    if torch_device != torch.device("cpu"):
        return

    from tmol import beta2016_score_function

    sfxn = beta2016_score_function(torch_device)

    # print([str(a).split('.')[1] for a in sfxn.all_score_types()])

    score_data = []
    failures = []
    i = 0
    for file_path in dir_path.iterdir():
        exclude = [
            # pathlib.PosixPath('/home/jflat06/pdbs/in/7k2e__1__1.A__1.C.cif'),
            # pathlib.PosixPath(
            # "/home/jflat06/pdbs/in/4tlm__1__1.A_1.B__1.I.cif"
            # ),  # 581 missing sidechains
            # pathlib.PosixPath('/home/jflat06/pdbs/in/6h9v__1__1.A_1.B__1.C.cif'), # OXT
            # pathlib.PosixPath('/home/jflat06/pdbs/in/3n0i__1__1.A_1.B_1.C__1.D.cif'), # OXT
            # pathlib.PosixPath('/home/jflat06/pdbs/in/6c4c__2__1.E_1.F__1.X.cif'), # H
            # pathlib.PosixPath('/home/jflat06/pdbs/in/4krm__2__1.C_1.D__1.O.cif'), # H1 H2 H3
        ]
        # i+=1
        # if i > 5:
        # break
        if (
            file_path.is_file()
            and file_path.suffix == ".cif"
            and file_path not in exclude
        ):
            print(file_path, end=" ")
            biotite_structure = biotite.structure.io.load_structure(
                file_path, extra_fields=["occupancy", "b_factor"]
            )
            try:
                pose_stack = pose_stack_from_biotite(biotite_structure, torch_device)
                scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
                scores = (
                    scorer.unweighted_scores(pose_stack.coords)
                    .squeeze(-1)
                    .detach()
                    .numpy()
                )
                score_data += [numpy.concat([[file_path.name], scores])]

                print("succesful pose_stack")
                bio = biotite_from_pose_stack(pose_stack)
                file = CIFFile()
                set_structure(file, bio)
                file.write(out_path / pathlib.PosixPath(file_path.name))
            except Exception as e:
                print(e)
                print("CRASH")
                failures += [(file_path.name, e)]
                print(failures)

    df = pd.DataFrame(numpy.stack(score_data))
    df.columns = ["filename"] + [str(a).split(".")[1] for a in sfxn.all_score_types()]
    with open("scores.html", "w") as f:
        f.write(df.style.to_html())

    with open(failures_file, "w") as fail_file:
        for failure in failures:
            fail_file.write(failure[0])
            fail_file.write(str(failure[1]))


def test_canonical_form_from_biotite(biotite_1r21, torch_device):
    pdb = canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


def test_pose_stack_from_biotite_4tlm_cif(biotite_4tlm, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_4tlm, torch_device=torch_device)


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
