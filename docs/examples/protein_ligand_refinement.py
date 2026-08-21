"""
Protein-ligand refinement
=========================

Load a protein-ligand complex through Biotite, build a ligand-aware tmol
``PoseStack``, visualize it, score it, repack protein side chains, run Cartesian
minimization, and report the score again.

The example uses checked-in ``protein_ligand_test`` fixtures. The ligand
chemistry is supplied by a prepared ``.tmol`` params file, which makes the run
deterministic and avoids regenerating ligand conformers during docs work. For a
new ligand, pass a CIF/MOL2/SMILES input through tmol's ligand preparation APIs
first, or call ``pose_stack_from_biotite(..., prepare_ligands=True)`` without
``ligand_params_files`` when OpenBabel is available.

This tutorial-scale repack-and-minimize workflow illustrates the API; it is not
a convergence-checked production refinement protocol.
"""

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io
import torch

import tmol
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.optimization import run_cart_min
from tmol.pack import pack_rotamers
from tmol.pack import PackerPalette, PackerTask
from tmol.pack.rotamer.dunbrack import (
    create_dunbrack_sampler_from_database,
)
from tmol.pack.rotamer import FixedAAChiSampler
from tmol.pack.rotamer import IncludeCurrentSampler
from tmol.score import beta2016_score_function

# sphinx_gallery_thumbnail_path = "_static/examples/protein_ligand_refinement_01.png"

LIGAND_RES_NAME = "LG1"


def load_complex(target: str):
    """Load a fixture complex as a Biotite AtomArray."""
    repo_root = Path(tmol.__file__).resolve().parents[1]
    data_dir = repo_root / "tmol" / "tests" / "data" / "protein_ligand_test"
    structure = biotite.structure.io.load_structure(
        str(data_dir / f"{target}.tmol.nomin.cif"),
        model=1,
        include_bonds=True,
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    return structure, data_dir / f"{target}.xtal-lig.mmff94.tmol"


def ligand_block_mask(pose_stack, device):
    """Select ligand blocks by residue name."""
    pbt = pose_stack.packed_block_types
    mask = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=device,
    )
    for pose_i in range(pose_stack.n_poses):
        for block_i in range(pose_stack.max_n_blocks):
            bt_ind = int(pose_stack.block_type_ind[pose_i, block_i])
            if bt_ind >= 0 and pbt.active_block_types[bt_ind].name3 == LIGAND_RES_NAME:
                mask[pose_i, block_i] = True
    if not bool(mask.any()):
        raise RuntimeError(f"{LIGAND_RES_NAME} ligand block not found")
    return mask


def total_score(pose_stack, sfxn):
    """Score a pose stack and return the first pose's weighted total."""
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    score = scorer(pose_stack.coords).detach().cpu()
    return float(score[0])


def pose_center(pose_stack):
    """Return the geometric center of the first pose for viewer labels."""
    coords = pose_stack.coords[0].detach().cpu()
    finite_coords = coords[torch.isfinite(coords).all(dim=-1)]
    center = finite_coords.mean(dim=0)
    return {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}


def view_pose_stack(pose_stack, score_label):
    """Return a py3Dmol viewer for use in notebooks, IPython, or docs."""
    import py3Dmol

    viewer = tmol.view(
        pose_stack,
        width=760,
        height=520,
        zoom_to={"resn": LIGAND_RES_NAME},
    )
    viewer.setStyle(
        {"resn": LIGAND_RES_NAME},
        {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}},
    )
    viewer.addSurface(
        py3Dmol.VDW,
        {"opacity": 0.25, "color": "white"},
        {"resn": LIGAND_RES_NAME},
    )
    viewer.addLabel(
        score_label,
        {
            "position": pose_center(pose_stack),
            "backgroundColor": "white",
            "fontColor": "black",
            "fontSize": 14,
            "inFront": True,
            "showBackground": True,
        },
    )
    viewer.zoomTo({"resn": LIGAND_RES_NAME})
    return viewer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
structure, ligand_params = load_complex("ada")

pose_stack, context = pose_stack_from_biotite(
    structure,
    device,
    prepare_ligands=True,
    ligand_params_files=[str(ligand_params)],
    param_db=ParameterDatabase.get_default(),
    return_context=True,
)

# ``pose_stack_from_biotite`` has rebuilt missing protein side chains when
# possible and, by default, has optimized protein and ligand hydrogens. The
# score function must use the ligand-extended database from the build context.
sfxn = beta2016_score_function(device, param_db=context.parameter_database)
start_score = total_score(pose_stack, sfxn)
print(f"score before refinement: {start_score:.3f}")

ligand_mask = ligand_block_mask(pose_stack, device)

task = PackerTask(pose_stack, PackerPalette())
task.restrict_to_repacking()
task.add_conformer_sampler(
    create_dunbrack_sampler_from_database(context.parameter_database, device)
)
task.add_conformer_sampler(FixedAAChiSampler())
task.add_conformer_sampler(IncludeCurrentSampler())

# Repack protein residues while keeping the ligand block fixed.
task.disable_packing_by_block_mask(ligand_mask)
packed_pose_stack = pack_rotamers(pose_stack, sfxn, task)

minimized_pose_stack = run_cart_min(packed_pose_stack, sfxn)
final_score = total_score(minimized_pose_stack, sfxn)
print(f"score after repack + minimize: {final_score:.3f}")

viewer = view_pose_stack(
    minimized_pose_stack,
    f"score before: {start_score:.3f}; after: {final_score:.3f}",
)

viewer
