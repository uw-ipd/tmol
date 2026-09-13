"""Regressions from AtomWorks' modified-component structures, without metals."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch

from tmol.io import atom_array_from_cif, build_context_from_biotite
from tmol.ligand import chem_comp_types_from_cif
from tmol.ligand._registry import _applied_patch
from tmol.score.elec._params import ElecParamResolver

DATA = Path(__file__).parents[1] / "data" / "atomworks_regressions"


def test_aromatic_acyl_cap_keeps_every_heavy_atom_in_its_tree():
    path = DATA / "modified_components_6q9t.cif"
    array = atom_array_from_cif(path)
    # Retain 4SO and its directly attached A1IJ4 partner. The complete source
    # also contains zinc; this regression concerns the organic atom tree.
    indices = struc.get_residue_positions(array, np.arange(len(array)))
    selected = set(indices[array.res_name == "4SO"])
    bonds = array.bonds.as_array()
    linked = bonds[np.isin(indices[bonds[:, :2]], list(selected)).any(axis=1), :2]
    selected.update(indices[linked].flatten())
    array = array[np.isin(indices, list(selected))]
    assert set(array.res_name) == {"4SO", "A1IJ4"}
    context = build_context_from_biotite(
        array,
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_seed=20260909,
        chem_comp_types=chem_comp_types_from_cif(path),
    )
    residue = next(r for r in context.restype_set.residue_types if r.name == "4SO")
    observed = set(array.atom_name[array.res_name == "4SO"])
    assert observed <= set(residue.atom_to_idx)
    assert observed <= {ic.name for ic in residue.icoors}
    assert np.isfinite(residue.compute_ideal_coords()).all()


def test_plp_lysine_termini_do_not_borrow_nucleotide_patch_identity():
    path = DATA / "plp_enzyme_7mkv.cif"
    context = build_context_from_biotite(
        atom_array_from_cif(path),
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_seed=20260909,
        chem_comp_types=chem_comp_types_from_cif(path),
    )
    db = context.parameter_database
    base = next(r for r in db.chemical.residues if r.name == "LLP")
    terminal = next(r for r in db.chemical.residues if r.name == "LLP:cterm")
    patch = _applied_patch(db.chemical, base, terminal)
    assert patch.display_name == "cterm"
    assert patch.applies_to.matches(base)
    charges = {
        q.atom: q.charge
        for q in db.scoring.elec.atom_charge_parameters
        if q.res == "LLP:cterm"
    }
    assert np.isfinite(charges["OXT"])
    resolver = ElecParamResolver.from_database(db.scoring.elec, torch.device("cpu"))
    for residue in context.restype_set.residue_types:
        if residue.name.split(":")[0] == "LLP":
            assert np.isfinite(resolver.get_partial_charges_for_block(residue)).all()
