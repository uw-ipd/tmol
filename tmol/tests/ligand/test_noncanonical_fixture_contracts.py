"""Contracts every noncanonical fixture must satisfy.

Two properties, checked across the whole noncanonical corpus rather than on one
hand-picked structure:

a. Parameters prepared from a CIF make a *coordinate-only* PDB sufficient. A
   caller who prepares chemistry once must be able to load plain coordinates
   afterwards and get the same complete structure back, with no dictionary
   lookup.

b. A component's name does not change its chemistry. Once the atoms and bonds
   are in hand, renaming a residue to something no dictionary knows must
   produce the same structure -- otherwise a name is silently supplying
   chemistry the file was supposed to carry.
"""

import numpy as np
import pytest
import torch
from biotite.structure.io import pdb

from tmol.io import (
    atom_array_from_cif,
    canonical_ordering_for_biotite,
    pose_stack_from_biotite,
    pose_stack_from_cif,
    pose_stack_from_file,
)
from tmol.tests.data import data_path

SEED = 20260915

# Every noncanonical fixture, by directory. Adding a fixture adds it here.
FIXTURES = sorted(
    f"{directory}/{path.name}"
    for directory in ("ncaa_fixtures", "covalent_fixtures")
    for path in data_path(directory).glob("*.cif")
)


def _noncanonical_names(array):
    """Residue names the default ordering does not already describe."""
    known = set(canonical_ordering_for_biotite().restype_io_equiv_classes)
    return sorted({str(n) for n in np.unique(array.res_name)} - known)


def _residue_atom_names(array):
    """``[(res_name, (atom names...))]`` in file order."""
    import biotite.structure as struc

    return [
        (str(residue.res_name[0]), tuple(str(a) for a in residue.atom_name))
        for residue in struc.residue_iter(array)
    ]


def _pose_atom_composition(pose):
    """Atom names per block, independent of how the blocks were named."""
    types = pose.packed_block_types.active_block_types
    return [
        tuple(atom.name for atom in types[int(index)].atoms)
        for index in pose.block_type_ind[0]
        if int(index) >= 0
    ]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_prepared_parameters_make_coordinate_only_pdb_complete(
    fixture, tmp_path, torch_device
):
    """Parameters prepared from the CIF load a bare PDB without a dictionary."""
    cif = data_path(*fixture.split("/"))
    prepared, context = pose_stack_from_cif(
        cif,
        torch_device,
        prepare_ligands=True,
        ligand_seed=SEED,
        no_optH=True,
        return_context=True,
    )
    assert torch.isfinite(prepared.coords[prepared.real_atoms]).all()

    # A PDB with coordinates only: no CONECT records, no component blocks.
    source = atom_array_from_cif(cif)
    source.bonds = None
    pdb_path = tmp_path / "coordinates.pdb"
    written = pdb.PDBFile()
    written.set_structure(source)
    written.write(pdb_path)

    reloaded = pose_stack_from_file(
        pdb_path,
        torch_device,
        param_db=context.parameter_database,
        use_ccd=False,
        no_optH=True,
    )

    assert torch.isfinite(reloaded.coords[reloaded.real_atoms]).all()
    assert reloaded.n_poses == prepared.n_poses
    assert _pose_atom_composition(reloaded) == _pose_atom_composition(prepared)
    assert int(reloaded.real_atoms.sum()) == int(prepared.real_atoms.sum())


@pytest.mark.parametrize("fixture", FIXTURES)
def test_renamed_components_process_identically(fixture, torch_device):
    """A component renamed beyond any dictionary keeps its chemistry."""
    cif = data_path(*fixture.split("/"))
    array = atom_array_from_cif(cif)
    noncanonical = _noncanonical_names(array)
    if not noncanonical:
        pytest.skip(f"{fixture} carries no noncanonical component to rename")

    named, named_context = pose_stack_from_biotite(
        array,
        torch_device,
        prepare_ligands=True,
        ligand_seed=SEED,
        no_optH=True,
        return_context=True,
    )

    # Underscored codes cannot collide with a dictionary entry, so the renamed
    # run has only the file's own atoms and bonds to work from.
    renamed_array = array.copy()
    aliases = {name: f"_{index:02d}" for index, name in enumerate(noncanonical)}
    for original, alias in aliases.items():
        renamed_array.res_name[renamed_array.res_name == original] = alias

    renamed, renamed_context = pose_stack_from_biotite(
        renamed_array,
        torch_device,
        prepare_ligands=True,
        ligand_seed=SEED,
        no_optH=True,
        use_ccd=False,
        return_context=True,
    )

    assert torch.isfinite(renamed.coords[renamed.real_atoms]).all()
    assert renamed.n_poses == named.n_poses
    assert int(renamed.real_atoms.sum()) == int(named.real_atoms.sum())
    # Names differ by construction; the atoms they carry must not.
    assert _pose_atom_composition(renamed) == _pose_atom_composition(named)
    torch.testing.assert_close(
        renamed.coords[renamed.real_atoms], named.coords[named.real_atoms]
    )
    del named_context, renamed_context
