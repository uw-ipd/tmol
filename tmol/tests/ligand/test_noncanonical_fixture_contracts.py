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

import biotite.structure as struc
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


def _fixtures(directory):
    return sorted(f"{directory}/{p.name}" for p in data_path(directory).glob("*.cif"))


# The two directories separate noncanonical *residues* from components joined
# by a bond that crosses a residue boundary. That distinction matters below:
# parameters describe a residue's chemistry, not which residues a particular
# structure links together.
RESIDUE_FIXTURES = _fixtures("ncaa_fixtures")
LINKED_FIXTURES = _fixtures("covalent_fixtures")
FIXTURES = sorted(RESIDUE_FIXTURES + LINKED_FIXTURES)


def _described_from_an_unplaced_copy(array, names):
    """Whether a renamed component would be described from an incomplete copy.

    A residue type is described from the copy of that residue declaring the most
    heavy atoms, preferring one that places them all. Renaming past every
    dictionary leaves nothing else to resolve the rest from, so preparation
    refuses exactly when every copy at that fullest declaration leaves an atom
    unplaced -- a 5' nucleotide declaring the phosphate its terminus does not
    carry, say. A residue whose unresolved copy declares no more than a complete
    sibling, like the partly resolved TYS side chain in oglycan_sia_1g1s, is
    described from that sibling and prepares normally.

    This models the count and the finiteness test, not the further requirement
    that the chosen copy carry every connection atom. Checked against all of the
    fixtures above by renaming each and running preparation; a fixture where the
    fullest copy is missing a connection atom name would need that clause too.
    """
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    for name in names:
        copies = [
            array[start:stop]
            for start, stop in zip(starts[:-1], starts[1:])
            if str(array.res_name[start]) == name
        ]
        declared = [int((~np.isin(copy.element, ("H", "D"))).sum()) for copy in copies]
        if copies and not any(
            np.isfinite(copy.coord).all()
            for copy, count in zip(copies, declared)
            if count == max(declared)
        ):
            return True
    return False


def _unresolved_atom_names(array):
    """Atoms the file declares but never places.

    ``atom_array_from_cif`` leaves a declared heavy atom at NaN when the
    structure resolves no position for it. A PDB record *is* a position, so
    these are exactly the atoms no coordinate-only file can carry.
    """
    absent = ~np.isfinite(array.coord).all(axis=-1)
    return sorted({str(name) for name in array.atom_name[absent]})


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


@pytest.mark.parametrize("fixture", RESIDUE_FIXTURES)
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
    unresolved = _unresolved_atom_names(source)
    source.bonds = None
    pdb_path = tmp_path / "coordinates.pdb"
    written = pdb.PDBFile()
    if unresolved:
        # This file declares atoms it never places, and a PDB record is a
        # position -- there is no way to write them. Replaying it from
        # coordinates alone is not something the format can express, and
        # placing them would assert geometry nothing observed. Assert that
        # boundary rather than a round trip that cannot happen.
        with pytest.raises(struc.BadStructureError):
            written.set_structure(source)
            written.write(pdb_path)
        return
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


@pytest.mark.parametrize("fixture", LINKED_FIXTURES)
def test_coordinate_only_pdb_cannot_carry_cross_residue_links(
    fixture, tmp_path, torch_device
):
    """Parameters carry residue chemistry; they do not carry a structure's links.

    An attachment that is neither a polymer backbone bond nor a disulfide lives
    in the CIF's connection records. A coordinate-only PDB has nowhere to put
    it, so the components come back in their free forms -- the attachment site
    regains the hydrogens the bond displaced. That is the correct outcome:
    inventing the link from proximity would be a guess. Callers who need it
    must declare the connectivity.
    """
    cif = data_path(*fixture.split("/"))
    prepared, context = pose_stack_from_cif(
        cif,
        torch_device,
        prepare_ligands=True,
        ligand_seed=SEED,
        no_optH=True,
        return_context=True,
    )

    source = atom_array_from_cif(cif)
    unresolved = _unresolved_atom_names(source)
    source.bonds = None
    pdb_path = tmp_path / "coordinates.pdb"
    written = pdb.PDBFile()
    if unresolved:
        # Declared-but-unplaced atoms have no PDB representation; see the
        # residue-fixture contract above.
        with pytest.raises(struc.BadStructureError):
            written.set_structure(source)
            written.write(pdb_path)
        return
    written.set_structure(source)
    written.write(pdb_path)

    reloaded = pose_stack_from_file(
        pdb_path,
        torch_device,
        param_db=context.parameter_database,
        use_ccd=False,
        no_optH=True,
    )

    # The chemistry still loads with no dictionary, which is the point of
    # injecting parameters; only the structure's own connectivity is missing.
    assert torch.isfinite(reloaded.coords[reloaded.real_atoms]).all()
    assert int(reloaded.real_atoms.sum()) > int(prepared.real_atoms.sum())
    assert _pose_atom_composition(reloaded) != _pose_atom_composition(prepared)


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

    if _described_from_an_unplaced_copy(array, aliases):
        from tmol.ligand._preparation import LigandPreparationError

        with pytest.raises(LigandPreparationError, match="declared but unresolved"):
            pose_stack_from_biotite(
                renamed_array,
                torch_device,
                prepare_ligands=True,
                ligand_seed=SEED,
                no_optH=True,
                use_ccd=False,
                return_context=True,
            )
        del named_context
        return

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
