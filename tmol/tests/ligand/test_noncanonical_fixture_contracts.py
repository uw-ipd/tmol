"""Contracts every noncanonical fixture must satisfy.

Checked across the whole noncanonical corpus rather than on one hand-picked
structure: a component's name does not change its chemistry. Once the atoms and
bonds are in hand, renaming a residue to something no dictionary knows must
produce the same structure -- otherwise a name is silently supplying chemistry
the file was supposed to carry.
"""

import biotite.structure as struc
import numpy as np
import pytest
import torch

from tmol.io import (
    atom_array_from_cif,
    canonical_ordering_for_biotite,
    pose_stack_from_biotite,
)
from tmol.tests.data import data_path

SEED = 20260915


def _fixtures(directory):
    return sorted(f"{directory}/{p.name}" for p in data_path(directory).glob("*.cif*"))


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


def _heavy_atoms(pose):
    """``pose.real_atoms`` without the hydrogens."""
    heavy = pose.real_atoms.clone()
    is_h = pose.packed_block_types.atom_is_hydrogen.bool()
    for p in range(pose.n_poses):
        for block, index in enumerate(pose.block_type_ind[p].tolist()):
            if index < 0:
                continue
            offset = int(pose.block_coord_offset[p, block])
            n_atoms = pose.packed_block_types.n_atoms[index]
            heavy[p, offset : offset + n_atoms] &= ~is_h[index, :n_atoms]
    return heavy


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
        return_context=True,
    )

    assert torch.isfinite(renamed.coords[renamed.real_atoms]).all()
    assert renamed.n_poses == named.n_poses
    assert int(renamed.real_atoms.sum()) == int(named.real_atoms.sum())
    # Names differ by construction; the atoms they carry must not. AtomWorks
    #    places hydrogens by dictionary geometry where it knows the component.
    assert _pose_atom_composition(renamed) == _pose_atom_composition(named)
    torch.testing.assert_close(
        renamed.coords[_heavy_atoms(renamed)], named.coords[_heavy_atoms(named)]
    )
    del named_context, renamed_context
