"""What a noncanonical residue takes from the rotamer library it borrows.

A residue with no library of its own is sampled from a canonical one, which
means more than reusing that library's chi values: it takes the library's chi
*definitions*, transferred through the correspondence the reference matcher
already found. Which bond each chi turns and which atoms it is measured
between both come from the reference, because a chi value is meaningless
without the frame it was measured in.

A nucleotide borrows the same way, from the base its own skeleton is closest
to, and its glycosidic chi is derived rather than transferred.
"""

from __future__ import annotations


import biotite.structure.info as info
import numpy
import pytest

from tmol.database import ParameterDatabase
from tmol.ligand import prepare_ligands, prepare_polymer_residue
from tmol.ligand._registry import rebuild_canonical_ordering
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("ncaa_fixtures")

# code -> (the library it should borrow, the atoms that bond to its neighbours)
#
# Diverse in what each one tests: HYP closes a ring onto the backbone, MLE
# substitutes the backbone nitrogen, B3K and BIL sit on beta backbones whose
# sidechain atom names do not line up with the library's, BIL branches at the
# first sidechain atom, and AIB has no sidechain to sample at all.
_BORROWERS: dict[str, tuple[str | None, frozenset]] = {
    "HYP": ("PRO", frozenset({"N", "C"})),
    "MLE": ("LEU", frozenset({"N", "C"})),
    "B3K": ("LYS", frozenset({"N", "C"})),
    "BIL": ("ILE", frozenset({"N", "C"})),
    "AIB": (None, frozenset({"N", "C"})),
}

# the borrowers whose own atom names match the library's, so the transferred
#    chi can be compared to it directly
_SAME_NAMES = ("HYP", "MLE")


def _chi(residue_type):
    """(name, (a, b, c, d)) for each chi, in order."""
    return [
        (t.name, (str(t.a.atom), str(t.b.atom), str(t.c.atom), str(t.d.atom)))
        for t in residue_type.torsions
        if t.name.startswith("chi")
    ]


def _leaving_group(atom_array, connection_atom):
    """The OXT/H a free component carries where a chain would continue."""
    bonds, _types = atom_array.bonds.get_all_bonds()
    names = atom_array.atom_name
    index = {n: i for i, n in enumerate(names)}
    if connection_atom not in index:
        return set()
    neighbours = [names[b] for b in bonds[index[connection_atom]] if b >= 0]
    for neighbour in neighbours:
        if not neighbour.startswith(("O", "H")):
            continue
        if neighbour == "O":
            continue
        protons = {
            names[b]
            for b in bonds[index[neighbour]]
            if b >= 0 and names[b].startswith("H")
        }
        if neighbour.startswith("O"):
            return {neighbour, *protons}
    return set()


def _prepared_component(code: str):
    """A CCD component prepared as it would sit mid-chain."""
    param_db = ParameterDatabase.get_default()
    atom_array = info.residue(code)
    atom_array.res_name[:] = code
    dropped: set[str] = set()
    for connection_atom in _BORROWERS[code][1]:
        dropped |= _leaving_group(atom_array, connection_atom)
    keep = ~numpy.isin(atom_array.atom_name, sorted(dropped))
    return prepare_polymer_residue(
        atom_array[keep],
        rebuild_canonical_ordering(param_db),
        param_db,
        connection_atoms=_BORROWERS[code][1],
    ).residue_type


@pytest.mark.parametrize("code", sorted(_BORROWERS))
def test_a_borrowed_library_brings_its_chi_definitions(code: str) -> None:
    """The reference decides which bonds are chi and how they are measured."""
    reference_name, _connections = _BORROWERS[code]
    residue_type = _prepared_component(code)
    assert residue_type.dunbrack_reference == reference_name

    chi = _chi(residue_type)
    if reference_name is None:
        # nothing to borrow from, so nothing is claimed to have been borrowed
        assert chi == []
        return

    chemdb = ParameterDatabase.get_default().chemical
    reference_chi = _chi(next(r for r in chemdb.residues if r.name == reference_name))
    assert len(chi) >= len(reference_chi)

    borrowed = chi[: len(reference_chi)]
    # chi are numbered from one, in order, with no gaps
    assert [name for name, _atoms in chi] == [f"chi{i + 1}" for i in range(len(chi))]
    # each chi walks one atom further out than the one before it: chi i+1 is
    #    measured from the three atoms chi i ended on
    for (_n, before), (_m, after) in zip(borrowed, borrowed[1:]):
        assert after[:3] == before[1:], f"{code}: {before} does not lead to {after}"
    # no bond is turned twice, which would undo whatever the first chi placed
    bonds = [frozenset(atoms[1:3]) for _name, atoms in chi]
    assert len(set(bonds)) == len(bonds), f"{code}: a bond carries two chi"

    if code in _SAME_NAMES:
        # the bond each chi turns and the atom it is measured from; the fourth
        #    atom picks between branch atoms whose labelling is not fixed
        assert [atoms[:3] for _n, atoms in borrowed] == [
            atoms[:3] for _n, atoms in reference_chi
        ]


def test_a_modified_nucleotide_borrows_the_base_it_resembles() -> None:
    """Its glycosidic chi is derived from its own chemistry, not from a name.

    Pseudouridine is joined to the sugar through C5 rather than N1, so a chi
    read off the canonical uracil's atom names would turn the wrong bond.
    """
    from tmol.io import atom_array_from_cif

    param_db = ParameterDatabase.get_default()
    structure = atom_array_from_cif(FIXTURE_DIR / "na_rna_psu_1bzt.cif")
    prepared, _canonical_ordering = prepare_ligands(structure, param_db=param_db)
    residue_type = next(r for r in prepared.chemical.residues if r.name == "PSU")

    assert residue_type.na_base_reference == "U"
    chi = _chi(residue_type)
    assert chi[0][1] == ("O4'", "C1'", "C5", "C4")
    # the sugar ring and the base are not sampled: their torsions belong to
    #    the nucleic acid term, which is calibrated on them as they are
    assert len(chi) == 2
    assert chi[1][1][1:3] == ("C2'", "O2'")


# code -> the fixture its structure comes from. RU is the canonical
# nucleotide sitting beside PSU in the same structure, and SEP an alpha
# noncanonical: both are controls for what a modified residue is compared to.
_STRUCTURES = {
    "HYP": "collagen_hyp_1bkv",
    "MLE": "nmethyl_peptide_6mvz",
    "B3K": "beta_peptide_3c3g",
    "BIL": "beta_peptide_3c3g",
    "SEP": "phosphopeptide_5ema",
    "PSU": "na_rna_psu_1bzt",
    "RU": "na_rna_psu_1bzt",
}

_NUCLEOTIDES = ("PSU", "RU")

# code -> how many rotamers one of these blocks should get.
#
# Below the library's own rotamer count, because the sampler drops the
# rotamers whose probability is under its cutoff -- 26 of lysine's 73 survive,
# 3 of isoleucine's 9. A borrowed library is read at a neutral backbone rather
# than the residue's own, so the count is a property of the library and the
# cutoff, not of the structure the residue came from. HYP is proline's two,
# both of which survive, times the three samples of its own hydroxyl chi.
_EXPECTED_ROTAMERS = {
    "HYP": 2 * 3,
    "MLE": 4,
    "B3K": 26,
    "BIL": 3,
}


def _pose_with(code: str, torch_device):
    """A pose built from the fixture, and the block index of ``code``."""
    from tmol.io import atom_array_from_cif, pose_stack_from_biotite

    param_db = ParameterDatabase.get_default()
    structure = atom_array_from_cif(FIXTURE_DIR / f"{_STRUCTURES[code]}.cif")
    poses, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        param_db=param_db,
        return_context=True,
    )
    block_types = poses.packed_block_types.active_block_types
    blocks = [
        (int(pose), int(block))
        for pose, block in zip(
            *numpy.nonzero(poses.block_type_ind64.cpu().numpy() >= 0)
        )
        if block_types[int(poses.block_type_ind64[int(pose), int(block)])].base_name
        == code
    ]
    assert blocks, f"{code} is not in the built pose"
    return poses, context, blocks[0]


def _rotamers(poses, samplers):
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import build_rotamers

    task = PackerTask(poses, PackerPalette())
    task.restrict_to_repacking()
    for sampler in samplers:
        task.add_conformer_sampler(sampler)
    poses, rotamer_set = build_rotamers(
        poses,
        SetPackerTask.from_packer_task(task),
        ParameterDatabase.get_default().chemical,
    )
    return poses, rotamer_set


@pytest.mark.parametrize("code", sorted(_STRUCTURES))
def test_the_packer_builds_rotamers_for_a_noncanonical(
    code: str, torch_device, dun_sampler
) -> None:
    """End to end: a residue no library describes still reaches the packer.

    Uses the sampler set a packing run assembles, IncludeCurrentSampler among
    them, so the input conformation is one of the rotamers offered.
    """
    from tmol.pack.rotamer import (
        FixedAAChiSampler,
        IncludeCurrentSampler,
        NaChiRotamerSampler,
    )

    poses, _context, (pose, block) = _pose_with(code, torch_device)
    samplers = [dun_sampler, FixedAAChiSampler(), IncludeCurrentSampler()]
    if code in _NUCLEOTIDES:
        samplers.append(
            NaChiRotamerSampler.from_database(
                ParameterDatabase.get_default(), torch_device
            )
        )
    _poses, rotamer_set = _rotamers(poses, samplers)

    assert rotamer_set is not None
    assert int(rotamer_set.n_rots_for_block[pose, block]) > 0


@pytest.mark.parametrize("code", sorted(_EXPECTED_ROTAMERS))
def test_a_borrowed_library_yields_its_own_number_of_rotamers(
    code: str, torch_device, dun_sampler
) -> None:
    """Borrowing a library means sampling every rotamer that library defines."""
    from tmol.pack.rotamer import FixedAAChiSampler

    poses, _context, (pose, block) = _pose_with(code, torch_device)
    _poses, rotamer_set = _rotamers(poses, [dun_sampler, FixedAAChiSampler()])

    assert int(rotamer_set.n_rots_for_block[pose, block]) == _EXPECTED_ROTAMERS[code]


def _signed_volume(coords, stereocenter, amine, forward, tip):
    """Handedness of the sidechain at the atom that carries it."""
    center = coords[stereocenter]
    return float(
        numpy.dot(
            numpy.cross(coords[amine] - center, coords[forward] - center),
            coords[tip] - center,
        )
    )


@pytest.mark.parametrize("code", sorted(_EXPECTED_ROTAMERS))
def test_a_built_rotamer_keeps_its_sidechain_on_the_same_side(
    code: str, torch_device, dun_sampler
) -> None:
    """A rotamer built from a borrowed library must not be the mirror image.

    Chirality is read as a signed volume rather than by naming atoms, so this
    holds for a backbone the canonical residues do not have.
    """
    from tmol.pack.rotamer import FixedAAChiSampler

    poses, _context, (pose, block) = _pose_with(code, torch_device)
    _poses, rotamer_set = _rotamers(poses, [dun_sampler, FixedAAChiSampler()])

    block_type = poses.packed_block_types.active_block_types[
        int(poses.block_type_ind64[pose, block])
    ]
    chi1 = next(t for t in block_type.torsions if t.name == "chi1")
    mainchain = list(block_type.properties.polymer.mainchain_atoms)
    # chi1 turns the bond from the stereocenter to the first sidechain atom
    stereocenter = str(chi1.b.atom)
    index = mainchain.index(stereocenter)
    names = (stereocenter, mainchain[index - 1], mainchain[index + 1], str(chi1.c.atom))
    where = [block_type.atom_to_idx[n] for n in names]

    start = int(poses.block_coord_offset64[pose, block])
    pose_coords = poses.coords[pose].cpu().numpy()
    expected = _signed_volume(pose_coords, *[start + i for i in where])

    offsets = rotamer_set.coord_offset_for_rot
    first = int(rotamer_set.rot_offset_for_block[pose, block])
    n_rots = int(rotamer_set.n_rots_for_block[pose, block])
    coords = rotamer_set.coords.cpu().numpy()
    for rot in range(first, first + n_rots):
        at = int(offsets[rot])
        built = _signed_volume(coords, *[at + i for i in where])
        assert built * expected > 0, f"{code}: rotamer {rot - first} is mirrored"
