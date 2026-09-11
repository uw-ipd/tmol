"""Glycans and conjugated ligands: chemistry that crosses a residue boundary.

The bond joining a glycan to its protein is neither a backbone link nor a
disulfide, so nothing in tmol's connection machinery reaches it today. These
fixtures pin what the input actually declares, and what the pose is expected
to make of it.
"""

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, default_canonical_ordering
from tmol.io._pose_stack_from_biotite import canonical_form_from_biotite
from tmol.ligand import prepare_ligands
from tmol.chemical import ResidueTypeSet
from tmol.ligand._detect import detect_nonstandard_residues
from tmol.score.elec._params import ElecParamResolver
from tmol.tests.data import data_path

AMINO_ACIDS = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR "
    "TRP TYR VAL".split()
)

FIXTURES = {
    "nglycan_tree": "nglycan_tree_1ax2",
    "oglycan_sia": "oglycan_sia_1g1s",
    "lys_ligand": "lys_biotin_1bdo",
}

# generate with fixed seed
_CONFORMER_SEED = 20260909

# every bond the input declares across a residue boundary that is not a
#    backbone link, as (res1, atom1, res2, atom2)
EXPECTED_LINKS = {
    "nglycan_tree_1ax2": {
        ("ASN", "ND2", "NAG", "C1"),
        ("NAG", "O4", "NAG", "C1"),
        ("NAG", "O3", "FUC", "C1"),
        ("NAG", "O4", "BMA", "C1"),
        ("BMA", "O2", "XYP", "C1"),
        ("BMA", "O3", "MAN", "C1"),
        ("BMA", "O6", "MAN", "C1"),
        ("NDG", "O4", "GAL", "C1"),
    },
    "oglycan_sia_1g1s": {
        ("THR", "OG1", "NGA", "C1"),
        ("NGA", "O6", "NAG", "C1"),
        ("NGA", "O3", "GAL", "C1"),
        ("NAG", "O4", "GAL", "C1"),
        ("NAG", "O3", "FUC", "C1"),
        ("GAL", "O3", "SIA", "C2"),
    },
    "lys_biotin_1bdo": {
        ("LYS", "NZ", "BTN", "C11"),
    },
}

# attachment atoms each component carries, which decide the variants it needs
EXPECTED_ATTACHMENTS = {
    "nglycan_tree_1ax2": {
        "NAG": ["C1", "O3", "O4"],
        "BMA": ["C1", "O2", "O3", "O6"],
        "MAN": ["C1"],
        "XYP": ["C1"],
        "FUC": ["C1"],
        "NDG": ["O4"],
        "GAL": ["C1"],
    },
    "oglycan_sia_1g1s": {
        "NGA": ["C1", "O3", "O6"],
        "NAG": ["C1", "O3", "O4"],
        "GAL": ["C1", "O3"],
        "SIA": ["C2"],
        "FUC": ["C1"],
    },
    "lys_biotin_1bdo": {
        "BTN": ["C11"],
    },
}


def structure(stem):
    return atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))


# the atom pairs a polymer backbone link joins, in chain order
BACKBONE_LINKS = frozenset({("C", "N"), ("O3'", "P")})


def cross_residue_links(atom_array):
    """Declared bonds spanning two residues, minus the polymer backbone.

    Backbone links are told apart by their atoms rather than by their residue
    names, so a modified residue's peptide bonds are excluded too.
    """
    links = set()
    for i, j, _order in atom_array.bonds.as_array():
        key_i = (str(atom_array.chain_id[i]), int(atom_array.res_id[i]))
        key_j = (str(atom_array.chain_id[j]), int(atom_array.res_id[j]))
        if key_i == key_j:
            continue
        atom_i = str(atom_array.atom_name[i])
        atom_j = str(atom_array.atom_name[j])
        if (atom_i, atom_j) in BACKBONE_LINKS or (atom_j, atom_i) in BACKBONE_LINKS:
            continue
        links.add(
            (str(atom_array.res_name[i]), atom_i, str(atom_array.res_name[j]), atom_j)
        )
    return links


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_fixture_declares_its_covalent_links(fixture):
    """Guards the fixtures: the attachment bonds survive trimming."""
    stem = FIXTURES[fixture]
    assert cross_residue_links(structure(stem)) == EXPECTED_LINKS[stem]


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_attachment_atoms_are_detected(fixture):
    """Detection already finds every site; only the route taken is wrong."""
    stem = FIXTURES[fixture]
    found = {
        lig.res_name: sorted(lig.connection_atom_names or [])
        for lig in detect_nonstandard_residues(
            structure(stem), default_canonical_ordering()
        )
    }
    for name, atoms in EXPECTED_ATTACHMENTS[stem].items():
        assert found.get(name) == atoms, (name, found.get(name))


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_covalent_components_prepare(fixture):
    """Every attached component gets a residue type of its own."""
    stem = FIXTURES[fixture]
    prepared, _ordering = prepare_ligands(
        structure(stem), param_db=ParameterDatabase.get_default(), seed=_CONFORMER_SEED
    )
    built = {residue.name.split(":")[0] for residue in prepared.chemical.residues}
    for name in EXPECTED_ATTACHMENTS[stem]:
        assert name in built, name


def test_free_oligosaccharide_still_prepares():
    """1AX2's unattached NDG-GAL must keep working as a plain ligand."""
    prepared, _ordering = prepare_ligands(
        structure("nglycan_tree_1ax2"),
        param_db=ParameterDatabase.get_default(),
        seed=_CONFORMER_SEED,
    )
    built = {residue.name.split(":")[0] for residue in prepared.chemical.residues}
    assert {"NDG", "GAL"} <= built


def declared_covalent_bonds(atom_array):
    return canonical_form_from_biotite(
        atom_array, torch.device("cpu"), co=default_canonical_ordering()
    ).covalent_bonds


@pytest.mark.parametrize("stem", ["cyclic_peptide_1jbl", "1UBQ"])
def test_backbone_and_disulfide_bonds_are_not_declared(stem):
    """Only bonds no other channel carries reach the covalent table.

    1JBL supplies both exclusions that matter: a disulfide, and a backbone
    link between residues that are not adjacent by index.
    """
    atom_array = atom_array_from_cif(data_path("cif", stem + ".cif"))
    assert declared_covalent_bonds(atom_array).shape[0] == 0


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_attachment_bonds_reach_canonical_form(fixture):
    """Declared attachments survive as far as the canonical form.

    Only once their components are prepared: an unrecognized residue is
    filtered out before the bond table is read, so the ordering has to be the
    one preparation extended.
    """
    stem = FIXTURES[fixture]
    _prepared, ordering = prepare_ligands(
        structure(stem), param_db=ParameterDatabase.get_default(), seed=_CONFORMER_SEED
    )
    bonds = canonical_form_from_biotite(
        structure(stem), torch.device("cpu"), co=ordering
    ).covalent_bonds
    assert bonds.shape[0] == len(EXPECTED_LINKS[stem])


# canonical residues a component attaches to, and the atom it attaches at
CANONICAL_SITES = {
    "nglycan_tree_1ax2": {"ASN": "ND2"},
    "oglycan_sia_1g1s": {"THR": "OG1"},
    "lys_biotin_1bdo": {"LYS": "NZ"},
}


def prepared_database(stem):
    prepared, _ordering = prepare_ligands(
        structure(stem), param_db=ParameterDatabase.get_default(), seed=_CONFORMER_SEED
    )
    return prepared


def conjugated_names(chemdb):
    return {residue.name for residue in chemdb.residues if ":conj_" in residue.name}


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_every_attachment_site_becomes_a_connection(fixture):
    """Each component gains a connection at each atom it is attached through."""
    stem = FIXTURES[fixture]
    names = conjugated_names(prepared_database(stem).chemical)
    for component, atoms in EXPECTED_ATTACHMENTS[stem].items():
        for atom in atoms:
            assert any(
                name.split(":")[0] == component
                and f"conj_{atom}" in name.split(":")[1:]
                for name in names
            ), (component, atom)


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_canonical_partner_gains_its_own_connection(fixture):
    """A glycan's protein anchor is patched too; its backbone is not."""
    stem = FIXTURES[fixture]
    names = conjugated_names(prepared_database(stem).chemical)
    canonical = {n.split(":")[0] for n in names} & AMINO_ACIDS
    assert canonical == set(CANONICAL_SITES[stem])
    for residue, atom in CANONICAL_SITES[stem].items():
        assert f"{residue}:conj_{atom}" in names


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_a_component_gets_one_variant_per_combination_of_sites(fixture):
    """Sites are independent, so n of them give 2**n forms counting the base.

    The count is what makes this design affordable or not, so it is pinned:
    a change in how sites are collected shows up here before it shows up in
    a score.
    """
    stem = FIXTURES[fixture]
    chemdb = prepared_database(stem).chemical
    for component, atoms in EXPECTED_ATTACHMENTS[stem].items():
        forms = [
            name
            for name in {r.name for r in chemdb.residues}
            if name.split(":")[0] == component
        ]
        assert len(forms) == 2 ** len(atoms), (component, atoms, sorted(forms))


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_conjugation_conserves_charge(fixture):
    """An attachment takes the charge of the hydrogen it displaces.

    Checked through the resolver rather than the raw entries, since a variant
    falls back to the unpatched charge for every atom it does not name.
    """
    stem = FIXTURES[fixture]
    prepared = prepared_database(stem)
    types = ResidueTypeSet.from_database(prepared.chemical)
    resolver = ElecParamResolver.from_database(
        prepared.scoring.elec, torch.device("cpu")
    )
    by_name = {block.name: block for block in types.residue_types}
    for residue, atom in CANONICAL_SITES[stem].items():
        for variants in ([], ["nterm"], ["cterm"], ["cterm", "nterm"]):
            base = ":".join([residue, *variants])
            conjugated = ":".join([residue, *variants, f"conj_{atom}"])
            if base not in by_name or conjugated not in by_name:
                continue
            net = [
                float(resolver.get_partial_charges_for_block(by_name[n]).sum())
                for n in (base, conjugated)
            ]
            assert net[0] == pytest.approx(net[1], abs=1e-4), (base, net)


def test_retyping_leaves_the_unconjugated_residue_alone():
    """The attachment atom is retyped only in the conjugated form."""
    chemdb = prepared_database("nglycan_tree_1ax2").chemical
    by_name = {residue.name: residue for residue in chemdb.residues}

    def atom_type(name, atom):
        return next(a.atom_type for a in by_name[name].atoms if a.name == atom)

    assert atom_type("ASN", "ND2") == "NH2O"
    assert atom_type("ASN:conj_ND2", "ND2") == "Nglyc"
