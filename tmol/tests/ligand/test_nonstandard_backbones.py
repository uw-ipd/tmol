"""Classification of polymer residues whose backbone is not an alpha amino acid.

A residue reaches the polymer path because the file places it in a chain,
which says only that it links into one -- not what its backbone is. These
fixtures cover the backbone classes that arrive there: the alpha backbone
modified at its amide nitrogen, backbones a carbon or two longer, and one that
is a peptide only by courtesy.

Only an unmodified alpha backbone may be typed and scored as protein. Two
conditions define it, both structural:

* the mainchain is a bonded ``N-CA-C(=O)`` path, and
* every heavy substituent on the amide nitrogen closes a ring back onto that
  mainchain.

Proline and hydroxyproline pass on the second clause, which is why they keep
peptide typing; an N-methyl or a peptoid sidechain does not. Everything that
fails either clause is prepared as a ligand: ligand atom types throughout, and
a mainchain read from the bonds between its polymer connections.

The assertions are sanity checks rather than goldens: what is pinned is which
class a residue lands in, and that the class decides the typing.
"""

from __future__ import annotations

import logging

import numpy
import pytest

import biotite.structure.info as info

from tmol.database import ParameterDatabase
from tmol.ligand import LigandPreparationError, prepare_polymer_residue
from tmol.ligand._polymer_profile import alpha_profile, profile_for_atom_array
from tmol.ligand._preparation import _ideal_coords_by_name
from tmol.ligand._registry import rebuild_canonical_ordering

# code -> (what the backbone is, whether it is an unmodified alpha backbone,
#          the atoms that bond to the neighbouring residues)
#
# The connection atoms are what a structure supplies and a bare component does
# not. Gamma-glutamate is the case that needs them: it carries an intact alpha
# fragment, and only the fact that the chain runs through CD says the residue
# is not an alpha amino acid.
_BACKBONE_CLASSES: dict[str, tuple[str, bool, frozenset]] = {
    "MLE": ("N-methylated alpha (N-methylleucine)", False, frozenset({"N", "C"})),
    "SAR": (
        "peptoid, N-substituted glycine (sarcosine)",
        False,
        frozenset({"N", "C"}),
    ),
    "B3K": ("beta amino acid (beta-lysine)", False, frozenset({"N", "C"})),
    "FGA": (
        "gamma-linked peptide (gamma-D-glutamate)",
        False,
        frozenset({"N", "CD"}),
    ),
    "HAO": (
        "peptide-like, no amino-acid backbone at all",
        False,
        frozenset({"N", "C"}),
    ),
    "AIB": (
        "alpha, disubstituted at CA (aminoisobutyrate)",
        True,
        frozenset({"N", "C"}),
    ),
    "HYP": (
        "alpha, ring closed onto N (hydroxyproline)",
        True,
        frozenset({"N", "C"}),
    ),
}

_NO_GENERIC_PROFILE = (
    "a backbone that is not alpha is classified but has no profile to build from yet"
)


def _cases(failing=None, reason=_NO_GENERIC_PROFILE):
    """Every class, with the ones that do not yet behave marked as expected to fail.

    ``failing`` defaults to every backbone that is not an unmodified alpha.
    """
    if failing is None:
        failing = [c for c, (_l, alpha, _c) in _BACKBONE_CLASSES.items() if not alpha]
    return [
        pytest.param(
            code,
            marks=(
                [pytest.mark.xfail(strict=True, reason=reason)]
                if code in failing
                else []
            ),
        )
        for code in sorted(_BACKBONE_CLASSES)
    ]


def _adjacency(atom_array):
    names = {i: str(n) for i, n in enumerate(atom_array.atom_name)}
    adj: dict[str, set[str]] = {}
    for i, j, _order in atom_array.bonds.as_array():
        adj.setdefault(names[i], set()).add(names[j])
        adj.setdefault(names[j], set()).add(names[i])
    return adj


def _leaving_group(atom_array, connection_atom: str) -> set[str]:
    """Atoms the CCD carries that a residue in a chain gives up at this atom.

    A component is defined as the free molecule, so each connection point still
    holds what the peptide bond would displace: a proton on the amine, a
    hydroxyl on the acid. Which one depends on the atom the chain leaves from,
    and gamma-glutamate makes that concrete -- it links through CD, so its
    alpha carboxyl keeps its OXT.
    """
    adj = _adjacency(atom_array)
    element = {str(n): str(e) for n, e in zip(atom_array.atom_name, atom_array.element)}
    neighbours = sorted(adj.get(connection_atom, ()))
    if element.get(connection_atom) == "N":
        protons = [n for n in neighbours if element.get(n) == "H"]
        return {protons[-1]} if protons else set()
    for neighbour in neighbours:
        if element.get(neighbour) != "O":
            continue
        protons = [n for n in adj.get(neighbour, ()) if element.get(n) == "H"]
        if protons:
            return {neighbour, *protons}
    return set()


def _residue(code: str):
    """A CCD component as it appears mid-chain, with its bond table."""
    atom_array = info.residue(code)
    atom_array.res_name[:] = code
    dropped: set[str] = set()
    for connection_atom in _BACKBONE_CLASSES[code][2]:
        dropped |= _leaving_group(atom_array, connection_atom)
    keep = ~numpy.isin(atom_array.atom_name, sorted(dropped))
    return atom_array[keep]


def _connection_atoms(code: str) -> frozenset:
    return _BACKBONE_CLASSES[code][2]


def _prepare(code: str):
    param_db = ParameterDatabase.get_default()
    return prepare_polymer_residue(
        _residue(code),
        rebuild_canonical_ordering(param_db),
        param_db,
        connection_atoms=_connection_atoms(code),
    )


def test_every_class_is_reachable() -> None:
    """The fixtures come from the CCD, so they must actually be defined there."""
    for code in _BACKBONE_CLASSES:
        residue = _residue(code)
        assert residue.array_length() > 0, code
        assert residue.bonds is not None and residue.bonds.get_bond_count() > 0, code


@pytest.mark.parametrize("code", _cases(failing=()))
def test_only_an_unmodified_alpha_backbone_matches_the_alpha_profile(code: str) -> None:
    """A backbone is classified by its bonds, not by the names it happens to use.

    Every one of these carries atoms called N, CA and C, so a name-based test
    cannot separate a beta amino acid, or an aromatic hydrazide, from alanine.
    """
    expected_alpha = _BACKBONE_CLASSES[code][1]
    matched = profile_for_atom_array(_residue(code), _connection_atoms(code))
    alpha = alpha_profile(ParameterDatabase.get_default().chemical)
    assert (matched is alpha) == expected_alpha


# types that describe a protein backbone; every other atom of every residue,
# and every atom of a residue that is not an unmodified alpha amino acid,
# carries a ligand type instead
_PEPTIDE_BACKBONE_TYPES = frozenset({"Nbb", "Npro", "CAbb", "CObb", "OCbb", "HNbb"})


@pytest.mark.parametrize("code", _cases(failing=()))
def test_typing_matches_the_backbone_class(code: str) -> None:
    """Peptide types belong to an unmodified alpha backbone and nowhere else.

    A modified backbone typed as protein reads as a peptide bond it is not:
    Nbb donates as hbdon_PBA and OCbb accepts as hbacc_PBA, so a peptoid or a
    beta amino acid would hydrogen bond with a canonical backbone's parameters.
    """
    types = {a.name: a.atom_type for a in _prepare(code).residue_type.atoms}
    peptide_typed = {
        name: atom_type
        for name, atom_type in types.items()
        if atom_type in _PEPTIDE_BACKBONE_TYPES
    }
    if _BACKBONE_CLASSES[code][1]:
        assert set(peptide_typed) >= {"N", "CA", "C", "O"}, peptide_typed
    else:
        assert peptide_typed == {}, f"{code} typed as protein backbone: {peptide_typed}"


@pytest.mark.parametrize("code", _cases(failing=()))
def test_a_declared_mainchain_is_a_bonded_path(code: str) -> None:
    """Consecutive mainchain atoms must be bonded to each other.

    A beta amino acid declares ('N', 'CA', 'C') while its backbone runs
    N-CA-CB-C, so phi, psi and omega are measured across a bond that is not
    there and the real backbone is picked up as sidechain chi.
    """
    restype = _prepare(code).residue_type
    bonded = {frozenset((b[0], b[1])) for b in restype.bonds}
    mainchain = restype.properties.polymer.mainchain_atoms
    assert mainchain
    for first, second in zip(mainchain, mainchain[1:]):
        assert frozenset((first, second)) in bonded, f"{code}: {first}-{second}"


@pytest.mark.parametrize(
    "code", sorted(c for c, v in _BACKBONE_CLASSES.items() if v[1])
)
def test_an_alpha_backbone_prepares_to_something_usable(code: str) -> None:
    """Smoke test on the two that should keep peptide typing."""
    prep = _prepare(code)
    restype = prep.residue_type

    assert {c.name for c in restype.connections} == {"down", "up"}
    types = {a.name: a.atom_type for a in restype.atoms}
    assert types["CA"] == "CAbb"
    assert types["C"] == "CObb"
    assert types["O"] == "OCbb"
    assert types["N"] in ("Nbb", "Npro")

    assert {"phi", "psi", "omega"} <= {t.name for t in restype.torsions}
    assert numpy.isfinite(sum(prep.partial_charges.values()))


def test_the_amide_nitrogen_rule_separates_proline_from_n_methyl() -> None:
    """The clause that keeps HYP as alpha and sends N-methyl to the ligand path.

    A substituent on the amide nitrogen is allowed only where it closes a ring
    back onto the mainchain, as proline's CD does.
    """

    def exocyclic_n_substituents(code):
        atom_array = _residue(code)
        adj = _adjacency(atom_array)
        element = {
            str(n): str(e) for n, e in zip(atom_array.atom_name, atom_array.element)
        }
        out = []
        for substituent in adj.get("N", ()):
            if element.get(substituent) == "H" or substituent == "CA":
                continue
            seen, stack, reaches = {"N"}, [substituent], False
            while stack:
                current = stack.pop()
                if current == "CA":
                    reaches = True
                    break
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(adj.get(current, ()))
            if not reaches:
                out.append(substituent)
        return out

    assert exocyclic_n_substituents("HYP") == []
    assert exocyclic_n_substituents("AIB") == []
    assert exocyclic_n_substituents("MLE") == ["CN"]
    assert exocyclic_n_substituents("SAR") == ["CN"]


# --------------------------------------------------------------------------- #
# backbones spelled with names other than N/CA/C/O
# --------------------------------------------------------------------------- #

# the backbone atoms cartbonded's wildcard rows name; those rows carry the terms
# that span the peptide bond and are looked up by name
_RESERVED_BACKBONE_NAMES = ("N", "CA", "C", "O")

_ALPHA_CONNECTIONS = frozenset({"N", "C"})


def _alpha_residue_spelled(renames: dict, res_name: str = "ALX"):
    """Alanine with some of its atoms renamed, as a residue mid-chain."""
    atom_array = info.residue("ALA")
    atom_array = atom_array[~numpy.isin(atom_array.atom_name, ["OXT", "HXT", "H2"])]
    atom_array.res_name[:] = res_name
    for i, name in enumerate(atom_array.atom_name):
        if str(name) in renames:
            atom_array.atom_name[i] = renames[str(name)]
    return atom_array


def _prepare_alpha(atom_array, renames: dict | None = None):
    """Prepare it, with the connections named the way the input spells them."""
    param_db = ParameterDatabase.get_default()
    renames = renames or {}
    connections = frozenset(renames.get(a, a) for a in _ALPHA_CONNECTIONS)
    return prepare_polymer_residue(
        atom_array,
        rebuild_canonical_ordering(param_db),
        param_db,
        connection_atoms=connections,
    )


def test_a_backbone_named_otherwise_is_still_recognized() -> None:
    """The backbone is found by its bonds, so its names need not be canonical."""
    residue = _alpha_residue_spelled({"CA": "CX"})
    assert "CA" not in set(residue.atom_name)
    assert profile_for_atom_array(residue, _ALPHA_CONNECTIONS) is alpha_profile(
        ParameterDatabase.get_default().chemical
    )


@pytest.mark.parametrize(
    "renames",
    [{"CA": "CX"}, {"O": "OQ"}, {"CA": "CX", "O": "OQ"}, {"N": "N1", "C": "C9"}],
)
def test_backbone_takes_canonical_names_and_keeps_the_input_as_an_alias(
    renames: dict,
) -> None:
    """Renamed onto the reserved names, with the input spelling recorded."""
    restype = _prepare_alpha(_alpha_residue_spelled(renames), renames).residue_type

    names = {str(a.name) for a in restype.atoms}
    assert set(_RESERVED_BACKBONE_NAMES) <= names
    assert not names & set(renames.values())

    aliases = {str(a.alt_name): str(a.name) for a in restype.atom_aliases}
    assert aliases == {spelled: canonical for canonical, spelled in renames.items()}


def test_a_canonically_named_backbone_is_not_renamed() -> None:
    """Nothing is renamed when nothing needs to be, so no alias is invented."""
    restype = _prepare_alpha(_alpha_residue_spelled({})).residue_type
    assert restype.atom_aliases == ()


def test_the_input_spelling_resolves_to_the_canonical_atom() -> None:
    """The point of the alias: coordinates land on the renamed atom.

    Resolution maps an input atom name to a canonical-ordering index, so a
    structure that spells the backbone its own way would otherwise drop those
    atoms and leave the residue with holes it cannot rebuild.
    """
    from tmol.ligand._registry import inject_ligand_preparations

    param_db = ParameterDatabase.get_default()
    prep = _prepare_alpha(_alpha_residue_spelled({"CA": "CX"}))
    canonical_ordering = rebuild_canonical_ordering(
        inject_ligand_preparations(param_db, [prep])
    )

    mapping = canonical_ordering.restypes_atom_index_mapping["ALX"]
    assert mapping["CX"] == mapping["CA"]
    # only the real name is a slot; the alias is a second way to reach it
    ordered = [str(n) for n in canonical_ordering.restypes_ordered_atom_names["ALX"]]
    assert "CA" in ordered and "CX" not in ordered


def test_a_reserved_name_used_for_another_atom_is_refused() -> None:
    """A residue cannot keep CA for something that is not the backbone CA."""
    # the alpha carbon is spelled CX, and the sidechain has taken the name CA
    residue = _alpha_residue_spelled({"CA": "CX", "CB": "CA"})
    with pytest.raises(ValueError, match="reserved backbone name"):
        _prepare_alpha(residue)


# --------------------------------------------------------------------------- #
# the angle at a ring nitrogen
# --------------------------------------------------------------------------- #


def _hyp_structure(renames: dict):
    """The collagen fixture, with hydroxyproline's atoms optionally renamed."""

    from tmol.tests.data import data_path

    from tmol.io import atom_array_from_cif

    atom_array = atom_array_from_cif(
        data_path("ncaa_fixtures") / "collagen_hyp_1bkv.cif"
    )
    for i, name in enumerate(atom_array.atom_name):
        if str(atom_array.res_name[i]) == "HYP" and str(name) in renames:
            atom_array.atom_name[i] = renames[str(name)]
    return atom_array


def _hyp_residue(renames: dict):
    atom_array = _hyp_structure(renames)
    first = numpy.nonzero(atom_array.res_name == "HYP")[0][0]
    mask = (atom_array.res_name == "HYP") & (
        atom_array.res_id == atom_array.res_id[first]
    )
    return atom_array[mask]


def _cross_residue_angles(prep):
    return {
        (str(p.atm1), str(p.atm2), str(p.atm3)): (p.x0, p.K)
        for p in prep.cartbonded_params.angle_parameters
        if str(p.atm3).startswith("+") or str(p.atm1).startswith("+")
    }


def test_a_ring_atom_called_cd_needs_no_injected_angle() -> None:
    """The wildcard row already matches, so a second copy would be redundant."""
    prep = _prepare_alpha(_hyp_residue({}))
    assert _cross_residue_angles(prep) == {}


def test_a_ring_atom_called_otherwise_carries_its_own_angle() -> None:
    """Cartbonded finds the row by name, so the residue supplies one that matches.

    The value is proline's rather than measured: the ideal-coordinate placement
    of the down connection puts this angle at 91 degrees even for proline
    itself, because the icoor that places it is shared with every alpha residue.
    """
    prep = _prepare_alpha(_hyp_residue({"CD": "CX"}))
    assert _cross_residue_angles(prep) == {("CX", "N", "+C"): (1.9548, 125.184)}


# conformer generation is stochastic and the bonded parameters are measured off
# the geometry it produces, so two preparations of one input do not agree. A
# fixed seed is what makes two preparations comparable at all
_CONFORMER_SEED = 20250828


def test_a_ring_atom_named_otherwise_scores_the_same(torch_device) -> None:
    """Hydroxyproline scores alike whether its ring atom is called CD or not.

    Called CD, the angle at the ring nitrogen comes from cartbonded's wildcard
    row; called anything else, from the copy the residue carries. The two paths
    have to arrive at the same number.
    """
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand._registry import inject_ligand_preparations
    from tmol.score import beta2016_score_function

    param_db = ParameterDatabase.get_default()

    def total(renames):
        prep = prepare_polymer_residue(
            _hyp_residue(renames),
            rebuild_canonical_ordering(param_db),
            param_db,
            connection_atoms=_ALPHA_CONNECTIONS,
            seed=_CONFORMER_SEED,
        )
        injected = inject_ligand_preparations(param_db, [prep])
        pose_stack = pose_stack_from_biotite(
            _hyp_structure(renames), torch_device, param_db=injected
        )
        sfxn = beta2016_score_function(torch_device, param_db=injected)
        module = sfxn.render_whole_pose_scoring_module(pose_stack)
        return float(module(pose_stack.coords).sum())

    assert total({"CD": "CX"}) == pytest.approx(total({}), abs=1e-3)


def test_an_injected_residue_gets_charges_for_its_termini() -> None:
    """A terminus patch adds atoms the base residue has no charge for.

    Elec looks charges up as ``[residue][atom][variant]`` and falls back to the
    unpatched value, but OXT exists only in the cterm variant, so without an
    entry the lookup has nothing to fall back to and scoring fails outright.
    The values are backbone chemistry, borrowed from proline where the nitrogen
    is already substituted and from alanine where it is not.
    """
    from tmol.ligand._registry import inject_ligand_preparations
    from tmol.tests.ligand.test_noncanonical_residues import _load, _residue

    param_db = ParameterDatabase.get_default()

    def charges_for(residue):
        prep = _prepare_alpha(residue)
        injected = inject_ligand_preparations(param_db, [prep])
        by_res: dict = {}
        for entry in injected.scoring.elec.atom_charge_parameters:
            by_res.setdefault(str(entry.res), {})[str(entry.atom)] = entry.charge
        return prep.residue_type.name, by_res

    name, by_res = charges_for(_hyp_residue({}))
    # hydroxyproline's nitrogen is already in a ring: two ammonium hydrogens
    assert set(by_res[f"{name}:nterm"]) >= {"N", "H2", "H3"}
    assert "H1" not in by_res[f"{name}:nterm"]
    assert by_res[f"{name}:cterm"] == pytest.approx(
        {"C": 0.35448, "O": -0.67724, "OXT": -0.67724}
    )

    name, by_res = charges_for(_residue(_load("phosphopeptide_5ema"), "SEP"))
    # phosphoserine's nitrogen carries a hydrogen: three
    assert set(by_res[f"{name}:nterm"]) >= {"N", "H1", "H2", "H3"}

    # the reference's own sidechain is not copied onto the borrowing residue
    assert not {"CB", "CG"} & set(by_res[f"{name}:nterm"])


# --------------------------------------------------------------------------- #
# residues seen only at a chain terminus
# --------------------------------------------------------------------------- #

# code -> (the atom it is bonded through, the mainchain that should be found)
# code -> (the connection the structure shows, the mainchain inferred from it).
# FGA links through CD and ACB through CG, but at a terminus nothing says so:
# both carry an intact alpha backbone, which is what an unambiguous chemistry
# test finds and what a residue linked conventionally almost always uses.
_TERMINAL_CASES = {
    "SAR": ("C", ("N", "CA", "C")),
    "FGA": ("N", ("N", "CA", "C")),
    "ACB": ("N", ("N", "CA", "C")),
}


def _residue_bonded_only_at(code: str, connection_atom: str):
    """A component as it appears at a chain end: one connection open, one intact."""
    atom_array = info.residue(code)
    atom_array.res_name[:] = code
    dropped = _leaving_group(atom_array, connection_atom)
    return atom_array[~numpy.isin(atom_array.atom_name, sorted(dropped))]


@pytest.mark.parametrize("code", sorted(_TERMINAL_CASES))
def test_a_terminal_residue_finds_its_other_end(code: str) -> None:
    """One connection does not say where the backbone stops; chemistry does.

    A single candidate for the other end settles it. Where more than one
    qualifies the conventional backbone wins, which is why gamma-glutamate and
    beta-aspartate resolve to their alpha backbones here: seen only at a
    terminus, nothing distinguishes the sidechain carbon they really link
    through. A copy of either in a chain is read correctly, from its bonds.
    """
    connection_atom, expected = _TERMINAL_CASES[code]
    residue = _residue_bonded_only_at(code, connection_atom)

    profile = profile_for_atom_array(residue, frozenset({connection_atom}))
    assert profile is not None, f"{code} was refused"
    assert profile.mainchain_atoms == expected


@pytest.mark.parametrize("code", ["GLU", "ASP", "ASN", "GLN"])
def test_a_free_sidechain_acid_does_not_outrank_the_backbone(code: str) -> None:
    """The chain continues through the backbone, not through a sidechain acid.

    A residue at a C-terminus whose terminal hydroxyl was never modeled carries
    one oxygen on its backbone carbonyl, where a free sidechain carboxylate
    carries two -- so counting oxygens picks the sidechain and silently reads
    the backbone as a gamma or beta one. What separates them is the nitrogen an
    amide sidechain carbon carries and a backbone carbonyl does not.
    """
    residue = info.residue(code)
    residue.res_name[:] = code
    residue = residue[~numpy.isin(residue.atom_name, ["OXT", "HXT"])]

    profile = profile_for_atom_array(residue, frozenset({"N"}))
    assert profile is not None, f"{code} was refused"
    assert profile.mainchain_atoms == ("N", "CA", "C")


def test_a_terminal_residue_is_completed_under_an_unknown_name() -> None:
    """The other end comes from chemistry, so the residue code cannot matter."""
    from tmol.ligand._polymer_profile import completed_connection_atoms

    residue = _residue_bonded_only_at("SAR", "C")
    # a name no dictionary defines, so nothing is declared about it
    residue.res_name[:] = "X_"
    assert completed_connection_atoms(residue, frozenset({"C"})) == frozenset(
        {"N", "C"}
    )


def test_an_undeterminable_terminal_residue_is_refused() -> None:
    """Candidate ends with no conventional backbone among them are refused."""
    param_db = ParameterDatabase.get_default()
    # beta-lysine carries a sidechain amine as well as its backbone one, and
    #    neither completes an alpha backbone, so nothing breaks the tie
    residue = _residue_bonded_only_at("B3K", "C")
    residue.res_name[:] = "X_"

    assert profile_for_atom_array(residue, frozenset({"C"})) is None
    with pytest.raises(LigandPreparationError, match="where the backbone's other"):
        prepare_polymer_residue(
            residue,
            rebuild_canonical_ordering(param_db),
            param_db,
            connection_atoms=frozenset({"C"}),
        )


def test_declaring_both_ends_prepares_what_could_not_be_inferred() -> None:
    """The way out of the refusal above: say where the backbone ends.

    Naming both connections skips inference entirely, which is what the error
    message tells the caller to do before saving the result as a params file.
    """
    param_db = ParameterDatabase.get_default()
    residue = _residue_bonded_only_at("B3K", "C")
    residue.res_name[:] = "X_"

    prep = prepare_polymer_residue(
        residue,
        rebuild_canonical_ordering(param_db),
        param_db,
        connection_atoms=frozenset({"N", "C"}),
    )
    assert prep.residue_type.properties.polymer.mainchain_atoms == (
        "N",
        "CA",
        "CB",
        "C",
    )


# --------------------------------------------------------------------------- #
# whole structures, through the whole pipeline
# --------------------------------------------------------------------------- #

# The tests above hand preparation the connection atoms, so they say what a
# residue becomes given the right answer. These start from a deposited entry
# and say nothing: detection reads the connections off the bonds, routing sends
# the residue down the polymer path, and the result has to reach a score.
#
# stem -> the nonstandard codes it carries, and how many atoms of each residue
#         its backbone runs through
_PIPELINE_FIXTURES: dict[str, dict[str, int]] = {
    # cyclic-peptide analogue, N-methylleucine at two positions
    "nmethyl_peptide_6mvz": {"MLE": 3},
    # peptidoglycan stem peptide; the chain leaves gamma-glutamate through CD,
    #    four bonds from the alpha carbon it would leave from as an alpha residue
    "gamma_peptide_1gac": {"FGA": 5},
    # a beta-peptide foldamer: eight distinct beta backbones in one chain
    "beta_peptide_3c3g": {
        "B3D": 4,
        "B3E": 4,
        "B3K": 4,
        "B3L": 4,
        "B3Q": 4,
        "BAL": 4,
        "BIL": 4,
        "HMR": 4,
    },
}


def _structure(stem: str):
    """A fixture as tmol reads it: bonded, and complete where the density was not."""
    from tmol.io import atom_array_from_cif
    from tmol.tests.data import data_path

    return atom_array_from_cif(data_path("ncaa_fixtures") / f"{stem}.cif")


# --------------------------------------------------------------------------- #
# where a residue's complete chemistry comes from
# --------------------------------------------------------------------------- #
# 3C3G resolves no sidechain for B3K, so a residue type built from its atoms
# alone would be a different, shorter molecule -- the pipeline protonates the
# truncation rather than leaving it open. What the residue actually is has to
# come from somewhere else, and there are three sources in decreasing
# authority: the file's own chem_comp_atom/chem_comp_bond, the component
# dictionary, and nothing at all.
_TRUNCATED = ("B3K", 25, 4)


def _prepared_beta_peptide(structure):
    """``{name: (n_atoms, n_chi)}`` for the residue types 3C3G adds."""
    from tmol.ligand import prepare_ligands
    from tmol.chemical import ResidueTypeSet

    param_db = ParameterDatabase.get_default()
    known = {r.name for r in param_db.chemical.residues}
    prepared, _ordering = prepare_ligands(structure, param_db=param_db, seed=1234)
    restypes = ResidueTypeSet.from_database(prepared.chemical)
    return {
        rt.name: (
            len(rt.atoms),
            len([t for t in rt.torsion_to_uaids if t.startswith("chi")]),
        )
        for rt in restypes.residue_types
        if rt.name not in known and ":" not in rt.name
    }


def test_declared_chemistry_completes_an_unresolved_sidechain() -> None:
    """The file declares the whole component, whatever the density resolved."""
    from tmol.io import component_chemistry_from_cif
    from tmol.tests.data import data_path

    code, n_atoms, n_chi = _TRUNCATED
    path = data_path("ncaa_fixtures") / "beta_peptide_3c3g.cif"
    declared = component_chemistry_from_cif(path)
    assert code in declared, "the fixture must carry its chem_comp_atom block"

    assert _prepared_beta_peptide(_structure("beta_peptide_3c3g"))[code] == (
        n_atoms,
        n_chi,
    )


def test_the_component_dictionary_completes_what_the_file_does_not_declare(
    tmp_path,
) -> None:
    """A file with no chem_comp_atom reaches the same residue via the dictionary."""
    import biotite.structure.io.pdbx as pdbx

    from tmol.io import atom_array_from_cif
    from tmol.tests.data import data_path

    code, n_atoms, n_chi = _TRUNCATED
    cif = pdbx.CIFFile.read(str(data_path("ncaa_fixtures") / "beta_peptide_3c3g.cif"))
    block = cif[next(iter(cif.keys()))]
    for category in ("chem_comp", "chem_comp_atom", "chem_comp_bond"):
        del block[category]
    stripped = tmp_path / "undeclared.cif"
    cif.write(str(stripped))

    assert _prepared_beta_peptide(atom_array_from_cif(stripped))[code] == (
        n_atoms,
        n_chi,
    )


def test_a_dictionary_entry_for_a_different_molecule_is_not_used(caplog) -> None:
    """A code invented for a residue usually names something else entirely.

    The dictionary defines tens of thousands of components, so completing by
    code alone would take atoms from an unrelated molecule -- X3K is a real
    entry, and not this residue. Refusing it as a description leaves nothing to
    say whether what was resolved is the whole residue, so the residue is taken
    as it stands and the doubt is reported.
    """
    from tmol.io._cif import with_unresolved_atoms

    structure = _structure("beta_peptide_3c3g")
    structure.res_name[structure.res_name == _TRUNCATED[0]] = "X3K"

    with caplog.at_level(logging.WARNING, logger="tmol.io._cif"):
        taken_as_is = with_unresolved_atoms(structure, {}, use_ccd=True)

    assert taken_as_is.array_length() == structure.array_length()
    assert any("does not account for" in r.message for r in caplog.records)


def test_a_residue_nothing_describes_is_assumed_complete(caplog) -> None:
    """With no declaration and no dictionary entry, there is nothing to check.

    The residue is taken as it was resolved, which is right whenever it is
    complete and wrong silently otherwise, so it says so.
    """
    from tmol.io._cif import with_unresolved_atoms
    from tmol.ligand import prepare_ligands

    structure = _structure("nmethyl_peptide_6mvz")
    # a code the component dictionary does not define
    structure.res_name[structure.res_name == "MLE"] = "QXJ"

    with caplog.at_level(logging.WARNING, logger="tmol.io._cif"):
        structure = with_unresolved_atoms(structure, {}, use_ccd=True)
    assert any("taken to be complete as resolved" in r.message for r in caplog.records)

    param_db = ParameterDatabase.get_default()
    known = {r.name for r in param_db.chemical.residues}
    prepared, _ordering = prepare_ligands(structure, param_db=param_db, seed=1234)
    assert "QXJ" in {r.name for r in prepared.chemical.residues} - known


@pytest.mark.parametrize("stem", sorted(_PIPELINE_FIXTURES))
def test_a_nonstandard_backbone_is_found_and_prepared_from_a_structure(
    stem: str,
) -> None:
    """Nothing tells the pipeline where the chain runs; it works it out."""
    from tmol.ligand import prepare_ligands

    param_db = ParameterDatabase.get_default()
    known = {r.name for r in param_db.chemical.residues}
    prepared, canonical_ordering = prepare_ligands(_structure(stem), param_db=param_db)

    expected = _PIPELINE_FIXTURES[stem]
    added = {
        r.name: r
        for r in prepared.chemical.residues
        if r.name not in known and ":" not in r.name
    }
    assert set(added) == set(expected), stem

    for code, mainchain_length in expected.items():
        restype = added[code]
        assert code in canonical_ordering.restype_io_equiv_classes

        # a chain runs through it, so it has both connections
        connections = {c.name: c.atom for c in restype.connections}
        assert set(connections) == {"down", "up"}, code

        # none of these is an unmodified alpha backbone, so none is typed as one
        peptide_typed = {
            a.name for a in restype.atoms if a.atom_type in _PEPTIDE_BACKBONE_TYPES
        }
        assert peptide_typed == set(), f"{code} typed as protein backbone"

        # the backbone found is the one the connections are on, and it is bonded
        mainchain = restype.properties.polymer.mainchain_atoms
        assert len(mainchain) == mainchain_length, f"{code}: {mainchain}"
        assert (mainchain[0], mainchain[-1]) == (connections["down"], connections["up"])
        bonded = {frozenset((b[0], b[1])) for b in restype.bonds}
        for first, second in zip(mainchain, mainchain[1:]):
            assert frozenset((first, second)) in bonded, f"{code}: {first}-{second}"


# 3C3G leaves B3K, B3Q and HMR with unresolved sidechains, and their residue
# types are completed against the chemistry the file declares. Pose
# construction rebuilds a missing sidechain with DunbrackChiSampler, which
# defines no rotamers for these: their sidechains are lysine's, glutamine's and
# arginine's, but hang off a beta backbone, so what a canonical calls chi1 is a
# backbone torsion here and no canonical library's chi line up. The sampler
# produces nothing and the atoms stay NaN. Lifting this needs a rotamer
# reference for a nonstandard backbone.
_NO_ROTAMERS_FOR_A_BETA_SIDECHAIN = "beta_peptide_3c3g"


@pytest.mark.parametrize(
    "stem",
    [
        pytest.param(
            stem,
            marks=(
                [
                    pytest.mark.xfail(
                        strict=True,
                        reason="no rotamer library reaches a beta backbone's "
                        "sidechain, so unresolved atoms stay NaN",
                    )
                ]
                if stem == _NO_ROTAMERS_FOR_A_BETA_SIDECHAIN
                else []
            ),
        )
        for stem in sorted(_PIPELINE_FIXTURES)
    ],
)
def test_a_nonstandard_backbone_scores(stem: str, torch_device) -> None:
    """Smoke test: preparation has to survive into a pose and a number."""
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand import prepare_ligands
    from tmol.score import beta2016_score_function

    structure = _structure(stem)
    prepared, _canonical_ordering = prepare_ligands(
        structure, param_db=ParameterDatabase.get_default()
    )
    pose_stack = pose_stack_from_biotite(structure, torch_device, param_db=prepared)
    sfxn = beta2016_score_function(torch_device, param_db=prepared)
    module = sfxn.render_whole_pose_scoring_module(pose_stack)
    total = float(module(pose_stack.coords).sum())

    assert numpy.isfinite(total), stem
    assert total != 0.0, stem


# --------------------------------------------------------------------------- #
# termini patches a residue brings with it
# --------------------------------------------------------------------------- #

# The database's termini patches are written for an alpha backbone and are
# scoped to one, so a residue whose backbone is not alpha generates its own.


def _prepared(stem: str):
    from tmol.ligand import prepare_ligands

    param_db = ParameterDatabase.get_default()
    known = {r.name for r in param_db.chemical.residues}
    prepared, _co = prepare_ligands(_structure(stem), param_db=param_db)
    return prepared, known


def test_a_nonstandard_backbone_brings_its_own_termini_patches() -> None:
    """It cannot use the database's, so preparation generates a pair."""
    prep = _prepare("MLE")
    assert prep.residue_type.properties.polymer.backbone_type == "nonstandard_aa"

    patches = prep.adds_patches
    assert {p.display_name for p in patches} == {"nterm", "cterm"}
    # scoped to this residue alone, so they can never reach another
    for patch in patches:
        assert patch.applies_to.base_names == ("MLE",)
        assert patch.applies_to.backbone_types is None


def test_an_alpha_backbone_keeps_using_the_database_patches() -> None:
    """An unmodified alpha backbone's terminus is the canonical amide."""
    prep = _prepare("HYP")
    assert prep.residue_type.properties.polymer.backbone_type == "alpha_aa"
    assert prep.adds_patches == ()


def test_a_generated_patch_avoids_names_the_residue_already_uses() -> None:
    """Gamma-glutamate keeps an alpha carbonyl O the patch would collide with.

    The database's C-terminus patch adds atoms called O and OXT. FGA links
    through CD, so the O it would replace is on the sidechain and the alpha
    backbone's own O survives -- two atoms of one name, which no residue type
    can hold.
    """
    prepared, known = _prepared("gamma_peptide_1gac")
    cterm = next(r for r in prepared.chemical.residues if r.name == "FGA:cterm")
    names = [a.name for a in cterm.atoms]
    assert len(names) == len(set(names)), "duplicate atom name"
    assert "O" in names


def test_a_cap_has_no_termini_variants() -> None:
    """A cap's one connection is the chain's, so it has no other end.

    Patching it would make the cap a free molecule, which is the ligand path's
    business rather than a variant of this residue.
    """
    prepared, known = _prepared("capped_peptide_ace_nme")
    added = {r.name for r in prepared.chemical.residues if r.name not in known}
    assert added == {"ACE", "NME"}


@pytest.mark.parametrize(
    "stem", ["nmethyl_peptide_6mvz", "gamma_peptide_1gac", "collagen_hyp_1bkv"]
)
def test_every_variant_carries_charges_for_all_of_its_atoms(stem: str) -> None:
    """A missing charge fails the whole structure, used variant or not.

    PackedBlockTypes packs every variant of every residue it is given, so a
    terminal form nothing sits in still has to resolve.
    """
    prepared, known = _prepared(stem)
    charges: dict = {}
    for entry in prepared.scoring.elec.atom_charge_parameters:
        res, _, variant = str(entry.res).partition(":")
        charges.setdefault(res, {}).setdefault(str(entry.atom), {})[
            variant
        ] = entry.charge

    missing, nets = [], {}
    for restype in prepared.chemical.residues:
        if restype.name in known:
            continue
        base_name, *variants = restype.name.split(":")
        variants.append("")  # unpatched last, as the elec resolver does
        total = 0.0
        for atom in restype.atoms:
            by_variant = charges.get(base_name, {}).get(atom.name)
            hit = next((v for v in variants if by_variant and v in by_variant), None)
            if hit is None:
                missing.append(f"{restype.name}/{atom.name}")
            else:
                total += by_variant[hit]
        nets[restype.name] = total

    assert missing == []
    # charges are not renormalized: a terminal form is charged as the molecule
    #    it is, and what its stub carries is not the residue's to redistribute
    assert nets


def test_a_params_file_carries_the_patches_it_needs(tmp_path) -> None:
    """A saved residue that loses its patches cannot sit at a chain end."""
    from tmol.ligand import load_params_file, write_params_file
    from tmol.ligand._registry import inject_ligand_preparations

    prep = _prepare("MLE")
    path = tmp_path / "mle.tmol"
    write_params_file([prep], str(path), format="tmol")
    assert "adds_patches" in path.read_text()

    loaded = load_params_file(path)
    assert len(loaded) == 1
    assert {p.display_name for p in loaded[0].adds_patches} == {"nterm", "cterm"}
    assert all(p.applies_to.base_names == ("MLE",) for p in loaded[0].adds_patches)

    # and the round-tripped residue still takes both termini
    injected = inject_ligand_preparations(ParameterDatabase.get_default(), loaded)
    names = {r.name for r in injected.chemical.residues}
    assert {"MLE", "MLE:nterm", "MLE:cterm"} <= names


# atom types that describe a peptide backbone or its termini; a residue
# prepared as a ligand carries ligand types throughout, its chain ends included
_PEPTIDE_TERMINUS_TYPES = _PEPTIDE_BACKBONE_TYPES | {"OOC", "Hpol", "Nlys"}


@pytest.mark.parametrize("code", ["MLE", "B3K", "HAO"])
def test_a_generated_patch_types_its_atoms_as_a_ligand(code: str) -> None:
    """A terminus typed as a peptide's is a different type system from the rest.

    The residue's own atoms come from the ligand typer, so its chain ends have
    to as well -- an OOC oxygen and a ligand carboxylate oxygen are different
    hbond acceptors.
    """
    prep = _prepare(code)
    assert prep.adds_patches
    for patch in prep.adds_patches:
        typed = [a.atom_type for a in (*patch.add_atoms, *patch.modify_atoms)]
        assert typed
        assert not set(typed) & _PEPTIDE_TERMINUS_TYPES, (code, patch.name, typed)


def _nterm_patch(code: str):
    return next(p for p in _prepare(code).adds_patches if p.display_name == "nterm")


def test_a_terminus_is_protonated_by_its_own_chemistry() -> None:
    """The protonation state comes from Dimorphite, per residue.

    The database's patch adds as many protons as an alpha backbone's amide
    nitrogen takes once charged, which is the only state it can express. A
    chain end that is not an amine of that kind takes a different one, and
    what it takes is Dimorphite's call rather than this pipeline's.
    """
    aliphatic = _nterm_patch("B3K")
    aromatic = _nterm_patch("HAO")

    assert len(aliphatic.add_atoms) == 3
    assert len(aromatic.add_atoms) < len(aliphatic.add_atoms)

    # and the nitrogen itself is typed differently for it
    def site_type(patch):
        return next(a.atom_type for a in patch.modify_atoms)

    assert site_type(aromatic) != site_type(aliphatic)


def test_a_terminal_group_is_built_at_the_angle_its_geometry_calls_for() -> None:
    """What the icoors produce, rather than what they say.

    A carboxylate carbon is trigonal, so its two oxygens sit 120 degrees apart.
    An icoor stores the supplement of the angle it builds, so a value carried
    over from a tetrahedral site would show up here as roughly 109.
    """
    prepared, _known = _prepared("nmethyl_peptide_6mvz")
    restype = next(r for r in prepared.chemical.residues if r.name == "MLE:cterm")
    coords = _ideal_coords_by_name(restype)

    connection = next(c.atom for c in restype.connections if c.name == "down")
    assert connection  # the acid end is patched, so the amine end remains

    oxygens = sorted(
        a.name
        for a in restype.atoms
        if a.name not in {x.name for x in _prepare("MLE").residue_type.atoms}
    )
    assert len(oxygens) == 2, oxygens
    carbon = next(
        b[1] if b[0] == oxygens[0] else b[0]
        for b in restype.bonds
        if oxygens[0] in b[:2] and not b[0].startswith("H") and not b[1].startswith("H")
    )
    first, second = (coords[o] - coords[carbon] for o in oxygens)
    cosine = numpy.dot(first, second) / (
        numpy.linalg.norm(first) * numpy.linalg.norm(second)
    )
    angle = numpy.degrees(numpy.arccos(numpy.clip(cosine, -1.0, 1.0)))
    assert angle == pytest.approx(120.0, abs=8.0), angle


def test_a_terminus_that_cannot_be_built_falls_back_to_the_residue() -> None:
    """Best effort rather than refusal, and never in another type system.

    Where the terminal form cannot be built, its atoms are typed and charged
    like the residue's own atoms of the same kind. Only the proton count is
    the database patch's, since nothing else can supply it. Asking for patches
    without a structure to build the terminal form from takes the same path.
    """
    from tmol.ligand._terminus_patches import terminus_patches

    param_db = ParameterDatabase.get_default()
    prep = _prepare("MLE")
    residue = _residue("MLE")
    profile = profile_for_atom_array(residue, _connection_atoms("MLE"))

    generated = terminus_patches(
        param_db.chemical,
        prep.residue_type,
        profile,
        base_charges=prep.partial_charges,
    )
    patches = {patch.display_name: patch for patch, _t, _c in generated}
    assert set(patches) == {"nterm", "cterm"}

    own = {a.atom_type for a in prep.residue_type.atoms}
    for patch in patches.values():
        typed = {a.atom_type for a in (*patch.add_atoms, *patch.modify_atoms)}
        assert typed
        # the residue's own types, never the database patch's peptide ones
        assert not typed & _PEPTIDE_TERMINUS_TYPES, (patch.name, typed)
        assert typed <= own, (patch.name, typed - own)

    # and it says so, rather than quietly standing in for a measured one
    assert all(chemistry["measured"] is False for _p, _t, chemistry in generated)
