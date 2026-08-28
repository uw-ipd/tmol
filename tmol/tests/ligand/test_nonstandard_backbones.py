"""Classification of polymer residues whose backbone is not an alpha amino acid.

A residue reaches the polymer path on its CCD type, which says only that it
links into a chain -- not what its backbone is. These fixtures cover the
backbone classes that arrive there: the alpha backbone modified at its amide
nitrogen, backbones a carbon or two longer, and one that is a peptide only by
courtesy.

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

import numpy
import pytest

import biotite.structure.info as info

from tmol.database import ParameterDatabase
from tmol.ligand import LigandPreparationError, prepare_polymer_residue
from tmol.ligand._polymer_profile import alpha_profile, profile_for_atom_array
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
    import biotite.structure.io.pdbx as pdbx

    from tmol.tests.data import data_path

    cif = pdbx.CIFFile.read(str(data_path("ncaa_fixtures") / "collagen_hyp_1bkv.cif"))
    atom_array = pdbx.get_structure(cif, model=1, include_bonds=True)
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
_TERMINAL_CASES = {
    "B3K": ("C", ("N", "CA", "CB", "C")),
    "SAR": ("C", ("N", "CA", "C")),
    "FGA": ("N", ("N", "CA", "CB", "CG", "CD")),
    "ACB": ("N", ("N", "CA", "CB", "CG")),
}


def _residue_bonded_only_at(code: str, connection_atom: str):
    """A component as it appears at a chain end: one connection open, one intact."""
    atom_array = info.residue(code)
    atom_array.res_name[:] = code
    dropped = _leaving_group(atom_array, connection_atom)
    return atom_array[~numpy.isin(atom_array.atom_name, sorted(dropped))]


@pytest.mark.parametrize("code", sorted(_TERMINAL_CASES))
def test_a_terminal_residue_finds_its_other_end(code: str) -> None:
    """One connection does not say where the backbone stops; the CCD does.

    A component definition flags the atoms it gives up on polymerizing, so the
    atoms those hang off are the chain's ends. Gamma-glutamate and beta-
    aspartate are why this cannot be left to the atom names: both carry an
    intact alpha backbone and link through a sidechain carbon instead.
    """
    connection_atom, expected = _TERMINAL_CASES[code]
    residue = _residue_bonded_only_at(code, connection_atom)

    profile = profile_for_atom_array(residue, frozenset({connection_atom}))
    assert profile is not None, f"{code} was refused"
    assert profile.mainchain_atoms == expected


def test_a_terminal_residue_is_completed_by_chemistry_without_the_ccd() -> None:
    """Where the component flags nothing, the residue's own chemistry answers."""
    from tmol.ligand._polymer_profile import completed_connection_atoms

    residue = _residue_bonded_only_at("SAR", "C")
    # a name the CCD does not define, so nothing is declared about it
    residue.res_name[:] = "X_"
    assert completed_connection_atoms(residue, frozenset({"C"})) == frozenset(
        {"N", "C"}
    )


def test_an_undeterminable_terminal_residue_is_refused() -> None:
    """Two candidate ends are not resolved by guessing between them."""
    param_db = ParameterDatabase.get_default()
    # beta-lysine carries a sidechain amine as well as its backbone one, so
    #    with no component definition to consult there are two candidates
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
