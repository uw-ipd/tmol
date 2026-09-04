"""Rotatable-bond (CHI / PROTON_CHI) classification for ligand residue types.

Ports RosettaVS ``generic_potential`` ``define_rotable_torsions``
(``SetupTopology.py``) and the PROTON_CHI sample tables (``Molecule.py``)
into tmol's ligand-preparation pipeline.  The goal is *semantic* parity with
RosettaVS: the same set of rotatable bond axes plus correct proton-chi sample
sets, not byte-identical CHI numbering.

The classifier is pure: it consumes the RDKit ``Mol``, the deterministic
atom-tree already built by :mod:`tmol.ligand._residue_builder`
(``order``/``parent``/``grandparents``), the per-atom names, and the
``RosettaTypingState`` produced by :func:`tmol.ligand._atom_typing.assign_tmol_atom_types`
(``return_state=True``).  It emits named ``Torsion`` objects (``chi1``..``chiN``)
and, for polar-hydrogen rotations, matching ``ChiSamples``.

Hard-coded RosettaVS default flags (see ``BasicClasses.py``):
``report_Hapol_chi=False``, ``report_amide_chi=False``,
``report_nbonded_chi=False``, ``report_ringring_chi=True``,
``report_puckering_chi=False``, ``max_confs=5000``.

Validated against the RosettaVS ground truth (``ref1``/``ref2`` via the SMILES
path in ``TestGroundTruthRegression``): emitted CHI axes and PROTON_CHI
samples/expansions match. The ``EXTRA`` encoding (``EXTRA 1 20`` ->
``expansions=(20.0,)``, ``EXTRA 0`` -> ``()``) is consistent with
``OptHSampler``'s ``len(samples) * (1 + 2 * len(expansions))`` expansion.

Scope notes / latitude:
- Conjugated-polar-H skipping is a faithful port of ``assign_bond_conjugation``'s
  core: a bond is conjugated only when both atom classes are in
  :data:`_CONJUGATING_ACLASSES` (and neither is sp3), plus Rosetta's all-but-one-H
  test. So phenol/acid C-OH and aniline/amide -NH are classified like
  mol2genparams (verified). The geometry-based planarity refinement (``is_planar``)
  is not ported.
- ``border > 1`` biaryl-pivot CHIs (ring <-> conjugated functional group) ARE
  emitted via a port of ``search_special_biaryl_ring`` (the hard-coded
  :data:`_SPECIAL_BIARYL_PAIRS` class-pair list). The remaining pivots through
  ring-like conjugated groups (guanidinium ``Ngu1``, tertiary amide ``Nad3``,
  ``NG2``, furan ``Ofu``) are recovered not by the ``is_planar`` geometry port
  but by honoring the source mol2's literal single-bond order
  (``original_single_bonds``): RDKit kekulization promotes those aryl-X bonds to
  DOUBLE, which the ``border > 1`` rule would skip, whereas Rosetta reads the
  mol2 order ``1`` verbatim. Restoring ``border = 1`` for those bonds matches
  mol2genparams and closes the DUD-80 parity set (80/80 full CHI).
- NU / ring-pucker DOFs are unsupported (RosettaVS default
  ``report_puckering_chi=False``); none are emitted by any preparation path.

RosettaVS rules that are handled *implicitly* by emitting CHIs only for
atom-tree edges (each non-root atom ``c`` with parent ``b``), rather than over
a separately-enumerated torsion list:
- ``ring_cuts``: a ring's closure bond is a non-tree (back) edge, so it is never
  a parent->child edge and never a CHI candidate.
- ``FT_connected``: fold-tree-disconnected torsions cannot arise — every emitted
  axis is a tree edge by construction.
- ``atms_puckering`` (default-off): puckering-ring internal bonds are
  ring-internal and are already skipped by ``_share_ring``.
num_H_confs is computed over the pre-skip polar-H set (so EXTRA matches
RosettaVS even when a polar-H chi is counted but later skipped).
"""

from __future__ import annotations

import attr
from rdkit import Chem

from tmol.database.chemical import ChiSamples, Torsion, UnresolvedAtom

# RosettaVS hard-coded constant (Molecule.py): controls EXTRA expansion.
MAX_CONFS = 5000

# How many conformers a residue's own sampled chi may multiply its rotamers
# by. The +/-20 degree expansions are kept while the product stays under the
# first limit and dropped at the second; past it, chi are frozen from the tip
# inward. A borrowed rotamer library multiplies this again, so its own count
# is folded in as 3 ** n_library_chi -- exact for a rotameric library and low
# by at most four for a semirotameric one, which is close enough to bound.
EXPANDED_CONF_LIMIT = 100
CONF_LIMIT = 1000

# Heteroatoms that can carry a rotatable polar hydrogen (O, N, S).
_POLAR_HEAVY = {7, 8, 16}

# RosettaVS Types.py CONJUGATING_ACLASSES: a single bond is treated as
# conjugated (for the proton-chi skip) only when BOTH central atom types are in
# this set. Notably it excludes hydroxyl ``Ohx`` and most carbonyl/carboxyl
# oxygens, so phenol C-OH and aniline C-NH stay rotatable — unlike RDKit's
# blanket ``GetIsConjugated()``.
_CONJUGATING_ACLASSES = frozenset(
    {
        "CD",
        "CD1",
        "CD2",
        "CR",
        "CDp",
        "CRp",
        "Nad",
        "Nin",
        "Nim",
        "Ngu1",
        "Ngu2",
        "NG2",
        "NG21",
        "NG22",
        "Nad3",
        "Ofu",
        "OG2",
        "Ssl",
        "SG2",
    }
)

# RosettaVS ``search_special_biaryl_ring`` hard-coded ``special_biaryl_to_ring``
# list (SetupTopology.py). An ordered ``(a2_aclass, a3_aclass)`` pair, where a2
# is a non-ring atom bonded to an aromatic ring atom a1 and a3 is a2's further
# neighbour. When matched (and a3 is further-connected), the a1-a2 bond is a
# biaryl pivot: a rotatable CHI that survives the ``border > 1`` skip.
_SPECIAL_BIARYL_PAIRS = frozenset(
    {
        ("CDp", "OG2"),
        ("CDp", "Oal"),
        ("CDp", "Oad"),
        ("NG21", "CR"),
        ("NG21", "CRp"),
        ("NG21", "CD"),
        ("NG21", "CDp"),
        ("NG21", "CSp"),
        ("NG21", "CS"),
        ("NG21", "CS1"),
        ("NG21", "CS2"),
        ("NG21", "SG5"),
        ("Nad", "CDp"),
        ("CDp", "Nad"),
        ("CDp", "Nam2"),
        ("CDp", "NG2"),
        ("CDp", "Nin"),
        ("CD1", "CD1"),
        ("CD1", "CD"),
        ("CD", "CD"),
        ("CD", "CD1"),
        ("CD", "CR"),
        ("CD1", "CR"),
        ("CDp", "CR"),
        ("CD", "F"),
        ("CD", "Cl"),
        ("CD", "Br"),
        ("CD", "I"),
    }
)


def _is_heavy(mol: Chem.Mol, idx: int) -> bool:
    """True if atom ``idx`` is not a hydrogen."""
    return mol.GetAtomWithIdx(idx).GetAtomicNum() != 1


def _is_polar_hydrogen(mol: Chem.Mol, idx: int) -> bool:
    """True if ``idx`` is a hydrogen bonded to O/N/S."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 1:
        return False
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() in _POLAR_HEAVY:
            return True
    return False


def _bond_order(bond: Chem.Bond) -> int:
    """Return the integer bond order (aromatic and unknown treated as 2)."""
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return 1
    if bt == Chem.BondType.DOUBLE:
        return 2
    if bt == Chem.BondType.TRIPLE:
        return 3
    # AROMATIC and anything else are treated as order > 1 for chi selection.
    return 2


def _share_ring(ring_membership: dict[int, set[int]], a: int, b: int) -> bool:
    """True if atoms ``a`` and ``b`` belong to a common ring."""
    return bool(ring_membership.get(a, set()) & ring_membership.get(b, set()))


def _symmetry_order(mol: Chem.Mol, b: int, c: int) -> int:
    """How many times the b-c torsion turns the tip onto itself in a full turn.

    A tip of interchangeable terminal substituents -- a carboxylate's two
    oxygens, a phosphate's three -- repeats every 360/order degrees, so only
    that much of the circle holds distinct structures.
    """
    others = [
        n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b
    ]
    if len(others) < 2:
        return 1
    # equivalent tips differ only by where the formal charge was written down,
    #    so the element alone decides
    elements = set()
    for idx in others:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetDegree() != 1:
            return 1
        elements.add(atom.GetAtomicNum())
    return len(others) if len(elements) == 1 else 1


def build_chi_topology(  # noqa: C901
    mol: Chem.Mol,
    order: list[int],
    parent: dict[int, int],
    grandparents: dict[int, tuple[int, int]],
    atom_names: list,
    typing_state,
    *,
    atype_by_idx: dict[int, str] | None = None,
    original_single_bonds: frozenset[frozenset[str]] | None = None,
    assign_ring_chis: bool = False,
    generate_heavy_chi_samples: bool = False,
    logger=None,
) -> tuple[tuple[Torsion, ...], tuple[ChiSamples, ...]]:
    """Classify rotatable bonds and return ``(torsions, chi_samples)``.

    ``order``/``parent``/``grandparents`` are the kept-atom tree from
    ``build_residue_type`` (indices are RDKit atom indices; ``parent[root]``
    is the root itself).  ``atom_names[idx]`` is the final residue atom name
    (or ``None`` for dropped atoms).  ``typing_state`` is a
    :class:`~tmol.ligand._atom_typing.RosettaTypingState`.

    ``assign_ring_chis`` keeps single bonds inside a non-aromatic ring as chi,
    the way proline's chi2/chi3 are defined. Polymer residues set it; ligands
    do not.

    ``original_single_bonds`` (optional) is a set of ``frozenset({name_a,
    name_b})`` pairs that the source mol2 records as literal single bonds.
    For those bonds the bond order is forced to 1 — overriding RDKit's
    post-kekulization promotion of some aromatic/conjugated single bonds to
    DOUBLE — so the ``border > 1`` skips match Rosetta's ``mol2genparams``,
    which reads the literal mol2 order.
    """
    valid = set(order)
    ring_membership = typing_state.ring_membership_by_idx
    atms_aro = typing_state.atms_aro
    atms_strained = typing_state.atms_strained
    hyb_by_idx = typing_state.hyb_by_idx
    atype_by_idx = atype_by_idx or {}

    def _rosetta_bond_conjugated(b: int, c: int) -> bool:
        """Approximate RosettaVS ``assign_bond_conjugation`` for a single bond.

        A bond is conjugated only when neither atom is sp3 and BOTH atom
        classes are in :data:`_CONJUGATING_ACLASSES`. This excludes hydroxyl
        ``Ohx`` and most carbonyl oxygens, so phenol C-OH / aniline C-NH are NOT
        conjugated (kept as rotatable chis), matching mol2genparams — unlike the
        blanket RDKit ``GetIsConjugated()`` this replaces. (Geometry-based
        planarity and biaryl-pivot refinements are not yet ported.)
        """
        if hyb_by_idx.get(b, 3) == 3 or hyb_by_idx.get(c, 3) == 3:
            return False
        return (
            atype_by_idx.get(b) in _CONJUGATING_ACLASSES
            and atype_by_idx.get(c) in _CONJUGATING_ACLASSES
        )

    def _all_but_one_h(idx: int) -> bool:
        """True if all but one of ``idx``'s bonds are to hydrogen (Rosetta nH test)."""
        atom = mol.GetAtomWithIdx(idx)
        n_h = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
        return n_h == atom.GetDegree() - 1

    def _detect_biaryl_pivots() -> set[frozenset]:
        """Port RosettaVS ``search_special_biaryl_ring``: ring<->functional-group
        rotatable bonds detected via the hard-coded ``_SPECIAL_BIARYL_PAIRS``.

        For each aromatic ring atom ``a1`` and non-ring neighbour ``a2``, if some
        ``(a2_aclass, a3_aclass)`` matches the list (``a3`` a further neighbour of
        ``a2`` that is itself connected on), the ``a1-a2`` bond is a biaryl pivot.
        These survive the ``border > 1`` skip. (The ``is_planar`` geometry
        refinement is not yet ported, so a few planar pivots may be over-kept.)
        """
        pivots: set[frozenset] = set()
        for a1 in atms_aro:
            atom1 = mol.GetAtomWithIdx(a1)
            for nb2 in atom1.GetNeighbors():
                a2 = nb2.GetIdx()
                if a2 not in valid or _share_ring(ring_membership, a1, a2):
                    continue
                a2c = atype_by_idx.get(a2)
                special = False
                further_connected = False
                for nb3 in nb2.GetNeighbors():
                    a3 = nb3.GetIdx()
                    if a3 == a1:
                        continue
                    if (a2c, atype_by_idx.get(a3)) in _SPECIAL_BIARYL_PAIRS:
                        special = True
                    if nb3.GetDegree() > 1:
                        further_connected = True
                if special and further_connected:
                    pivots.add(frozenset((a1, a2)))
        return pivots

    biaryl_pivots = _detect_biaryl_pivots()

    # Map each atom to its tree children (atoms whose parent is this atom).
    children: dict[int, list[int]] = {i: [] for i in order}
    for c in order:
        b = parent.get(c, c)
        if b != c and b in children:
            children[b].append(c)

    def pick_a(b: int, c: int) -> int | None:
        """An atom on ``b``'s side of the bond, distinct from ``c``."""
        gp = grandparents.get(c, (b, b))[0]
        if gp in valid and gp not in (b, c) and atom_names[gp] is not None:
            return gp
        # Fall back to a deterministic neighbor of b (heavy preferred).
        nbrs = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(b).GetNeighbors()
            if n.GetIdx() in valid
            and n.GetIdx() != c
            and atom_names[n.GetIdx()] is not None
        ]
        nbrs.sort(key=lambda n: (mol.GetAtomWithIdx(n).GetAtomicNum() == 1, n))
        return nbrs[0] if nbrs else None

    def _trace(c, b, msg):
        if logger is not None:
            nb = atom_names[b] if b is not None else "?"
            nc = atom_names[c] if c is not None else "?"
            logger.debug("chi-edge %s-%s: %s", nb, nc, msg)

    # Pass 1: collect candidate chis (after the RosettaVS default-flag skips).
    # Each candidate: (b, c, a, d, is_proton, is_sp2).
    # polar_h_factors keys every polar-H rotatable bond to its H-conformer
    # factor (6 for sp2, 9 for sp3) BEFORE the later CHI skip filters, matching
    # RosettaVS num_H_confs (SetupTopology.py:862-873 counts the hpol torsion
    # set before applying ring-cut/conjugation/puckering skips).
    candidates: list[tuple[int, int, int, int, bool, bool]] = []
    polar_h_factors: dict[frozenset, int] = {}
    for c in order:
        b = parent.get(c, c)
        if b == c or b not in valid:
            continue  # root has no parent bond
        bond = mol.GetBondBetweenAtoms(b, c)
        if bond is None:
            _trace(c, b, "skip: no rdkit bond")
            continue
        if atom_names[b] is None or atom_names[c] is None:
            _trace(c, b, "skip: dropped atom name")
            continue

        # Determine the tip atom d (on c's side) and the chi kind.
        c_children = children.get(c, [])
        heavy_children = [x for x in c_children if _is_heavy(mol, x)]
        if assign_ring_chis and not heavy_children:
            # the last atom of a ring reaches the rest of it by a back edge, so
            #    its tip is a neighbour rather than a tree child
            heavy_children = [
                x.GetIdx()
                for x in mol.GetAtomWithIdx(c).GetNeighbors()
                if x.GetIdx() != b
                and x.GetIdx() in valid
                and _is_heavy(mol, x.GetIdx())
            ]
        polar_h_children = [x for x in c_children if _is_polar_hydrogen(mol, x)]
        if heavy_children:
            is_proton = False
            d = heavy_children[0]
        elif mol.GetAtomWithIdx(c).GetAtomicNum() in _POLAR_HEAVY and polar_h_children:
            is_proton = True
            d = polar_h_children[0]
        else:
            # Terminal heavy atom (no rotatable tip) or apolar-H-only rotation.
            # report_Hapol_chi=False -> apolar-H chis are not emitted.
            _trace(
                c,
                b,
                f"skip: no tip (c_children={[atom_names[x] for x in c_children]})",
            )
            continue

        if atom_names[d] is None:
            _trace(c, b, "skip: tip dropped")
            continue
        a = pick_a(b, c)
        if a is None:
            _trace(c, b, "skip: no 'a' atom on b's side")
            continue
        # If the only reference atom on b's side is an apolar hydrogen, every
        # torsion across this bond has an apolar-H endpoint (e.g. a methyl
        # carbon's bond to a ring). RosettaVS classifies these as apolar-H
        # (hapol) torsions and skips them with report_Hapol_chi=False.
        if mol.GetAtomWithIdx(a).GetAtomicNum() == 1 and not _is_polar_hydrogen(mol, a):
            _trace(c, b, "skip: apolar-H reference atom (hapol)")
            continue

        # RosettaVS sp2 proton-chi requires the heteroatom (stem) be 2-coordinate
        # sp2 (`stem.hyb == 2 and len(stem.bonds) < 3`); otherwise it is sp3.
        is_sp2 = is_proton and (
            hyb_by_idx.get(c, 3) == 2 and mol.GetAtomWithIdx(c).GetDegree() < 3
        )
        # Record this polar-H stem's conformer factor for num_H_confs BEFORE the
        # later skip filters, so EXTRA matches RosettaVS even when a polar-H chi
        # is present but later skipped (e.g. conjugated).
        if is_proton:
            polar_h_factors[frozenset((b, c))] = 6 if is_sp2 else 9

        # --- RosettaVS define_rotable_torsions skip rules (default flags) ---
        border = _bond_order(bond)
        # Honor the source mol2's literal single-bond order: RDKit kekulization
        # promotes some aromatic/conjugated single bonds (e.g. aryl-Ngu1) to
        # DOUBLE, which the border>1 rule would wrongly skip. Rosetta reads the
        # mol2 order verbatim, so restore border=1 for those bonds.
        if original_single_bonds and (
            frozenset((atom_names[b], atom_names[c])) in original_single_bonds
        ):
            border = 1
        is_pivot = frozenset((b, c)) in biaryl_pivots

        # strained torsion inside a ring
        if (
            _share_ring(ring_membership, b, c)
            and b in atms_strained
            and c in atms_strained
        ):
            _trace(c, b, "skip: strained ring")
            continue

        if b in atms_aro and c in atms_aro:
            if border > 1:
                # biaryl border>1 chi: requires biaryl_pivot detection (gap).
                _trace(c, b, "skip: aromatic border>1 (biaryl gap)")
                continue
            if _share_ring(ring_membership, b, c):
                _trace(c, b, "skip: same aromatic ring")
                continue
            # ring-ring single bond: report_ringring_chi=True -> keep
        elif _share_ring(ring_membership, b, c) and not assign_ring_chis:
            _trace(c, b, "skip: non-aromatic ring-internal")
            continue

        if border > 1 and not is_pivot:
            # amide/nbonded border>1: default flags skip these
            # (report_amide_chi=False, report_nbonded_chi=False). Biaryl pivots
            # (ring<->conjugated functional group) are kept (report_ringring_chi).
            _trace(c, b, "skip: border>1 (amide/nbonded default off)")
            continue

        # RosettaVS: skip a conjugated polar-H chi only when the bond is
        # conjugated (both atom classes conjugating) AND one central atom is
        # all-but-one hydrogen. Phenol C-OH / aniline C-NH are NOT conjugated
        # (Ohx / amine-N outside CONJUGATING_ACLASSES), so they survive.
        if (
            is_proton
            and _rosetta_bond_conjugated(b, c)
            and (_all_but_one_h(b) or _all_but_one_h(c))
        ):
            _trace(c, b, "skip: conjugated polar-H")
            continue

        # Enforce four distinct atoms in the bonded torsion path.
        if len({a, b, c, d}) != 4:
            _trace(c, b, "skip: non-distinct quad")
            continue

        _trace(
            c,
            b,
            f"EMIT {'proton' if is_proton else 'heavy'} "
            f"a={atom_names[a]} d={atom_names[d]}",
        )
        candidates.append((b, c, a, d, is_proton, is_sp2))

    # Pass 2: EXTRA expansion factor. num_H_confs is the product over every
    # (pre-skip) polar-H rotatable bond of its conformer factor (sp2->6, sp3->9).
    num_h_confs = 1
    for factor in polar_h_factors.values():
        num_h_confs *= factor
    # "1 20" => one extra sample expanded by +/-20 degrees; "0" => none.
    extra_expansions: tuple[float, ...] = (20.0,) if num_h_confs <= MAX_CONFS else ()

    # Pass 3: build torsions + proton-chi samples.
    torsions: list[Torsion] = []
    chi_samples: list[ChiSamples] = []
    for n, (b, c, a, d, is_proton, is_sp2) in enumerate(candidates, start=1):
        name = f"chi{n}"
        torsions.append(
            Torsion(
                name=name,
                a=UnresolvedAtom(atom=atom_names[a]),
                b=UnresolvedAtom(atom=atom_names[b]),
                c=UnresolvedAtom(atom=atom_names[c]),
                d=UnresolvedAtom(atom=atom_names[d]),
            )
        )
        # sp2 (2-coordinate) heteroatom -> samples 0/180; sp3 -> 60/-60/180.
        samples = (0.0, 180.0) if is_sp2 else (60.0, -60.0, 180.0)
        if is_proton:
            chi_samples.append(
                ChiSamples(
                    chi_dihedral=name,
                    samples=samples,
                    expansions=extra_expansions,
                    is_proton=True,
                )
            )
        elif generate_heavy_chi_samples:
            # a ring torsion cannot be turned on its own without tearing the
            #    ring open, whatever the chi definition says
            bond = mol.GetBondBetweenAtoms(b, c)
            if bond is not None and not bond.IsInRing():
                # spread the samples over the distinct part of the circle, so a
                #    symmetric tip is covered once rather than repeatedly
                step = 360.0 / _symmetry_order(mol, b, c) / len(samples)
                chi_samples.append(
                    ChiSamples(
                        chi_dihedral=name,
                        samples=tuple(
                            samples[0] + i * step for i in range(len(samples))
                        ),
                        expansions=extra_expansions,
                        is_proton=False,
                    )
                )

    return tuple(torsions), tuple(chi_samples)


def rigid_central_bonds(
    mol: Chem.Mol,
    atom_names: list,
    typing_state,
    *,
    original_single_bonds: frozenset[frozenset[str]] | None = None,
) -> frozenset:
    """Heavy-atom bonds that no chi may turn, as frozensets of atom names.

    A bond-level verdict, unlike the chi emitted by :func:`build_chi_topology`:
    which bonds a tree walk reaches depends on where it is rooted, and a ring
    containing backbone atoms is entered from both ends. Callers that enumerate
    chi themselves need the chemistry without the tree.

    Rigid means a bond order above one, a bond inside an aromatic ring, or one
    inside a strained ring. Biaryl pivots are not detected here, so a ring-to-
    conjugated-group pivot is reported rigid.
    """
    ring_membership = typing_state.ring_membership_by_idx
    atms_aro = typing_state.atms_aro
    atms_strained = typing_state.atms_strained

    rigid = set()
    for bond in mol.GetBonds():
        b = bond.GetBeginAtomIdx()
        c = bond.GetEndAtomIdx()
        if not _is_heavy(mol, b) or not _is_heavy(mol, c):
            continue
        name_b, name_c = atom_names[b], atom_names[c]
        if name_b is None or name_c is None:
            continue
        border = _bond_order(bond)
        if (
            original_single_bonds
            and frozenset((name_b, name_c)) in original_single_bonds
        ):
            border = 1
        shares_ring = _share_ring(ring_membership, b, c)
        if (
            border > 1
            or (shares_ring and b in atms_aro and c in atms_aro)
            or (shares_ring and b in atms_strained and c in atms_strained)
        ):
            rigid.add(frozenset((name_b, name_c)))
    return frozenset(rigid)


def planar_heavy_atoms(mol: Chem.Mol, atom_names: list) -> frozenset:
    """Heavy atoms carrying a bond that is not a plain single, by name.

    A torsion turning between two such atoms is two-fold and planar, where one
    between sp3 centres is three-fold; a rotamer library measured on the one
    cannot supply the other.
    """
    planar = set()
    for bond in mol.GetBonds():
        b, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not _is_heavy(mol, b) or not _is_heavy(mol, c):
            continue
        if _bond_order(bond) > 1 or bond.GetIsAromatic():
            for idx in (b, c):
                if atom_names[idx] is not None:
                    planar.add(atom_names[idx])
    return frozenset(planar)


def _conformers(chi_samples, expanded: bool) -> int:
    """How many conformers this set of sampled chi enumerates."""
    total = 1
    for cs in chi_samples:
        if cs is None:  # frozen
            continue
        total *= len(cs.samples) * (1 + 2 * len(cs.expansions) if expanded else 1)
    return total


def apply_chi_sample_budget(chi_samples, n_library_chi: int = 0) -> tuple:
    """Trim sampled chi so the rotamers they enumerate stay bounded.

    A proton chi is never frozen: its hydrogen has no other source of
    placement, and optH reads the same samples. Heavy chi are frozen from the
    tip inward, since a chi near the backbone swings the whole sidechain where
    one at the tip moves a couple of atoms.
    """
    # a chi the borrowed library defines is read from the library, not sampled
    samples = [
        cs
        for cs in chi_samples
        if cs.is_proton or int(cs.chi_dihedral[3:]) > n_library_chi
    ]
    library = 3**n_library_chi
    if library * _conformers(samples, True) <= EXPANDED_CONF_LIMIT:
        return tuple(samples)

    samples = [attr.evolve(cs, expansions=()) for cs in samples]
    # chi are numbered outward from the backbone, so the last is the tip
    heavy = sorted(
        (i for i, cs in enumerate(samples) if not cs.is_proton),
        key=lambda i: int(samples[i].chi_dihedral[3:]),
        reverse=True,
    )
    for index in heavy:
        if library * _conformers(samples, False) <= CONF_LIMIT:
            break
        samples[index] = None
    return tuple(cs for cs in samples if cs is not None)
