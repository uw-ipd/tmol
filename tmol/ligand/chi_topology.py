"""Rotatable-bond (CHI / PROTON_CHI) classification for ligand residue types.

Ports RosettaVS ``generic_potential`` ``define_rotable_torsions``
(``SetupTopology.py``) and the PROTON_CHI sample tables (``Molecule.py``)
into tmol's ligand-preparation pipeline.  The goal is *semantic* parity with
RosettaVS: the same set of rotatable bond axes plus correct proton-chi sample
sets, not byte-identical CHI numbering.

The classifier is pure: it consumes the RDKit ``Mol``, the deterministic
atom-tree already built by :mod:`tmol.ligand.residue_builder`
(``order``/``parent``/``grandparents``), the per-atom names, and the
``RosettaTypingState`` produced by :func:`tmol.ligand.atom_typing.assign_tmol_atom_types`
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
- ``border > 1`` biaryl-pivot CHIs are intentionally not emitted; faithful
  ``biaryl_pivots`` detection (``is_biaryl_ring`` / ``search_special_biaryl_ring``)
  is out of scope for this stage and does not affect ref1/ref2 (ref1's
  ``#biaryl`` CHI is a single bond, handled correctly).
- Conjugated-polar-H skipping uses RDKit ``bond.GetIsConjugated()`` as a close
  approximation of Rosetta's per-atom H-count test.
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

from rdkit import Chem

from tmol.database.chemical import ChiSamples, Torsion, UnresolvedAtom

# RosettaVS hard-coded constant (Molecule.py): controls EXTRA expansion.
MAX_CONFS = 5000

# Heteroatoms that can carry a rotatable polar hydrogen (O, N, S).
_POLAR_HEAVY = {7, 8, 16}


def _is_heavy(mol: Chem.Mol, idx: int) -> bool:
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
    return bool(ring_membership.get(a, set()) & ring_membership.get(b, set()))


def build_chi_topology(
    mol: Chem.Mol,
    order: list[int],
    parent: dict[int, int],
    grandparents: dict[int, tuple[int, int]],
    atom_names: list,
    typing_state,
    *,
    logger=None,
) -> tuple[tuple[Torsion, ...], tuple[ChiSamples, ...]]:
    """Classify rotatable bonds and return ``(torsions, chi_samples)``.

    ``order``/``parent``/``grandparents`` are the kept-atom tree from
    ``build_residue_type`` (indices are RDKit atom indices; ``parent[root]``
    is the root itself).  ``atom_names[idx]`` is the final residue atom name
    (or ``None`` for dropped atoms).  ``typing_state`` is a
    :class:`~tmol.ligand.atom_typing.RosettaTypingState`.
    """
    valid = set(order)
    ring_membership = typing_state.ring_membership_by_idx
    atms_aro = typing_state.atms_aro
    atms_strained = typing_state.atms_strained
    hyb_by_idx = typing_state.hyb_by_idx

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
        polar_h_children = [x for x in c_children if _is_polar_hydrogen(mol, x)]
        if heavy_children:
            is_proton = False
            d = heavy_children[0]
        elif (
            mol.GetAtomWithIdx(c).GetAtomicNum() in _POLAR_HEAVY and polar_h_children
        ):
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
        if mol.GetAtomWithIdx(a).GetAtomicNum() == 1 and not _is_polar_hydrogen(
            mol, a
        ):
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
        elif _share_ring(ring_membership, b, c):
            _trace(c, b, "skip: non-aromatic ring-internal")
            continue

        if border > 1:
            # amide/nbonded/biaryl border>1: default flags skip these
            # (report_amide_chi=False, report_nbonded_chi=False); kept biaryl
            # pivots are a documented gap.
            _trace(c, b, "skip: border>1 (amide/nbonded/biaryl default off)")
            continue

        if is_proton and bond.GetIsConjugated():
            # conjugated polar-H chi skipped (approximation of Rosetta's
            # per-atom hydrogen-count test)
            _trace(c, b, "skip: conjugated polar-H")
            continue

        # quad validity (AC-9): four distinct, bonded atoms
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
    extra_expansions: tuple[float, ...] = (
        (20.0,) if num_h_confs <= MAX_CONFS else ()
    )

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
        if is_proton:
            # sp2 (2-coordinate) heteroatom -> samples 0/180; sp3 -> 60/-60/180.
            samples = (0.0, 180.0) if is_sp2 else (60.0, -60.0, 180.0)
            chi_samples.append(
                ChiSamples(
                    chi_dihedral=name,
                    samples=samples,
                    expansions=extra_expansions,
                )
            )

    return tuple(torsions), tuple(chi_samples)
