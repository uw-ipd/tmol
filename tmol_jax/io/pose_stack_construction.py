"""Construct a ``PoseStack`` from a canonical-form dict -- pure numpy.

Upstream reference: ``tmol/io/pose_stack_construction.py`` (torch, with
helper modules in ``tmol/io/details/`` doing disulfide resolution, HIS
tautomer resolution, left-justification, missing-atom building, etc.).

The JAX port does a lean version of those details:

* Chain termini are detected from per-residue ``chain_id`` (when the
  canonical-form dict carries one) and the first / last residue of each
  chain is promoted to the matching ``:nterm`` / ``:cterm`` variant
  block type (e.g. ``MET`` -> ``MET:nterm``, ``GLY`` -> ``GLY:cterm``).
  Single-residue "chains" use the ``:cterm:nterm`` variant when it
  exists in the palette. Termini resolution requires the palette to
  contain the variant block types -- the default palette does, via
  :meth:`tmol_jax.pose.packed_block_types.PackedBlockTypes.default`.
* HIS tautomer detection: for each HIS residue, inspect which of HD1/HE2
  are provided and assign either HIS (NE2-protonated, default), HIS_D
  (ND1-protonated) or HIS_POS (both protonated). Mirrors upstream's
  ``resolve_his_tautomerization``.
* Disulfide detection: if two CYS residues' SG atoms are within 2.5 Å of
  each other, both are promoted to CYD. Mirrors upstream's
  ``find_disulfides``.
* Atoms whose canonical-slot is not present in the input (the common
  case is H1/H2/H3 added by the N-term variant and OXT added by the
  C-term variant when the source PDB does not provide them) are built
  from the residue's icoor tree, superposed onto the available
  backbone frame via a Kabsch-like alignment. When a PDB *does*
  provide those atoms (as 1ubq does), their authored coordinates are
  used directly.
* Non-leaf atoms (any atom that is a parent of another atom in the icoor
  tree) must be present in the input; if they are missing the builder
  raises ``ValueError`` with the same wording upstream uses. Leaf atoms
  (hydrogens, backbone O, OXT, ND1/NE2 termini on HIS variants) may be
  absent and are built from ideal coords.
* ``left_justify_canonical_form`` has no analogue -- we accept the
  residues in the layout the caller provides.

The primary API:

    pose_stack_from_canonical_form(
        canonical_form_dict,
        packed_block_types,
        chain_id=None,
        res_num=None,
    ) -> PoseStack

with ``canonical_form_dict`` having:
- ``"coords"``: ``float32[n_poses, max_n_res, max_n_can_atoms, 3]``
- ``"atom_is_present"`` (optional): ``bool[n_poses, max_n_res, max_n_can_atoms]``
  -- overrides the NaN-sniffing fallback. An atom is treated as "present"
  iff ``atom_is_present[p, r, a]`` is True.
- ``"res_name"`` (or ``"res_names"``): ``list[list[str]]`` of 3-letter
  codes. If omitted we fall back to ``"res_types"`` (an int array of
  canonical-ordering indices) combined with the provided
  :class:`CanonicalOrdering` (passed via the dict under the key
  ``"canonical_ordering"``).
- ``"chain_id"`` (optional): ``int[n_poses, max_n_res]`` per-residue
  chain index. Used for termini detection; if absent we fall back to
  the ``chain_id`` kwarg (per-pose sequences) or assume a single chain.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _normalise_res_names(
    cf: Dict[str, Any],
    n_poses: int,
    max_n_res: int,
) -> List[List[str]]:
    """Pull per-residue 3-letter names from the canonical-form dict.

    Prefers ``"res_name"`` / ``"res_names"`` when provided. Falls back to
    resolving ``"res_types"`` (integer indices) through a
    :class:`CanonicalOrdering` stored under ``"canonical_ordering"``.
    Padding positions come back as the empty string.
    """

    if "res_name" in cf or "res_names" in cf:
        raw = cf.get("res_name", cf.get("res_names"))
        if isinstance(raw, np.ndarray):
            res_names = [
                [str(x) if x is not None else "" for x in row] for row in raw.tolist()
            ]
        else:
            res_names = [[str(x) if x is not None else "" for x in row] for row in raw]
        # Pad rows to max_n_res.
        for row in res_names:
            while len(row) < max_n_res:
                row.append("")
        return res_names

    if "res_types" in cf:
        types = np.asarray(cf["res_types"])
        co = cf.get("canonical_ordering", None)
        if co is None or not hasattr(co, "restype_io_equiv_classes"):
            raise ValueError(
                "pose_stack_from_canonical_form: got res_types but no "
                "canonical_ordering to resolve them. Either pass "
                "'res_name' directly or include 'canonical_ordering'."
            )
        classes = list(co.restype_io_equiv_classes)
        out: List[List[str]] = []
        for p in range(n_poses):
            row: List[str] = []
            for r in range(max_n_res):
                idx = int(types[p, r])
                row.append(classes[idx] if 0 <= idx < len(classes) else "")
            out.append(row)
        return out

    raise ValueError(
        "pose_stack_from_canonical_form: canonical_form_dict must include "
        "'res_name' (list[list[str]]) or 'res_types' (+ 'canonical_ordering')."
    )


def _bt_name_index(pbt: Any) -> Dict[str, int]:
    """Map every block-type ``name`` (including variants like
    ``"MET:nterm"``) to its index in ``pbt.active_block_types``.
    """
    mapping: Dict[str, int] = {}
    for i, bt in enumerate(getattr(pbt, "active_block_types", ()) or ()):
        name = getattr(bt, "name", None)
        if name is not None:
            mapping[str(name)] = i
    return mapping


def _bt_name3_index(pbt: Any) -> Dict[str, int]:
    """Map ``name3`` -> first-matching *base* block-type index in ``pbt``.

    For the default palette (which now includes termini variants) we
    prefer the base residue (``name == name3``) over any variant so
    non-terminal residues resolve to the un-patched block type.
    """
    mapping: Dict[str, int] = {}
    # First pass: base (un-patched) residues, which have name == name3.
    for i, bt in enumerate(getattr(pbt, "active_block_types", ()) or ()):
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", None)
        name = getattr(bt, "name", None)
        if n3 is None or name is None:
            continue
        if str(name) == str(n3):
            mapping.setdefault(str(n3), i)
    # Second pass: anything else that hasn't been registered yet.
    for i, bt in enumerate(getattr(pbt, "active_block_types", ()) or ()):
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", None)
        if n3 is not None and str(n3) not in mapping:
            mapping[str(n3)] = i
    return mapping


def _build_variant_lookup(
    pbt: Any, bt_name_index: Dict[str, int]
) -> Dict[str, Dict[str, int]]:
    """Return ``{base_name: {variant_tag: bt_ind}}`` where ``variant_tag``
    is one of ``""``, ``"nterm"``, ``"cterm"``, ``"cterm:nterm"``.

    ``base_name`` is the colon-prefix of the block type name, so
    ``HIS_POS:cterm`` contributes to ``{"HIS_POS": {"cterm": ...}}``.
    Missing variants simply don't appear in the inner dict -- callers
    fall back to the base block type.
    """
    out: Dict[str, Dict[str, int]] = {}
    for name, i in bt_name_index.items():
        parts = name.split(":")
        base = parts[0]
        tag = ":".join(parts[1:])  # "" for base, "cterm:nterm" for both, etc.
        out.setdefault(base, {})[tag] = i
    return out


def _build_name3_special_case_lookup(pbt: Any) -> Dict[str, Dict[str, str]]:
    """Return ``{name3: {special_case_key: base_name}}``.

    Used to pick a base-name (``HIS_POS``, ``HIS_D``, ``HIS``, ``CYS``,
    ``CYD``, ...) for a residue whose 3-letter code is ``name3``. The
    inner dict keys mirror upstream's ``_map_spcase_var_to_int``:

    - ``"default"``       -- base type (e.g. HIS, CYS)
    - ``"cyd"``           -- CYD (disulfide CYS)
    - ``"hisd"``          -- HIS_D (ND1 protonated)
    - ``"hispos"``        -- HIS_POS (both NE2 and ND1 protonated)

    Entries without an explicit key inherit the ``"default"`` base.
    For residues that only have one base (ALA, GLY, ...) the inner
    dict will just have ``{"default": "ALA"}`` etc.
    """
    # Map from base_name -> name3 (3-letter code that the base belongs to)
    # using the block-type metadata that ships with the palette.
    bases_for_name3: Dict[str, List[str]] = {}
    for bt in getattr(pbt, "active_block_types", ()) or ():
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", None)
        base = getattr(bt, "base_name", None) or getattr(bt, "name", None)
        if n3 is None or base is None:
            continue
        n3 = str(n3)
        base = str(base)
        # Only track *base* (no-termini) rows so we don't double-add.
        if ":" in str(getattr(bt, "name", "")):
            continue
        bases_for_name3.setdefault(n3, [])
        if base not in bases_for_name3[n3]:
            bases_for_name3[n3].append(base)

    def _spcase_key(base: str, name3: str) -> str:
        if base == "CYD":
            return "cyd"
        if base == "HIS_D":
            return "hisd"
        if base == "HIS_POS":
            return "hispos"
        # The "default" base is whichever one matches the name3 literally
        # (HIS for HIS, CYS for CYS, ALA for ALA, ...).
        if base == name3:
            return "default"
        return "other:" + base

    out: Dict[str, Dict[str, str]] = {}
    for n3, bases in bases_for_name3.items():
        entry: Dict[str, str] = {}
        for base in bases:
            entry[_spcase_key(base, n3)] = base
        # Ensure a default exists: fall back to the name3 itself or the
        # first base-name we saw.
        if "default" not in entry:
            entry["default"] = n3 if n3 in bases else bases[0]
        out[n3] = entry
    return out


def _variant_bt_for_terminus(
    variants: Dict[str, int],
    is_nterm: bool,
    is_cterm: bool,
) -> Optional[int]:
    """Pick the right block-type index for a (nterm, cterm) state.

    Returns ``None`` if no suitable variant exists (caller should
    fall back to the base block type).
    """
    if is_nterm and is_cterm:
        # Single-residue chain; try both-terminal variant first.
        for key in ("cterm:nterm", "nterm:cterm"):
            if key in variants:
                return variants[key]
        # Fall back to one-terminal variant (cterm > nterm arbitrary).
        if "cterm" in variants:
            return variants["cterm"]
        if "nterm" in variants:
            return variants["nterm"]
        return None
    if is_nterm:
        return variants.get("nterm")
    if is_cterm:
        return variants.get("cterm")
    return None  # non-terminal -> use base


def _canonical_to_variant_map(
    co_atom_names: Tuple[str, ...],
    co_aliases: Dict[str, str],
    bt_atoms: Tuple[str, ...],
    bt_aliases: Dict[str, str],
) -> np.ndarray:
    """Map each canonical atom-slot index to the variant block-type's
    atom-slot index (``-1`` when the canonical atom doesn't exist in
    the variant).

    ``co_atom_names`` gives the canonical atom name at each canonical
    slot (this is ``CanonicalOrdering.restypes_ordered_atom_names[n3]``).
    ``co_aliases`` maps alternate canonical names to the canonical
    name (used when callers query via an alias).

    ``bt_atoms`` is the variant block type's own atom-name ordering;
    ``bt_aliases`` is its ``aliases_map``. The returned array has
    length ``len(bt_atoms)``, and each entry tells us which canonical
    slot to read from for that variant-atom position.
    """
    can_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(co_atom_names)}
    # Aliases: make the canonical name for each alias resolve to the
    # same slot; don't override existing mappings.
    for alias, canonical in co_aliases.items():
        if alias not in can_name_to_idx and canonical in can_name_to_idx:
            can_name_to_idx[alias] = can_name_to_idx[canonical]

    n_bt_atoms = len(bt_atoms)
    variant_to_canonical = np.full(n_bt_atoms, -1, dtype=np.int32)
    for j, name in enumerate(bt_atoms):
        if name in can_name_to_idx:
            variant_to_canonical[j] = can_name_to_idx[name]
        elif name in bt_aliases:
            canon_name = bt_aliases[name]
            if canon_name in can_name_to_idx:
                variant_to_canonical[j] = can_name_to_idx[canon_name]
    return variant_to_canonical


def _resolve_his_spcase(
    res_name: str,
    co_atom_names: Tuple[str, ...],
    atom_is_present_row: np.ndarray,
    coords_row: np.ndarray,
) -> str:
    """Return ``"default"`` / ``"hisd"`` / ``"hispos"`` for a HIS residue.

    Reproduces upstream's ``resolve_his_tautomerization`` branching:

    1. HD1 present and HE2 absent       -> HIS_D (``"hisd"``)
    2. HE2 present and HD1 absent       -> HIS   (``"default"``)
    3. Both HD1 and HE2 present         -> HIS_POS (``"hispos"``)
    4. HN present with ND1 and NE2      -> assign to whichever of ND1/NE2
       is closer to HN (HIS_D vs HIS).
    5. NH/NN/HN/CG all present          -> assign to whichever of NH/NN is
       closer to CG (HIS_D vs HIS).
    6. Otherwise                        -> HIS (arbitrary default).

    ``co_atom_names`` is the canonical-ordering atom-name tuple for the
    HIS equivalence class (includes HD1 / HE2 / HN / NH / NN / CG).
    """
    if res_name != "HIS":
        return "default"
    name_to_slot = {n: i for i, n in enumerate(co_atom_names)}

    def _has(atom: str) -> bool:
        slot = name_to_slot.get(atom, -1)
        if slot < 0 or slot >= atom_is_present_row.shape[0]:
            return False
        return bool(atom_is_present_row[slot])

    hd1 = _has("HD1")
    he2 = _has("HE2")
    hn = _has("HN")
    nd1 = _has("ND1")
    ne2 = _has("NE2")
    nh = _has("NH")
    nn = _has("NN")
    cg = _has("CG")

    if hd1 and he2:
        return "hispos"
    if hd1 and not he2:
        return "hisd"
    if he2 and not hd1:
        return "default"
    if hn and not hd1 and not he2 and nd1 and ne2:
        s_hn = name_to_slot["HN"]
        s_nd1 = name_to_slot["ND1"]
        s_ne2 = name_to_slot["NE2"]
        d_nd1 = float(np.sum((coords_row[s_nd1] - coords_row[s_hn]) ** 2))
        d_ne2 = float(np.sum((coords_row[s_ne2] - coords_row[s_hn]) ** 2))
        return "hisd" if d_nd1 < d_ne2 else "default"
    if nh and nn and hn and cg:
        s_nh = name_to_slot["NH"]
        s_nn = name_to_slot["NN"]
        s_cg = name_to_slot["CG"]
        d_nh = float(np.sum((coords_row[s_nh] - coords_row[s_cg]) ** 2))
        d_nn = float(np.sum((coords_row[s_nn] - coords_row[s_cg]) ** 2))
        # If NH is closer to CG, NH is ND1 (ND1-protonated -> HIS_D).
        # If NN is closer to CG, NN is ND1 (NE2-protonated -> HIS).
        return "hisd" if d_nh < d_nn else "default"
    # No discriminators -> NE2-protonated HIS.
    return "default"


def _resolve_disulfide_pairs(
    real_r_inds: List[int],
    real_name3s: List[str],
    co_atom_names_for_cys: Tuple[str, ...],
    atom_is_present: np.ndarray,
    coords: np.ndarray,
    pose_idx: int,
    cutoff_dis: float = 2.5,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return ``(paired_flags, pairs)`` for CYS residues at this pose.

    * ``paired_flags`` is a list-of-ints (0/1) indexed by ``real_r_inds``
      that is True for CYS residues that should be promoted to CYD.
    * ``pairs`` is a list of ``(k_i, k_j)`` tuples giving the
      block-index-among-real-residues of the two halves of each
      disulfide. These are needed downstream to wire the ``dslf``
      inter-residue connection so the cart-bonded SG-SG bond term sees
      the disulfide as a real bond (otherwise 1cgi etc. blow up the
      ``cart_lengths`` term with ~1.95 Å SG-SG distances treated as
      "unbonded").

    Greedy nearest-neighbour matching: walks the CYS residues in order
    and pairs each unpaired CYS with the closest unpaired CYS whose SG
    is within ``cutoff_dis``. Mirrors upstream's ``find_disulf_numba``.
    """
    paired = [0] * len(real_r_inds)
    pairs: List[Tuple[int, int]] = []
    if "SG" not in co_atom_names_for_cys:
        return paired, pairs
    sg_slot = co_atom_names_for_cys.index("SG")
    cys_ks = [k for k, n3 in enumerate(real_name3s) if n3 == "CYS"]
    if len(cys_ks) < 2:
        return paired, pairs

    cutoff2 = cutoff_dis * cutoff_dis
    for i_idx, k_i in enumerate(cys_ks):
        if paired[k_i]:
            continue
        r_i = real_r_inds[k_i]
        if not bool(atom_is_present[pose_idx, r_i, sg_slot]):
            continue
        c_i = coords[pose_idx, r_i, sg_slot]
        best_j = -1
        best_d2 = None
        for j_idx in range(i_idx + 1, len(cys_ks)):
            k_j = cys_ks[j_idx]
            if paired[k_j]:
                continue
            r_j = real_r_inds[k_j]
            if not bool(atom_is_present[pose_idx, r_j, sg_slot]):
                continue
            c_j = coords[pose_idx, r_j, sg_slot]
            d2 = float(np.sum((c_i - c_j) ** 2))
            if d2 < cutoff2 and (best_d2 is None or d2 < best_d2):
                best_d2 = d2
                best_j = k_j
        if best_j >= 0:
            paired[k_i] = 1
            paired[best_j] = 1
            pairs.append((k_i, best_j))
    return paired, pairs


def _kabsch_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(R, t)`` such that ``src @ R.T + t`` best-aligns ``src``
    onto ``dst`` (same shape ``(n, 3)``). Standard Kabsch; works for
    any ``n >= 1`` but requires ``n >= 3`` for a unique rotation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    # Handle reflection.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(3)
    D[2, 2] = d
    R = Vt.T @ D @ U.T
    t = dst_c - src_c @ R.T
    return R.astype(np.float32), t.astype(np.float32)


def _is_leaf_atom_for_bt(bt: Any) -> np.ndarray:
    """Return ``bool[n_atoms]`` marking each atom as "leaf" in the icoor
    tree (i.e. not a parent of any other atom).

    Mirrors upstream's ``_annotate_block_type_atom_is_leaf_atom``:
    an atom is a leaf iff nothing else uses it as a parent/grandparent.
    We also mark any atom that serves as the fourth atom of a named
    dihedral (e.g. OE1 on GLU) as non-leaf so the upstream behaviour of
    refusing to build a missing chi-defining atom is preserved -- but
    only for heavy atoms; hydroxyl hydrogens are still leaves.
    """
    atoms = list(bt.atoms)
    n = len(atoms)
    atom_name_to_idx = {a.name: i for i, a in enumerate(atoms)}

    is_parent = np.zeros(n, dtype=bool)
    icoors = list(getattr(bt, "icoors", ()) or ())
    # Only the *immediate* parent link makes an atom non-leaf (mirrors
    # upstream's ``icoors_ancestors[:, 0]`` indexing). Grandparent /
    # great-grandparent references are just coordinate-frame anchors
    # and don't imply that the referenced atom has a dependent.
    for ic in icoors:
        p = getattr(ic, "parent", None)
        if p is None:
            continue
        # Skip self-referential root records (parent == name).
        if p == getattr(ic, "name", None):
            continue
        if p in atom_name_to_idx:
            is_parent[atom_name_to_idx[p]] = True

    # Fourth torsion atom on dihedrals (GLU OE1 / ASP OD1 / ...); upstream
    # counts these as non-leaf because rebuilding them with a 180-degree
    # guess gives nonsense energies. Hydrogens are the exception because
    # the hydroxyl-dihedral build is already fuzzy.
    torsions = getattr(bt, "torsions", None) or getattr(bt, "ordered_torsions", None)
    is_hydrogen = np.array(
        [
            (getattr(a, "element", None) == "H")
            or (getattr(a, "atom_type", "") in ("Hpol", "Hapo", "HN", "Haro"))
            # fallback: atom name starts with H (robust across type-lib variants)
            or str(getattr(a, "name", "")).startswith("H")
            for a in atoms
        ],
        dtype=bool,
    )
    if torsions is not None:
        for tors in torsions:
            fourth = getattr(tors, "d", None)
            if fourth is None:
                continue
            fname = getattr(fourth, "atom", None) or getattr(fourth, "name", None)
            if fname is not None and fname in atom_name_to_idx:
                idx = atom_name_to_idx[fname]
                if not is_hydrogen[idx]:
                    is_parent[idx] = True

    is_leaf = np.logical_not(is_parent)
    # Fewer than 3 atoms -> we can't build anything from ideal coords.
    if n <= 3:
        is_leaf[:] = False
    return is_leaf


def _fill_missing_from_ideal(
    ideal: np.ndarray,
    actual: np.ndarray,
    present_mask: np.ndarray,
) -> np.ndarray:
    """Build a filled-in coord array aligned with ``actual`` where
    ``present_mask`` is True, and aligned-ideal values elsewhere.

    ``ideal``, ``actual`` both shape ``(n, 3)``; ``present_mask`` is
    ``bool[n]``. When there are < 3 present atoms, the returned
    coord for absent atoms is ``actual`` (i.e. unchanged -- leaves
    the zero fill in place) so callers can decide to fall back.
    """
    filled = actual.copy()
    ok = np.where(present_mask)[0]
    if ok.shape[0] >= 3:
        R, t = _kabsch_align(ideal[ok], actual[ok])
        missing = np.where(~present_mask)[0]
        if missing.size > 0:
            filled[missing] = ideal[missing] @ R.T + t
    return filled


def _fill_missing_from_icoors(
    bt: Any,
    actual: np.ndarray,
    present_mask: np.ndarray,
    conn_atom_xyz: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build missing atoms from the icoor tree using present parent /
    grand-parent / great-grand-parent coordinates.

    This mirrors upstream's ``gen_pose_leaf_atoms``: for each missing
    atom, fetch the placed coordinates of its three ancestors (parent,
    grand_parent, great_grand_parent) and apply ``build_coordinate`` with
    the atom's stored ``(phi, theta, d)``. Atoms whose ancestors are all
    present are built in a single sweep; atoms whose ancestors are
    themselves missing can still be built in subsequent passes after
    their immediate ancestor has been placed (H1 -> H2 -> H3 on
    MET:nterm is the canonical example).

    When an ancestor name is a connection name (e.g. ``"down"`` for the
    previous residue's C), the optional ``conn_atom_xyz`` dict supplies
    the cross-block coordinate. This is the difference between
    backbone-H placement working (an ALA at the middle of a chain has
    a real prev-C to reference) and failing (the first residue in a
    chain has no prev-C; we mark the H as "leaf" and let it fall
    through to Kabsch alignment, which gives a sane geometry).

    Returns ``(new_coords, newly_filled_mask)``. Atoms whose ancestors
    can't be fully resolved are left unchanged.
    """
    from tmol_jax.chemical.ideal_coords import frame_from_coords, rot_x, rot_z, trans_z

    filled = actual.copy()
    n_bt = filled.shape[0]
    newly_filled = np.zeros(n_bt, dtype=bool)

    # Collect icoor records keyed by atom name.
    icoors = list(getattr(bt, "icoors", ()) or ())
    atoms = list(bt.atoms)
    atom_name_to_idx = {a.name: i for i, a in enumerate(atoms)}

    ic_for_atom: Dict[int, Any] = {}
    for ic in icoors:
        nm = getattr(ic, "name", None)
        if nm is not None and nm in atom_name_to_idx:
            ic_for_atom.setdefault(atom_name_to_idx[nm], ic)

    def _ancestor_xyz(name: str, placed_mask: np.ndarray) -> Optional[np.ndarray]:
        """Resolve an ancestor name to a 3-vec, or None if unresolvable.

        Two cases:
        1. ``name`` is an atom name in this block — read from ``filled``
           if the atom is currently placed.
        2. ``name`` is a connection name (e.g. ``"down"`` or ``"up"``) —
           look it up in ``conn_atom_xyz``, which the caller pre-fills
           with the corresponding cross-block atom position.
        """
        idx = atom_name_to_idx.get(name, -1)
        if idx >= 0:
            if placed_mask[idx]:
                return filled[idx, :3].astype(np.float32)
            return None
        if conn_atom_xyz is not None and name in conn_atom_xyz:
            xyz = conn_atom_xyz[name]
            if xyz is not None:
                return np.asarray(xyz, dtype=np.float32)
        return None

    # A simple iterative sweep: in each pass, try to build each missing
    # atom whose three ancestors are now placed. Loop until no atom
    # makes progress. N passes is O(tree depth), small in practice.
    def _place_one(i: int) -> bool:
        ic = ic_for_atom.get(i)
        if ic is None:
            return False
        p_name = getattr(ic, "parent", None)
        gp_name = getattr(ic, "grand_parent", None)
        ggp_name = getattr(ic, "great_grand_parent", None)
        if p_name is None or gp_name is None or ggp_name is None:
            return False
        # Root: parent == name (only N on the block-type root) -- nothing to do.
        if p_name == getattr(ic, "name", None):
            return False
        placed_mask = present_mask | newly_filled
        p_xyz = _ancestor_xyz(p_name, placed_mask)
        gp_xyz = _ancestor_xyz(gp_name, placed_mask)
        ggp_xyz = _ancestor_xyz(ggp_name, placed_mask)
        if p_xyz is None or gp_xyz is None or ggp_xyz is None:
            return False
        phi = float(getattr(ic, "phi", 0.0))
        theta = float(getattr(ic, "theta", 0.0))
        d = float(getattr(ic, "d", 0.0))
        ht = frame_from_coords(ggp_xyz, gp_xyz, p_xyz)
        ht = ht @ rot_z(phi) @ rot_x(-theta) @ trans_z(d)
        filled[i, 0] = ht[0, 3]
        filled[i, 1] = ht[1, 3]
        filled[i, 2] = ht[2, 3]
        newly_filled[i] = True
        return True

    progress = True
    while progress:
        progress = False
        for i in range(n_bt):
            if present_mask[i] or newly_filled[i]:
                continue
            if _place_one(i):
                progress = True

    return filled, newly_filled


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #


def pose_stack_from_canonical_form(
    canonical_form_dict: Dict[str, Any],
    packed_block_types: Any,
    chain_id: Optional[Sequence[Sequence[Any]]] = None,
    res_num: Optional[Sequence[Sequence[int]]] = None,
) -> Any:
    """Build a :class:`PoseStack` from a canonical-form dict.

    See the module docstring for the expected keys. Returns a
    :class:`tmol_jax.pose.pose_stack.PoseStack` with
    ``inter_residue_connections`` and ``inter_block_bondsep`` populated
    via :meth:`PoseStackBuilder.from_block_types_and_coords`'s chain
    adjacency approximation.

    Termini detection uses ``canonical_form_dict["chain_id"]`` when
    present; otherwise falls back to the ``chain_id`` kwarg. If neither
    is supplied, every residue is treated as belonging to the same
    chain "A" and only the first / last overall residues are tagged as
    terminal.
    """
    # Local imports so this module imports cleanly even before sibling
    # ports finish populating.
    from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

    pbt = packed_block_types
    cf = canonical_form_dict

    # ------------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------------
    coords = np.asarray(cf["coords"], dtype=np.float32)
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(
            f"coords must be [n_poses, max_n_res, max_n_can_atoms, 3]; "
            f"got shape {coords.shape}."
        )
    n_poses, max_n_res, max_n_can_atoms, _ = coords.shape

    # atom_is_present falls back to "is finite / not NaN".
    if "atom_is_present" in cf and cf["atom_is_present"] is not None:
        atom_is_present = np.asarray(cf["atom_is_present"], dtype=bool)
        if atom_is_present.shape != (n_poses, max_n_res, max_n_can_atoms):
            raise ValueError(
                "atom_is_present shape mismatches coords; "
                f"got {atom_is_present.shape} vs "
                f"({n_poses}, {max_n_res}, {max_n_can_atoms})."
            )
    else:
        atom_is_present = np.all(np.isfinite(coords), axis=-1)

    res_names = _normalise_res_names(cf, n_poses, max_n_res)

    bt_name_index = _bt_name_index(pbt)
    name3_to_bt = _bt_name3_index(pbt)
    variant_lookup = _build_variant_lookup(pbt, bt_name_index)
    name3_spcase_lookup = _build_name3_special_case_lookup(pbt)

    n_atoms_per_bt = np.asarray(pbt.n_atoms, dtype=np.int32)

    # Canonical ordering (for per-residue atom names / aliases). Prefer
    # an explicit override from the caller.
    co = cf.get("canonical_ordering", None)
    if co is None:
        from tmol_jax.io.canonical_ordering import default_canonical_ordering

        co = default_canonical_ordering()
    co_atom_names_map = getattr(co, "restypes_ordered_atom_names", None)
    # CanonicalOrdering stores aliases as part of the index mapping, not
    # as a standalone alt_name->canonical dict. Construct one now.
    co_alias_maps: Dict[str, Dict[str, str]] = {}
    if co_atom_names_map is not None:
        idx_map = getattr(co, "restypes_atom_index_mapping", {})
        for n3, names in co_atom_names_map.items():
            name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(names)}
            aliases: Dict[str, str] = {}
            for alt_name, idx in idx_map.get(n3, {}).items():
                if alt_name in name_to_idx:
                    continue  # primary canonical name
                if 0 <= idx < len(names):
                    aliases[alt_name] = names[idx]
            co_alias_maps[n3] = aliases

    # Cache per-variant "variant-atom -> canonical-atom" index maps so
    # we only compute them once per (name3, variant_tag) pair.
    canon_map_cache: Dict[int, np.ndarray] = {}

    def _map_for_bt(bt_ind: int) -> np.ndarray:
        cached = canon_map_cache.get(bt_ind)
        if cached is not None:
            return cached
        bt = pbt.active_block_types[bt_ind]
        n3 = getattr(bt, "name3", None) or getattr(bt, "name", "")
        co_names = (
            co_atom_names_map.get(n3, tuple())
            if co_atom_names_map is not None
            else tuple()
        )
        co_aliases = co_alias_maps.get(n3, {})
        bt_atom_names = tuple(a.name for a in bt.atoms)
        bt_aliases = dict(getattr(bt, "aliases_map", {}) or {})
        m = _canonical_to_variant_map(co_names, co_aliases, bt_atom_names, bt_aliases)
        canon_map_cache[bt_ind] = m
        return m

    # Cache per-variant ideal coords.
    ideal_cache: Dict[int, np.ndarray] = {}

    def _ideal_for_bt(bt_ind: int) -> np.ndarray:
        cached = ideal_cache.get(bt_ind)
        if cached is not None:
            return cached
        bt = pbt.active_block_types[bt_ind]
        ic = getattr(bt, "ideal_coords", None)
        if ic is None:
            from tmol_jax.chemical.ideal_coords import build_ideal_coords

            ic = build_ideal_coords(bt)
        ic_arr = np.asarray(ic, dtype=np.float32)
        # The residue's ideal_coords is defined over all icoors
        # (atoms + connections); take only the first n_atoms rows.
        n_a = len(bt.atoms)
        if ic_arr.shape[0] > n_a:
            ic_arr = ic_arr[:n_a]
        ideal_cache[bt_ind] = ic_arr
        return ic_arr

    # ------------------------------------------------------------------
    # Per-pose chain id lookup (for termini detection).
    # ------------------------------------------------------------------
    cf_chain_id_arr: Optional[np.ndarray] = None
    if "chain_id" in cf and cf["chain_id"] is not None:
        cf_chain_id_arr = np.asarray(cf["chain_id"], dtype=np.int64)
        if cf_chain_id_arr.ndim == 1:
            cf_chain_id_arr = cf_chain_id_arr[None, :]
        if cf_chain_id_arr.shape[0] != n_poses:
            cf_chain_id_arr = None

    # ------------------------------------------------------------------
    # Resolve block_type_ind per (pose, residue)
    # ------------------------------------------------------------------
    block_type_inds_per_pose: List[np.ndarray] = []
    coords_per_pose: List[np.ndarray] = []
    chain_per_pose: List[List[Any]] = []
    resnum_per_pose: List[List[int]] = []
    res_name_per_pose: List[List[str]] = []
    # Disulfide pairs collected per pose: list-of-tuples ``(k_i, k_j)``
    # giving the two CYD half indices in the *real-residue* list (i.e.
    # the same index space as ``block_type_inds_per_pose[p]``). Passed
    # through to PoseStackBuilder so the dslf inter-residue connection
    # gets wired before bondsep is computed.
    disulfide_pairs_per_pose: List[List[Tuple[int, int]]] = []

    chain_input_is_per_pose = chain_id is not None and len(chain_id) == n_poses
    resnum_input_is_per_pose = res_num is not None and len(res_num) == n_poses

    def _default_chain_row(p: int, n_real: int) -> List[str]:
        return ["A"] * n_real

    def _default_resnum_row(p: int, n_real: int) -> List[int]:
        return list(range(1, n_real + 1))

    for p in range(n_poses):
        # -- First pass: collect the 'real' residue indices (those with
        #    a known name3) and their per-residue chain id.
        real_r_inds: List[int] = []
        real_chain_ids: List[int] = []
        real_name3s: List[str] = []
        for r in range(max_n_res):
            name3 = (
                res_names[p][r].strip()
                if res_names[p] and r < len(res_names[p])
                else ""
            )
            if name3 == "" or name3 not in name3_to_bt:
                continue
            real_r_inds.append(r)
            real_name3s.append(name3)
            if cf_chain_id_arr is not None:
                real_chain_ids.append(int(cf_chain_id_arr[p, r]))
            else:
                real_chain_ids.append(0)

        real_res_count = len(real_r_inds)

        # -- Determine termini flags per real residue index.
        is_nterm_flags: List[bool] = [False] * real_res_count
        is_cterm_flags: List[bool] = [False] * real_res_count
        if real_res_count > 0:
            is_nterm_flags[0] = True
            is_cterm_flags[-1] = True
            for k in range(1, real_res_count):
                if real_chain_ids[k] != real_chain_ids[k - 1]:
                    is_nterm_flags[k] = True
                    is_cterm_flags[k - 1] = True

        # -- Special-case variant selection: HIS tautomer, disulfide CYD.
        # Defaults to the name3's "default" base (CYS, HIS, ALA, ...).
        spcase_key_per_k: List[str] = ["default"] * real_res_count

        # HIS tautomer resolution -- mirrors upstream resolve_his_tautomerization.
        his_co_names = (
            co_atom_names_map.get("HIS", tuple()) if co_atom_names_map else tuple()
        )
        for k in range(real_res_count):
            if real_name3s[k] != "HIS":
                continue
            r = real_r_inds[k]
            spcase_key_per_k[k] = _resolve_his_spcase(
                "HIS", his_co_names, atom_is_present[p, r], coords[p, r]
            )

        # Disulfide resolution -- mirrors upstream find_disulfides.
        cys_co_names = (
            co_atom_names_map.get("CYS", tuple()) if co_atom_names_map else tuple()
        )
        disulfide_flags, disulfide_pairs_p = _resolve_disulfide_pairs(
            real_r_inds, real_name3s, cys_co_names, atom_is_present, coords, p
        )
        for k, flag in enumerate(disulfide_flags):
            if flag:
                spcase_key_per_k[k] = "cyd"

        # -- Pick block-type index per real residue (variant-aware).
        bt_inds_p: List[int] = []
        for k in range(real_res_count):
            n3 = real_name3s[k]
            # Resolve base name (HIS / HIS_D / HIS_POS / CYS / CYD / ...).
            spcase_entry = name3_spcase_lookup.get(n3, {})
            base_name = spcase_entry.get(
                spcase_key_per_k[k], spcase_entry.get("default", n3)
            )
            variant_dict = variant_lookup.get(base_name, {})
            chosen = _variant_bt_for_terminus(
                variant_dict, is_nterm_flags[k], is_cterm_flags[k]
            )
            if chosen is None:
                # Fall back to the base (mid-chain) variant of this base_name.
                chosen = variant_dict.get("", None)
            if chosen is None:
                # Last-resort fallback: the default base for this name3.
                chosen = name3_to_bt[n3]
            bt_inds_p.append(int(chosen))

        # -- Pack atoms for real residues into a flat [sum(n_atoms), 3]
        #    array. Use ``variant_to_canonical`` mapping so variant
        #    atom order is honoured, and fill missing atoms from the
        #    variant's ideal coords aligned onto the available backbone.
        n_atoms_this_pose = int(sum(n_atoms_per_bt[bi] for bi in bt_inds_p))
        coords_flat = np.zeros((n_atoms_this_pose, 3), dtype=np.float32)

        # Collect non-leaf-missing errors so we can raise once with the
        # full list (matches upstream's build_missing_leaf_atoms error
        # formatting).
        nonleaf_err_msgs: List[str] = []

        # Cache of leaf-atom masks keyed by block-type index.
        leaf_mask_cache: Dict[int, np.ndarray] = {}

        def _leaf_mask_for_bt(bt_ind: int) -> np.ndarray:
            cached = leaf_mask_cache.get(bt_ind)
            if cached is not None:
                return cached
            bt = pbt.active_block_types[bt_ind]
            mask = _is_leaf_atom_for_bt(bt)
            leaf_mask_cache[bt_ind] = mask
            return mask

        # First pass: collect each block's variant-mapped atoms + present
        # mask, **without** filling missing atoms yet. We need every
        # block's atoms placed (or absent) before we can resolve
        # cross-block ``down`` / ``up`` references during icoor build.
        block_var_coords: List[np.ndarray] = []
        block_var_present: List[np.ndarray] = []
        block_atom_name_to_idx: List[Dict[str, int]] = []
        for k in range(real_res_count):
            r = real_r_inds[k]
            bt_ind = bt_inds_p[k]
            n_bt_atoms = int(n_atoms_per_bt[bt_ind])
            bt = pbt.active_block_types[bt_ind]
            bt_name_str = str(getattr(bt, "name", ""))

            var_to_can = _map_for_bt(bt_ind)  # variant_slot -> canonical_slot
            var_present = np.zeros(n_bt_atoms, dtype=bool)
            var_coords = np.zeros((n_bt_atoms, 3), dtype=np.float32)
            for j in range(n_bt_atoms):
                can_a = int(var_to_can[j]) if j < len(var_to_can) else -1
                if can_a < 0 or can_a >= max_n_can_atoms:
                    continue
                if atom_is_present[p, r, can_a]:
                    var_present[j] = True
                    var_coords[j, 0] = coords[p, r, can_a, 0]
                    var_coords[j, 1] = coords[p, r, can_a, 1]
                    var_coords[j, 2] = coords[p, r, can_a, 2]

            # Validate that all non-leaf atoms are present.
            leaf_mask = _leaf_mask_for_bt(bt_ind)[:n_bt_atoms]
            missing_mask = ~var_present
            nonleaf_missing = np.logical_and(missing_mask, ~leaf_mask)
            if np.any(nonleaf_missing):
                bt_atoms = list(bt.atoms)
                for j in np.where(nonleaf_missing)[0]:
                    at_name = bt_atoms[j].name if j < len(bt_atoms) else f"atom_{j}"
                    nonleaf_err_msgs.append(
                        f"Error: missing non-leaf atom {at_name} on residue {k} "
                        f"{bt_name_str} on pose {p} real res? True"
                    )

            block_var_coords.append(var_coords)
            block_var_present.append(var_present)
            block_atom_name_to_idx.append({a.name: i for i, a in enumerate(bt.atoms)})

        # Second pass: fill missing atoms with cross-block-aware icoor
        # resolution, then pack into the flat output.
        idx_ptr = 0
        for k in range(real_res_count):
            bt_ind = bt_inds_p[k]
            n_bt_atoms = int(n_atoms_per_bt[bt_ind])
            bt = pbt.active_block_types[bt_ind]

            var_coords = block_var_coords[k]
            var_present = block_var_present[k]

            # If any variant atoms are missing (most often H1/H2/H3 or
            # OXT) fill them in. Primary path: icoor-tree build (mirrors
            # upstream's ``gen_pose_leaf_atoms``); fallback: Kabsch-align
            # the ideal coords onto the present frame.
            if not var_present.all():
                # Resolve cross-block connection-atom xyzs for this block.
                # ``down`` -> previous residue's "up" connection atom
                # (typically C of the prev residue for amino acids).
                # ``up``   -> next residue's "down" connection atom
                # (typically N of the next residue).
                conn_atom_xyz: Dict[str, np.ndarray] = {}
                # ``down`` connection: previous-residue's "up" atom.
                # Only valid when (k > 0) AND not an n-terminal break
                # (i.e. is_nterm_flags[k] is False because we set it on
                # chain breaks too).
                if k > 0 and not is_nterm_flags[k]:
                    prev_bt = pbt.active_block_types[bt_inds_p[k - 1]]
                    up_conn = None
                    for c in getattr(prev_bt, "connections", ()) or ():
                        if getattr(c, "name", None) == "up":
                            up_conn = c
                            break
                    if up_conn is not None:
                        prev_at_name = getattr(up_conn, "atom", None)
                        prev_atom_to_idx = block_atom_name_to_idx[k - 1]
                        if prev_at_name in prev_atom_to_idx:
                            prev_idx = prev_atom_to_idx[prev_at_name]
                            if block_var_present[k - 1][prev_idx]:
                                conn_atom_xyz["down"] = block_var_coords[k - 1][
                                    prev_idx
                                ]
                if k < real_res_count - 1 and not is_cterm_flags[k]:
                    next_bt = pbt.active_block_types[bt_inds_p[k + 1]]
                    down_conn = None
                    for c in getattr(next_bt, "connections", ()) or ():
                        if getattr(c, "name", None) == "down":
                            down_conn = c
                            break
                    if down_conn is not None:
                        next_at_name = getattr(down_conn, "atom", None)
                        next_atom_to_idx = block_atom_name_to_idx[k + 1]
                        if next_at_name in next_atom_to_idx:
                            next_idx = next_atom_to_idx[next_at_name]
                            if block_var_present[k + 1][next_idx]:
                                conn_atom_xyz["up"] = block_var_coords[k + 1][next_idx]

                # First try icoor-based build (matches upstream exactly).
                var_coords, newly_filled = _fill_missing_from_icoors(
                    bt, var_coords, var_present, conn_atom_xyz=conn_atom_xyz
                )
                var_present = var_present | newly_filled
                # Any residual missing atoms fall back to Kabsch-align.
                if not var_present.all() and var_present.sum() >= 3:
                    ideal = _ideal_for_bt(bt_ind)
                    if ideal.shape[0] >= n_bt_atoms:
                        var_coords = _fill_missing_from_ideal(
                            ideal[:n_bt_atoms], var_coords, var_present
                        )

                # Keep our cached entries in sync so a later block
                # querying ``conn_atom_xyz["down"]`` against this block's
                # "up" connection atom sees the filled value (rarely
                # matters because heavy ``up`` atoms are always present,
                # but defensive).
                block_var_coords[k] = var_coords
                block_var_present[k] = var_present

            coords_flat[idx_ptr : idx_ptr + n_bt_atoms] = var_coords
            idx_ptr += n_bt_atoms

        if nonleaf_err_msgs:
            raise ValueError("\n".join(nonleaf_err_msgs))

        block_type_inds_per_pose.append(np.asarray(bt_inds_p, dtype=np.int32))
        coords_per_pose.append(coords_flat)

        # Chain / resnum rows: derive from cf preferentially, else kwargs,
        # else synthesise.
        if "chain_labels" in cf and cf["chain_labels"] is not None:
            cl_raw = cf["chain_labels"]
            cl_arr = np.asarray(cl_raw, dtype=object)
            if cl_arr.ndim == 1:
                cl_arr = cl_arr[None, :]
            chain_row = [str(cl_arr[p, ri]) for ri in real_r_inds]
        elif chain_input_is_per_pose:
            row = list(chain_id[p])
            if len(row) < real_res_count:
                row = row + ["A"] * (real_res_count - len(row))
            chain_row = row[:real_res_count]
        else:
            chain_row = _default_chain_row(p, real_res_count)
        chain_per_pose.append(chain_row)

        if "res_num" in cf and cf["res_num"] is not None:
            rn_raw = np.asarray(cf["res_num"])
            if rn_raw.ndim == 1:
                rn_raw = rn_raw[None, :]
            resnum_row = [int(rn_raw[p, ri]) for ri in real_r_inds]
        elif resnum_input_is_per_pose:
            row = list(res_num[p])
            if len(row) < real_res_count:
                row = row + list(range(len(row) + 1, real_res_count + 1))
            resnum_row = [int(x) for x in row[:real_res_count]]
        else:
            resnum_row = _default_resnum_row(p, real_res_count)
        resnum_per_pose.append(resnum_row)

        res_name_per_pose.append(list(real_name3s))
        disulfide_pairs_per_pose.append(disulfide_pairs_p)

    # ------------------------------------------------------------------
    # Delegate to PoseStackBuilder.from_block_types_and_coords
    # ------------------------------------------------------------------
    # Score only residue types used by this batch. All-palette pair setup is
    # quadratic in the number of types and otherwise retains unused parameters.
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    used = np.unique(np.concatenate(block_type_inds_per_pose))
    remap = np.full(len(pbt.active_block_types), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    pbt = PackedBlockTypes.from_restypes(
        pbt.chem_db, [pbt.active_block_types[i] for i in used]
    )
    block_type_inds_per_pose = [remap[indices] for indices in block_type_inds_per_pose]

    return PoseStackBuilder.from_block_types_and_coords(
        pbt=pbt,
        block_type_inds_per_pose=block_type_inds_per_pose,
        coords_per_pose=coords_per_pose,
        chain_id_per_pose=chain_per_pose,
        res_num_per_pose=resnum_per_pose,
        res_name_per_pose=res_name_per_pose,
        disulfide_pairs_per_pose=disulfide_pairs_per_pose,
    )


__all__ = ["pose_stack_from_canonical_form"]
