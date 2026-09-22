"""Build tmol RawResidueType definitions from RDKit molecules.

Converts a Chem.Mol with assigned atom types into a complete RawResidueType
suitable for registration in tmol's ChemicalDatabase. Handles atom tree
construction, internal coordinate computation, rotatable bond detection,
and non-polymer property assignment.

The atom tree and internal coordinates come from :mod:`atomworks.geometry`,
so a structure keeps the same geometry whichever library placed it.
"""

import logging

import numpy as np
from atomworks.protonation import (
    build_atom_tree,
    find_root_atom,
    icoor_geometry_from_coords,
)
from rdkit import Chem

from tmol.database.chemical import (
    Atom,
    ChemicalProperties,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
)
from tmol.ligand._atom_typing import AtomTypeAssignment, RosettaTypingState
from tmol.ligand._chi_topology import build_chi_topology

logger = logging.getLogger(__name__)


def _mol_coords(mol: Chem.Mol) -> np.ndarray:
    """Return an (N, 3) float array of 3D coordinates from mol's first conformer.

    If the mol has no conformer (e.g. a SMILES-only unit-test input),
    return zeros -- the caller's icoor geometry will be degenerate but the
    atom tree and topology outputs still build correctly.
    """
    n = mol.GetNumAtoms()
    if mol.GetNumConformers() == 0:
        return np.zeros((n, 3), dtype=float)
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


def _connectivity(mol: Chem.Mol) -> tuple[list[tuple[int, int]], list[bool]]:
    """The bonds and heavy-atom flags the shared geometry works from."""
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return bonds, [a.GetAtomicNum() != 1 for a in mol.GetAtoms()]


def _find_nbr_atom(
    mol: Chem.Mol, coords: np.ndarray, skip_indices: set[int] | None = None
) -> int:
    """Find the neighbor atom (root) for the atom tree."""
    bonds, is_heavy = _connectivity(mol)
    return find_root_atom(coords, bonds, is_heavy, skip_indices)


def _build_atom_tree(
    mol: Chem.Mol, root_idx: int, frame_excluded_indices=()
) -> tuple[list[int], dict[int, int], dict[int, tuple[int, int]]]:
    """Build an atom tree via BFS from the root."""
    bonds, is_heavy = _connectivity(mol)
    return build_atom_tree(
        mol.GetNumAtoms(), bonds, is_heavy, root_idx, frame_excluded_indices
    )


def _compute_icoors(
    mol: Chem.Mol,
    order: list[int],
    parent: dict[int, int],
    grandparents: dict[int, tuple[int, int]],
    atom_names: list[str],
    coords: np.ndarray | None = None,
) -> list[Icoor]:
    """Compute internal coordinates for all atoms, in BFS traversal order."""
    if coords is None:
        coords = _mol_coords(mol)
    geometry = icoor_geometry_from_coords(coords, order, parent, grandparents)
    return [
        Icoor(
            name=atom_names[idx],
            phi=geom.phi,
            theta=geom.theta,
            d=geom.d,
            parent=atom_names[parent[idx]],
            grand_parent=atom_names[grandparents[idx][0]],
            great_grand_parent=atom_names[grandparents[idx][1]],
        )
        for idx, geom in zip(order, geometry)
    ]


_AMIDE_N_TYPES = {"Nad", "Nad3"}
_GUANIDINIUM_N_TYPES = {"Ngu1", "Ngu2"}
_PLANAR_N_TYPES = _AMIDE_N_TYPES | _GUANIDINIUM_N_TYPES

# Rosetta convention: rings larger than this are treated as
# non-cyclic for the purposes of the bond's is_in_ring flag.
_MAX_RING_SIZE_TREATED_AS_RING = 8


def _is_in_large_ring(mol: Chem.Mol, bond: Chem.Bond) -> bool:
    """True if every cycle this bond participates in is >8 atoms."""
    if not bond.IsInRing():
        return False
    ri = mol.GetRingInfo()
    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    for ring in ri.AtomRings():
        if a in ring and b in ring and len(ring) <= _MAX_RING_SIZE_TREATED_AS_RING:
            return False
    return True


def _is_planar_resonance_bond(
    atom_a: Chem.Atom, atom_b: Chem.Atom, type_a: str, type_b: str
) -> bool:
    """True iff this N-C bond is the planar partner in an acyclic
    resonance group — i.e. the amide N-C(=O) or the guanidinium N-C(N)(N)
    bond, but NOT other N-C connections incident to the same N.

    Reuses the atom-type classifier's existing detection: ``Nad`` /
    ``Nad3`` mark amide nitrogens and ``Ngu1`` / ``Ngu2`` mark
    guanidinium nitrogens. We only upgrade the specific N-C bond whose
    C is the resonance partner (a carbonyl C for amide; a multi-N C for
    guanidinium); methylene tethers etc. stay SINGLE.
    """
    if type_a in _PLANAR_N_TYPES and atom_b.GetAtomicNum() == 6:
        n_atom, c_atom, n_type = atom_a, atom_b, type_a
    elif type_b in _PLANAR_N_TYPES and atom_a.GetAtomicNum() == 6:
        n_atom, c_atom, n_type = atom_b, atom_a, type_b
    else:
        return False

    other_neighbors = (
        nbr for nbr in c_atom.GetNeighbors() if nbr.GetIdx() != n_atom.GetIdx()
    )
    if n_type in _AMIDE_N_TYPES:
        # C must have a C=O carbonyl on its other side.
        return any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(c_atom).GetAtomicNum() == 8
            for b in c_atom.GetBonds()
            if b.GetOtherAtom(c_atom).GetIdx() != n_atom.GetIdx()
        )
    # guanidinium: C must have ≥2 other N neighbors.
    return sum(1 for nbr in other_neighbors if nbr.GetAtomicNum() == 7) >= 2


def build_residue_type(  # noqa: C901
    mol: Chem.Mol,
    res_name: str,
    atom_types: list[AtomTypeAssignment],
    atom_aliases: tuple = (),
    *,
    typing_state: RosettaTypingState,
    assign_ring_chis: bool = False,
    generate_heavy_chi_samples: bool = False,
    original_single_bonds: frozenset[frozenset[str]] | None = None,
    frame_excluded_atoms: frozenset[str] = frozenset(),
) -> RawResidueType:
    """Build a complete RawResidueType from a Chem.Mol.

    Constructs atoms, bonds, internal coordinates, and non-polymer
    properties suitable for registration in tmol's ChemicalDatabase.

    Atoms with unknown elements (atomic number 0, e.g. metals that
    lost identity during SMILES roundtrip) are silently dropped.

    Args:
        mol: An RDKit Mol with 3D coordinates and bonds.
        res_name: Residue name (e.g. "L_1", "ATP").
        atom_types: Atom type assignments from assign_tmol_atom_types().
        atom_aliases: Optional tuple of AtomAlias for CIF name mapping.
        typing_state: The perception state from the atom type assignment,
            needed for rotatable bond classification.
        assign_ring_chis: Keep single bonds inside a non-aromatic ring as chi,
            the way proline's are. Polymer residues set it; ligands do not.
        generate_heavy_chi_samples: Emit samples for heavy chi.
        original_single_bonds: Bonds the source mol2 records as literal single
            bonds, whose order kekulization must not promote.
        frame_excluded_atoms: Declared leaving atoms that must not root the
            construction frame; their retained neighbors are visited first.

    Returns:
        A fully populated RawResidueType.
    """
    dropped = [at for at in atom_types if at.element == "*"]
    if dropped:
        logger.warning(
            "%s: dropping %d atom(s) with unknown element: %s",
            res_name,
            len(dropped),
            ", ".join(at.atom_name for at in dropped),
        )
        keep_indices = {at.index for at in atom_types if at.element != "*"}
        atom_types = [at for at in atom_types if at.element != "*"]
    else:
        keep_indices = None

    idx_to_name = {at.index: at.atom_name for at in atom_types}
    atom_names = [idx_to_name.get(i) for i in range(mol.GetNumAtoms())]
    atom_type_by_name = {at.atom_name: at.atom_type for at in atom_types}

    atoms = tuple(Atom(name=at.atom_name, atom_type=at.atom_type) for at in atom_types)

    bonds: list[tuple[str, str, str]] = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if atom_names[a] is None or atom_names[b] is None:
            continue

        # Kekulé first — Frank's reference .tmol files emit DOUBLE/SINGLE
        # for ring bonds when they carry a definite bond order, and only
        # fall back to AROMATIC when the bond was never kekulized (e.g.
        # acyclic resonance groups like amide N-C(=O)). Reading the bond
        # order directly catches the kekulized case; the GetIsAromatic
        # branch only fires when the bond type is still AROMATIC at this
        # point, i.e. kekulization was skipped or failed.
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            b_type = "SINGLE"
        elif bt == Chem.BondType.DOUBLE:
            b_type = "DOUBLE"
        elif bt == Chem.BondType.TRIPLE:
            b_type = "TRIPLE"
        elif bt == Chem.BondType.AROMATIC or bond.GetIsAromatic():
            b_type = "AROMATIC"
        else:
            b_type = "SINGLE"

        # Planar acyclic resonance (amide / guanidinium) — RDKit reports
        # the bond as SINGLE because it isn't on a Hückel ring, but the
        # geometry is planar and the downstream scoring treats this as
        # an aromatic bond.
        if b_type == "SINGLE" and _is_planar_resonance_bond(
            mol.GetAtomWithIdx(a),
            mol.GetAtomWithIdx(b),
            atom_type_by_name.get(atom_names[a], ""),
            atom_type_by_name.get(atom_names[b], ""),
        ):
            b_type = "AROMATIC"

        # 4th field: is_in_ring (for scoring; rings >8 atoms count as
        # not-in-a-ring, matching Rosetta convention).
        is_in_ring = bond.IsInRing() and not _is_in_large_ring(mol, bond)
        bonds.append((atom_names[a], atom_names[b], b_type, is_in_ring))

    coords = _mol_coords(mol)
    dropped_indices = (
        (set(range(mol.GetNumAtoms())) - keep_indices) if keep_indices else None
    )
    frame_excluded = {
        i for i, name in enumerate(atom_names) if name in frame_excluded_atoms
    }
    nbr_idx = _find_nbr_atom(
        mol, coords, skip_indices=(dropped_indices or set()) | frame_excluded
    )
    order, parent, grandparents = _build_atom_tree(mol, nbr_idx, frame_excluded)
    if keep_indices is not None:
        order = [i for i in order if i in keep_indices]

    valid_set = set(order)
    for idx in order:
        if parent[idx] not in valid_set:
            parent[idx] = idx
        gp, ggp = grandparents[idx]
        if gp not in valid_set:
            gp = parent[idx]
        if ggp not in valid_set:
            ggp = gp
        grandparents[idx] = (gp, ggp)

    icoors = _compute_icoors(mol, order, parent, grandparents, atom_names, coords)

    # Rotatable-bond (CHI / PROTON_CHI) topology, classified against the
    # perception state atom typing already built.
    atype_by_idx = {at.index: at.atom_type for at in atom_types}
    torsions, chi_samples = build_chi_topology(
        mol,
        order,
        parent,
        grandparents,
        atom_names,
        typing_state,
        atype_by_idx=atype_by_idx,
        original_single_bonds=original_single_bonds,
        assign_ring_chis=assign_ring_chis,
        generate_heavy_chi_samples=generate_heavy_chi_samples,
        logger=logger,
    )
    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=False,
            polymer_type=None,
            backbone_type=None,
            mainchain_atoms=None,
            sidechain_chirality="NA",
            termini_variants=(),
        ),
        chemical_modifications=(),
        connectivity=(),
        protonation=ProtonationProperties(
            protonated_atoms=(),
            protonation_state="neutral",
            pH=7,
        ),
        virtual=(),
    )

    return RawResidueType(
        name=res_name,
        base_name=res_name,
        name3=res_name,
        io_equiv_class=res_name,
        atoms=atoms,
        atom_aliases=atom_aliases,
        bonds=tuple(bonds),
        connections=(),
        torsions=torsions,
        icoors=tuple(icoors),
        properties=properties,
        chi_samples=chi_samples,
        default_jump_connection_atom=atom_names[nbr_idx],
        # Autogen re-protonates: input H names/coords are stale, so pose build
        # must rebuild them (see take_block_type_atoms_from_canonical).
        hydrogens_regenerated=True,
    )
