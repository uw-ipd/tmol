"""SMILES for a residue type, or for one residue of a PoseStack.

Bonds come from the type's Lewis structure (chemical_bond_orders), every
hydrogen is explicit, and formal charges follow from valence. Each
inter-residue connection becomes a [*] attachment point; metal coordination
is left out. Stereochemistry comes from 3D coordinates.
"""

import numpy
from rdkit import Chem

from tmol.chemical import BondType, build_ideal_coords, chemical_bond_orders
from tmol.database.chemical import site_connections
from tmol.ligand._rdkit_mol import assign_stereochemistry_from_3d

_PERIODIC_TABLE = Chem.GetPeriodicTable()

_RDKIT_ORDER = {
    "SINGLE": Chem.BondType.SINGLE,
    "DOUBLE": Chem.BondType.DOUBLE,
    "TRIPLE": Chem.BondType.TRIPLE,
    "AROMATIC": Chem.BondType.AROMATIC,
}


def _lewis_charge(atom: Chem.Atom) -> int:
    """Formal charge of an atom whose hydrogens are all explicit, from its valence."""
    bonds = atom.GetBonds()
    if any(b.GetBondType() == Chem.BondType.AROMATIC for b in bonds):
        return 0
    valence = int(round(sum(b.GetBondTypeAsDouble() for b in bonds)))
    allowed = [v for v in _PERIODIC_TABLE.GetValenceList(atom.GetAtomicNum()) if v >= 0]
    if not allowed or valence in allowed:
        return 0
    lower = [v for v in allowed if v < valence]
    if not lower:
        return valence - min(allowed)
    excess = valence - max(lower)
    # Electron-poor elements (B, Al) take a negative charge for an extra bond.
    if _PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum()) < 4:
        return -excess
    return excess


def _recombine_charges(mol: Chem.RWMol) -> None:
    """Turn X+ - Y- into X=Y where X allows the higher valence (P+ - O- -> P=O)."""
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 1:
            continue
        allowed = _PERIODIC_TABLE.GetValenceList(atom.GetAtomicNum())
        valence = int(round(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())))
        if valence + 1 not in allowed:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (
                other.GetFormalCharge() == -1
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                bond.SetBondType(Chem.BondType.DOUBLE)
                atom.SetFormalCharge(0)
                other.SetFormalCharge(0)
                break


def _sanitize(mol: Chem.RWMol) -> Chem.Mol:
    """Sanitize; if an aromatic ring cannot be kekulized, charge one aromatic N:
    +1 if three-connected (imidazolium), -1 if two-connected (imidazolate)."""
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Chem.KekulizeException:
        pass
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic() or atom.GetAtomicNum() != 7:
            continue
        trial = Chem.RWMol(mol)
        trial.GetAtomWithIdx(atom.GetIdx()).SetFormalCharge(
            1 if atom.GetDegree() == 3 else -1
        )
        try:
            Chem.SanitizeMol(trial)
            return trial
        except Chem.KekulizeException:
            continue
    Chem.SanitizeMol(mol)
    return mol


def _clear_resonance_stereo(mol: Chem.Mol) -> None:
    """Unset centers with two terminal neighbors of one element (phosphate,
    carboxylate-like oxygens): Kekule orders make them look distinct."""
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        terminal = [
            n.GetAtomicNum()
            for n in atom.GetNeighbors()
            if n.GetDegree() == 1 and n.GetAtomicNum() > 1
        ]
        if len(terminal) != len(set(terminal)):
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _is_element(symbol: str) -> bool:
    try:
        return _PERIODIC_TABLE.GetAtomicNumber(symbol) > 0
    except RuntimeError:
        return False


def _smiles(block_type, chem_db, atom_coords, attachments, hydrogens, atom_map):
    """SMILES from a block type, its atom coordinates, and (connection index,
    coordinate) for each attachment point."""
    element_for_type = {at.name: at.element for at in chem_db.atom_types}
    virtual = set(block_type.properties.virtual)

    mol = Chem.RWMol()
    index = {}
    positions = []
    for i, atom in enumerate(block_type.atoms):
        element = element_for_type[atom.atom_type]
        if atom.name in virtual or not _is_element(element):
            continue
        rd_atom = Chem.Atom(element)
        rd_atom.SetNoImplicit(True)
        if atom_map and (hydrogens or rd_atom.GetAtomicNum() != 1):
            rd_atom.SetAtomMapNum(i + 1)
        index[atom.name] = mol.AddAtom(rd_atom)
        positions.append(atom_coords[i])

    for key, order in chemical_bond_orders(block_type, chem_db).items():
        a, b = tuple(key)
        if a not in index or b not in index:
            continue
        mol.AddBond(index[a], index[b], _RDKIT_ORDER[order])
        if order == "AROMATIC":
            mol.GetBondBetweenAtoms(index[a], index[b]).SetIsAromatic(True)
            mol.GetAtomWithIdx(index[a]).SetIsAromatic(True)
            mol.GetAtomWithIdx(index[b]).SetIsAromatic(True)

    for conn, xyz in attachments:
        dummy = mol.AddAtom(Chem.Atom(0))
        order = BondType(int(block_type.connection_bond_types[conn])).name
        mol.AddBond(
            index[block_type.connections[conn].atom],
            dummy,
            _RDKIT_ORDER["SINGLE" if order == "AROMATIC" else order],
        )
        positions.append(xyz)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 0:
            atom.SetFormalCharge(_lewis_charge(atom))
    _recombine_charges(mol)
    mol = _sanitize(mol)

    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for i, xyz in enumerate(positions):
        conformer.SetAtomPosition(i, [float(x) for x in xyz])
    mol.AddConformer(conformer, assignId=True)
    assign_stereochemistry_from_3d(mol)
    _clear_resonance_stereo(mol)

    if not hydrogens:
        mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol)
    if not smiles:
        raise ValueError(f"{block_type.name} has no real atoms")
    return smiles


def smiles_from_block_type(
    block_type, chem_db, *, hydrogens: bool = False, atom_map: bool = False
) -> str:
    """Canonical SMILES of a residue type, with stereochemistry from its ideal
    geometry and a [*] for each connection that is not a metal site.

    Args:
        block_type: A RefinedResidueType (patched types describe patched chemistry).
        chem_db: The chemical database the type was built from (element lookup).
        hydrogens: Keep explicit hydrogens in the SMILES.
        atom_map: Number each atom by its block-type atom index + 1 (hydrogens
            only when kept).
    """
    ideal = build_ideal_coords(block_type)
    metal = set(site_connections(block_type))
    attachments = []
    for c, conn in enumerate(block_type.connections):
        if conn.name in metal:
            continue
        icoor = block_type.icoors_index.get(conn.name)
        xyz = numpy.full(3, numpy.nan) if icoor is None else ideal[icoor]
        attachments.append((c, xyz))
    return _smiles(
        block_type,
        chem_db,
        ideal[block_type.at_to_icoor_ind],
        attachments,
        hydrogens,
        atom_map,
    )


def smiles_from_pose_stack(
    pose_stack,
    pose_ind: int,
    block_ind: int,
    *,
    hydrogens: bool = False,
    atom_map: bool = False,
) -> str:
    """Canonical SMILES of one residue of a PoseStack, with stereochemistry from its
    coordinates and a [*] for each connection that is not a metal bond.

    Args:
        pose_stack: The PoseStack.
        pose_ind: Pose index.
        block_ind: Residue (block) index within the pose; must be a real block.
        hydrogens: Keep explicit hydrogens in the SMILES.
        atom_map: Number each atom by its block-type atom index + 1 (hydrogens
            only when kept).
    """
    pbt = pose_stack.packed_block_types
    block_type = pose_stack.block_type(pose_ind, block_ind)
    coords = pose_stack.coords[pose_ind].detach().cpu().numpy()
    offsets = pose_stack.block_coord_offset[pose_ind].tolist()
    block_types = pose_stack.block_type_ind64[pose_ind].tolist()
    partners = pose_stack.inter_residue_connections64[pose_ind, block_ind].tolist()
    offset = offsets[block_ind]

    own_sites = set(site_connections(block_type))
    attachments = []
    for c, conn in enumerate(block_type.connections):
        if conn.name in own_sites:
            continue
        partner, partner_conn = partners[c]
        if partner < 0:
            attachments.append((c, numpy.full(3, numpy.nan)))
            continue
        partner_type = pbt.active_block_types[block_types[partner]]
        if partner_type.connections[partner_conn].name in site_connections(
            partner_type
        ):
            continue
        partner_atom = partner_type.ordered_connection_atoms[partner_conn]
        attachments.append((c, coords[offsets[partner] + partner_atom]))

    return _smiles(
        block_type,
        chem_db=pbt.chem_db,
        atom_coords=coords[offset : offset + block_type.n_atoms],
        attachments=attachments,
        hydrogens=hydrogens,
        atom_map=atom_map,
    )
