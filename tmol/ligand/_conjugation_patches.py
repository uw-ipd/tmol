"""Patches giving a residue a connection where a covalent bond attaches.

A glycan's link to its protein, or a ligand's to a sidechain, is a bond the
residue types know nothing about: serine declares no connection at OG, and a
sugar declares none at its hydroxyls. Each attachment site becomes a patch that
removes the hydrogen standing in the bond's place and puts a connection there,
so a residue linked at several sites is the combination of its patches.

Patches name atoms outright rather than matching a pattern: a pattern cannot
tell one hydroxyl of a sugar from another. They are therefore scoped to the one
residue type they were generated for.
"""

import attr

from tmol.database.chemical import (
    Atom,
    ChiSamples,
    Connection,
    IcoorVariant,
    Torsion,
    UnresolvedAtom,
    VariantScope,
    VariantType,
)

CONNECTION_PREFIX = "conj_"

# the bond a conjugation makes is sp3-sp3, so its torsion is sampled
#    staggered, as the proton chi it replaces was
LINKAGE_SAMPLES = (60.0, -60.0, 180.0)


def connection_name(atom: str) -> str:
    """The name a conjugation connection takes at an atom."""
    return CONNECTION_PREFIX + atom


def _element_for_atom(residue_type, chemdb):
    element = {atom_type.name: atom_type.element for atom_type in chemdb.atom_types}
    return {atom.name: element.get(atom.atom_type) for atom in residue_type.atoms}


def _icoor_leaves(residue_type):
    """Atoms no other atom is built from."""
    referenced = {
        ref
        for icoor in residue_type.icoors
        for ref in (icoor.parent, icoor.grand_parent, icoor.great_grand_parent)
    }
    return {atom.name for atom in residue_type.atoms} - referenced


def _hydrogens_on(residue_type, atom: str, chemdb):
    """The hydrogens bonded to ``atom``, by name."""
    element = _element_for_atom(residue_type, chemdb)
    return sorted(
        other
        for first, second, *_ in residue_type.bonds
        for this, other in ((first, second), (second, first))
        if this == atom and element.get(other) == "H"
    )


def _leaves_excluding(residue_type, removed):
    """Atoms nothing still present is built from."""
    referenced = {
        ref
        for icoor in residue_type.icoors
        if icoor.name not in removed
        for ref in (icoor.parent, icoor.grand_parent, icoor.great_grand_parent)
    }
    return {a.name for a in residue_type.atoms if a.name not in removed} - referenced


def leaving_atoms(residue_type, atom: str, chemdb, n_leaving: int = 1):
    """The hydrogens the bond displaces, or None where the site has none.

    Preparation fills the free valence a covalent bond left behind, so whatever
    the site is -- a hydroxyl oxygen, an anomeric carbon -- the atom in the
    bond's way is a hydrogen. How many go depends on what the bond makes of the
    site: acylating a lysine gives an amide, which keeps one of the amine's
    three hydrogens, where etherifying a hydroxyl displaces its only one.

    They are peeled from the tip: each hydrogen taken is one nothing still
    present is built from, since lysine's three are placed one against the next
    and taking the first would leave the others with no frame.
    """
    candidates = _hydrogens_on(residue_type, atom, chemdb)
    if not candidates:
        return None
    removed = []
    for _ in range(max(min(n_leaving, len(candidates)), 1)):
        remaining = [h for h in candidates if h not in removed]
        if not remaining:
            break
        leaves = _leaves_excluding(residue_type, set(removed))
        removed.append(next((h for h in remaining if h in leaves), remaining[0]))
    return tuple(removed)


def leaving_atom(residue_type, atom: str, chemdb) -> str:
    """The single hydrogen standing where the bond attaches, or None."""
    leaving = leaving_atoms(residue_type, atom, chemdb, 1)
    return leaving[0] if leaving else None


def _icoor_for(residue_type, name):
    return next((ic for ic in residue_type.icoors if ic.name == name), None)


def _linkage_torsion(name, frame, atom, chi_name):
    """The torsion the new bond turns, framed as the departing hydrogen was.

    Rotating it swings whatever is attached, so it is not a proton chi: the
    packer owns it, optH does not.
    """
    return (
        Torsion(
            name=chi_name,
            a=UnresolvedAtom(atom=frame.great_grand_parent),
            b=UnresolvedAtom(atom=frame.grand_parent),
            c=UnresolvedAtom(atom=atom),
            d=UnresolvedAtom(connection=name, bond_sep_from_conn=0),
        ),
        ChiSamples(
            chi_dihedral=chi_name,
            samples=LINKAGE_SAMPLES,
            expansions=(),
            is_proton=False,
        ),
    )


def conjugation_patch(
    residue_type,
    atom: str,
    chemdb,
    distance=None,
    chi_name=None,
    n_hydrogens=None,
):
    """A patch replacing an atom's hydrogens with a connection, or None.

    The connection inherits the first hydrogen's internal coordinates, which
    already point along the bond; only the length differs, and the caller
    supplies it where the input measured one.

    ``n_hydrogens`` is how many the site keeps once bonded, as the conjugated
    molecule itself reports it. Without it one hydrogen goes, which is right
    wherever the bond does not change what the site is; acylating an amine
    does change it, and leaving the spare hydrogen behind puts an atom where
    the partner already is.
    """
    present = _hydrogens_on(residue_type, atom, chemdb)
    n_leaving = 1 if n_hydrogens is None else max(len(present) - n_hydrogens, 1)
    leaving_all = leaving_atoms(residue_type, atom, chemdb, n_leaving)
    if not leaving_all:
        return None
    # the connection takes a departing hydrogen's frame, so it has to be one
    #    framed on atoms that stay: the hydrogens are built one against the
    #    next, and the ones peeled first are framed on the ones peeled after
    gone = set(leaving_all)

    def _survives(name):
        ic = _icoor_for(residue_type, name)
        return ic is not None and not (
            {ic.parent, ic.grand_parent, ic.great_grand_parent} & gone
        )

    leaving = next(
        (name for name in reversed(leaving_all) if _survives(name)), leaving_all[-1]
    )
    frame = _icoor_for(residue_type, leaving)
    if frame is None:
        return None

    name = connection_name(atom)
    # a hydroxyl that loses its proton to a bond is an ether; the database
    #    says what each type becomes, or says nothing and it is left alone
    conjugated = {t.name: t.conjugated_type for t in chemdb.atom_types}
    current = next(a.atom_type for a in residue_type.atoms if a.name == atom)
    becomes = conjugated.get(current)
    modify_atoms = (Atom(name=f"<{atom}>", atom_type=becomes),) if becomes else ()
    # the connection takes the departing hydrogen's frame outright, which
    #    already points along the bond; only the length differs. An omitted
    #    field is inherited from the source, so pass nothing else.
    icoors = (IcoorVariant(name=name, source=f"<{leaving}>", d=distance),)

    torsions, chi_samples = (), ()
    if chi_name is not None and frame.grand_parent and frame.great_grand_parent:
        torsion, sample = _linkage_torsion(name, frame, atom, chi_name)
        torsions, chi_samples = (torsion,), (sample,)

    return VariantType(
        name=f"{CONNECTION_PREFIX}{residue_type.base_name}_{atom}",
        display_name=name,
        pattern="",
        remove_atoms=tuple(f"<{name}>" for name in leaving_all),
        add_atoms=(),
        add_atom_aliases=(),
        modify_atoms=modify_atoms,
        add_connections=(Connection(name=name, atom=f"<{atom}>", type="SINGLE"),),
        add_bonds=(),
        icoors=icoors,
        add_torsions=torsions,
        add_chi_samples=chi_samples,
        applies_to=VariantScope(base_names=(residue_type.base_name,)),
    )


def conjugation_patches(residue_type, atoms, chemdb, distances=None, hydrogens=None):
    """One patch per attachment site, skipping sites with nothing to displace.

    Each site's torsion gets a chi number past every one the residue already
    uses, and past the other sites': two patches can apply at once, and a
    number claimed twice would name one torsion twice.
    """
    distances = distances or {}
    taken = {torsion.name for torsion in residue_type.torsions}
    patches = []
    for atom in sorted(atoms):
        number = 1
        while f"chi{number}" in taken:
            number += 1
        chi_name = f"chi{number}"
        taken.add(chi_name)
        patch = conjugation_patch(
            residue_type,
            atom,
            chemdb,
            distances.get(atom),
            chi_name,
            (hydrogens or {}).get(atom),
        )
        if patch is not None:
            patches.append(patch)
    return tuple(patches)


def with_conjugation_patches(chemdb, residue_type, atoms, distances=None):
    """``chemdb`` extended with the patches ``residue_type`` needs."""
    patches = conjugation_patches(residue_type, atoms, chemdb, distances)
    if not patches:
        return chemdb
    return attr.evolve(chemdb, variants=(*chemdb.variants, *patches))


def conjugation_charge_entries(
    residue_type, atoms, chemdb, base_charges, hydrogens=None
):
    """``{variant name: {atom: charge}}`` for a residue's conjugation patches.

    The attachment atom takes the charge of every hydrogen that left, so the
    residue keeps the net charge it had. Nothing else moves: every other atom
    falls back to the unpatched entry, and only one atom per patch needs one.

    A generated component's charges do not sum to an integer to begin with --
    they are MMFF94's -- and conserving the total keeps it that way rather than
    imposing a round number the chemistry does not have.
    """
    entries = {}
    for atom in sorted(atoms):
        present = _hydrogens_on(residue_type, atom, chemdb)
        wanted = (hydrogens or {}).get(atom)
        n_leaving = 1 if wanted is None else max(len(present) - wanted, 1)
        leaving = leaving_atoms(residue_type, atom, chemdb, n_leaving)
        if not leaving:
            continue
        if atom not in base_charges or any(
            name not in base_charges for name in leaving
        ):
            continue
        variant = f"{residue_type.name}:{connection_name(atom)}"
        entries[variant] = {
            atom: base_charges[atom] + sum(base_charges[n] for n in leaving)
        }
    return entries


def charges_for_database_residue(param_db, res_name):
    """The unpatched partial charges of a residue already in the database."""
    return {
        entry.atom: entry.charge
        for entry in param_db.scoring.elec.atom_charge_parameters
        if entry.res == res_name
    }


def conjugated_fragment(residue_array, atom, partner_array, partner_atom):
    """The residue joined to enough of its partner to carry the chemistry.

    One shell of the partner is enough and is needed: whether a lysine stays
    protonated turns on the partner atom's own bonds, since an acyl group
    neutralizes it where a plain carbon does not.
    """
    import numpy

    def heavy(array):
        return array[numpy.array([str(e) != "H" for e in array.element])]

    residue = heavy(residue_array)
    partner = heavy(partner_array)
    names = [str(n) for n in partner.atom_name]
    if partner_atom not in names:
        return None, None
    site = names.index(partner_atom)

    neighbours = {site}
    if partner.bonds is not None:
        for first, second, *_ in partner.bonds.as_array():
            if first == site:
                neighbours.add(int(second))
            elif second == site:
                neighbours.add(int(first))
    stub = partner[numpy.array(sorted(neighbours))]

    residue_names = [str(n) for n in residue.atom_name]
    if atom not in residue_names:
        return None, None
    combined = residue + stub
    stub_site = len(residue) + sorted(neighbours).index(site)
    combined.bonds.add_bond(residue_names.index(atom), stub_site, 1)
    return combined, residue_names.index(atom)


def conjugated_chemistry(residue_array, atom, partner_array, partner_atom, ph=7.4):
    """Protonation, atom type and charges the attachment site takes on.

    Built the way a terminal form is: the conjugated molecule is only ever a
    SMILES, since Dimorphite, the ligand typer and MMFF94 all work from
    topology. Returns None where the molecule cannot be built.
    """
    from rdkit import Chem

    from tmol.ligand._atom_typing import assign_tmol_atom_types, sanitize_tolerant
    from tmol.ligand._detect import _dimorphite_protonate_smiles
    from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array
    from tmol.ligand._terminus_patches import _mmff94_charges

    combined, site_index = conjugated_fragment(
        residue_array, atom, partner_array, partner_atom
    )
    if combined is None:
        return None
    smiles = ligand_smiles_from_atom_array(combined, with_atom_map=True)
    mol = Chem.MolFromSmiles(_dimorphite_protonate_smiles(smiles, ph=ph))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    sanitize_tolerant(mol)
    site = next(
        (a for a in mol.GetAtoms() if a.GetAtomMapNum() == site_index + 1), None
    )
    if site is None:
        return None
    charges = _mmff94_charges(mol, mol.GetNumAtoms())
    types = {t.index: t for t in assign_tmol_atom_types(mol)}
    hydrogens = [n for n in site.GetNeighbors() if n.GetAtomicNum() == 1]
    return {
        "n_hydrogens": len(hydrogens),
        "site_type": types[site.GetIdx()].atom_type if site.GetIdx() in types else None,
        "site_charge": charges[site.GetIdx()] if charges else None,
        "hydrogen_charges": (
            [charges[h.GetIdx()] for h in hydrogens] if charges else []
        ),
    }
