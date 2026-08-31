"""Terminus patches generated for one nonstandard polymer residue.

The database's termini patches are written for an alpha backbone: they assume
the residue's only carbonyl is the one the chain leaves through, that the names
they give the atoms they add are free, and that a chain end is an ammonium and
a carboxylate. None of that holds generally -- a gamma-linked acid keeps an
intact alpha carbonyl called O, and an aromatic amine is not basic enough to be
charged at physiological pH.

A residue whose backbone is not alpha therefore carries its own copies. The
terminal form is built as a molecule and put through the same steps the residue
itself went through: Dimorphite decides its protonation, the ligand typer its
atom types, and MMFF94 its charges. None of those need a conformer, so no
structure is generated for it.
"""

import logging

import math

import attr

from tmol.database.chemical import Atom, VariantScope

logger = logging.getLogger(__name__)

_PLACEHOLDER = "<"

# Bond angle at the atom a terminal group hangs off, by its geometry. The
#    database's patches carry one tetrahedral value because an alpha backbone's
#    amide nitrogen is always sp3 once protonated; a planar nitrogen is not.
_TETRAHEDRAL_ANGLE = 109.5
_PLANAR_ANGLE = 120.0


def _renamed(name, renames):
    """A patch's own atom name, remapped; pattern placeholders pass through."""
    if not isinstance(name, str) or name.startswith(_PLACEHOLDER):
        return name
    return renames.get(name, name)


def _free_name(wanted, taken):
    """``wanted`` if the residue does not use it, else the first free suffix."""
    if wanted not in taken:
        return wanted
    for suffix in range(1, 10):
        candidate = f"{wanted}{suffix}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"no free name for {wanted}")


def terminus_templates(chemdb, profile):
    """The database patches this backbone's chain ends are modelled on.

    Which connection a patch acts on is read from the one it removes, so a
    nucleotide's 5' and 3' ends are found the same way a peptide's two are,
    without either being named here.
    """
    base = profile.terminus_template_backbone
    grouped: dict = {}
    for variant in chemdb.variants:
        types = variant.applies_to.backbone_types
        if types is None or base not in types:
            continue
        removed = " ".join(str(a) for a in variant.remove_atoms)
        if "{down}" in removed:
            connection = profile.down
        elif "{up}" in removed:
            connection = profile.up
        else:
            continue
        if connection is None:
            continue
        grouped.setdefault((variant.display_name, connection), []).append(variant)
    return grouped


def _template_for(residue_type, connection_atom, candidates):
    """Which of several patches sharing a display name this residue takes.

    They are told apart by their pattern -- an amine's from a substituted
    amine's. The pattern is not matched here, so they are told apart by the
    same thing it tests: whether the connection carries a hydrogen.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    hydrogens = {a.name for a in residue_type.atoms if a.atom_type.startswith("H")}
    protonated = any(
        connection_atom in bond[:2] and set(bond[:2]) & hydrogens
        for bond in residue_type.bonds
    )
    # the patch that removes a hydrogen is the one written for a connection
    #    that has one; where none is removed, the connection is substituted
    for candidate in candidates:
        removes_hydrogen = any(
            r.strip("<>").startswith("H") for r in candidate.remove_atoms
        )
        if removes_hydrogen == protonated:
            return candidate
    return candidates[0]


def _rename_patch(
    template,
    residue_type,
    renames,
    name,
    add_atoms,
    chemistry,
    connection_atom,
):
    """``template`` with its added atoms renamed and scoped to one residue."""
    added_names = {a.name for a in add_atoms}
    modify_atoms = _retyped_atoms(
        template, residue_type, chemistry, connection_atom, add_atoms
    )
    return attr.evolve(
        template,
        name=name,
        add_atoms=tuple(
            attr.evolve(a, name=_renamed(a.name, renames)) for a in add_atoms
        ),
        add_atom_aliases=tuple(
            attr.evolve(a, name=_renamed(a.name, renames))
            for a in template.add_atom_aliases
            if a.name in added_names
        ),
        modify_atoms=tuple(modify_atoms),
        add_bonds=tuple(
            tuple(_renamed(x, renames) for x in bond)
            for bond in template.add_bonds
            if all(
                not isinstance(x, str)
                or x.startswith(_PLACEHOLDER)
                or x in added_names
                or x not in {a.name for a in template.add_atoms}
                for x in bond[:2]
            )
        ),
        icoors=tuple(
            attr.evolve(
                ic,
                name=_renamed(ic.name, renames),
                parent=_renamed(ic.parent, renames),
                grand_parent=_renamed(ic.grand_parent, renames),
                great_grand_parent=_renamed(ic.great_grand_parent, renames),
                theta=_terminal_theta(ic.theta, chemistry),
            )
            for ic in template.icoors
            if ic.name.startswith(_PLACEHOLDER) or ic.name in added_names
        ),
        applies_to=VariantScope(base_names=(residue_type.name,)),
    )


def _terminal_theta(theta, chemistry):
    """The icoor angle at the connection, set by whether it is planar.

    An icoor's theta is the supplement of the bond angle it builds, so a
    tetrahedral site's 109.5 degrees is stored as 70.5. The database's patches
    carry the tetrahedral value throughout, which is wrong at a nitrogen that
    stays planar because it is not basic enough to protonate.
    """
    if not chemistry["site_is_planar"]:
        return theta
    return math.radians(180.0 - _PLANAR_ANGLE)


def _connection_placeholder(template, add_atoms):
    """How the template's pattern refers to the atom the group hangs off.

    A patch names atoms of the residue by the placeholders its pattern binds,
    never by their real names, since it has to work on every residue it
    matches. The connection is whatever the added atoms are built against.
    """
    added = {a.name for a in add_atoms}
    for icoor in template.icoors:
        if icoor.name in added and str(icoor.parent).startswith(_PLACEHOLDER):
            return icoor.parent
    return None


def _retyped_atoms(template, residue_type, chemistry, connection_atom, add_atoms):
    """The connection's type, where the terminal form changes it.

    Nothing further out: a terminal form's typing beyond the site is the
    residue's own, and the patch has no business restating it.
    """
    retyped = {
        a.name: attr.evolve(a, atom_type=chemistry["site_type"])
        for a in template.modify_atoms
    }
    if retyped:
        return list(retyped.values())

    current = {a.name: a.atom_type for a in residue_type.atoms}
    if current.get(connection_atom) == chemistry["site_type"]:
        return []
    placeholder = _connection_placeholder(template, add_atoms)
    if placeholder is None:
        return []
    return [Atom(name=placeholder, atom_type=chemistry["site_type"])]


def _cap_subtree(profile, root):
    """The cap atoms hanging off ``root``, including it."""
    if root is None:
        return set()
    by_parent = {}
    for cap in profile.caps:
        by_parent.setdefault(cap.bond_to, []).append(cap.name)
    reached, stack = set(), [root]
    while stack:
        name = stack.pop()
        if name in reached:
            continue
        reached.add(name)
        stack.extend(by_parent.get(name, ()))
    return reached


def terminal_cap_profile(profile, chemdb, template, terminus):
    """``profile`` with one connection capped as a chain end, not as a stub.

    A residue at a terminus is a different molecule from the same residue in a
    chain. The stub standing in for the absent neighbour is replaced by what
    the terminus patch puts there: an acid's oxygen where the patch adds one,
    and nothing where it adds only hydrogens, since protonation supplies those.
    """
    partner = profile.down_partner if terminus == "nterm" else profile.up_partner
    if partner is None:
        return None
    elements = {at.name: at.element for at in chemdb.atom_types}
    added = {elements.get(a.atom_type, "") for a in template.add_atoms}
    dropped = _cap_subtree(profile, partner)
    kept = [cap for cap in profile.caps if cap.name not in dropped]
    heavy = sorted(added - {"H"})
    if heavy:
        if len(heavy) != 1:
            return None
        stub = next(cap for cap in profile.caps if cap.name == partner)
        kept.append(attr.evolve(stub, element=heavy[0], bond_order="SINGLE"))
    return attr.evolve(profile, caps=tuple(kept))


def _terminal_chemistry(
    atom_array, residue_type, profile, chemdb, template, terminus, ph
):
    """Protonation, atom types and charges of the residue's terminal form.

    The molecule is only ever a SMILES here: Dimorphite, the ligand typer and
    MMFF94 all work from topology, so nothing is embedded.
    """
    from rdkit import Chem

    from tmol.ligand._atom_typing import assign_tmol_atom_types, sanitize_tolerant
    from tmol.ligand._detect import _dimorphite_protonate_smiles
    from tmol.ligand._polymer_profile import cap_residue
    from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array

    terminal = terminal_cap_profile(profile, chemdb, template, terminus)
    if terminal is None:
        return None
    connection = profile.down if terminus == "nterm" else profile.up
    capped, _cap_names = cap_residue(atom_array, terminal)

    names = [str(n) for n in capped.atom_name]
    if connection[1] not in names:
        return None
    connection_index = names.index(connection[1])

    smiles = ligand_smiles_from_atom_array(capped, with_atom_map=True)
    mol = Chem.MolFromSmiles(_dimorphite_protonate_smiles(smiles, ph=ph))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    sanitize_tolerant(mol)
    types = assign_tmol_atom_types(mol)
    charges = _mmff94_charges(Chem.MolToSmiles(mol), mol.GetNumAtoms())
    if charges is None:
        return None

    by_index = {t.index: t for t in types}
    site = next(
        (a for a in mol.GetAtoms() if a.GetAtomMapNum() == connection_index + 1), None
    )
    if site is None:
        return None
    hydrogens = [n for n in site.GetNeighbors() if n.GetAtomicNum() == 1]
    element_of = {at.name: at.element for at in chemdb.atom_types}
    return {
        "measured": True,
        "element_of": element_of,
        "added_elements": {
            element_of.get(a.atom_type, "")
            for a in template.add_atoms
            if element_of.get(a.atom_type, "") not in ("", "H")
        },
        "n_hydrogens": len(hydrogens),
        "hydrogen_type": (
            by_index[hydrogens[0].GetIdx()].atom_type if hydrogens else None
        ),
        "hydrogen_charge": (charges[hydrogens[0].GetIdx()] if hydrogens else 0.0),
        "site_type": by_index[site.GetIdx()].atom_type,
        "site_charge": charges[site.GetIdx()],
        "site_is_planar": _is_planar(site),
        "heavy": _terminal_heavy_atoms(
            site, names, by_index, charges, {a.name for a in residue_type.atoms}
        ),
        **_charges_by_atom(mol, names, charges),
    }


def _charges_by_atom(mol, names, charges):
    """The terminal molecule's charges, keyed by what the residue calls its atoms.

    Its hydrogens were all regenerated, so they have no names of their own and
    are keyed by the heavy atom they hang off. Hydrogens on one atom are
    interchangeable, so the order within a parent does not matter.
    """
    heavy_charge, hydrogen_charge = {}, {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            parents = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            if not parents:
                continue
            source = parents[0].GetAtomMapNum() - 1
            if 0 <= source < len(names):
                hydrogen_charge.setdefault(names[source], []).append(
                    charges[atom.GetIdx()]
                )
            continue
        source = atom.GetAtomMapNum() - 1
        if 0 <= source < len(names):
            heavy_charge[names[source]] = charges[atom.GetIdx()]
    return {"heavy_charge": heavy_charge, "hydrogen_charge_by_parent": hydrogen_charge}


def _is_planar(atom):
    """Whether the atom's geometry is trigonal rather than tetrahedral."""
    from rdkit import Chem

    return atom.GetHybridization() in (
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP,
    )


def _terminal_heavy_atoms(site, names, by_index, charges, base_names):
    """The heavy atoms bonded to the connection, typed and charged.

    ``in_base`` says whether the residue already has the atom: the ones it does
    not are what the patch adds, and the ones it does may be retyped by having
    a terminal group next to them.
    """
    out = []
    for neighbour in site.GetNeighbors():
        if neighbour.GetAtomicNum() == 1:
            continue
        source = neighbour.GetAtomMapNum() - 1
        name = names[source] if 0 <= source < len(names) else None
        typed = by_index.get(neighbour.GetIdx())
        if typed is None:
            continue
        out.append(
            {
                "name": name,
                "in_base": name in base_names,
                "element": neighbour.GetSymbol(),
                "atom_type": typed.atom_type,
                "charge": charges[neighbour.GetIdx()],
            }
        )
    return out


def _mmff94_charges(smiles, n_atoms):
    """Per-atom MMFF94 charges, in the order the SMILES lists its atoms.

    MMFF94 charges are bond-charge increments, so no conformer is needed. The
    EEM fallback the free-ligand path allows is not taken here: it is a
    different model, and a polymer residue that needs it should say so.
    """
    from openbabel import openbabel, pybel

    try:
        mol = pybel.readstring("smi", smiles)
        mol.addh()
        model = openbabel.OBChargeModel.FindType("mmff94")
        if model is None or not model.ComputeCharges(mol.OBMol):
            return None
    except Exception as err:  # noqa: BLE001 - report and fall back to a copy
        logger.warning("MMFF94 charges unavailable for %r: %s", smiles, err)
        return None
    charges = [a.partialcharge for a in mol.atoms]
    return charges if len(charges) == n_atoms else None


def _fallback_chemistry(residue_type, base_charges, template, connection_atom, chemdb):
    """A terminal group described from the residue's own atoms.

    Used where the terminal form could not be built, so Dimorphite, the typer
    and the charge model never saw it. Everything except the proton count
    comes from an atom the residue already has of the same kind: the oxygen a
    C-terminus adds is typed and charged like the carbonyl oxygen already
    there, and the protons an N-terminus adds like the one already on its
    nitrogen. The proton count cannot be recovered and is the database
    patch's, which assumes an alpha backbone.
    """
    element_of = {at.name: at.element for at in chemdb.atom_types}
    types = {a.name: a.atom_type for a in residue_type.atoms}
    added_elements = {
        element_of.get(a.atom_type, "")
        for a in template.add_atoms
        if element_of.get(a.atom_type, "") not in ("", "H")
    }
    neighbours = [
        other
        for bond in residue_type.bonds
        for atom, other in (bond[:2], bond[1::-1])
        if atom == connection_atom
    ]

    def like(predicate, candidates):
        for name in candidates:
            if name in types and predicate(types[name]):
                return name
        return None

    heavy_source = like(lambda t: element_of.get(t, "") in added_elements, neighbours)
    hydrogen_source = like(lambda t: element_of.get(t, "") == "H", neighbours) or like(
        lambda t: element_of.get(t, "") == "H", list(types)
    )
    if heavy_source is None and added_elements:
        return None
    if hydrogen_source is None and not added_elements:
        return None

    n_heavy = sum(
        1
        for a in template.add_atoms
        if element_of.get(a.atom_type, "") in added_elements
    )
    return {
        "measured": False,
        "element_of": element_of,
        "added_elements": added_elements,
        "n_hydrogens": sum(
            1 for a in template.add_atoms if element_of.get(a.atom_type, "") == "H"
        ),
        "hydrogen_type": types.get(hydrogen_source),
        "hydrogen_charge": base_charges.get(hydrogen_source, 0.0),
        "site_type": types.get(connection_atom),
        "site_charge": base_charges.get(connection_atom, 0.0),
        "site_is_planar": False,
        "heavy": [
            {
                "name": None,
                "in_base": False,
                "element": element_of.get(types[heavy_source], ""),
                "atom_type": types[heavy_source],
                "charge": base_charges.get(heavy_source, 0.0),
            }
            for _ in range(n_heavy)
        ],
    }


def terminus_patches(
    chemdb, residue_type, profile, atom_array=None, ph=7.4, base_charges=None
):
    """Patches for this residue's chain ends, and what each was modelled on.

    Returns ``[(patch, template, chemistry), ...]``, one entry per chain end,
    and the two are independent -- a residue can end up able to sit at one and
    not the other. A terminus the residue cannot be described at is left
    without a patch, and says so; without one it cannot sit there at all,
    which beats sitting there described as something it is not.
    """
    # a terminus is one end of a chain that continues at the other, so a
    #    residue with a single connection -- a cap -- has none. Patching its
    #    one connection would make it a free molecule, which is the ligand
    #    path's business rather than a variant of this residue
    if profile.down is None or profile.up is None:
        return []

    taken = {a.name for a in residue_type.atoms}
    generated = []
    for (display_name, connection), candidates in terminus_templates(
        chemdb, profile
    ).items():
        template = _template_for(residue_type, connection[1], candidates)
        if template is None:
            logger.warning(
                "%s %s patch: no template. The database describes no %s for a "
                "connection at %s, so the residue cannot sit at that end.",
                residue_type.name,
                display_name,
                display_name,
                connection[1],
            )
            continue
        chemistry = None
        if atom_array is not None:
            try:
                chemistry = _terminal_chemistry(
                    atom_array,
                    residue_type,
                    profile,
                    chemdb,
                    template,
                    display_name,
                    ph,
                )
            except Exception as err:  # noqa: BLE001 - reported, and no patch
                logger.warning(
                    "%s %s patch: building the terminal molecule failed (%s). "
                    "Dimorphite, the atom typer and the charge model never saw "
                    "it; falling back to the residue's own atoms.",
                    residue_type.name,
                    display_name,
                    err,
                )

        if chemistry is None:
            chemistry = _fallback_chemistry(
                residue_type, base_charges or {}, template, connection[1], chemdb
            )
            if chemistry is None:
                logger.warning(
                    "%s %s patch: fallback failed. The residue has no atom of "
                    "the kind this terminus adds to model one on, so it cannot "
                    "sit at that end.",
                    residue_type.name,
                    display_name,
                )
                continue
            logger.warning(
                "%s %s patch: built from the residue, not from its terminal "
                "form. Types and charges come from the residue's own atoms of "
                "the same kind; the proton count is the database patch's and "
                "assumes an alpha backbone, which may not be this residue's.",
                residue_type.name,
                display_name,
            )

        add_atoms = _terminal_add_atoms(template, chemistry)
        renames = {}
        for atom in add_atoms:
            free = _free_name(atom.name, taken)
            taken.add(free)
            if free != atom.name:
                renames[atom.name] = free
        patch = _rename_patch(
            template,
            residue_type,
            renames,
            f"{residue_type.name}_{template.name}",
            add_atoms=add_atoms,
            chemistry=chemistry,
            connection_atom=connection[1],
        )
        generated.append((patch, template, chemistry))
    return generated


def _terminal_group_heavy(chemistry):
    """The heavy atoms of the terminal group the patch is responsible for.

    A C-terminus patch removes the carbonyl oxygen it matched and adds both of
    the acid's, so the atom the residue already had is one of the patch's too.
    They are told apart from the rest of the connection's neighbours by
    element, which is what the patch declares.
    """
    heavy = [
        e for e in chemistry["heavy"] if e["element"] in chemistry["added_elements"]
    ]
    # the atom the residue does not have goes first; within an element the
    #    atoms are equivalent, so which takes which name does not matter
    return sorted(heavy, key=lambda e: (e["in_base"], e["name"] or ""))


def _terminal_add_atoms(template, chemistry):
    """The atoms the patch adds, typed and counted by the terminal molecule.

    The database's patch adds as many hydrogens as an alpha backbone's amide
    nitrogen takes once protonated. A nitrogen that is not that basic takes
    fewer, and the terminal molecule says how many.
    """
    heavy = [a for a in template.add_atoms if not a.atom_type.startswith("H")]
    hydrogens = [a for a in template.add_atoms if a.atom_type.startswith("H")]

    spare = _terminal_group_heavy(chemistry)
    out = []
    for atom, replacement in zip(heavy, spare + [None] * len(heavy)):
        atom_type = replacement["atom_type"] if replacement else atom.atom_type
        out.append(attr.evolve(atom, atom_type=atom_type))

    # a hydrogen the patch adds is one the terminal molecule has and the base
    #    residue does not; how many that is depends on how basic the site is
    wanted = chemistry["n_hydrogens"]
    for atom in hydrogens[:wanted]:
        out.append(
            attr.evolve(atom, atom_type=chemistry["hydrogen_type"] or atom.atom_type)
        )
    return out


def _with_every_added_atom(delta, variant, patch, chemistry):
    """``delta`` with a charge for every atom the patch added.

    A terminus patch can put an atom somewhere other than on the connection --
    a 5' phosphate patch leaves its proton on the oxygen the phosphate was
    attached to -- and an atom with no charge cannot be scored at all. What it
    is bonded to describes it better than nothing does.
    """
    present = {a.name for a in variant.atoms}
    by_name = {a.name: a.atom_type for a in variant.atoms}
    element_of = chemistry["element_of"]
    neighbours: dict = {}
    for bond in variant.bonds:
        neighbours.setdefault(bond[0], []).append(bond[1])
        neighbours.setdefault(bond[1], []).append(bond[0])

    for atom in patch.add_atoms:
        if atom.name not in present or atom.name in delta:
            continue
        if element_of.get(by_name.get(atom.name, ""), "") == "H":
            delta[atom.name] = chemistry["hydrogen_charge"]
            continue
        parent = next((n for n in neighbours.get(atom.name, ()) if n in delta), None)
        delta[atom.name] = delta[parent] if parent else chemistry["site_charge"]
    return delta


def patch_charge_entries(param_db, patched_chemdb, residue_type, generated):
    """``{variant name: {atom: charge}}`` for the generated patches.

    A terminal form's charges are measured on it, never borrowed: a residue
    whose backbone is not alpha has no counterpart in the database to borrow
    from, and copying an alpha one's numbers would describe a chain end this
    residue does not have.
    """
    by_name = {r.name: r for r in patched_chemdb.residues}
    entries = {}
    for patch, _template, chemistry in generated:
        variant_name = f"{residue_type.name}:{patch.display_name}"
        if variant_name not in by_name or chemistry is None:
            continue
        delta = _computed_delta(by_name[variant_name], patch, chemistry)
        if delta:
            entries[variant_name] = _with_every_added_atom(
                delta, by_name[variant_name], patch, chemistry
            )
    return entries


def _fallback_delta(variant, patch, chemistry):
    """Charges for the atoms the fallback added, and nothing else.

    Every other atom has no entry and falls back to the in-chain row, which is
    what the elec resolver does for a variant that does not mention an atom.
    """
    spare = _terminal_group_heavy(chemistry)
    hydrogens = {
        a.name for a in patch.add_atoms if a.atom_type == chemistry["hydrogen_type"]
    }
    present = {a.name for a in variant.atoms}
    delta = {
        name: chemistry["hydrogen_charge"] for name in hydrogens if name in present
    }
    heavy_names = [a.name for a in patch.add_atoms if a.name not in hydrogens]
    for name, entry in zip(heavy_names, _stretched(spare, len(heavy_names))):
        if name in present:
            delta[name] = entry["charge"]
    return delta


def _stretched(entries, wanted):
    """``entries`` extended to ``wanted`` by repeating the last one.

    A patch can add more atoms than the terminal form has to model them on --
    a phosphate patch adds three oxygens where the molecule offers two. They
    are equivalent, so the last one describes the rest.
    """
    if not entries or len(entries) >= wanted:
        return entries
    return [*entries, *([entries[-1]] * (wanted - len(entries)))]


def _computed_delta(variant, patch, chemistry):
    """The terminal form's charges, mapped onto the variant's atom names.

    Every atom it can account for, not only the ones the patch touched: a
    charge model apportions a molecule's whole charge at once, so an atom near
    the chain end is not the only one the terminus moves.
    """
    if not chemistry["measured"]:
        return _fallback_delta(variant, patch, chemistry)

    heavy_charge = dict(chemistry["heavy_charge"])
    # the patch's own atoms are named by the patch; the terminal molecule knows
    #    them by the cap names it was built with, in the same order
    spare = _terminal_group_heavy(chemistry)
    hydrogens = {
        a.name for a in patch.add_atoms if a.atom_type == chemistry["hydrogen_type"]
    }
    heavy_names = [a.name for a in patch.add_atoms if a.name not in hydrogens]
    for atom, entry in zip(heavy_names, _stretched(spare, len(heavy_names))):
        heavy_charge[atom] = entry["charge"]

    by_parent = dict(chemistry["hydrogen_charge_by_parent"])
    delta = {}
    neighbours = {}
    for bond in variant.bonds:
        neighbours.setdefault(bond[0], []).append(bond[1])
        neighbours.setdefault(bond[1], []).append(bond[0])
    element_of = chemistry["element_of"]
    is_hydrogen = {
        a.name: element_of.get(a.atom_type, a.atom_type[:1]) == "H"
        for a in variant.atoms
    }

    taken = {}
    for atom in variant.atoms:
        if atom.name in heavy_charge:
            delta[atom.name] = heavy_charge[atom.name]
            continue
        if not is_hydrogen.get(atom.name):
            continue
        for parent in neighbours.get(atom.name, ()):
            source = by_parent.get(parent)
            if not source:
                continue
            index = taken.get(parent, 0)
            if index < len(source):
                delta[atom.name] = source[index]
                taken[parent] = index + 1
            break
    # an atom the terminal molecule has no counterpart for keeps its in-chain
    #    charge, which is the right value for it: the terminal form is only
    #    consulted about the chain end, and the residue's own protonation is
    #    what the variant's atoms actually are
    return delta
