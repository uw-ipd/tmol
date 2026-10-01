"""Read supplied structure information before tmol validates parameter requirements."""

import warnings
from string import ascii_uppercase

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdb.hybrid36 import decode_hybrid36
import numpy as np
from rdkit import Chem

from atomworks.constants import (
    BOND_DISTANCE_THRESHOLD_CHNO,
    BOND_DISTANCE_THRESHOLD_CHNOPS,
    BOND_DISTANCE_THRESHOLD_OTHER,
    METAL_ELEMENTS,
)
from atomworks.io import load_pdb
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse, parse_atom_array, prepare_atom_array
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.tools.rdkit import atom_array_to_rdkit, BIOTITE_BOND_TYPE_TO_RDKIT
from atomworks.io.utils.bonds import get_struct_conn_bonds
from atomworks.io.utils.ccd import (
    build_ccd_entries_from_cif_block,
    custom_ccd_residues,
    get_polymerization_atoms,
)
from atomworks.io.utils.io_utils import get_structure, infer_pdb_file_type, read_any

_AUTHOR_FIELDS = {
    "atom_name": "auth_atom_id",
    "res_name": "auth_comp_id",
    "chain_id": "auth_asym_id",
    "res_id": "auth_seq_id",
    "ins_code": "pdbx_PDB_ins_code",
}
_FIELDS = [
    *_AUTHOR_FIELDS.values(),
    "label_entity_id",
    "pdbx_formal_charge",
    "partial_charge",
]


def _polymer_from_backbone_bonds(array):
    """Mark as polymer the residues a polymer bond joins to a neighbour in their chain.

    A PDB writes a modified residue in a chain as HETATM, as a free ligand (5EMA SEP).
    """
    residue_of = struc.get_all_residue_positions(array)
    polymer = ~array.hetero[struc.get_residue_starts(array)]

    if array.bonds is not None and not polymer.all():
        bonds = array.bonds.as_array()[:, :2]
        first, second = residue_of[bonds[:, 0]], residue_of[bonds[:, 1]]
        crossing = bonds[(first != second) & ~(polymer[first] & polymer[second])]
        ports = {
            str(name): get_polymerization_atoms(str(name))
            for name in np.unique(array.res_name[crossing])
        }
        for i, j in crossing:
            if array.auth_asym_id[i] != array.auth_asym_id[j]:
                continue
            near, far = ports[str(array.res_name[i])], ports[str(array.res_name[j])]
            joined = (str(array.atom_name[i]), str(array.atom_name[j]))
            if joined in ((near[0], far[1]), (near[1], far[0])):
                polymer[residue_of[[i, j]]] = True

    array.set_annotation("is_polymer", polymer[residue_of])
    return array


def _with_pdb_author_chains(array, path, model):
    """Each atom's chain as its PDB record names it, as ``auth_asym_id``.

    The loader moves a chain's HETATM residues to a new chain (5EMA SEP as PDB).
    """
    authored = read_any(path).get_structure(
        model=model, altloc="first", extra_fields=["atom_id"]
    )
    chain_of = dict(zip(authored.atom_id.tolist(), authored.chain_id.tolist()))
    array.set_annotation(
        "auth_asym_id", np.array([chain_of[i] for i in array.atom_id.tolist()])
    )
    return array


def _with_metal_coordination(array, block):
    """The bond table plus the file's metalc bonds, typed COORDINATION.

    The reader kept one conformer, so a row's alternate locations name it (3F7L, 7ADR).
    """
    struct_conn = category_to_dict(block, "struct_conn")
    for partner in (1, 2):
        struct_conn.pop(f"pdbx_ptnr{partner}_label_alt_id", None)
    bonds = array.bonds if array.bonds is not None else struc.BondList(len(array))
    return bonds.merge(
        get_struct_conn_bonds(
            array,
            struct_conn,
            add_bond_types=("metalc",),
            distance_policy="keep",
        )
    )


def _one_disulfide_per_sulfur(array):
    """The bond table keeping, of the S-S bonds between residues, one per sulfur.

    A deposit can declare one cysteine in two disulfides (6CNB L:51) or a Zn(Cys)4
    site as a ring of them (5N5Y). The bonds nearest 2.04 A are kept.
    """
    bonds = array.bonds.as_array()
    sulfur = np.char.upper(array.element.astype(str)) == "S"
    residue = struc.get_all_residue_positions(array)
    i, j = bonds[:, 0], bonds[:, 1]
    rows = np.flatnonzero(sulfur[i] & sulfur[j] & (residue[i] != residue[j]))
    ends = bonds[rows, :2].ravel()
    if len(np.unique(ends)) == len(ends):
        return array.bonds
    length = np.linalg.norm(array.coord[i[rows]] - array.coord[j[rows]], axis=-1)
    bonded, dropped = set(), []
    for k in np.argsort(
        np.nan_to_num(np.abs(length - 2.04), nan=np.inf), kind="stable"
    ):
        pair = {int(i[rows[k]]), int(j[rows[k]])}
        if pair & bonded:
            dropped.append(k)
        else:
            bonded |= pair
    warnings.warn(
        "Dropping disulfides that share a sulfur with one nearer 2.04 A: "
        + ", ".join(
            "{}:{}-{}:{} ({:.2f} A)".format(
                array.chain_id[i[rows[k]]],
                array.res_id[i[rows[k]]],
                array.chain_id[j[rows[k]]],
                array.res_id[j[rows[k]]],
                length[k],
            )
            for k in dropped
        ),
        stacklevel=2,
    )
    return struc.BondList(len(array), np.delete(bonds, rows[dropped], axis=0))


def renumbered_decreasing_chains(array):
    """(res_id, ins_code, chains): chains whose numbering decreases renumbered 1..N.

    Their insertion codes are cleared; other chains' are load-bearing (antibody CDRs).
    """
    res_id = array.res_id.copy()
    ins_code = array.ins_code.copy()
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    chains = []
    for chain in dict.fromkeys(array.chain_id.tolist()):
        in_chain = array.chain_id == chain
        if not (np.diff(res_id[in_chain]) < 0).any():
            continue
        number = 0
        for begin, end in zip(starts[:-1], starts[1:]):
            if array.chain_id[begin] == chain:
                number += 1
                res_id[begin:end] = number
        ins_code[in_chain] = ""
        chains.append(str(chain))
    return res_id, ins_code, chains


def _renumber_decreasing_author_ids(array):
    """Renumber, in file order, any chain whose author numbering decreases.

    AtomWorks refuses such a chain (5XNL numbers its waters backwards).
    """
    res_id, ins_code, repaired = renumbered_decreasing_chains(array)
    if not repaired:
        return array
    warnings.warn(
        f"Renumbering chain(s) {', '.join(repaired)}: author numbering that decreases "
        "within a chain is not an ordering that can be relied on downstream.",
        stacklevel=2,
    )
    array = array.copy()
    array.res_id = res_id
    array.ins_code = ins_code
    return array


def _link_bonds(array, lines):
    """The LINK records as bonds, skipping symmetry mates and links longer than a covalent bond.

    A link to a metal is a coordination bond (1HZY A:401 ZN), as in the mmCIF
    struct_conn path, whose distance limits these are.
    """
    index = {}
    for i, key in enumerate(
        zip(
            array.auth_asym_id,
            array.res_id,
            array.ins_code,
            array.res_name,
            array.atom_name,
            strict=True,
        )
    ):
        index.setdefault(tuple(str(k) for k in key), i)
    elements = np.char.upper(array.element.astype(str))
    bonds = []
    for line in lines:
        if not line.startswith("LINK  "):
            continue
        line = line.ljust(80)
        partners = [
            (line[at + 9], line[at + 10 : at + 14], line[at + 14])
            + (line[at + 5 : at + 8], line[at : at + 4])
            for at in (12, 42)
        ]
        i, j = (index.get(tuple(field.strip() for field in p)) for p in partners)
        mate = {line[59:65].strip(), line[66:72].strip()} - {"", "1555"}
        if i is None or j is None or mate:
            continue
        pair = {elements[i], elements[j]}
        if pair <= set("CHNO"):
            limit = BOND_DISTANCE_THRESHOLD_CHNO
        elif pair <= set("CHNOPS"):
            limit = BOND_DISTANCE_THRESHOLD_CHNOPS
        else:
            limit = BOND_DISTANCE_THRESHOLD_OTHER
        if np.linalg.norm(array.coord[i] - array.coord[j]) <= limit:
            metal = bool(pair & METAL_ELEMENTS)
            bond = struc.BondType.COORDINATION if metal else struc.BondType.ANY
            bonds.append((i, j, bond))
    return struc.BondList(len(array), np.array(bonds, dtype=np.int64).reshape(-1, 3))


def _stated_hetero_bonds(array, lines):
    """The bond table with only the CONECT bonds, of no stated order, within HETATM residues.

    The loader also bonds a HETATM residue by the CCD entry of its name when an ATOM
    residue shares it (3URI's 65-atom PRO ligand gains 179 bonds).
    """
    residue = struc.get_all_residue_positions(array)
    shared = array.hetero & np.isin(array.res_name, array.res_name[~array.hetero])
    bonds = array.bonds.as_array()
    inside = shared[bonds[:, 0]] & (residue[bonds[:, 0]] == residue[bonds[:, 1]])
    if not inside.any():
        return array.bonds
    index = {int(i): n for n, i in enumerate(array.atom_id.tolist())}
    stated = set()
    for line in lines:
        if line.startswith("CONECT"):
            ids = [line[k : k + 5] for k in range(6, 31, 5) if line[k : k + 5].strip()]
            at = [index.get(decode_hybrid36(i), -1) for i in ids]
            stated |= {frozenset((at[0], j)) for j in at[1:]}
    kept = [frozenset((int(i), int(j))) in stated for i, j in bonds[inside, :2]]
    bonds[np.flatnonzero(inside)[kept], 2] = struc.BondType.ANY
    inside[np.flatnonzero(inside)[kept]] = False
    return struc.BondList(len(array), bonds[~inside])


def _extends_polymer(array, starts, anchor, residue, step):
    """Whether *residue* bonds to the polymer atom of its neighbour *anchor*."""
    port = get_polymerization_atoms(str(array.res_name[starts[anchor]]))[step < 0]
    at = starts[anchor] + np.flatnonzero(
        array.atom_name[starts[anchor] : starts[anchor + 1]] == port
    )
    atoms = array.coord[starts[residue] : starts[residue + 1]]
    distance = np.linalg.norm(atoms[:, None] - array.coord[at], axis=-1)
    return bool((distance <= BOND_DISTANCE_THRESHOLD_CHNO).any())


def _read_pdb(path, model):
    """A PDB's atoms with its LINK records as bonds, HETATM chains in residue-number order.

    The loader moves a chain's HETATM residues off it; those within its ATOM residues
    (3SVU A:63 NRQ), or bonded at its ends (caps: 1COI A:0 ACE, A:30 NH2), rejoin it.
    PDB files may list the rest out of order (1HZY A:369 FMT after A:401 ZN).
    """
    from tmol.ligand._mol2_names import disambiguated_atom_names

    array, _ = load_pdb(path, model=model)
    if array.coord.ndim == 3:
        array = array[0]
    array = _with_pdb_author_chains(array, path, model)
    lines = read_any(path).lines
    array.bonds = _stated_hetero_bonds(array, lines).merge(_link_bonds(array, lines))
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    author = array.auth_asym_id[starts[:-1]]
    for chain in np.unique(author[~array.hetero[starts[:-1]]]):
        records = np.flatnonzero((author == chain) & ~array.hetero[starts[:-1]])
        joined = list(range(records[0] + 1, records[-1]))
        for anchor, step in ((records[0], -1), (records[-1], 1)):
            while (
                0 <= anchor + step < len(author)
                and author[anchor + step] == chain
                and _extends_polymer(array, starts, anchor, anchor + step, step)
            ):
                anchor += step
                joined.append(anchor)
        for residue in joined:
            if author[residue] == chain:
                array.chain_id[starts[residue] : starts[residue + 1]] = chain
    order = np.arange(len(array))
    for chain in np.unique(array.chain_id[array.hetero]):
        in_chain = np.flatnonzero(array.chain_id == chain)
        if array.hetero[in_chain].all():
            by_number = np.argsort(array.res_id[in_chain], kind="stable")
            order[in_chain] = in_chain[by_number]
    # PDBbind 10GS: a residue naming atoms alike (a peptidic ligand written as
    # one) takes the MOL2 reader's names for them
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    for begin, end in zip(starts[:-1], starts[1:]):
        names = array.atom_name[begin:end]
        if len(set(names)) < len(names):
            array.atom_name[begin:end] = disambiguated_atom_names(names.tolist())
    # PDBbind 1GPK pocket: a blank chain ID is valid in a PDB but names no chain
    blank = array.auth_asym_id == ""
    if blank.any():
        free = next(c for c in ascii_uppercase if c not in array.auth_asym_id)
        array.chain_id[blank] = array.auth_asym_id[blank] = free
    return array[order]


def _heavy(array):
    return ~np.isin(np.char.upper(array.element.astype(str)), ("H", "D"))


def _atoms(array):
    """``(atom name, element)`` of each atom of ``array``."""
    return set(zip(array.atom_name.tolist(), np.char.upper(array.element.astype(str))))


def _bond_table(array, rows=slice(None)):
    """``{atom-name pair: bond type}`` of the bonds ``rows`` of ``array``."""
    names = array.atom_name.astype(str)
    bonds = array.bonds.as_array()[rows]
    return {frozenset((names[i], names[j])): int(t) for i, j, t in bonds}


def _own_template(residue, entry):
    """``residue`` as its own component, and whether its heavy atoms are the entry's.

    Then the component is the entry's but for the bonds it states to its hydrogens
    (1E66 HUX names them apart).
    """
    from tmol.io._assemble import _component_template

    template = _component_template(residue)
    heavy, pairs = _heavy(template), _bond_table(entry)
    bonds = template.bonds.as_array()
    to_h = ~(heavy[bonds[:, 0]] & heavy[bonds[:, 1]])
    fits = _atoms(template[heavy]) <= _atoms(entry)
    fits &= _compatible_bonds(_bond_table(template, ~to_h), entry)
    kind = str(entry.chem_comp_type[0]) if fits else "NON-POLYMER"
    template.set_annotation("chem_comp_type", np.full(len(template), kind))
    if fits:
        index = {n: i for i, n in enumerate(template.atom_name.tolist())}
        stated = set(bonds[to_h, :2].ravel().tolist()) - set(np.flatnonzero(heavy))
        known = [
            (index[a], index[b], t)
            for (a, b), t in ((tuple(p), t) for p, t in pairs.items())
            if a in index and b in index and not {index[a], index[b]} & stated
        ]
        bonds[to_h, 2] = struc.BondType.SINGLE
        bonds = np.concatenate([bonds[to_h], np.reshape(known, (-1, 3))])
        template.bonds = struc.BondList(len(template), bonds.astype(np.uint32))
        charge = dict(zip(entry.atom_name.tolist(), entry.charge.tolist()))
        template.charge[heavy] = [charge[n] for n in template.atom_name[heavy]]
    return template, fits


def _compatible_bonds(stated, entry):
    """Explicit bond orders must agree, allowing unknown orders and Kekule equivalents."""
    known = _bond_table(entry)
    if not stated.keys() <= known.keys():
        return False
    explicit = {
        pair: kind for pair, kind in stated.items() if kind != struc.BondType.ANY
    }
    if all(kind == known[pair] for pair, kind in explicit.items()):
        return True
    try:
        reference = atom_array_to_rdkit(
            entry,
            sanitize=False,
            attempt_fixing_corrupted_molecules=False,
            annotations_to_keep=[],
        )
        Chem.SanitizeMol(reference)
        molecule = Chem.Mol(reference)
        Chem.Kekulize(molecule, clearAromaticFlags=True)
        index = {name: i for i, name in enumerate(entry.atom_name)}
        for pair, kind in explicit.items():
            a, b = pair
            bond = molecule.GetBondBetweenAtoms(index[a], index[b])
            bond.SetBondType(BIOTITE_BOND_TYPE_TO_RDKIT[kind][0])
        Chem.SanitizeMol(molecule)
        return all(
            bond.GetBondType() == reference.GetBondWithIdx(bond.GetIdx()).GetBondType()
            for bond in molecule.GetBonds()
        )
    except (ValueError, RuntimeError):
        return False


def _own_components(array, names=None):
    """``array`` and templates of residues whose heavy atoms or bonds the CCD entry of their name lacks.

    PDBbind names ligands with codes of other molecules (3UDH MOL, 3OV1 ACT). ``names``
    selects the components whose heavy atoms to check (default: hetero residues,
    hydrogen bonds included); a PDB ligand whose name a polymer uses is renamed (3URI PRO).
    """
    from tmol.io._cif import _component_dictionary_template

    bounds = struc.get_residue_starts(array, add_exclusive_stop=True)
    residue = struc.get_all_residue_positions(array)
    chosen = array.hetero if names is None else np.isin(array.res_name, list(names))
    bonds = array.bonds.as_array()
    within = residue[bonds[:, 0]] == residue[bonds[:, 1]]
    if names is not None:
        within &= _heavy(array)[bonds[:, 0]] & _heavy(array)[bonds[:, 1]]
    templates = {}
    for name in np.unique(array.res_name[chosen]).tolist():
        entry = _component_dictionary_template(name)
        mine = chosen & (array.res_name == name)
        if entry is None or (
            _atoms(array[mine & _heavy(array)]) <= _atoms(entry)
            and _compatible_bonds(_bond_table(array, within & mine[bonds[:, 0]]), entry)
        ):
            continue
        r = max(np.unique(residue[mine]), key=lambda r: bounds[r + 1] - bounds[r])
        template, fits = _own_template(array[bounds[r] : bounds[r + 1]], entry)
        if names is None and name in array.res_name[~array.hetero]:
            if fits:
                # 3B3S: a free leucine beside the chain's is the CCD's LEU
                continue
            from tmol.ligand._preparation import unused_ligand_name

            name = unused_ligand_name(set(array.res_name.tolist()))
            array.res_name[mine] = name
            template.res_name[:] = name
        templates[name] = template
    return array, templates


def _own_cif_components(block, model):
    """Templates of the components a CIF bonds in ``chem_comp_bond`` without declaring their atoms."""
    if block is None or "chem_comp_bond" not in block:
        return {}
    bonded = set(block["chem_comp_bond"]["comp_id"].as_array(str))
    if "chem_comp_atom" in block:
        bonded -= set(block["chem_comp_atom"]["comp_id"].as_array(str))
    if not bonded:
        return {}
    atoms = pdbx.get_structure(
        block, model=model, altloc="first", include_bonds=True, extra_fields=["charge"]
    )
    return _own_components(atoms, bonded)[1]


def _parse_repairing_author_numbering(path, config, model, assembly_id):
    """``parse`` the file, renumbering refused author ids in the asymmetric unit first.

    Also returns the templates of the residues parsed as their own components. An
    assembly that needs renumbering is refused with AtomWorks' own message.
    """
    is_pdb = infer_pdb_file_type(path) == "pdb"
    try:
        if is_pdb:
            array, templates = _own_components(_read_pdb(path, model))
            with custom_ccd_residues(templates):
                return parse_atom_array(array, config=config), templates
        result = parse(path, config=config)
        templates = _own_cif_components(result["cif_block"], model)
        if templates:
            with custom_ccd_residues(templates):
                result = parse(path, config=config)
        return result, templates
    except ValueError as refused:
        if assembly_id is not None or "non-decreasing order" not in str(refused):
            raise
        block = None
        if is_pdb:
            array, templates = _own_components(_read_pdb(path, model))
        else:
            file = read_any(path)
            block = getattr(file, "block", None)
            array = get_structure(file, model=model, extra_fields=_FIELDS)
            templates = {
                **build_ccd_entries_from_cif_block(block, on_mismatch="ignore"),
                **_own_cif_components(block, model),
            }
        repaired = _renumber_decreasing_author_ids(array)
        if repaired is array:
            raise
        with custom_ccd_residues(templates):
            atoms = prepare_atom_array(repaired, config=config, cif_block=block)
        return {"asym_unit": atoms, "cif_block": block}, templates


def read_structure(path, *, model=1, assembly_id=None):
    """Read PDB/CIF atoms and available bonds, supplemented from the CCD."""
    if model is None or model < 1:
        raise ValueError("The structure reader requires a positive model number")
    config = ParseConfig(
        model=model,
        add_missing_atoms=False,
        build_assembly=None if assembly_id is None else [assembly_id],
        extra_fields=_FIELDS,
        remove_ccds=[],
        remove_waters=False,
        fix_arginines=False,
        fix_ligands_at_symmetry_centers=False,
        add_bond_types_from_struct_conn=["covale", "disulf"],
        hydrogen_policy="keep",
        ccd_mirror_path=None,
        cif_ccd_on_mismatch="ignore",
        add_id_and_entity_annotations=False,
        keep_cif_block=True,
        return_atom_array_plus=True,
        long_bond_policy="keep",
        struct_conn_distance_policy="keep",
    )
    result, templates = _parse_repairing_author_numbering(
        path, config, model, assembly_id
    )
    array = (
        result["asym_unit"]
        if assembly_id is None
        else result["assemblies"][assembly_id]
    )
    array._custom_ccd_registry = {
        **getattr(array, "_custom_ccd_registry", {}),
        **templates,
    }
    if array.coord.ndim == 3:
        array = array[0]
    block = result.get("cif_block")
    if block is None and array.bonds is not None:
        from tmol.io._cif import _component_dictionary_template

        # CONECT gives orders only when a compatible component template supplies them.
        unknown_names = set()
        heavy = ~np.isin(array.element, ["H", "D"])
        for name in np.unique(array.res_name):
            template = _component_dictionary_template(str(name))
            observed = set(array.atom_name[heavy & (array.res_name == name)])
            if template is None or not observed <= set(template.atom_name):
                unknown_names.add(name)
        if unknown_names:
            residue = struc.spread_residue_wise(
                array, np.arange(struc.get_residue_count(array))
            )
            bonds = array.bonds.as_array()
            i, j = bonds[:, 0], bonds[:, 1]
            unknown = np.isin(array.res_name, list(unknown_names))
            bonds[unknown[i] & (residue[i] == residue[j]), 2] = struc.BondType.ANY
            array.bonds = struc.BondList(len(array), bonds)
    if array.bonds is not None:
        array.bonds = _one_disulfide_per_sulfur(array)
    if block is None:
        # A PDB, where the loader has moved off whatever its records called non-polymer.
        array = _polymer_from_backbone_bonds(array)
    elif assembly_id is None and "struct_conn" in block:
        array.bonds = _with_metal_coordination(array, block)
    if assembly_id is not None:
        array.set_annotation("chain_id", array.chain_iid.copy())
    for target, source in _AUTHOR_FIELDS.items():
        # Assembly expansion owns chain_id: it distinguishes symmetry copies
        # that share an author chain. Nothing else does, so a file without
        # one keeps its author chains -- a reader that derives them instead
        # splits an unrecognised residue into a chain of its own, and every
        # neighbour then looks like a chain end.
        if target == "chain_id" and assembly_id is not None:
            continue
        if source in array.get_annotation_categories():
            array.set_annotation(target, array.get_annotation(source).copy())
    array.ins_code[np.isin(array.ins_code, (".", "?"))] = ""
    retained = {
        "chain_id",
        "res_id",
        "ins_code",
        "res_name",
        "hetero",
        "atom_name",
        "element",
        "charge",
        "label_entity_id",
        "pdbx_formal_charge",
        "partial_charge",
        "is_polymer",
        "chain_type",
        "chem_comp_type",
        "auth_asym_id",
        "occupancy",
        "b_factor",
        "chain_iid",
        "transformation_id",
    }
    for name in set(array.get_annotation_categories()) - retained:
        array.del_annotation(name)
    return array, block
