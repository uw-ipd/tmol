"""Read supplied structure information before tmol validates parameter requirements."""

import warnings

import biotite.structure as struc
import numpy as np

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
from atomworks.io.utils.bonds import get_struct_conn_bonds
from atomworks.io.utils.ccd import get_polymerization_atoms
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

    A PDB writes a modified residue in a chain as HETATM, as it writes a free ligand.
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

    The loader moves the HETATM residues of a chain that also has polymer ones to a new chain.
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

    The reader has kept one conformer, so a row's alternate locations all name it.
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

    A deposit can declare one cysteine in two disulfides (6CNB) or a Zn(Cys)4
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


def _renumber_decreasing_author_ids(array):
    """Renumber, in file order, any chain whose author numbering decreases.

    AtomWorks refuses such a chain (5XNL numbers its waters backwards); renumbering
    keeps residue identity and order. Returns the array unchanged where none decreases.
    """
    res_id = array.res_id.copy()
    ins_code = array.ins_code.copy()
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    repaired = []
    for chain in dict.fromkeys(array.chain_id.tolist()):
        in_chain = array.chain_id == chain
        if not (np.diff(res_id[in_chain]) < 0).any():
            continue
        number = 0
        for begin, end in zip(starts[:-1], starts[1:], strict=False):
            if array.chain_id[begin] == chain:
                number += 1
                res_id[begin:end] = number
        # renumbering supersedes this chain's insertion codes and no others:
        #    elsewhere they are load-bearing, as an antibody's CDRs are
        ins_code[in_chain] = ""
        repaired.append(str(chain))
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

    A link to a metal is a coordination bond.
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

    The loader moves a chain's HETATM residues off it; those within its ATOM residues, or
    bonded at its ends (caps), rejoin it. PDB files may list the rest out of order (1HZY).
    """
    array, _ = load_pdb(path, model=model)
    if array.coord.ndim == 3:
        array = array[0]
    array = _with_pdb_author_chains(array, path, model)
    array.bonds = array.bonds.merge(_link_bonds(array, read_any(path).lines))
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
    return array[order]


def _parse_repairing_author_numbering(path, config, model, assembly_id):
    """``parse`` the file, renumbering refused author ids in the asymmetric unit first.

    An assembly that needs renumbering is refused with AtomWorks' own message.
    """
    is_pdb = infer_pdb_file_type(path) == "pdb"
    try:
        if is_pdb:
            return parse_atom_array(_read_pdb(path, model), config=config)
        return parse(path, config=config)
    except ValueError as refused:
        if assembly_id is not None or "non-decreasing order" not in str(refused):
            raise
        block = None
        if is_pdb:
            array = _read_pdb(path, model)
        else:
            file = read_any(path)
            block = getattr(file, "block", None)
            array = get_structure(file, model=model, extra_fields=_FIELDS)
        repaired = _renumber_decreasing_author_ids(array)
        if repaired is array:
            raise
        atoms = prepare_atom_array(repaired, config=config, cif_block=block)
        return {"asym_unit": atoms, "cif_block": block}


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
    result = _parse_repairing_author_numbering(path, config, model, assembly_id)
    array = (
        result["asym_unit"]
        if assembly_id is None
        else result["assemblies"][assembly_id]
    )
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
