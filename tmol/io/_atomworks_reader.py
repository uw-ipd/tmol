"""Read supplied structure information before tmol validates parameter requirements."""

import io
import warnings
from string import ascii_uppercase

import biotite.structure as struc
import numpy as np
from scipy.spatial import cKDTree

from atomworks.constants import (
    BOND_DISTANCE_THRESHOLD_CHNO,
    BOND_DISTANCE_THRESHOLD_CHNOPS,
    BOND_DISTANCE_THRESHOLD_OTHER,
    CCD_MIRROR_PATH,
    METAL_ELEMENTS,
)
from atomworks.io import load_pdb
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse, parse_atom_array, prepare_atom_array
from atomworks.io.transforms.categories import category_to_dict
from atomworks.io.utils.bonds import get_struct_conn_bonds
from atomworks.io.utils.ccd import (
    get_atom_names_for_residue,
    get_polymerization_atoms,
)
from atomworks.io.utils.io_utils import get_structure, infer_pdb_file_type, read_any

from tmol.io._alternates import (
    NO_ALTERNATE,
    one_alternate_per_group,
    selected_residue_names,
)

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

    A row naming an alternate location binds that conformer: it is dropped where
    the reader kept another conformer of the atom (3P1O names GLU A:86 conformer
    B for MG A:237; the kept conformer A is 6 A away), and read where the atom
    has no alternates.
    """
    struct_conn = _rows_binding_kept_conformers(
        array, category_to_dict(block, "struct_conn")
    )
    bonds = array.bonds if array.bonds is not None else struc.BondList(len(array))
    return bonds.merge(
        get_struct_conn_bonds(
            array,
            struct_conn,
            add_bond_types=("metalc",),
            distance_policy="keep",
        )
    )


def _rows_binding_kept_conformers(array, struct_conn):
    """struct_conn without its alt ids, less the rows naming a conformer not kept."""
    named = [struct_conn.pop(f"pdbx_ptnr{p}_label_alt_id", None) for p in (1, 2)]
    if "label_alt_id" not in array.get_annotation_categories() or all(
        n is None for n in named
    ):
        return struct_conn
    alt = array.label_alt_id
    lettered = np.flatnonzero(~np.isin(alt, (".", "?", " ", "")))
    kept = {
        (str(array.chain_id[i]), str(array.res_id[i]), str(array.res_name[i]))
        + (str(array.atom_name[i]),): str(alt[i])
        for i in lettered
    }
    keep = np.ones(len(struct_conn["conn_type_id"]), dtype=bool)
    for partner, letters in zip((1, 2), named):
        if letters is None:
            continue
        seq = struct_conn[f"ptnr{partner}_label_seq_id"]
        seq = np.where(
            seq == ".", struct_conn.get(f"ptnr{partner}_auth_seq_id", seq), seq
        )
        for row, key in enumerate(
            zip(
                struct_conn[f"ptnr{partner}_label_asym_id"],
                seq,
                struct_conn[f"ptnr{partner}_label_comp_id"],
                struct_conn[f"ptnr{partner}_label_atom_id"],
                strict=True,
            )
        ):
            letter = str(letters[row])
            if letter not in (".", "?", " ", ""):
                keep[row] &= kept.get(tuple(str(k) for k in key), letter) == letter
    return {name: column[keep] for name, column in struct_conn.items()}


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


# closer than the shortest metal-metal contact of any site (Cu-Cu in CuA, 2.4 A)
_ION_SITE = 2.0


def _one_ion_per_site(array):
    """The array keeping, of metal ions closer than ``_ION_SITE``, the more occupied.

    Two ions that close are alternates of one site the file does not label as
    such: 8A7K models Mn and Mg at half occupancy on each of its sites, 3F7L puts
    the two conformers of a Cu in two chains. Ties keep the first in the file.
    """
    elements = np.char.upper(array.element.astype(str))
    residue = struc.get_all_residue_positions(array)
    heavy = ~np.isin(elements, ("H", "D", "T"))
    n_heavy = np.bincount(residue[heavy], minlength=residue.max() + 1)
    ion = np.flatnonzero(
        heavy
        & np.isin(elements, list(METAL_ELEMENTS))
        & (n_heavy[residue] == 1)
        & np.isfinite(array.coord).all(axis=-1)
    )
    if len(ion) < 2:
        return array
    near = cKDTree(array.coord[ion]).query_ball_point(array.coord[ion], _ION_SITE)
    if all(len(n) == 1 for n in near):
        return array
    occupancy = (
        array.occupancy[ion]
        if "occupancy" in array.get_annotation_categories()
        else np.ones(len(ion))
    )
    kept, dropped = set(), []
    for k in np.lexsort((ion, -occupancy)):
        if kept.isdisjoint(near[k]):
            kept.add(k)
        else:
            dropped.append(ion[k])
    warnings.warn(
        "Keeping one metal ion per site: dropping "
        + ", ".join(
            f"{array.chain_id[i]}:{array.res_id[i]} {array.res_name[i]}"
            for i in sorted(dropped)
        ),
        stacklevel=2,
    )
    return array[~np.isin(residue, residue[dropped])]


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


def _extends_polymer(array, starts, anchor, residue, step):
    """Whether *residue* bonds to the polymer atom of its neighbour *anchor*."""
    port = get_polymerization_atoms(str(array.res_name[starts[anchor]]))[step < 0]
    at = starts[anchor] + np.flatnonzero(
        array.atom_name[starts[anchor] : starts[anchor + 1]] == port
    )
    atoms = array.coord[starts[residue] : starts[residue + 1]]
    distance = np.linalg.norm(atoms[:, None] - array.coord[at], axis=-1)
    return bool((distance <= BOND_DISTANCE_THRESHOLD_CHNO).any())


def _pdb_lines_with_one_alternate(path, model):
    """The PDB's lines keeping one alternate per linked group, altloc columns blank.

    Atoms without a letter at a microheterogeneity site are named after the kept
    residue, or dropped where it lacks them, as AtomWorks does for mmCIF. LINK
    records naming an alternate that is not kept are dropped. None when the file
    has no alternates.
    """
    lines = list(read_any(path).lines)
    records = [
        i for i, line in enumerate(lines) if line.startswith(("ATOM  ", "HETATM"))
    ]
    if all(len(lines[i]) < 17 or lines[i][16] in NO_ALTERNATE for i in records):
        return None
    pdb_file = read_any(path)
    atoms = pdb_file.get_structure(model=model, altloc="all")
    if atoms.coord.ndim == 3:
        atoms = atoms[0]
    in_model, current = [], 0
    for i, line in enumerate(lines):
        if line.startswith("MODEL "):
            current += 1
        elif line.startswith(("ATOM  ", "HETATM")) and max(current, 1) == model:
            in_model.append(i)
    residue = np.char.add(
        np.char.add(atoms.chain_id.astype(str), "|"),
        np.char.add(atoms.res_id.astype(str), atoms.ins_code.astype(str)),
    )
    heavy = ~np.isin(np.char.upper(atoms.element.astype(str)), ("H", "D", "T"))
    keep = one_alternate_per_group(
        residue, atoms.chain_id, atoms.altloc_id, heavy, atoms.coord
    )
    name = selected_residue_names(residue, atoms.res_name, atoms.altloc_id, keep)
    kept_alt = keep & ~np.isin(atoms.altloc_id, NO_ALTERNATE)
    letters = {
        r: set(atoms.altloc_id[kept_alt & (residue == r)]) for r in residue[kept_alt]
    }
    for k in np.flatnonzero(keep & (name != atoms.res_name)):
        taken = (
            keep
            & (residue == residue[k])
            & (atoms.res_name == name[k])
            & (atoms.atom_name == atoms.atom_name[k])
        )
        standard, alternative, _ = get_atom_names_for_residue(
            str(name[k]), str(CCD_MIRROR_PATH or "")
        )
        keep[k] = not taken.any() and atoms.atom_name[k] in standard | alternative
    edited = {
        i: (line[:16] + " " + f"{name[k]:>3}" + line[20:]).rstrip() if keep[k] else None
        for k, (i, line) in enumerate((i, lines[i].ljust(80)) for i in in_model)
    }
    out, previous = [], None
    for i, line in enumerate(lines):
        if i in edited:
            previous = edited[i]
            if previous is not None:
                out.append(previous)
        elif line.startswith("ANISOU"):
            if previous is not None:
                out.append(line[:16] + " " + line[17:])
        elif line.startswith("LINK  "):
            line = line.ljust(80)
            partners = [
                (f"{line[at + 9]}|{line[at + 10 : at + 15].strip()}", line[at + 4])
                for at in (12, 42)
            ]
            if all(
                alt in NO_ALTERNATE or alt in letters.get(key, {alt})
                for key, alt in partners
            ):
                out.append((line[:16] + " " + line[17:46] + " " + line[47:]).rstrip())
        else:
            out.append(line)
    return out


def _read_pdb(path, model):
    """A PDB's atoms with its LINK records as bonds, HETATM chains in residue-number order.

    The loader moves a chain's HETATM residues off it; those within its ATOM residues
    (3SVU A:63 NRQ), or bonded at its ends (caps: 1COI A:0 ACE, A:30 NH2), rejoin it.
    PDB files may list the rest out of order (1HZY A:369 FMT after A:401 ZN).
    """
    from tmol.ligand._mol2_names import disambiguated_atom_names

    lines = _pdb_lines_with_one_alternate(path, model)
    if lines is not None:
        text = "\n".join(lines) + "\n"
        path = io.StringIO(text)
    array, _ = load_pdb(path, model=model)
    if array.coord.ndim == 3:
        array = array[0]
    if lines is not None:
        path = io.StringIO(text)
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
    array = _one_ion_per_site(array)
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
