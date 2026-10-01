"""Alternate locations: one per linked group in PDB files, one residue per site."""

import io
import warnings
from collections import Counter, defaultdict

import biotite.structure as struc
import numpy as np
from atomworks.constants import (
    ALTLOC_DEFAULT_IDS,
    CCD_MIRROR_PATH,
    HYDROGEN_LIKE_SYMBOLS,
    METAL_ELEMENTS,
)
from atomworks.io.utils.ccd import get_atom_names_for_residue
from biotite.structure.io.pdb import PDBFile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# heavy atoms of two residues this close are bonded, coordinated or overlapping
LINK_DISTANCE = 2.2
# shorter than any bond between two residues: the atoms are alternates of one site
OVERLAP_DISTANCE = 1.2


def select_altlocs(atom_array):
    """Keep one alternate per linked group of residues (AtomWorks' rule, first letter).

    Residues with alternates are linked when they share a chain or when an
    alternate heavy atom lies within ``LINK_DISTANCE`` of another residue; each
    group keeps its first letter. A residue without it keeps its first alternate
    that overlaps no kept atom; if all overlap, it is absent.
    """
    arr = atom_array
    altloc_ids = arr.altloc_id
    is_alt = ~np.isin(altloc_ids, ALTLOC_DEFAULT_IDS)
    if not is_alt.any():
        return atom_array
    _, res = np.unique(
        np.rec.fromarrays([arr.chain_id, arr.res_id, arr.ins_code]),
        return_inverse=True,
    )
    _, chain = np.unique(arr.chain_id, return_inverse=True)
    is_heavy = ~np.isin(arr.element, HYDROGEN_LIKE_SYMBOLS)
    heavy = np.flatnonzero(is_heavy & np.isfinite(arr.coord).all(-1))
    # hydrogens follow the heavy atoms of their residue
    defining = is_alt & (is_heavy | ~np.isin(res, res[is_alt & is_heavy]))
    alt_heavy = heavy[is_alt[heavy]]
    near = cKDTree(arr.coord[heavy]).query_ball_point(
        arr.coord[alt_heavy], LINK_DISTANCE
    )
    rows = np.concatenate(
        [np.repeat(res[alt_heavy], [len(n) for n in near]), res[is_alt]]
    )
    cols = np.concatenate(
        [res[heavy[[j for n in near for j in n]]], res.max() + 1 + chain[is_alt]]
    )
    size = res.max() + chain.max() + 2
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    group = connected_components(graph)[1][res]

    pick = np.full(res.max() + 1, "", dtype=altloc_ids.dtype)
    for g in np.unique(group[defining]):
        in_group = defining & (group == g)
        letter = min(altloc_ids[in_group])
        pick[res[in_group & (altloc_ids == letter)]] = letter

    lacking = np.flatnonzero(defining & (pick[res] == ""))
    if len(lacking):
        options, option = np.unique(
            np.rec.fromarrays([res[lacking], altloc_ids[lacking]]),
            return_inverse=True,
        )
        kept = heavy[~is_alt[heavy] | (altloc_ids[heavy] == pick[res[heavy]])]
        probe = np.isin(lacking, heavy)
        atoms = np.concatenate([kept, lacking[probe]])
        owner = np.concatenate([np.full(len(kept), -1), option[probe]])
        near = cKDTree(arr.coord[atoms]).query_ball_point(
            arr.coord[lacking[probe]], OVERLAP_DISTANCE
        )
        clash = [set() for _ in options]
        for o, i, n in zip(option[probe], lacking[probe], near, strict=True):
            clash[o] |= {owner[j] for j in n if res[atoms[j]] != res[i]}
        chosen = {-1}  # kept atoms, then the alternates chosen here in residue order
        for o, (r, letter) in enumerate(options.tolist()):
            if pick[r] == "" and not clash[o] & chosen:
                pick[r] = letter
                chosen.add(o)
    return atom_array[~is_alt | (altloc_ids == pick[res])]


def one_alternate(atoms):
    """``atoms``, read with every alternate, keeping one per linked group.

    Atoms without a letter at a microheterogeneity site take the kept residue's
    name, or are dropped where its CCD entry lacks them (1EJG A:22 PRO/SER).
    """
    atoms = select_altlocs(atoms)
    lettered = ~np.isin(atoms.altloc_id, ALTLOC_DEFAULT_IDS)
    _, residue = np.unique(
        np.rec.fromarrays([atoms.chain_id, atoms.res_id, atoms.ins_code]),
        return_inverse=True,
    )
    name = np.full(residue.max() + 1, "", dtype=object)
    name[residue[lettered]] = atoms.res_name[lettered]
    name = name[residue]
    keep = np.ones(len(atoms), dtype=bool)
    for i in np.flatnonzero(~lettered & (name != "") & (atoms.res_name != name)):
        standard, alternative, _ = get_atom_names_for_residue(
            name[i], str(CCD_MIRROR_PATH or "")
        )
        taken = (atoms.res_name == name) & (atoms.atom_name == atoms.atom_name[i])
        taken &= residue == residue[i]
        keep[i] = atoms.atom_name[i] in standard | alternative and not taken.any()
    atoms.res_name[name != ""] = name[name != ""].astype(atoms.res_name.dtype)
    return atoms[keep]


def pdb_lines_with_one_alternate(lines, model):
    """The PDB lines less the alternates ``one_alternate`` drops and their LINKs."""
    atom = np.array([line.startswith(("ATOM  ", "HETATM")) for line in lines])
    if not any(lines[i][16:17].strip() for i in np.flatnonzero(atom)):
        return lines
    text = io.StringIO("\n".join(lines))
    atoms = PDBFile.read(text).get_structure(model=model, altloc="all")
    in_model = np.cumsum([line.startswith("MODEL ") for line in lines]).clip(1) == model
    atoms.set_annotation("line", np.flatnonzero(in_model & atom))
    kept = one_alternate(atoms)
    edited = [
        None if a else line for line, a in zip(lines, in_model & atom, strict=True)
    ]
    for i, name in zip(kept.line, kept.res_name, strict=True):
        edited[i] = f"{lines[i][:17]}{name:>3}{lines[i][20:]}"
    lettered = ~np.isin(kept.altloc_id, ALTLOC_DEFAULT_IDS)
    fields = (kept.chain_id, kept.res_id.astype(str), kept.ins_code, kept.altloc_id)
    letters = set(zip(*(f[lettered] for f in fields), strict=True))
    residues = {letter[:3] for letter in letters}
    for i, line in enumerate(lines):
        if line.startswith("LINK  "):
            line = line.ljust(80)
            # each partner's chain, residue number, insertion code and altloc
            partners = [
                (line[at + 9], line[at + 10 : at + 14].strip())
                + (line[at + 14].strip(), line[at + 4].strip())
                for at in (12, 42)
            ]
            if any(p[3] and p[:3] in residues and p not in letters for p in partners):
                edited[i] = None
    return [line for line in edited if line is not None]


def records_with_one_alternate(atom_records):
    """``parse_pdb`` records keeping one alternate per linked group in each model.

    Otherwise a later alternate overwrites an earlier one atom by atom (1EJG A:22).
    """
    if (atom_records["location"] == "").all():
        return atom_records
    atoms = struc.AtomArray(len(atom_records))
    atoms.coord = atom_records[["x", "y", "z"]].to_numpy(dtype=float)
    atoms.chain_id = atom_records["chain"].to_numpy(str)
    atoms.res_id = atom_records["resi"].to_numpy()
    atoms.ins_code = atom_records["insert"].to_numpy(str)
    atoms.res_name = atom_records["resn"].to_numpy(str)
    atoms.atom_name = atom_records["atomn"].to_numpy(str)
    atoms.element = atom_records["atomn"].str.lstrip("0123456789").str[:1].to_numpy(str)
    atoms.set_annotation("altloc_id", atom_records["location"].to_numpy(str))
    atoms.set_annotation("record", np.arange(len(atoms)))
    atoms = struc.concatenate(
        [
            one_alternate(atoms[indices])
            for indices in atom_records.groupby("modeli", sort=False).indices.values()
        ]
    )
    atoms = atoms[np.argsort(atoms.record)]
    kept = atom_records.iloc[atoms.record].assign(resn=atoms.res_name)
    return kept.reset_index(drop=True)


# closer than the shortest metal-metal contact of any site (Cu-Cu in CuA, 2.4 A)
ION_SITE = 2.0


def one_residue_per_site(array):
    """The array keeping one of the non-polymer residues occupying each site.

    Two residues occupy one site when half the heavy atoms of either lie within
    ``OVERLAP_DISTANCE`` of the other's, or both are ions closer than ``ION_SITE``:
    alternates without letters (8A7K Mn/Mg, 1P4K two GOL) or in other chains
    (3F7L Cu, 8CH1 LAO/VDF). Waters are left alone.
    """
    residue = struc.get_all_residue_positions(array)
    elements = np.char.upper(array.element.astype(str))
    heavy = ~np.isin(elements, HYDROGEN_LIKE_SYMBOLS) & np.isfinite(array.coord).all(-1)
    categories = array.get_annotation_categories()
    polymer = array.is_polymer if "is_polymer" in categories else ~array.hetero
    water = np.isin(array.res_name, ("HOH", "DOD", "WAT"))
    site = np.flatnonzero(heavy & ~polymer & ~water)
    if len(site) < 2:
        return array
    n_heavy = np.bincount(residue[heavy], minlength=residue.max() + 1)
    ion = np.isin(elements[site], list(METAL_ELEMENTS)) & (n_heavy[residue[site]] == 1)
    near = cKDTree(array.coord[site]).query_ball_point(
        array.coord[site], np.where(ion, ION_SITE, OVERLAP_DISTANCE)
    )
    covering = {
        (residue[site[k]], residue[site[h]], k)
        for k, hits in enumerate(near)
        for h in hits
        if residue[site[h]] != residue[site[k]] and (ion[h] or not ion[k])
    }
    partners = defaultdict(set)
    for (a, b), n in Counter((a, b) for a, b, _ in covering).items():
        if 2 * n >= n_heavy[a]:
            partners[a].add(b)
            partners[b].add(a)
    if not partners:
        return array
    first = struc.get_residue_starts(array)
    occupancy = array.occupancy if "occupancy" in categories else np.ones(len(array))
    alt = (
        array.label_alt_id if "label_alt_id" in categories else np.full(len(array), "")
    )
    letter = np.where(np.isin(alt, ALTLOC_DEFAULT_IDS), "", alt)
    kept, dropped = set(), []
    # the letter AtomWorks keeps elsewhere, then the more occupied, then file order
    for r in sorted(
        partners, key=lambda r: (letter[first[r]], -occupancy[first[r]], r)
    ):
        if partners[r] & kept:
            dropped.append(r)
        else:
            kept.add(r)
    warnings.warn(
        "Keeping one residue per site: dropping "
        + ", ".join(
            f"{array.chain_id[first[r]]}:{array.res_id[first[r]]} "
            f"{array.res_name[first[r]]}"
            for r in sorted(dropped)
        ),
        stacklevel=2,
    )
    return array[~np.isin(residue, dropped)]
