"""Read supplied structure information before tmol validates parameter requirements."""

import warnings

import biotite.structure as struc
import numpy as np

from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse

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
    """Mark which residues of a PDB are polymer, reading its bonds rather than its records.

    A PDB writes a modified residue inside a chain as HETATM, exactly as it writes a
    free ligand, so the record cannot tell the two apart. The bonds can: a residue
    joined to a neighbour through the atoms its component polymerises with is part of
    that chain. Restoring the chain itself is the author-field mapping's job.
    """
    from atomworks.io.utils.ccd import get_polymerization_atoms

    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    residue_of = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    polymer = ~array.hetero[starts[:-1]]

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


def _read_declared(path, model, assembly_id):
    """Read authored atoms and bonds without any dictionary supplementation."""
    from atomworks.io import load_pdb
    from atomworks.io.transforms.categories import category_to_dict
    from atomworks.io.utils.atom_array_plus import as_atom_array_plus
    from atomworks.io.utils.bonds import get_struct_conn_bonds
    from atomworks.io.utils.ccd import (
        bond_dict_from_cif_block,
        build_ccd_entries_from_cif_block,
    )
    from atomworks.io.utils.io_utils import get_structure, read_any

    file = read_any(path)
    block = getattr(file, "block", None)
    if block is None:
        # The shared PDB loader retains CONECT and TER chain boundaries.
        array, _ = load_pdb(path, model=model)
        entries = {}
    else:
        array = get_structure(file, model=model, extra_fields=_FIELDS)
        entries = build_ccd_entries_from_cif_block(block, use_ccd=False)
        bonds = {name: {} for name in np.unique(array.res_name)}
        bonds.update(bond_dict_from_cif_block(block))
        array.bonds = struc.connect_via_residue_names(
            array, inter_residue=False, custom_bond_dict=bonds
        )
        if "struct_conn" in block:
            array.bonds = array.bonds.merge(
                get_struct_conn_bonds(
                    array,
                    category_to_dict(block, "struct_conn"),
                    add_bond_types=("covale", "disulf"),
                    distance_policy="keep",
                    use_ccd=False,
                )
            )
    array = as_atom_array_plus(array)
    array._custom_ccd_registry = entries
    if assembly_id is not None:
        from atomworks.io.utils.assembly import build_assemblies_from_asym_unit

        if block is None or "pdbx_struct_assembly_gen" not in block:
            raise ValueError(f"Structure does not declare assembly {assembly_id!r}")
        array = build_assemblies_from_asym_unit(
            block["pdbx_struct_assembly_gen"],
            block["pdbx_struct_oper_list"],
            struc.stack([array]),
            fix_symmetry_centers=False,
            build_assembly=[assembly_id],
        )[assembly_id][0]
        array = as_atom_array_plus(array)
        array._custom_ccd_registry = entries
    return array, block


def _renumber_decreasing_author_ids(array):
    """Renumber, in file order, any chain whose author numbering runs backwards.

    Author numbering is the depositor's, and real entries do run backwards --
    5XNL numbers its waters that way. AtomWorks reads res_id as an ordering and
    refuses a chain that decreases, so supplying numbering it can rely on is the
    reader's job. Residue identity and order are what is wanted from res_id here,
    and renumbering keeps both. Returns the array unchanged where nothing decreases.
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


def _parse_repairing_author_numbering(path, config, model, assembly_id):
    """``parse`` the file, renumbering author ids first where it will not read them.

    Only the asymmetric unit: building an assembly is ``parse``'s to do, and the
    repaired array goes to ``prepare_atom_array``, which does not build one. A
    file that needs both is refused with AtomWorks' own message.
    """
    from atomworks.io import load_pdb
    from atomworks.io.parser import prepare_atom_array
    from atomworks.io.utils.io_utils import get_structure, read_any

    try:
        return parse(path, config=config)
    except ValueError as refused:
        if assembly_id is not None or "non-decreasing order" not in str(refused):
            raise
        file = read_any(path)
        block = getattr(file, "block", None)
        if block is None:
            array, _ = load_pdb(path, model=model)
            if array.coord.ndim == 3:
                array = array[0]
        else:
            array = get_structure(file, model=model, extra_fields=_FIELDS)
        repaired = _renumber_decreasing_author_ids(array)
        if repaired is array:
            raise
        atoms = prepare_atom_array(repaired, config=config, cif_block=block)
        return {"asym_unit": atoms, "cif_block": block}


def read_structure(path, *, model=1, assembly_id=None, use_ccd=True):
    """Read PDB/CIF atoms and available bonds, optionally supplemented from CCD."""
    if model is None or model < 1:
        raise ValueError("The structure reader requires a positive model number")
    if use_ccd:
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

    else:
        array, block = _read_declared(path, model, assembly_id)
    if block is None:
        # A PDB, where the loader has moved off whatever its records called non-polymer.
        array = _polymer_from_backbone_bonds(array)
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
