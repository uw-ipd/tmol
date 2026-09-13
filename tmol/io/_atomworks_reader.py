"""AtomWorks owns parsing and completion; tmol consumes its chemical annotations.

The parser supplies chemical identity and missing atoms. Observed histidine
ring protons remain available to tmol's coordinate-based tautomer selection.
"""

import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse


def read_cif(
    path,
    *,
    model=1,
    assembly_id=None,
    author_fields=False,
    extra_fields=None,
    hydrogen_policy="rebuild",
):
    """Parse bonds before selecting author identifiers; return atoms and CIF data.

    Author identifiers are restored on observed atoms before tmol's legacy
    completion policy runs. Label identifiers use AtomWorks completion directly.
    """
    if model is None or model < 1:
        raise ValueError("The AtomWorks CIF reader requires a positive model number")
    if hydrogen_policy not in ("preserve", "rebuild"):
        raise ValueError("hydrogen_policy must be 'preserve' or 'rebuild'")
    options = dict(
        model=model,
        add_missing_atoms=not author_fields,
        build_assembly=None if assembly_id is None else [assembly_id],
        remove_ccds=[],
        remove_waters=False,
        fix_arginines=False,
        fix_ligands_at_symmetry_centers=False,
        add_bond_types_from_struct_conn=["covale", "disulf"],
        hydrogen_policy="keep",
        ccd_mirror_path=None,
        add_id_and_entity_annotations=False,
        keep_cif_block=True,
        return_atom_array_plus=True,
    )
    author_annotations = {
        "atom_name": "auth_atom_id",
        "res_name": "auth_comp_id",
        "chain_id": "auth_asym_id",
        "res_id": "auth_seq_id",
        "ins_code": "pdbx_PDB_ins_code",
    }
    if author_fields:
        options["extra_fields"] = list(
            dict.fromkeys(
                [*author_annotations.values(), "label_entity_id", *(extra_fields or [])]
            )
        )
    result = parse(
        path,
        config=ParseConfig(
            **options,
            long_bond_policy="keep",
            struct_conn_distance_policy="keep",
        ),
    )
    array = (
        result["asym_unit"]
        if assembly_id is None
        else result["assemblies"][assembly_id]
    )
    if array.coord.ndim == 3:
        array = array[0]
    block = result["cif_block"]
    if assembly_id is not None:
        # AtomWorks distinguishes copies by label chain and transformation.
        # Use that identity for Biotite residue boundaries, including one-residue
        # chains whose author identifiers repeat across copies.
        array.set_annotation("chain_id", array.chain_iid.copy())
    if author_fields:
        for target, source in author_annotations.items():
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
            "label_entity_id",
            *(extra_fields or []),
        }
        if assembly_id is not None:
            retained.update(("auth_asym_id", "chain_iid", "transformation_id"))
        for name in set(array.get_annotation_categories()) - retained:
            array.del_annotation(name)
    is_h = np.isin(np.char.upper(array.element), ["H", "D"])
    observed_his_h = (
        np.isin(array.res_name, ["HIS", "HIS_D", "DHI", "DHIS", "DHIS_D"])
        & np.isin(array.atom_name, ["HD1", "HE2", "HN"])
        & np.isfinite(array.coord).all(axis=-1)
    )
    if hydrogen_policy == "rebuild":
        array = array[~is_h | observed_his_h]

    # Consume chemical types from the already parsed category.
    from tmol.io._cif import _with_component_type_annotation

    return _with_component_type_annotation(array, block), block
