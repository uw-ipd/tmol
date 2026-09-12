"""AtomWorks owns parsing and completion; tmol consumes its chemical annotations.

The parser supplies chemical identity and missing atoms. Observed histidine
ring protons remain available to tmol's coordinate-based tautomer selection.
"""

import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse


def read_cif(path, *, model=1):
    if model is None or model < 1:
        raise ValueError("The AtomWorks CIF reader requires a positive model number")
    options = dict(
        model=model,
        build_assembly=None,
        remove_ccds=[],
        remove_waters=False,
        fix_arginines=False,
        fix_ligands_at_symmetry_centers=False,
        add_bond_types_from_struct_conn=["covale", "disulf"],
        hydrogen_policy="keep",
        ccd_mirror_path=None,
        add_id_and_entity_annotations=False,
        keep_cif_block=True,
    )
    result = parse(
        path,
        config=ParseConfig(
            **options,
            long_bond_policy="keep",
            struct_conn_distance_policy="keep",
        ),
    )
    array = result["asym_unit"]
    if array.coord.ndim == 3:
        array = array[0]
    is_h = np.isin(np.char.upper(array.element), ["H", "D"])
    observed_his_h = (
        np.isin(array.res_name, ["HIS", "HIS_D", "DHIS"])
        & np.isin(array.atom_name, ["HD1", "HE2", "HN"])
        & np.isfinite(array.coord).all(axis=-1)
    )
    array = array[~is_h | observed_his_h]
    block = result["cif_block"]
    _check_observed_heavy_atom_names(block, array, model)

    # Consume chemical types from the already parsed category.
    from tmol.io._cif import _with_component_type_annotation

    return _with_component_type_annotation(array, block)


def _check_observed_heavy_atom_names(block, array, model):
    """Catch parser template substitution that deletes observed atom identities.

    This is a component/name inventory check, not proof that every residue or
    bond survived. AtomWorks still owns altloc and leaving-group policies.
    Inspect its raw category in memory rather than parsing coordinates twice.
    """
    site = block["atom_site"]
    names = site["label_atom_id"].as_array(str)
    components = site["label_comp_id"].as_array(str)
    elements = site["type_symbol"].as_array(str)
    selected = ~np.isin(np.char.upper(elements), ("H", "D"))
    if "pdbx_PDB_model_num" in site:
        selected &= site["pdbx_PDB_model_num"].as_array(int) == model
    retained = set(zip(array.res_name.tolist(), array.atom_name.tolist()))
    lost = set(zip(components[selected], names[selected])) - retained
    if lost:
        details = ", ".join(f"{res}.{atom}" for res, atom in sorted(lost))
        raise ValueError(
            "AtomWorks completion removed source heavy-atom names: "
            f"{details}. Resolve the input chemical definition or explicitly "
            "select the intended atoms before pose construction."
        )
