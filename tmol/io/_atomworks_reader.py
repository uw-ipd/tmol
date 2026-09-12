"""AtomWorks owns parsing and completion; tmol consumes its chemical annotations.

This adapter supports the released keyword API and the local ParseConfig API.
No fallback to a different reader or force-field preparation is implicit.
"""

import inspect

import numpy as np


def read_cif(path, *, model=1):
    if model is None or model < 1:
        raise ValueError("The AtomWorks CIF reader requires a positive model number")
    try:
        from atomworks.io.parser import parse
    except ModuleNotFoundError as error:
        if error.name != "atomworks":
            raise
        raise ImportError(
            "The AtomWorks CIF reader requires the optional dependency: "
            "install 'tmol[atomworks]'."
        ) from error

    options = dict(
        model=model,
        build_assembly=None,
        remove_ccds=[],
        remove_waters=False,
        fix_arginines=False,
        fix_ligands_at_symmetry_centers=False,
        add_bond_types_from_struct_conn=["covale", "disulf"],
        hydrogen_policy="remove",
        ccd_mirror_path=None,
        add_id_and_entity_annotations=False,
        keep_cif_block=True,
    )
    if "config" in inspect.signature(parse).parameters:
        from atomworks.io.config import ParseConfig

        result = parse(
            path,
            config=ParseConfig(
                **options,
                long_bond_policy="keep",
                struct_conn_distance_policy="keep",
            ),
        )
    else:
        # Released 2.x infers a +1 charge on an acetyl carbon after attaching
        # its amide partner. Keep the template charges instead of that obsolete
        # valence heuristic; tmol subsequently owns protonation/MMFF charges.
        result = parse(path, **options, fix_formal_charges=False)
    array = result["asym_unit"]
    if array.coord.ndim == 3:
        array = array[0]
    block = result["cif_block"]
    _check_observed_heavy_atom_names(block, array, model)

    # Reuse the parsed category, including on older releases that do not
    # annotate chem_comp_type. Do not read the file again during preparation.
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
