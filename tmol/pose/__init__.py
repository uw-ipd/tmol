_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ConstraintSet": ("constraint_set", "ConstraintSet"),
    "residue_types_from_residues": (
        "packed_block_types",
        "residue_types_from_residues",
    ),
    "PackedBlockTypes": ("packed_block_types", "PackedBlockTypes"),
    "DEFAULT_ATOM_OCCUPANCY": ("pdb_info", "DEFAULT_ATOM_OCCUPANCY"),
    "DEFAULT_ATOM_B_FACTOR": ("pdb_info", "DEFAULT_ATOM_B_FACTOR"),
    "PDBInfo": ("pdb_info", "PDBInfo"),
    "PoseStack": ("pose_stack", "PoseStack"),
    "PoseStackBuilder": ("pose_stack_builder", "PoseStackBuilder"),
    "SeqToken": ("sequence", "SeqToken"),
    "tokenize_sequences": ("sequence", "tokenize_sequences"),
    "smiles_in_tokens": ("sequence", "smiles_in_tokens"),
    "resolve_block_type_names": ("sequence", "resolve_block_type_names"),
    "EXTENDED_BACKBONE_TORSIONS": ("util", "EXTENDED_BACKBONE_TORSIONS"),
    "extended_pose_stack_from_sequences": (
        "util",
        "extended_pose_stack_from_sequences",
    ),
    "get_torsion_names": ("util", "get_torsion_names"),
    "get_named_torsions": ("util", "get_named_torsions"),
    "set_named_torsions": ("util", "set_named_torsions"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
