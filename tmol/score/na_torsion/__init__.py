_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "SUGAR_SLOTS": ("na_torsion_energy_term", "SUGAR_SLOTS"),
    "NaTorsionEnergyTerm": ("na_torsion_energy_term", "NaTorsionEnergyTerm"),
    "eval_na_torsion_for_pose": ("na_torsion_energy_term", "eval_na_torsion_for_pose"),
    "na_torsion_subterms": ("na_torsion_energy_term", "na_torsion_subterms"),
    "eval_na_torsion_for_rotamers": (
        "na_torsion_energy_term",
        "eval_na_torsion_for_rotamers",
    ),
    "BACKBONE_TORSIONS": ("params", "BACKBONE_TORSIONS"),
    "SUGAR_TORSIONS": ("params", "SUGAR_TORSIONS"),
    "CHI_TORSION": ("params", "CHI_TORSION"),
    "TORSION_NAMES": ("params", "TORSION_NAMES"),
    "TORSION_IND": ("params", "TORSION_IND"),
    "POLYMERS": ("params", "POLYMERS"),
    "POLYMER_IND": ("params", "POLYMER_IND"),
    "BASES": ("params", "BASES"),
    "N_BASE_PER_POLYMER": ("params", "N_BASE_PER_POLYMER"),
    "N_BASE": ("params", "N_BASE"),
    "BASE_FOR_NAME3": ("params", "BASE_FOR_NAME3"),
    "N_PUCKER": ("params", "N_PUCKER"),
    "NORTH_PUCKERS": ("params", "NORTH_PUCKERS"),
    "SYN_MEAN": ("params", "SYN_MEAN"),
    "SYN_RANGE": ("params", "SYN_RANGE"),
    "N_TORSION": ("params", "N_TORSION"),
    "DELTA": ("params", "DELTA"),
    "CHI": ("params", "CHI"),
    "REQUIRED_TORSIONS": ("params", "REQUIRED_TORSIONS"),
    "sugar_ring_atoms": ("params", "sugar_ring_atoms"),
    "block_type_params": ("params", "block_type_params"),
    "polymer_index": ("params", "polymer_index"),
    "NaTorsionParams": ("params", "NaTorsionParams"),
    "RAD": ("potentials", "RAD"),
    "wrap_degrees": ("potentials", "wrap_degrees"),
    "dihedral": ("potentials", "dihedral"),
    "pucker_weights": ("potentials", "pucker_weights"),
    "blended_devsq": ("potentials", "blended_devsq"),
    "triple_bin_weights": ("potentials", "triple_bin_weights"),
    "bi_bii_weight": ("potentials", "bi_bii_weight"),
    "syn_weight": ("potentials", "syn_weight"),
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
