_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "distance_V": ("geom", "distance_V"),
    "distance_V_dV": ("geom", "distance_V_dV"),
    "interior_angle_V": ("geom", "interior_angle_V"),
    "interior_angle_V_dV": ("geom", "interior_angle_V_dV"),
    "cos_interior_angle_V": ("geom", "cos_interior_angle_V"),
    "cos_interior_angle_V_dV": ("geom", "cos_interior_angle_V_dV"),
    "dihedral_angle_V": ("geom", "dihedral_angle_V"),
    "dihedral_angle_V_dV": ("geom", "dihedral_angle_V_dV"),
    "test_distance": ("test_cuda", "test_distance"),
    "geom": ("test_geom", "geom"),
    "dist_vecs": ("test_geom", "dist_vecs"),
    "test_distance_values": ("test_geom", "test_distance_values"),
    "test_distance_gradcheck": ("test_geom", "test_distance_gradcheck"),
    "angle_vecs": ("test_geom", "angle_vecs"),
    "test_interior_angle_values": ("test_geom", "test_interior_angle_values"),
    "test_interior_angle_gradcheck": ("test_geom", "test_interior_angle_gradcheck"),
    "test_cos_interior_angle_values": ("test_geom", "test_cos_interior_angle_values"),
    "test_cos_interior_angle_gradcheck": (
        "test_geom",
        "test_cos_interior_angle_gradcheck",
    ),
    "dihedral_points": ("test_geom", "dihedral_points"),
    "test_dihedral_angle_values": ("test_geom", "test_dihedral_angle_values"),
    "test_dihedral_angle_values_gradcheck": (
        "test_geom",
        "test_dihedral_angle_values_gradcheck",
    ),
    "DihedralDat": ("test_geom", "DihedralDat"),
    "dihedral_test_data": ("test_geom", "dihedral_test_data"),
    "test_coord_dihedrals": ("test_geom", "test_coord_dihedrals"),
    "test_coord_dihedral_angle_gradcheck": (
        "test_geom",
        "test_coord_dihedral_angle_gradcheck",
    ),
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
