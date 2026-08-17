_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "build_acc_water_V": ("compiled", "build_acc_water_V"),
    "build_acc_water_dV": ("compiled", "build_acc_water_dV"),
    "build_don_water_V": ("compiled", "build_don_water_V"),
    "build_don_water_dV": ("compiled", "build_don_water_dV"),
    "lk_fraction_V": ("compiled", "lk_fraction_V"),
    "lk_fraction_dV": ("compiled", "lk_fraction_dV"),
    "lk_bridge_fraction_V": ("compiled", "lk_bridge_fraction_V"),
    "lk_bridge_fraction_dV": ("compiled", "lk_bridge_fraction_dV"),
    "lk_ball_score_V": ("compiled", "lk_ball_score_V"),
    "lk_ball_score_dV": ("compiled", "lk_ball_score_dV"),
    "LKBallTypeParams": ("compiled", "LKBallTypeParams"),
    "LKBallGlobalParams": ("compiled", "LKBallGlobalParams"),
    "detach_maybe_requires_grad": ("compiled", "detach_maybe_requires_grad"),
    "BuildAcceptorWater": ("compiled", "BuildAcceptorWater"),
    "BuildDonorWater": ("compiled", "BuildDonorWater"),
    "LKFraction": ("compiled", "LKFraction"),
    "LKBridgeFraction": ("compiled", "LKBridgeFraction"),
    "LKBallScore": ("compiled", "LKBallScore"),
    "LKBallScoreFun": ("compiled", "LKBallScoreFun"),
    "ljlk_params": ("test_compiled_lk_ball", "ljlk_params"),
    "atype_params": ("test_compiled_lk_ball", "atype_params"),
    "test_build_acc_waters": ("test_compiled_lk_ball", "test_build_acc_waters"),
    "test_build_don_water": ("test_compiled_lk_ball", "test_build_don_water"),
    "test_lk_fraction": ("test_compiled_lk_ball", "test_lk_fraction"),
    "test_lk_bridge_fraction": ("test_compiled_lk_ball", "test_lk_bridge_fraction"),
    "test_lk_bridge_fraction_overlapping_waters": (
        "test_compiled_lk_ball",
        "test_lk_bridge_fraction_overlapping_waters",
    ),
    "lkball_score_and_gradcheck": (
        "test_compiled_lk_ball",
        "lkball_score_and_gradcheck",
    ),
    "test_lk_ball_donor_donor_spotcheck": (
        "test_compiled_lk_ball",
        "test_lk_ball_donor_donor_spotcheck",
    ),
    "test_lk_ball_sp2_nonpolar_spotcheck": (
        "test_compiled_lk_ball",
        "test_lk_ball_sp2_nonpolar_spotcheck",
    ),
    "test_lk_ball_sp3_ring_spotcheck": (
        "test_compiled_lk_ball",
        "test_lk_ball_sp3_ring_spotcheck",
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
