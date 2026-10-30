"""Per-term exact-pairwise protein--ligand scoring characterization.

The CPU/float64 values below pin weighted beta2016 block-pair energies.  The
coordinates originate in the normal float32 pose-construction path and are cast
to the requested scoring dtype before the score function is rendered.  Thus the
dtype comparison characterizes scoring arithmetic without changing fixture
coordinates.

``_TERM_ORDER`` is the row order returned by
``ScoreFunction.render_block_pair_scoring_module(...)(sum_terms=False)``.  The
interface values sum both orientations of every ligand--receptor block pair;
the ligand-intra values sum only ligand-block diagonal entries.
"""

from __future__ import annotations

from pathlib import Path

from tmol.tests.data import data_path

import numpy
import pytest
import torch

from tmol.tests.ligand.test_protein_ligand_ddg import (
    LIGAND_RES_NAME,
    PLI_DATA_DIR,
    _GOLDEN_DDG,
    _ligand_block_mask,
    _load_complex_cif,
)

_TARGETS = ("cdk2", "ache", "hsp90", "egfr", "fxa")

# Rows from the rendered beta2016 block-pair scoring module, in order.
_TERM_ORDER = (
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "hbond",
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
    "disulfide",
    "rama",
    "omega",
    "dunbrack_rot",
    "dunbrack_rotdev",
    "dunbrack_semirot",
    "lk_ball_iso",
    "lk_ball",
    "lk_bridge",
    "lk_bridge_uncpl",
    "ref",
    "gen_torsions",
    "na_torsion",
    "na_torsion_well",
    "metal_coordination",
)

# Weighted per-term energies captured on CPU with float64 scoring arithmetic.
_GOLDEN_PER_TERM = {
    "cdk2": {
        "interface": (
            -25.57994719793408,
            7.733001951002542,
            3.204408370084265,
            -3.6748593118669985,
            -2.646444475872238,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -5.360572185831259,
            4.992669350254961,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "ligand_intra": (
            -6.4435003413029825,
            13.234962611206427,
            2.1708860276980757,
            -1.7914790534516132,
            0.0,
            6.275798793639213e-05,
            2.0794549302364826e-05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.15434299374764607,
            0.13450682189184834,
            0.0,
            0.0,
            0.0,
            0.7656795017136948,
            0.0,
            0.0,
        ),
    },
    "ache": {
        "interface": (
            -34.71272320594154,
            6.127934525283625,
            -0.7255291909495671,
            -3.1800032082265517,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -4.0577942028133895,
            3.472953157090236,
            -0.02213571075719007,
            -0.20064770774122723,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "ligand_intra": (
            -14.74182473058658,
            33.756858919475775,
            0.2100107591446559,
            0.6081140335169357,
            0.0,
            0.00020562074249141376,
            0.00010317600530060272,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.1813238439813437,
            0.0,
            0.0,
            0.0,
            0.0,
            4.395738252918556,
            0.0,
            0.0,
        ),
    },
    "hsp90": {
        "interface": (
            -33.92566997149779,
            8.415776455177602,
            -1.070959167383983,
            -2.090682803371836,
            -1.179686365963039,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -4.234987867412792,
            2.7718565021746198,
            -0.042205038438266285,
            -0.393361626636389,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "ligand_intra": (
            -14.527495577736872,
            24.47271604443152,
            -0.4812491259090431,
            -0.05348168964646799,
            0.0,
            0.00011762119922817827,
            8.899044999426378e-05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.11235190936931966,
            0.002981632255762639,
            0.0,
            0.0,
            0.0,
            19.04947754908467,
            0.0,
            0.0,
        ),
    },
    "egfr": {
        "interface": (
            -32.904613763466344,
            16.749796718400624,
            6.637264207033996,
            -2.0683949386254534,
            -2.132347675868545,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -6.561520471882825,
            6.618740447771341,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "ligand_intra": (
            -13.091177588657885,
            53.23623571644305,
            0.9396294391204676,
            -3.367657474793671,
            0.0,
            0.0001584601780466463,
            9.18208842524211e-05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.1689152117109205,
            0.08164037741952326,
            0.0,
            0.0,
            0.0,
            14.211709687073139,
            0.0,
            0.0,
        ),
    },
    "fxa": {
        "interface": (
            -38.84036186155282,
            4.070367802274902,
            15.671959640568954,
            -5.674336697721668,
            -1.3564967225867783,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -10.44107654252881,
            7.4289054334822335,
            -0.02035189328532385,
            -0.2089615310035006,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "ligand_intra": (
            -14.335343165857466,
            18.83626961686039,
            4.363591765578374,
            -3.4267800632614978,
            0.0,
            3.627618589199711e-08,
            3.465875313738949e-08,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.3241640663003018,
            0.0,
            -0.0017854937111996842,
            -0.009985818515823428,
            0.0,
            9.634915000414983,
            0.0,
            0.0,
        ),
    },
}
# ``metal_coordination`` post-dates the capture and is zero for these metal-free
# complexes, so the goldens gain a trailing zero row entry.
_GOLDEN_PER_TERM = {
    target: {name: (*values, 0.0) for name, values in rows.items()}
    for target, rows in _GOLDEN_PER_TERM.items()
}

_ROSETTA_INTERFACE_COLUMNS = {
    "fa_ljatr": ("dG_fa_atr",),
    "fa_ljrep": ("dG_fa_rep",),
    "fa_lk": ("dG_fa_sol",),
    "fa_elec": ("dG_fa_elec",),
    "hbond": (
        "dG_hbond_sc",
        "dG_hbond_bb_sc",
        "dG_hbond_sr_bb",
        "dG_hbond_lr_bb",
    ),
    "cart_hxltorsions": ("dG_hxl_tors",),
    "lk_ball_iso": ("dG_lk_ball_iso",),
    "lk_ball": ("dG_lk_ball",),
    "lk_bridge": ("dG_lk_ball_bridge",),
    "lk_bridge_uncpl": ("dG_lk_ball_bridge_uncpl",),
    "gen_torsions": ("dG_gen_bonded",),
}

_GRID_INTERACTION_TERMS = (
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "hbond",
    "lk_ball_iso",
    "lk_ball",
    "lk_bridge",
    "lk_bridge_uncpl",
)


def _rosetta_interface_terms(sc_path: Path) -> dict[str, float]:
    """Read weighted interface terms from a Rosetta scorefile.

    Ligand-intra ``dG_fa_intra_*`` columns are intentionally not read: they are
    bound-minus-unbound differences and are zero when intramolecular energies
    cancel, rather than isolated ligand-intra energies.
    """
    with open(sc_path) as scorefile:
        score_lines = [
            line.split()[1:] for line in scorefile if line.startswith("SCORE:")
        ]
    assert len(score_lines) == 2, f"expected one score row in {sc_path}"
    columns = dict(zip(score_lines[0], score_lines[1]))
    return {
        term: sum(float(columns[column]) for column in source_columns)
        for term, source_columns in _ROSETTA_INTERFACE_COLUMNS.items()
    }


def _weighted_perterm_from_fixtures(
    target: str, torch_device: torch.device, dtype: torch.dtype
):
    from tmol.database import ParameterDatabase
    from tmol.io._pose_stack_from_biotite import pose_stack_from_atom_array
    from tmol.score import beta2016_score_function
    from tmol.ops import calculate_block_pair_ddg

    pose_stack, context = pose_stack_from_atom_array(
        _load_complex_cif(target),
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol")],
        no_optH=True,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    pose_stack.coords = pose_stack.coords.to(dtype=dtype)
    ligand_mask = _ligand_block_mask(pose_stack, torch_device)
    sfxn = beta2016_score_function(torch_device, param_db=context.parameter_database)

    interface = calculate_block_pair_ddg(
        pose_stack,
        ligand_mask,
        sfxn=sfxn,
        sum_terms=False,
        minimize=False,
        pack=False,
        database=context.parameter_database,
    )
    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
    ligand_intra = (
        block_pair_scores.diagonal(dim1=2, dim2=3)
        .masked_fill(~ligand_mask.unsqueeze(0), 0)
        .sum(dim=2)
    )

    term_order = tuple(score_type.name for score_type in sfxn.all_score_types())
    ligand_types = [
        block_type
        for block_type in pose_stack.packed_block_types.active_block_types
        if block_type.name3 == LIGAND_RES_NAME
    ]
    ligand_charge = sum(
        float(pose_stack.packed_block_types.elec_partial_charge[block_type_ind].sum())
        for block_type_ind, block_type in enumerate(
            pose_stack.packed_block_types.active_block_types
        )
        if block_type.name3 == LIGAND_RES_NAME
    )
    return interface[:, 0], ligand_intra[:, 0], term_order, ligand_types, ligand_charge


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(
    "dtype", (torch.float64, torch.float32), ids=("float64", "float32")
)
def test_ligand_receptor_perterm_baseline(target, dtype, torch_device) -> None:
    """Pin weighted interface/intra rows across device and scoring dtype."""
    interface, ligand_intra, term_order, ligand_types, ligand_charge = (
        _weighted_perterm_from_fixtures(target, torch_device, dtype)
    )
    assert term_order == _TERM_ORDER
    assert interface.shape == ligand_intra.shape == (len(_TERM_ORDER),)
    assert torch.isfinite(interface).all()
    assert torch.isfinite(ligand_intra).all()

    # CPU fp32-vs-fp64 characterization over all targets measured a maximum
    # absolute row difference of 1.43e-5.  The fp32 bound is a COMBINED envelope
    # ``atol + rtol*|expected|``: with atol=5e-5, rtol=1e-6 the effective limit is
    # ~5e-5 for small rows and ~1.0e-4 for the largest row (~53 REU).  That
    # comfortably covers the measured CPU deviation plus device/reduction-order
    # variation (the CUDA lane, jobs 16248688, passes within this envelope).
    if dtype == torch.float64:
        rtol, atol = 1e-6, 1e-6
    else:
        rtol, atol = 1e-6, 5e-5
    numpy.testing.assert_allclose(
        interface.detach().cpu().numpy(),
        _GOLDEN_PER_TERM[target]["interface"],
        rtol=rtol,
        atol=atol,
    )
    numpy.testing.assert_allclose(
        ligand_intra.detach().cpu().numpy(),
        _GOLDEN_PER_TERM[target]["ligand_intra"],
        rtol=rtol,
        atol=atol,
    )

    # Keep the established weighted-total regression visible in this per-term
    # oracle.  This intentionally uses its original tolerance unchanged.
    numpy.testing.assert_allclose(
        float(interface.sum().detach()),
        _GOLDEN_DDG[target],
        rtol=1e-3,
        atol=1e-3,
    )

    term_index = {term: index for index, term in enumerate(term_order)}
    assert set(_GRID_INTERACTION_TERMS).issubset(term_index)
    assert torch.isfinite(
        interface[[term_index[term] for term in _GRID_INTERACTION_TERMS]]
    ).all()

    # Every selected ligand has at least one named rotatable torsion, and its
    # intra-block generic torsional energy is represented in the oracle.
    assert sum(block_type.n_torsions for block_type in ligand_types) > 0
    assert ligand_intra[term_index["gen_torsions"]] != 0

    # ache carries approximately +1 net partial charge in its committed params.
    if target == "ache":
        assert ligand_charge > 0.5
        assert interface[term_index["fa_elec"]] != 0

    rosetta = _rosetta_interface_terms(
        data_path("protein_ligand_test") / f"{target}_complex.sc"
    )
    actual_rosetta_terms = numpy.array(
        [float(interface[term_index[term]].detach()) for term in rosetta]
    )
    expected_rosetta_terms = numpy.array(list(rosetta.values()))
    # The scorefiles print three decimals.  Across these fixtures the largest
    # observed tmol/Rosetta interface spread is 0.1391 REU (fxa lk_ball), so
    # this secondary parity sanity check uses a documented 0.2 REU bound.
    numpy.testing.assert_allclose(
        actual_rosetta_terms, expected_rosetta_terms, rtol=0.0, atol=0.2
    )
