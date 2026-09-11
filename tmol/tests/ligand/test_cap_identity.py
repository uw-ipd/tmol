"""Capping must preserve source chemistry independently of coordinates."""

import attr
import numpy as np
import pytest

from tmol.ligand._polymer_profile import cap_residue, profile_for_atom_array
from tmol.tests.ligand.test_nonstandard_backbones import _residue, _connection_atoms


@pytest.mark.parametrize("code", ["HYP", "MLE", "B3K", "5CM"])
def test_caps_preserve_retained_annotations_and_topology(code):
    if code == "5CM":
        import biotite.structure as struc
        from tmol.io import atom_array_from_cif
        from tmol.tests.data import data_path

        array = atom_array_from_cif(data_path("ncaa_fixtures", "na_dna_5mc_1d17.cif"))
        source = next(
            res for res in struc.residue_iter(array) if res.res_name[0] == code
        ).copy()
        connections = {"P", "O3'"}
    else:
        source = _residue(code).copy()
        connections = _connection_atoms(code)
    profile = profile_for_atom_array(source, connections)
    source.set_annotation("source_index", np.arange(len(source)))
    source.set_annotation("charge", np.arange(len(source), dtype=np.int32) % 3 - 1)
    source.set_annotation(
        "source_chemistry", np.array([f"atom_{i}_tag" for i in range(len(source))])
    )
    source.ins_code[:] = "B"
    before = source.copy()
    capped, caps = cap_residue(source, profile)
    topology, topology_caps = cap_residue(source, profile, include_coordinates=False)
    assert caps == topology_caps
    assert np.isnan(topology.coord).all()
    np.testing.assert_array_equal(topology.bonds.as_array(), capped.bonds.as_array())
    for annotation in source.get_annotation_categories():
        np.testing.assert_array_equal(
            source.get_annotation(annotation), before.get_annotation(annotation)
        )
        np.testing.assert_array_equal(
            capped.get_annotation(annotation), topology.get_annotation(annotation)
        )
        original = {
            str(name): value
            for name, value in zip(source.atom_name, source.get_annotation(annotation))
        }
        for name, value in zip(capped.atom_name, capped.get_annotation(annotation)):
            if name not in caps.values():
                assert value == original[str(name)], (annotation, name)
    cap_indices = np.isin(capped.atom_name, list(caps.values()))
    assert np.all(capped.charge[cap_indices] == 0)
    assert np.all(capped.source_chemistry[cap_indices] == "")
    assert np.all(capped.ins_code == "B")


def test_topology_only_caps_do_not_construct_frames(monkeypatch):
    import tmol.ligand._polymer_profile as module

    source = _residue("HYP").copy()
    profile = profile_for_atom_array(source, _connection_atoms("HYP"))
    source.coord[:] = np.nan

    def fail(*args):
        raise AssertionError("Topology-only capping must not construct coordinates")

    monkeypatch.setattr(module, "place_atom", fail)
    capped, _ = cap_residue(source, profile, include_coordinates=False)
    assert np.isnan(capped.coord).all()
    assert capped.bonds.get_bond_count() > 0
    assert len(capped) > np.count_nonzero(source.element != "H")


def test_long_cap_names_are_not_truncated():
    source = _residue("HYP").copy()
    profile = profile_for_atom_array(source, _connection_atoms("HYP"))
    rename = {c.name: "synthetic_cap_" + c.name for c in profile.caps}
    profile = attr.evolve(
        profile,
        caps=tuple(
            attr.evolve(
                cap,
                name=rename[cap.name],
                bond_to=rename.get(cap.bond_to, cap.bond_to),
                refs=tuple(rename.get(name, name) for name in cap.refs),
            )
            for cap in profile.caps
        ),
    )
    capped, caps = cap_residue(source, profile)
    assert set(caps.values()) <= set(capped.atom_name)
    assert np.isfinite(capped.coord).all()
