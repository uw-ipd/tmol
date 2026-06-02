"""Unit tests for CHI / PROTON_CHI topology classification and params IO.

Complements the ground-truth parity assertions in
``test_ligand_pipeline.py::TestGroundTruthRegression`` with focused negative
cases, sp3 sample / EXTRA-expansion checks, quad validity, NU-unsupported
confirmation, and the ``params_io`` CHI/PROTON_CHI round-trip.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.ligand.atom_typing import assign_tmol_atom_types
from tmol.ligand.residue_builder import build_residue_type


def _restype_from_smiles(smi: str, name: str = "LIG"):
    """Prepare a RawResidueType from SMILES via the real typing+build path.

    3D coordinates are embedded so the atom-tree root is geometry-based (as in
    the real pipeline, where ligands always carry coords); without coords the
    root degenerates to atom index 0, which can land on a terminal heteroatom.
    """
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, smi
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0, f"embed failed: {smi}"
    atom_types, state = assign_tmol_atom_types(mol, return_state=True)
    return build_residue_type(mol, name, atom_types, typing_state=state)


def _axes(restype):
    return {frozenset((t.b.atom, t.c.atom)) for t in restype.torsions}


def _chi_signature(restype):
    """Name-independent CHI signature: (n_heavy, n_proton, sorted proton samples)."""
    n_proton = len(restype.chi_samples)
    n_heavy = len(restype.torsions) - n_proton
    proton = sorted(tuple(cs.samples) for cs in restype.chi_samples)
    return (n_heavy, n_proton, proton)


def _smiles_to_mol2(smi: str, name: str) -> str:
    """Write a TRIPOS .mol2 (3D + explicit bond orders) for a SMILES molecule.

    Used to exercise the mol2 prep path on a molecule whose SMILES-path CHI
    topology is known (openbabel is unavailable to generate one otherwise).
    """
    _HYB = {
        Chem.HybridizationType.SP3: "3",
        Chem.HybridizationType.SP2: "2",
        Chem.HybridizationType.SP: "1",
    }
    _ORDER = {
        Chem.BondType.SINGLE: "1",
        Chem.BondType.DOUBLE: "2",
        Chem.BondType.TRIPLE: "3",
        Chem.BondType.AROMATIC: "ar",
    }
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    conf = mol.GetConformer()
    lines = [
        "@<TRIPOS>MOLECULE",
        name,
        f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    counts: dict[str, int] = {}
    for a in mol.GetAtoms():
        el = a.GetSymbol()
        counts[el] = counts.get(el, 0) + 1
        p = conf.GetAtomPosition(a.GetIdx())
        if a.GetAtomicNum() == 1:
            sybyl = "H"
        elif a.GetIsAromatic():
            sybyl = f"{el}.ar"
        else:
            sybyl = f"{el}.{_HYB.get(a.GetHybridization(), '3')}"
        lines.append(
            f"{a.GetIdx() + 1} {el}{counts[el]} {p.x:.4f} {p.y:.4f} {p.z:.4f} "
            f"{sybyl} 1 {name} 0.0000"
        )
    lines.append("@<TRIPOS>BOND")
    for i, b in enumerate(mol.GetBonds(), start=1):
        lines.append(
            f"{i} {b.GetBeginAtomIdx() + 1} {b.GetEndAtomIdx() + 1} "
            f"{_ORDER.get(b.GetBondType(), '1')}"
        )
    return "\n".join(lines) + "\n"


def _adjacency(restype):
    adj = set()
    for a, b, *_rest in restype.bonds:
        adj.add(frozenset((a, b)))
    return adj


# --- AC-1 negative cases: no spurious rotatable bonds ---------------------


def test_benzene_emits_no_chi():
    rt = _restype_from_smiles("c1ccccc1", "BNZ")
    assert rt.torsions == ()
    assert rt.chi_samples == ()


def test_toluene_methyl_is_apolar_h_skipped():
    # Methyl-on-aromatic: the only reference atom is an apolar H -> hapol skip.
    rt = _restype_from_smiles("Cc1ccccc1", "TOL")
    assert rt.torsions == ()


def test_ethane_apolar_skipped():
    rt = _restype_from_smiles("CC", "ETA")
    assert rt.torsions == ()


# --- AC-7: ring-pucker (NU) is unsupported / not emitted ------------------


def test_saturated_ring_no_chi_and_no_nu():
    # Cyclohexane: all C-C bonds are ring-internal -> no rotatable CHI, and
    # tmol emits no NU / ring-pucker DOFs (NU is explicitly unsupported).
    rt = _restype_from_smiles("C1CCCCC1", "CHX")
    assert rt.torsions == ()
    assert rt.chi_samples == ()


# --- AC-1 / AC-2: heavy + proton chis, sp3 samples + EXTRA ----------------


def test_ethylene_glycol_heavy_and_proton_chis():
    # HOCH2-CH2OH: one heavy C-C chi + two sp3 hydroxyl proton chis.
    rt = _restype_from_smiles("OCCO", "EDO")
    proton_names = {cs.chi_dihedral for cs in rt.chi_samples}
    assert len(rt.chi_samples) == 2
    heavy = [t for t in rt.torsions if t.name not in proton_names]
    assert len(heavy) == 1
    for cs in rt.chi_samples:
        # sp3 heteroatom -> 60/-60/180 (NOT the sp2 0/180 set)
        assert sorted(cs.samples) == sorted((60.0, -60.0, 180.0))
        assert sorted(cs.samples) != sorted((0.0, 180.0))
        # 9^2 = 81 <= MAX_CONFS -> EXTRA 1 20 -> one 20-degree expansion
        assert cs.expansions == (20.0,)


def test_tetraol_extra_expansion_overflow():
    # Four sp3 hydroxyls -> 9^4 = 6561 > MAX_CONFS(5000) -> EXTRA 0.
    rt = _restype_from_smiles("OCC(O)C(O)CO", "TTL")
    assert len(rt.chi_samples) == 4
    for cs in rt.chi_samples:
        assert sorted(cs.samples) == sorted((60.0, -60.0, 180.0))
        assert cs.expansions == ()


def test_extra_counts_skipped_conjugated_polar_h():
    # num_H_confs is computed over the PRE-skip polar-H set (RosettaVS parity).
    # This triol-diacid has 3 emitted aliphatic hydroxyl chis (sp3, 9 each) plus
    # TWO conjugated carboxylic O-H that are skipped from emission but STILL
    # counted: 9^3 * (>=6)^2 = at least 26244 > MAX_CONFS(5000) -> EXTRA 0.
    # If the skipped acids were not counted, 9^3 = 729 <= 5000 would wrongly
    # give EXTRA 1 20.
    rt = _restype_from_smiles("OCC(O)C(O)C(C(=O)O)C(=O)O", "TDA")
    assert len(rt.chi_samples) == 3  # only the 3 aliphatic O-H emitted
    for cs in rt.chi_samples:
        assert cs.expansions == ()  # the 2 skipped acid O-H are counted


def test_fused_ring_emits_no_ring_internal_chi():
    # Ring-closure bonds are never atom-tree edges, so they are never CHI
    # candidates (implicit handling of RosettaVS ring_cuts / FT_connected).
    # Decalin and naphthalene have only ring-internal bonds -> no chi.
    for smi, name in [("C1CCC2CCCCC2C1", "DEC"), ("c1ccc2ccccc2c1", "NAP")]:
        rt = _restype_from_smiles(smi, name)
        assert rt.torsions == (), f"{name} should emit no ring-internal chi"


# --- AC-9: torsion-quad validity ------------------------------------------


def test_torsion_quads_are_valid_bonded_paths():
    rt = _restype_from_smiles("OCC(O)C(O)CO", "TTL")
    adj = _adjacency(rt)
    proton_names = {cs.chi_dihedral for cs in rt.chi_samples}
    h_names = {a.name for a in rt.atoms if a.atom_type.startswith("H")}
    for t in rt.torsions:
        quad = [t.a.atom, t.b.atom, t.c.atom, t.d.atom]
        assert None not in quad
        assert len(set(quad)) == 4, f"{t.name} quad not distinct: {quad}"
        # a-b, b-c, c-d are all real bonds (a bonded path)
        for x, y in zip(quad, quad[1:]):
            assert frozenset((x, y)) in adj, f"{t.name}: {x}-{y} not bonded"
        # proton chis: the tip d is the rotated polar hydrogen
        if t.name in proton_names:
            assert t.d.atom in h_names, f"{t.name} proton tip {t.d.atom} not H"


# --- AC-8: params_io CHI / PROTON_CHI round-trip --------------------------


def test_params_io_chi_proton_chi_roundtrip(tmp_path):
    from tmol.ligand.params_io import read_params_file, write_params_file

    rt = _restype_from_smiles("OCCO", "EDO")
    assert rt.torsions and rt.chi_samples  # has both heavy + proton chis
    out = tmp_path / "edo.params"
    write_params_file(rt, out)

    text = out.read_text()
    assert "CHI " in text and "PROTON_CHI " in text
    assert "#biaryl" not in text  # Rosetta-only annotation never written

    rt2 = read_params_file(out)
    # semantic axis set preserved
    assert _axes(rt2) == _axes(rt)
    # proton chi samples + expansions preserved (keyed by axis)
    def proton_by_axis(r):
        axis = {t.name: frozenset((t.b.atom, t.c.atom)) for t in r.torsions}
        return {
            axis[cs.chi_dihedral]: (tuple(cs.samples), tuple(cs.expansions))
            for cs in r.chi_samples
        }

    assert proton_by_axis(rt2) == proton_by_axis(rt)


def test_params_io_ignores_biaryl_comment(tmp_path):
    from tmol.ligand.params_io import read_params_file

    params = tmp_path / "x.params"
    params.write_text(
        "NAME LIG\nIO_STRING LIG Z\nTYPE LIGAND\nAA UNK\n"
        "ATOM  C1   CS  X 0.0\nATOM  N1   Nad X 0.0\n"
        "ATOM  C2   CS  X 0.0\nATOM  C3   CS  X 0.0\n"
        "CHI 1 C3 N1 C2 C1 #biaryl\n"
        "NBR_ATOM C1\nNBR_RADIUS 999.0\n"
    )
    rt = read_params_file(params)
    assert len(rt.torsions) == 1
    t = rt.torsions[0]
    assert t.name == "chi1"
    assert (t.a.atom, t.b.atom, t.c.atom, t.d.atom) == ("C3", "N1", "C2", "C1")


# --- AC-3: refine resolves chi torsions; AC-4: sampler activation ----------


def _refined(smi: str, name: str):
    import cattr

    from tmol.chemical.restypes import RefinedResidueType

    rt = _restype_from_smiles(smi, name)
    refined = cattr.structure(cattr.unstructure(rt), RefinedResidueType)
    return rt, refined


def test_refine_resolves_chi_torsion_uaids():
    # AC-3: every chi_samples.chi_dihedral references a named torsion, and the
    # RefinedResidueType exposes a resolvable torsion_to_uaids for chi1..chiN.
    rt, refined = _refined("OCCO", "EDO")
    torsion_names = {t.name for t in rt.torsions}
    for cs in rt.chi_samples:
        assert cs.chi_dihedral in torsion_names
    for t in rt.torsions:
        assert t.name in refined.torsion_to_uaids
        assert len(refined.torsion_to_uaids[t.name]) == 4


def test_opth_sampler_activates_for_polar_h_ligand():
    # AC-4: a hydroxyl ligand carries proton chi_samples -> OptHSampler active.
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    rt, refined = _refined("OCCO", "EDO")
    assert rt.torsions and rt.chi_samples
    assert OptHSampler().defines_rotamers_for_rt(refined) is True


def test_opth_sampler_inactive_without_polar_h():
    # AC-4 negative: benzene has no chi_samples -> OptHSampler inactive.
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    _rt, refined = _refined("c1ccccc1", "BNZ")
    assert not refined.chi_samples
    assert OptHSampler(flip_NHQ=False).defines_rotamers_for_rt(refined) is False


def test_polymer_samplers_skip_ligands(default_database, torch_device):
    # AC-4: heavy-chi rotamer samplers are polymer-guarded -> never fire on
    # non-polymer ligands, regardless of emitted torsions/chi_samples.
    from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
    from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
    from tmol.score.dunbrack.params import DunbrackParamResolver

    _rt, refined = _refined("OCCO", "EDO")
    assert FixedAAChiSampler().defines_rotamers_for_rt(refined) is False

    resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    assert DunbrackChiSampler(resolver).defines_rotamers_for_rt(refined) is False


# --- AC-1 conjugation / biaryl edge cases (negative + positive matrix) -----


def test_conjugated_polar_h_not_emitted():
    # Conjugated polar hydrogens (aromatic-attached N-H/O-H, carbonyl O-H/N-H)
    # are skipped, matching RosettaVS's conjugated-polar-H rule.
    for smi, name in [
        ("Nc1ccccc1", "ANI"),  # aniline
        ("Oc1ccccc1", "PHN"),  # phenol
        ("CC(=O)O", "ACD"),  # acetic acid
        ("CC(=O)N", "AMD"),  # primary amide
    ]:
        rt = _restype_from_smiles(smi, name)
        assert rt.chi_samples == (), f"{name}: conjugated polar-H should be skipped"


def test_aliphatic_polar_h_emitted():
    # Non-conjugated aliphatic O-H / N-H ARE emitted as proton chis.
    for smi, name in [("CCO", "EOH"), ("CCN", "EAM")]:
        rt = _restype_from_smiles(smi, name)
        assert len(rt.chi_samples) == 1, f"{name}: aliphatic polar-H expected"


def test_biaryl_single_bond_is_heavy_chi():
    # report_ringring_chi=True (default): an aromatic-aromatic single bond
    # (biphenyl pivot) is emitted as a heavy CHI, not skipped.
    rt = _restype_from_smiles("c1ccccc1-c2ccccc2", "BPH")
    assert len(rt.chi_samples) == 0
    assert len(rt.torsions) == 1  # the single inter-ring axis


# --- Round 2: strained ring negative; heavy-only OptHSampler negative -------


def test_strained_ring_emits_no_chi():
    # 3-membered ring: all ring bonds are strained ring-internal -> no chi.
    rt = _restype_from_smiles("C1CC1", "CPR")
    assert rt.torsions == ()
    assert rt.chi_samples == ()


def test_opth_inactive_for_heavy_only_chi_ligand():
    # AC-4 negative: biphenyl has a heavy CHI but NO proton chi_samples ->
    # OptHSampler must NOT define rotamers for it.
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    _rt, refined = _refined("c1ccccc1-c2ccccc2", "BPH")
    assert refined.torsions and not refined.chi_samples
    assert OptHSampler(flip_NHQ=False).defines_rotamers_for_rt(refined) is False


# --- Round 2: AC-8 params_io read->write->read + tmol YAML round-trip -------


def _proton_by_axis(restype):
    axis = {t.name: frozenset((t.b.atom, t.c.atom)) for t in restype.torsions}
    return {
        axis[cs.chi_dihedral]: (tuple(cs.samples), tuple(cs.expansions))
        for cs in restype.chi_samples
    }


def test_params_io_read_write_read_roundtrip(tmp_path):
    # Start from a hand-written .params with CHI + PROTON_CHI, write it back,
    # read again; the semantic content (axes, samples, expansions) is stable.
    from tmol.ligand.params_io import read_params_file, write_params_file

    src = tmp_path / "src.params"
    src.write_text(
        "NAME LIG\nIO_STRING LIG Z\nTYPE LIGAND\nAA UNK\n"
        "ATOM  C1  CS  X 0.0\nATOM  C2  CS  X 0.0\nATOM  C3  CS  X 0.0\n"
        "ATOM  O1  Ohx X 0.0\nATOM  HO1 HO  X 0.0\n"
        "CHI 1  C3 C2 C1 O1\n"
        "CHI 2  C2 C1 O1 HO1\nPROTON_CHI 2 SAMPLES 3 60 -60 180 EXTRA 1 20\n"
        "NBR_ATOM C1\nNBR_RADIUS 999.0\n"
    )
    rt1 = read_params_file(src)
    assert len(rt1.torsions) == 2 and len(rt1.chi_samples) == 1
    out = tmp_path / "out.params"
    write_params_file(rt1, out)
    rt2 = read_params_file(out)
    assert _axes(rt2) == _axes(rt1)
    assert _proton_by_axis(rt2) == _proton_by_axis(rt1)
    # the proton-chi expansion survived (EXTRA 1 20 -> (20.0,))
    assert rt2.chi_samples[0].expansions == (20.0,)


def test_tmol_yaml_roundtrip_preserves_chi(tmp_path):
    # tmol .tmol YAML path (params_file.py) round-trips non-empty torsions and
    # chi_samples. cartbonded is built with the same helper the prep pipeline
    # uses; charges are dummy (orthogonal to chi topology).
    from tmol.ligand.params_file import load_params_file, write_params_file
    from tmol.ligand.registry import _build_cartbonded_params

    rt = _restype_from_smiles("OCCO", "EDO")
    assert rt.torsions and rt.chi_samples
    charges = {a.name: 0.0 for a in rt.atoms}
    cart = _build_cartbonded_params(rt)

    out = tmp_path / "edo.tmol"
    write_params_file(out, [rt], {"EDO": charges}, {"EDO": cart})
    rt2 = load_params_file(out)[0].residue_type
    assert _axes(rt2) == _axes(rt)
    assert _proton_by_axis(rt2) == _proton_by_axis(rt)


# --- Round 2: AC-3 ref1/ref2 inject -> ParameterDatabase -> CanonicalOrdering


REF_SMILES = {
    "ref1": "Cc1cn([C@@H]2C[C@@H](O)[C@H](CO)O2)c(=O)[nH]c1=O",
    "ref2": "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C",
}


@pytest.mark.parametrize("name", ["ref1", "ref2"])
def test_ref_ligand_injects_and_builds_canonical_ordering(name):
    import cattr

    from tmol.chemical.restypes import RefinedResidueType
    from tmol.ligand.preparation import prepare_ligand_from_smiles

    # Full pipeline: parse -> protonate -> embed -> charges -> type -> build
    # -> inject into a fresh ParameterDatabase -> rebuild CanonicalOrdering.
    param_db, co = prepare_ligand_from_smiles(REF_SMILES[name], res_name=name)
    assert co is not None  # CanonicalOrdering built without error

    injected = next(r for r in param_db.chemical.residues if r.name == name)
    assert injected.torsions and injected.chi_samples
    refined = cattr.structure(cattr.unstructure(injected), RefinedResidueType)
    for t in injected.torsions:
        assert t.name in refined.torsion_to_uaids
    for cs in injected.chi_samples:
        assert cs.chi_dihedral in {t.name for t in injected.torsions}


# --- Round 2: mol2-path smoke (prepares + emits non-empty topology) ---------


def test_mol2_path_emits_topology(monkeypatch):
    # mol2 prep path produces a residue type with rotatable-bond DOFs. (We do
    # NOT compare to ref1.params: ref1.mol2 is a different molecule than the
    # SMILES-derived ref1.params; this is a topology-emission smoke test.)
    # Charges are orthogonal to chi topology and the host rdkit cannot MMFF
    # this aromatic ref mol2, so stub the charge step.
    from pathlib import Path

    import tmol.ligand.preparation as prep_mod
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    from tmol.ligand.preparation import prepare_single_ligand

    monkeypatch.setattr(
        prep_mod,
        "build_partial_charges",
        lambda mol, atom_types, **kw: {at.atom_name: 0.0 for at in atom_types},
    )

    mol2 = (
        Path(__file__).parent.parent / "data" / "ligand_ground_truth" / "ref1.mol2"
    )
    info = nonstandard_residue_info_from_mol2(str(mol2))
    rt = prepare_single_ligand(info).residue_type
    assert len(rt.torsions) > 0  # mol2 path emits CHI topology
    # invariant: every proton chi references a named torsion
    names = {t.name for t in rt.torsions}
    for cs in rt.chi_samples:
        assert cs.chi_dihedral in names


def test_mol2_path_matches_smiles_path_topology(tmp_path, monkeypatch):
    # Matching-molecule mol2-path parity: generate a mol2 from the SAME molecule
    # used on the SMILES path, prepare via the mol2 path, and assert the emitted
    # CHI topology (heavy/proton counts + proton samples) matches. Charges are
    # stubbed (orthogonal to chi topology).
    import tmol.ligand.preparation as prep_mod
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    from tmol.ligand.preparation import prepare_single_ligand

    monkeypatch.setattr(
        prep_mod,
        "build_partial_charges",
        lambda mol, atom_types, **kw: {at.atom_name: 0.0 for at in atom_types},
    )

    smi = "OCC(O)CO"  # glycerol: 2 heavy + 3 proton chis
    rt_smiles = _restype_from_smiles(smi, "GOL")

    mol2_path = tmp_path / "gol.mol2"
    mol2_path.write_text(_smiles_to_mol2(smi, "GOL"))
    rt_mol2 = prepare_single_ligand(
        nonstandard_residue_info_from_mol2(str(mol2_path))
    ).residue_type

    assert _chi_signature(rt_mol2) == _chi_signature(rt_smiles)
    assert _chi_signature(rt_mol2)[1] == 3  # 3 hydroxyl proton chis
