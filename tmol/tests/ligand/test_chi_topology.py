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
    # sample_proton_chi=True so proton chi_samples are emitted for these
    # classifier tests (the prep pipeline defaults this off to avoid the
    # pose-build NaN for sampled polar hydrogens; see the gating test below).
    return build_residue_type(
        mol, name, atom_types, typing_state=state, sample_proton_chi=True
    )


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


# --- negative cases: no spurious rotatable bonds ---------------------


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


# --- ring-pucker (NU) is unsupported / not emitted ------------------


def test_saturated_ring_no_chi_and_no_nu():
    # Cyclohexane: all C-C bonds are ring-internal -> no rotatable CHI, and
    # tmol emits no NU / ring-pucker DOFs (NU is explicitly unsupported).
    rt = _restype_from_smiles("C1CCCCC1", "CHX")
    assert rt.torsions == ()
    assert rt.chi_samples == ()


# --- heavy + proton chis, sp3 samples + EXTRA ----------------


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


def test_carboxylic_and_aliphatic_polar_h_all_emitted():
    # Rosetta (verified via mol2genparams) emits proton chis for BOTH aliphatic
    # hydroxyls AND carboxylic-acid O-H: the carboxyl O is Ohx, outside
    # CONJUGATING_ACLASSES, so the C-OH bond is not conjugated. 5 sp3 polar-H
    # chis (9-fold each) -> 9^5 = 59049 > MAX_CONFS(5000) -> EXTRA 0.
    rt = _restype_from_smiles("OCC(O)C(O)C(C(=O)O)C(=O)O", "TDA")
    assert len(rt.chi_samples) == 5  # 3 aliphatic + 2 carboxylic O-H
    for cs in rt.chi_samples:
        assert cs.expansions == ()


def test_fused_ring_emits_no_ring_internal_chi():
    # Ring-closure bonds are never atom-tree edges, so they are never CHI
    # candidates (implicit handling of RosettaVS ring_cuts / FT_connected).
    # Decalin and naphthalene have only ring-internal bonds -> no chi.
    for smi, name in [("C1CCC2CCCCC2C1", "DEC"), ("c1ccc2ccccc2c1", "NAP")]:
        rt = _restype_from_smiles(smi, name)
        assert rt.torsions == (), f"{name} should emit no ring-internal chi"


# --- torsion-quad validity ------------------------------------------


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


# --- params_io CHI / PROTON_CHI round-trip --------------------------


def test_params_io_chi_proton_chi_roundtrip(tmp_path):
    from tmol.ligand.params_file import _empty_cartres
    from tmol.ligand.params_io import read_params_file, write_params_file
    from tmol.ligand.registry import LigandPreparation

    rt = _restype_from_smiles("OCCO", "EDO")
    assert rt.torsions and rt.chi_samples  # has both heavy + proton chis
    out = tmp_path / "edo.params"
    prep = LigandPreparation(rt, {}, _empty_cartres())
    write_params_file(prep, out, format="rosetta")

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


# --- refine resolves chi torsions; sampler activation ----------


def _refined(smi: str, name: str):
    import cattr

    from tmol.chemical.restypes import RefinedResidueType

    rt = _restype_from_smiles(smi, name)
    refined = cattr.structure(cattr.unstructure(rt), RefinedResidueType)
    return rt, refined


def test_refine_resolves_chi_torsion_uaids():
    # every chi_samples.chi_dihedral references a named torsion, and the
    # RefinedResidueType exposes a resolvable torsion_to_uaids for chi1..chiN.
    rt, refined = _refined("OCCO", "EDO")
    torsion_names = {t.name for t in rt.torsions}
    for cs in rt.chi_samples:
        assert cs.chi_dihedral in torsion_names
    for t in rt.torsions:
        assert t.name in refined.torsion_to_uaids
        assert len(refined.torsion_to_uaids[t.name]) == 4


def test_opth_sampler_activates_for_polar_h_ligand():
    # a hydroxyl ligand carries proton chi_samples -> OptHSampler active.
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    rt, refined = _refined("OCCO", "EDO")
    assert rt.torsions and rt.chi_samples
    assert OptHSampler().defines_rotamers_for_rt(refined) is True


def test_opth_sampler_inactive_without_polar_h():
    # benzene has no chi_samples -> OptHSampler inactive.
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    _rt, refined = _refined("c1ccccc1", "BNZ")
    assert not refined.chi_samples
    assert OptHSampler(flip_NHQ=False).defines_rotamers_for_rt(refined) is False


def test_polymer_samplers_skip_ligands(default_database, torch_device):
    # heavy-chi rotamer samplers are polymer-guarded -> never fire on
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


# --- conjugation / biaryl edge cases (negative + positive matrix) -----


def test_conjugated_polar_h_not_emitted():
    # Conjugated polar hydrogens on an all-but-one-H center (aniline / primary
    # amide -NH2) ARE skipped: the C-N bond is conjugated (both classes in
    # CONJUGATING_ACLASSES) and the N is all-but-one hydrogen. Matches Rosetta.
    for smi, name in [
        ("Nc1ccccc1", "ANI"),  # aniline -NH2
        ("CC(=O)N", "AMD"),  # primary amide -NH2
    ]:
        rt = _restype_from_smiles(smi, name)
        assert rt.chi_samples == (), f"{name}: conjugated polar-H should be skipped"


def test_aromatic_and_acid_hydroxyl_polar_h_emitted():
    # Phenol C-OH and carboxylic-acid C-OH ARE emitted (the O is Ohx, outside
    # CONJUGATING_ACLASSES, so the bond is not conjugated) — verified against
    # mol2genparams. tmol previously over-skipped these via RDKit GetIsConjugated.
    for smi, name in [("Oc1ccccc1", "PHN"), ("CC(=O)O", "ACD")]:
        rt = _restype_from_smiles(smi, name)
        assert len(rt.chi_samples) == 1, f"{name}: aromatic/acid O-H expected"


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


def test_special_biaryl_pivot_ring_to_functional_group():
    # search_special_biaryl_ring: a ring<->conjugated-functional-group bond is a
    # biaryl pivot kept as a heavy CHI despite border>1. Verified vs mol2genparams
    # (both emit a single #biaryl CHI on the ring-C<->substituent-C axis).
    for smi, name, axis_types in [
        ("c1ccccc1C(=O)N", "BAM", {"CR", "CDp"}),  # aryl<->amide carbonyl
        ("c1ccccc1/C=C/C", "STY", {"CR", "CD1"}),  # aryl<->vinyl
    ]:
        rt = _restype_from_smiles(smi, name)
        assert len(rt.torsions) == 1, f"{name}: one biaryl-pivot CHI expected"
        assert rt.chi_samples == ()  # heavy chi, no proton samples
        t = rt.torsions[0]
        types = {a.atom_type for a in rt.atoms if a.name in (t.b.atom, t.c.atom)}
        assert types == axis_types, f"{name}: {types}"


# --- mol2 literal single-bond override (RDKit kekulization promotion) ------


def test_mol2_single_bond_ids_parses_only_order_one():
    # _mol2_single_bond_ids returns ONLY the order-'1' bonds as 1-based id pairs;
    # aromatic ('ar'), amide ('am'), double ('2') etc. are excluded.
    from tmol.ligand.detect import _mol2_single_bond_ids

    text = (
        "@<TRIPOS>MOLECULE\nx\n4 3\nSMALL\nUSER_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "1 C1 0 0 0 C.ar\n2 N1 0 0 0 N.pl3\n3 O1 0 0 0 O.2\n4 C2 0 0 0 C.3\n"
        "@<TRIPOS>BOND\n"
        "1 1 2 1\n"  # single -> kept
        "2 1 3 2\n"  # double -> excluded
        "3 2 4 ar\n"  # aromatic -> excluded
    )
    assert _mol2_single_bond_ids(text) == frozenset({frozenset({1, 2})})


def test_original_single_bonds_restores_border_one_chi():
    # A genuine non-ring C=C double bond between two heavy-substituted carbons is
    # skipped by the border>1 rule. Passing it in original_single_bonds (as the
    # mol2 reader does for mol2-'1' bonds RDKit promoted) forces border=1, so the
    # axis is recovered as a heavy CHI — the exact mechanism that closes the
    # aryl-Ngu1/Nad3/NG2/Ofu DUD-80 cases.
    smi = "CCC=CCC"  # hex-3-ene: ethyls flank the central C2=C3 double bond
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    atom_types, state = assign_tmol_atom_types(mol, return_state=True)
    name_by_idx = {at.index: at.atom_name for at in atom_types}
    dbl = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    axis = frozenset(
        {name_by_idx[dbl.GetBeginAtomIdx()], name_by_idx[dbl.GetEndAtomIdx()]}
    )

    base = build_residue_type(
        mol, "HEX", atom_types, typing_state=state, sample_proton_chi=True
    )
    assert axis not in _axes(base), "double bond should be skipped by default"

    forced = build_residue_type(
        mol,
        "HEX",
        atom_types,
        typing_state=state,
        sample_proton_chi=True,
        original_single_bonds=frozenset({axis}),
    )
    assert axis in _axes(forced), "honoring mol2 single order should add the CHI"


# --- strained ring negative; heavy-only OptHSampler negative -------


def test_strained_ring_emits_no_chi():
    # 3-membered ring: all ring bonds are strained ring-internal -> no chi.
    rt = _restype_from_smiles("C1CC1", "CPR")
    assert rt.torsions == ()
    assert rt.chi_samples == ()


def test_proton_chi_samples_default_on_opt_out_off():
    # The DEFAULT preparation path now emits proton chi_samples
    # (sample_proton_chi defaults to True); sample_proton_chi=False opts out
    # (e.g. to keep pose construction NaN-free) while still emitting torsions.
    from tmol.ligand.preparation import prepare_ligand_from_smiles

    # Allyl alcohol: one aliphatic hydroxyl (-> proton chi) plus a C=C double
    # bond, so it carries a chemistry-level bond-order signal and is accepted
    # by the full preparation path (an all-single-bond SMILES like "OCCO" is
    # rejected by the topology-only guard in rdkit_mol).
    db_default, _ = prepare_ligand_from_smiles("OCC=C", res_name="EDG")
    rt_default = next(r for r in db_default.chemical.residues if r.name == "EDG")
    assert rt_default.torsions  # torsions always emitted
    assert rt_default.chi_samples  # proton chi_samples present by default

    db_off, _ = prepare_ligand_from_smiles(
        "OCC=C", res_name="EDN", sample_proton_chi=False
    )
    rt_off = next(r for r in db_off.chemical.residues if r.name == "EDN")
    assert rt_off.torsions  # torsions still emitted
    assert rt_off.chi_samples == ()  # opt-out -> no proton chi_samples


@pytest.mark.parametrize(
    "loader_attr, func_attr, arg",
    [
        ("nonstandard_residue_info_from_cif", "prepare_ligand_from_cif", "x.cif"),
        ("nonstandard_residue_info_from_mol2", "prepare_ligand_from_mol2", "x.mol2"),
        ("nonstandard_residue_info_from_pdb", "prepare_ligand_from_pdb", "x.pdb"),
    ],
)
def test_sample_proton_chi_forwarded_file_paths(
    monkeypatch, loader_attr, func_attr, arg
):
    # the option must reach prepare_single_ligand from every file-based
    # public entry point, not just the SMILES path. Spy on the call kwargs.
    import tmol.ligand.preparation as prep

    captured: dict = {}

    def fake_prepare_single_ligand(ligand_info, **kwargs):
        captured.update(kwargs)
        return "PREP"

    monkeypatch.setattr(prep, loader_attr, lambda *a, **k: "LIG")
    monkeypatch.setattr(prep, "prepare_single_ligand", fake_prepare_single_ligand)
    monkeypatch.setattr(prep, "inject_ligand_preparations", lambda db, preps, **k: db)
    monkeypatch.setattr(prep, "rebuild_canonical_ordering", lambda db: "CO")

    getattr(prep, func_attr)(arg, param_db="DB", sample_proton_chi=False)
    assert captured.get("sample_proton_chi") is False

    captured.clear()
    getattr(prep, func_attr)(arg, param_db="DB")  # default (now True)
    assert captured.get("sample_proton_chi") is True


def test_sample_proton_chi_forwarded_prepare_ligands(monkeypatch):
    # the multi-ligand entry point forwards the option too.
    from types import SimpleNamespace

    import tmol.ligand.preparation as prep

    captured: dict = {}
    fake_lig = SimpleNamespace(
        res_name="LG1",
        ccd_type="UNKNOWN",
        covalently_linked=False,
        atom_names=("C1",),
        elements=("C",),
        partial_charges=None,
    )

    def fake_prepare_single_ligand(ligand_info, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            residue_type=SimpleNamespace(name="LG1"),
            partial_charges={},
            cartbonded_params=None,
            atom_type_elements=None,
        )

    monkeypatch.setattr(prep, "detect_nonstandard_residues", lambda *a, **k: [fake_lig])
    monkeypatch.setattr(prep, "get_cached_ligand_for_key", lambda *a, **k: None)
    monkeypatch.setattr(prep, "get_cached_charges_for_key", lambda *a, **k: None)
    monkeypatch.setattr(prep, "cache_ligand", lambda *a, **k: None)
    monkeypatch.setattr(prep, "prepare_single_ligand", fake_prepare_single_ligand)
    monkeypatch.setattr(prep, "inject_ligand_preparations", lambda db, preps, **k: db)
    monkeypatch.setattr(prep, "rebuild_canonical_ordering", lambda db: "CO")

    prep.prepare_ligands("AA", param_db="DB", cache="CACHE", sample_proton_chi=True)
    assert captured.get("sample_proton_chi") is True


def test_opth_inactive_for_heavy_only_chi_ligand():
    # biphenyl has a heavy CHI but NO proton chi_samples ->
    # OptHSampler must NOT define rotamers for it.
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    _rt, refined = _refined("c1ccccc1-c2ccccc2", "BPH")
    assert refined.torsions and not refined.chi_samples
    assert OptHSampler(flip_NHQ=False).defines_rotamers_for_rt(refined) is False


# --- params_io read->write->read + tmol YAML round-trip -------


def _proton_by_axis(restype):
    axis = {t.name: frozenset((t.b.atom, t.c.atom)) for t in restype.torsions}
    return {
        axis[cs.chi_dihedral]: (tuple(cs.samples), tuple(cs.expansions))
        for cs in restype.chi_samples
    }


def test_params_io_read_write_read_roundtrip(tmp_path):
    # Start from a hand-written .params with CHI + PROTON_CHI, write it back,
    # read again; the semantic content (axes, samples, expansions) is stable.
    from tmol.ligand.params_file import _empty_cartres
    from tmol.ligand.params_io import read_params_file, write_params_file
    from tmol.ligand.registry import LigandPreparation

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
    write_params_file(
        LigandPreparation(rt1, {}, _empty_cartres()), out, format="rosetta"
    )
    rt2 = read_params_file(out)
    assert _axes(rt2) == _axes(rt1)
    assert _proton_by_axis(rt2) == _proton_by_axis(rt1)
    # the proton-chi expansion survived (EXTRA 1 20 -> (20.0,))
    assert rt2.chi_samples[0].expansions == (20.0,)


def test_tmol_yaml_roundtrip_preserves_chi(tmp_path):
    # tmol .tmol YAML path (params_file.py) round-trips non-empty torsions and
    # chi_samples. cartbonded is built with the same helper the prep pipeline
    # uses; charges are dummy (orthogonal to chi topology).
    from tmol.ligand.params_file import load_params_file
    from tmol.ligand.params_io import write_params_file
    from tmol.ligand.registry import LigandPreparation, _build_cartbonded_params

    rt = _restype_from_smiles("OCCO", "EDO")
    assert rt.torsions and rt.chi_samples
    charges = {a.name: 0.0 for a in rt.atoms}
    cart = _build_cartbonded_params(rt)

    out = tmp_path / "edo.tmol"
    write_params_file(LigandPreparation(rt, charges, cart), out, format="tmol")
    rt2 = load_params_file(out)[0].residue_type
    assert _axes(rt2) == _axes(rt)
    assert _proton_by_axis(rt2) == _proton_by_axis(rt)


def test_write_rosetta_params_list_one_file_per_residue(tmp_path):
    # A list of preparations + format="rosetta" writes one .params per residue,
    # named by the residue type, into the directory `path`.
    from tmol.ligand.params_file import _empty_cartres
    from tmol.ligand.params_io import read_params_file, write_params_file
    from tmol.ligand.registry import LigandPreparation

    preps = [
        LigandPreparation(_restype_from_smiles("OCCO", "EDO"), {}, _empty_cartres()),
        LigandPreparation(_restype_from_smiles("CC", "ETA"), {}, _empty_cartres()),
    ]
    write_params_file(preps, tmp_path, format="rosetta")
    assert (tmp_path / "EDO.params").is_file()
    assert (tmp_path / "ETA.params").is_file()
    assert read_params_file(tmp_path / "EDO.params").name == "EDO"
    assert read_params_file(tmp_path / "ETA.params").name == "ETA"


# --- ref1/ref2 inject -> ParameterDatabase -> CanonicalOrdering


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
    param_db, co = prepare_ligand_from_smiles(
        REF_SMILES[name], res_name=name, sample_proton_chi=True
    )
    assert co is not None  # CanonicalOrdering built without error

    injected = next(r for r in param_db.chemical.residues if r.name == name)
    assert injected.torsions and injected.chi_samples
    refined = cattr.structure(cattr.unstructure(injected), RefinedResidueType)
    for t in injected.torsions:
        assert t.name in refined.torsion_to_uaids
    for cs in injected.chi_samples:
        assert cs.chi_dihedral in {t.name for t in injected.torsions}


# --- mol2-path smoke (prepares + emits non-empty topology) ---------


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

    mol2 = Path(__file__).parent.parent / "data" / "ligand_ground_truth" / "ref1.mol2"
    info = nonstandard_residue_info_from_mol2(str(mol2))
    rt = prepare_single_ligand(info, sample_proton_chi=True).residue_type
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
        nonstandard_residue_info_from_mol2(str(mol2_path)), sample_proton_chi=True
    ).residue_type

    assert _chi_signature(rt_mol2) == _chi_signature(rt_smiles)
    assert _chi_signature(rt_mol2)[1] == 3  # 3 hydroxyl proton chis
