"""Unit tests for CHI / PROTON_CHI topology classification and params IO.

Complements the ground-truth parity assertions in
``test_ligand_pipeline.py::TestGroundTruthRegression`` with focused negative
cases, sp3 sample / EXTRA-expansion checks, quad validity, NU-unsupported
confirmation, and the ``params_io`` CHI/PROTON_CHI round-trip.
"""

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
