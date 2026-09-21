"""Exercise shared parsing through chemical preparation and minimization."""

from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.score import beta2016_score_function

pytest.importorskip("atomworks")

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture
def forbid_ccd_access(monkeypatch):
    """Return a guard that rejects CCD inventory and template access."""
    import atomworks.io.utils.ccd as ccd

    @contextmanager
    def guard():
        def unexpected(*args, **kwargs):
            raise AssertionError("use_ccd=False must not access CCD data")

        with monkeypatch.context() as patch:
            patch.setattr(ccd, "get_available_ccd_codes", unexpected)
            patch.setattr(ccd, "atom_array_from_ccd_code", unexpected)
            yield

    return guard


def _write_halogen_cif(
    tmp_path,
    source_charge: str,
    component_charge: str | None = "-1",
    *,
    res_name: str = "QHX",
    atom_name: str = "X1",
    element: str = "Cl",
) -> Path:
    """Write an isolated atom with independently authored charge fields."""
    charge_column = "_chem_comp_atom.charge\n" if component_charge is not None else ""
    charge_value = f"{component_charge} " if component_charge is not None else ""
    path = tmp_path / f"{res_name}_{source_charge}_{component_charge}.cif"
    path.write_text(f"""data_halogen
_entry.id halogen
_entity.id 1
_entity.type non-polymer
loop_
_chem_comp.id
_chem_comp.name
_chem_comp.type
_chem_comp.formula
{res_name} 'synthetic atom' NON-POLYMER '{element}'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
{charge_column}_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
{res_name} {atom_name} {element} {charge_value}N N 0.0 0.0 0.0
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.id
_atom_site.B_iso_or_equiv
_atom_site.occupancy
_atom_site.pdbx_formal_charge
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
HETATM {element} {atom_name} . {res_name} A 1 . ? 1 {res_name} A {atom_name} 1 10.0 1.0 {source_charge} 1.0 2.0 3.0 1
""")
    return path


def test_file_hydrogen_policy_through_scoring(tmp_path, ubq_pdb):
    from biotite.structure.io import pdbx
    from tmol.io import biotite_from_pose_stack, pose_stack_from_pdb

    device = torch.device("cpu")
    source = biotite_from_pose_stack(pose_stack_from_pdb(ubq_pdb, device))
    index = np.flatnonzero((source.res_name == "ALA") & (source.element == "H"))[0]
    source.coord[index] += [0.2, 0.1, -0.1]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "hydrogens.cif"
    file.write(path)
    array = atom_array_from_cif(path)
    selected = (
        (array.res_id == source.res_id[index])
        & (array.atom_name == source.atom_name[index])
        & (array.chain_id == source.chain_id[index])
    )
    np.testing.assert_allclose(
        array.coord[selected], source.coord[index][None], atol=0.001
    )
    pose = pose_stack_from_cif(path, device, no_optH=True)
    restored = biotite_from_pose_stack(pose)
    selected = (
        (restored.res_id == source.res_id[index])
        & (restored.atom_name == source.atom_name[index])
        & (restored.chain_id == source.chain_id[index])
    )
    np.testing.assert_allclose(
        restored.coord[selected], source.coord[index][None], atol=0.001
    )
    coords = pose.coords.detach().clone().requires_grad_()
    energy = beta2016_score_function(device).render_whole_pose_scoring_module(pose)(
        coords
    )
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(coords.grad).all()


@pytest.mark.parametrize(
    "source_charge,should_fail", [("0", True), ("-1", False), ("?", False)]
)
def test_isolated_halogen_charge_provenance(
    tmp_path, source_charge: str, should_fail: bool
) -> None:
    """Reject authored neutral halogen without overriding source or CCD charge."""
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand._preparation import LigandPreparationError
    from tmol.tests.ligand.test_local_conjugate_params import _charges

    path = _write_halogen_cif(tmp_path, source_charge)
    source = atom_array_from_cif(path)
    expected_charge = 0 if source_charge == "0" else -1
    assert source.charge.tolist() == [expected_charge]
    assert source.tmol_source_formal_charge.tolist() == [source_charge]
    before = {
        name: source.get_annotation(name).copy()
        for name in source.get_annotation_categories()
    }
    bonds_before = source.bonds.as_array().copy()

    if should_fail:
        with pytest.raises(
            LigandPreparationError,
            match=(
                r"QHX: source explicitly declares formal charge 0.*"
                r"CCD component definition declares -1.*HCl-like.*"
                r"Correct the source.*params_files.*strict_ligands=False"
            ),
        ):
            pose_stack_from_biotite(
                source, torch.device("cpu"), prepare_ligands=True, ligand_seed=17
            )
    else:
        pose, context = pose_stack_from_biotite(
            source,
            torch.device("cpu"),
            prepare_ligands=True,
            ligand_seed=17,
            no_optH=True,
            return_context=True,
        )
        block_type = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, 0])
        ]
        assert set(block_type.atom_to_idx) == {"X1"}
        assert sum(_charges(context.parameter_database, block_type).values()) == (
            pytest.approx(-1, abs=1e-3)
        )
        assert torch.isfinite(pose.coords[pose.real_atoms]).all()

    assert source.get_annotation_categories() == list(before)
    for name, values in before.items():
        np.testing.assert_array_equal(source.get_annotation(name), values)
    np.testing.assert_array_equal(source.bonds.as_array(), bonds_before)


@pytest.mark.parametrize(
    "res_name,atom_name,element,expected_charge",
    [("XE", "XE", "Xe", 0), ("CL", "CL", "Cl", -1)],
)
def test_ccd_isolated_atom_charge_provenance(
    tmp_path,
    res_name: str,
    atom_name: str,
    element: str,
    expected_charge: int,
) -> None:
    """Treat neutral and charged CCD formal-charge fields as authoritative."""
    source = atom_array_from_cif(
        _write_halogen_cif(
            tmp_path,
            "?",
            None,
            res_name=res_name,
            atom_name=atom_name,
            element=element,
        )
    )

    assert source.charge.tolist() == [expected_charge]
    assert source.tmol_source_formal_charge.tolist() == ["?"]
    assert source.tmol_formal_charge_specified.tolist() == [True]


@pytest.mark.parametrize(
    "source_charge,component_charge,expected_charge,expected_specified",
    [
        ("-1", None, -1, True),
        ("0", None, 0, True),
        ("?", "-1", -1, True),
        ("?", "0", 0, True),
        ("?", None, 0, False),
    ],
)
def test_no_ccd_preserves_authored_charge_provenance(
    tmp_path,
    forbid_ccd_access,
    source_charge: str,
    component_charge: str | None,
    expected_charge: int,
    expected_specified: bool,
) -> None:
    """Read only authored charges and distinguish zero from unspecified."""
    with forbid_ccd_access():
        source = atom_array_from_cif(
            _write_halogen_cif(tmp_path, source_charge, component_charge),
            use_ccd=False,
        )

    assert source.charge.tolist() == [expected_charge]
    assert source.tmol_source_formal_charge.tolist() == [source_charge]
    assert source.tmol_formal_charge_specified.tolist() == [expected_specified]


def test_no_ccd_missing_isolated_charge_fails_only_for_preparation(
    tmp_path, forbid_ccd_access, torch_device
) -> None:
    """Require uninferable charge for generation but permit saved-parameter reuse."""
    from tmol.io import pose_stack_from_biotite
    from tmol.ligand import prepare_ligands
    from tmol.ligand._preparation import LigandPreparationError

    with forbid_ccd_access():
        missing = atom_array_from_cif(
            _write_halogen_cif(tmp_path, "?", None), use_ccd=False
        )
    assert missing.coord.tolist() == [[1.0, 2.0, 3.0]]

    expected = (
        "QHX: formal charge is unspecified for isolated Cl atom X1; no bond or "
        "hydrogen valence can establish its charge. Supply "
        "_atom_site.pdbx_formal_charge or _chem_comp_atom.charge, enable use_ccd "
        "for a CCD component, or load its prepared .tmol parameters. Pass "
        "strict_ligands=False to skip it with a warning, or supply prebuilt params "
        "via ligand_params_files."
    )
    with pytest.raises(LigandPreparationError) as error:
        pose_stack_from_biotite(
            missing, torch_device, prepare_ligands=True, use_ccd=False
        )
    assert str(error.value) == expected

    with forbid_ccd_access():
        charged = atom_array_from_cif(
            _write_halogen_cif(tmp_path, "?", "-1"), use_ccd=False
        )
    params_path = tmp_path / "halide.tmol"
    prepare_ligands(
        charged,
        params_output=str(params_path),
        use_ccd=False,
        seed=17,
    )
    pose = pose_stack_from_biotite(
        missing,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(params_path)],
        use_ccd=False,
        no_optH=True,
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()


@pytest.mark.parametrize(
    "fixture",
    [
        "ncaa_fixtures/capped_peptide_ace_nh2.cif",
        "ncaa_fixtures/beta_peptide_3c3g.cif",
        "ncaa_fixtures/na_dna_8og_183d.cif",
        "ncaa_fixtures/na_dna_5mc_1d17.cif",
        "ncaa_fixtures/na_rna_2ome_310d.cif",
        "ncaa_fixtures/na_dna_ttd_1ttd.cif",
        "atomworks_regressions/hydrolase_intermediate_1tqh.cif.gz",
        "atomworks_regressions/phosphate_charge_4js1.cif.gz",
        "atomworks_regressions/chloride_complex_4hbt.cif.gz",
        "atomworks_regressions/triphosphate_rna_4gxy.cif",
        "atomworks_regressions/unresolved_modified_polymer_1xj9.cif.gz",
        "atomworks_regressions/missing_phosphate_rna_5w1i.cif",
    ],
)
def test_shared_parser_builds_and_scores_general_chemistry(fixture, torch_device):
    import biotite.structure as struc
    from tmol.tests.io.test_atomworks_corpus_regressions import (
        _assert_all_source_connections,
        _score_and_minimize,
    )

    pose, context = pose_stack_from_cif(
        DATA / fixture,
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    array = atom_array_from_cif(DATA / fixture)
    array = array[array.res_name != "HOH"]
    if "1tqh" in fixture or "4hbt" in fixture:
        residues = list(struc.residue_iter(array))
        unresolved = np.array([not np.isfinite(r.coord).any() for r in residues])
        # AtomWorks also restores wholly unresolved protein residues.
        # Their absence from the constructed pose must not hide observed atoms.
        expected = 0
        assert int(unresolved.sum()) == expected
        keep = np.repeat(~unresolved, [len(r) for r in residues])
        assert not array.hetero[~keep].any()
        array = array[keep]
    if "1xj9" in fixture:
        from collections import Counter

        residues = list(struc.residue_iter(array))
        excluded = [
            r.res_name[0] == "LYS" or not np.isfinite(r.coord).any() for r in residues
        ]
        expected = {"LYS": 2}
        assert (
            Counter(r.res_name[0] for r, skip in zip(residues, excluded) if skip)
            == expected
        )
        keep = np.repeat(np.logical_not(excluded), [len(r) for r in residues])
        # Only the two isolated lysine nitrogens have observed coordinates.
        assert np.isfinite(array.coord[~keep]).all(-1).sum() == 2
        array = array[keep]
        assert len(list(struc.residue_iter(array))) == 16
    if "5w1i" in fixture:
        residues = list(struc.residue_iter(array))
        assert [r.res_name[0] for r in residues] == ["A", "G", "C", "C"]
        assert not np.isfinite(residues[1].coord[residues[1].atom_name == "O3'"]).any()
        # The backbone-incomplete G is excluded; C retains its phosphate
        # across that gap and builds it from its own resolved sugar frame.
        array = array[np.repeat([True, False, True, True], list(map(len, residues)))]
        cytidine = residues[2]
        # The author reader can also retain the template's unresolved
        # leaving oxygen; it is not part of an internal nucleotide type.
        leaving = cytidine.atom_name == "OP3"
        assert not np.isfinite(cytidine.coord[leaving]).any()
        cytidine = cytidine[~leaving]
        missing = ~np.isfinite(cytidine.coord).all(-1)
        assert set(cytidine.atom_name[missing]) == {"P", "OP1", "OP2"}
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 1])]
        offset = int(pose.block_coord_offset[0, 1])
        indices = [offset + bt.atom_to_idx[str(name)] for name in cytidine.atom_name]
        actual = pose.coords[0, indices].detach().cpu().numpy()
        np.testing.assert_array_equal(actual[~missing], cytidine.coord[~missing])
        assert np.isfinite(actual[missing]).all()
    _assert_all_source_connections(pose, array)
    if "/na_" in fixture:
        # Capping must displace only terminal oxygen, preserving both retained
        # phosphate oxygens and their supplied coordinates in the final pose.
        for i, residue in enumerate(struc.residue_iter(array)):
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, i])
            ]
            offset = int(pose.block_coord_offset[0, i])
            for name in ("OP1", "OP2"):
                observed = residue[
                    (residue.atom_name == name) & np.isfinite(residue.coord).all(-1)
                ]
                if len(observed):
                    assert name in bt.atom_to_idx
                    np.testing.assert_allclose(
                        pose.coords[0, offset + bt.atom_to_idx[name]].detach().cpu(),
                        observed.coord[0],
                        atol=1e-6,
                    )
        if "8og" in fixture:
            nucleotide = array[array.res_name == "8OG"]
            assert "OP2" in nucleotide.atom_name and "OP3" not in nucleotide.atom_name
    if "1tqh" in fixture:
        # The observed tetrahedral intermediate has four single bonds at CAI:
        # restoring the free component's carbonyl would overfill that carbon.
        ligand = array.res_name == "4PA"
        carbon = int(np.flatnonzero(ligand & (array.atom_name == "CAI"))[0])
        oxygen = int(np.flatnonzero(ligand & (array.atom_name == "OAD"))[0])
        neighbors, orders = array.bonds.get_bonds(carbon)
        assert len(neighbors) == 4 and np.all(orders == struc.BondType.SINGLE)
        assert oxygen in neighbors and array.charge[oxygen] == -1
        assert (
            np.count_nonzero(
                (array.res_name[neighbors] == "SER")
                & (array.atom_name[neighbors] == "OG")
            )
            == 1
        )
        for bi, residue in enumerate(struc.residue_iter(array)):
            if residue.res_name[0] == "4PA":
                bt = pose.packed_block_types.active_block_types[
                    int(pose.block_type_ind[0, bi])
                ]
                assert "conj_CAI" in bt.connection_to_cidx
                offset = int(pose.block_coord_offset[0, bi])
                indices = [offset + bt.atom_to_idx[str(n)] for n in residue.atom_name]
                np.testing.assert_array_equal(
                    pose.coords[0, indices].detach().cpu(), residue.coord
                )
    if "4js1" in fixture or "4hbt" in fixture:
        from tmol.tests.ligand.test_local_conjugate_params import _charges

        ion, names, charge = (
            ("PO4", {"P", "O1", "O2", "O3", "O4"}, -3)
            if "4js1" in fixture
            else ("CL", {"CL"}, -1)
        )
        for bi, residue in enumerate(struc.residue_iter(array)):
            if residue.res_name[0] != ion:
                continue
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, bi])
            ]
            assert set(bt.atom_to_idx) == names
            if ion == "PO4":
                from rdkit import Chem

                from tmol.ligand import ligand_smiles_from_atom_array

                generated_smiles = ligand_smiles_from_atom_array(residue, res_name=ion)
                generated = Chem.MolFromSmiles(generated_smiles)
                assert generated_smiles == "O=P([O-])([O-])[O-]"
                assert Chem.GetFormalCharge(generated) == charge
            else:
                assert residue.charge.tolist() == [charge]
                assert residue.tmol_formal_charge_specified.tolist() == [True]
            assert np.isfinite(
                list(_charges(context.parameter_database, bt).values())
            ).all()
            offset = int(pose.block_coord_offset[0, bi])
            indices = [offset + bt.atom_to_idx[str(n)] for n in residue.atom_name]
            np.testing.assert_array_equal(
                pose.coords[0, indices].detach().cpu(), residue.coord
            )
    if "4gxy" in fixture:
        nucleotide = next(struc.residue_iter(array))
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, 0])]
        assert bt.base_name == "GTP"
        assert {connection.name for connection in bt.connections} == {"up"}
        aliases = {alias.alt_name: alias.name for alias in bt.atom_aliases}
        elements = {
            at.name: at.element for at in context.parameter_database.chemical.atom_types
        }
        assert sum(elements[atom.atom_type] == "P" for atom in bt.atoms) == 3
        missing = ~np.isfinite(nucleotide.coord).all(-1)
        assert set(nucleotide.atom_name[missing]) == {"PG", "O1G", "O2G", "O3G"}
        indices = [
            bt.atom_to_idx[aliases.get(str(name), str(name))]
            for name in nucleotide.atom_name
        ]
        actual = pose.coords[0, indices].detach().cpu().numpy()
        np.testing.assert_array_equal(actual[~missing], nucleotide.coord[~missing])
        assert np.isfinite(actual[missing]).all()
    _score_and_minimize(pose, context, max_iter=100)


@pytest.mark.parametrize("state", ["HD1", "HE2", "both", "none"])
def test_reader_preserves_observed_histidine_tautomer_evidence(tmp_path, state):
    from biotite.structure import info
    from biotite.structure.io import pdbx

    source = info.residue("HIS")
    source.chain_id[:] = "A"
    source.res_id[:] = 1
    removed = {"HD1": ["HE2"], "HE2": ["HD1"], "both": [], "none": ["HD1", "HE2"]}[
        state
    ]
    source = source[~np.isin(source.atom_name, removed)]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "histidine.cif"
    file.write(path)
    parsed = atom_array_from_cif(path)
    for name in ("HD1", "HE2"):
        observed = source.atom_name == name
        retained = parsed.atom_name == name
        assert bool(retained.any()) == bool(observed.any())
        if observed.any():
            np.testing.assert_allclose(
                parsed.coord[retained], source.coord[observed], atol=0.001
            )


def test_pdb_modified_polymer_preserves_chain_and_scores(tmp_path, torch_device):
    from biotite.structure.io import pdb
    from tmol.io import atom_array_from_file, pose_stack_from_pdb
    from tmol.tests.io.test_atomworks_corpus_regressions import (
        _assert_all_source_connections,
        _score_and_minimize,
    )

    source = atom_array_from_file(DATA / "ncaa_fixtures/phosphopeptide_5ema.cif")
    source = source[
        np.isfinite(source.coord).all(-1) & ~np.isin(source.element, ["H", "D"])
    ]
    file = pdb.PDBFile()
    file.set_structure(source)
    path = tmp_path / "phosphopeptide.pdb"
    file.write(path)
    observed = file.get_structure(model=1)
    raw = atom_array_from_file(path)
    np.testing.assert_array_equal(raw.coord, observed.coord)
    assert raw.bonds is not None  # Preserve any authored CONECT records.
    array = atom_array_from_file(path)
    assert set(array.chain_id) == set(source.chain_id)
    assert np.all(array.tmol_polymer_entity)
    assert set(array.res_name) == set(source.res_name)
    pose, context = pose_stack_from_pdb(
        path, torch_device, prepare_ligands=True, return_context=True, ligand_seed=17
    )
    charges = context.parameter_database.scoring.elec.atom_charge_parameters
    # Database partial charges are rounded; their sum retains phosphate's -2 state.
    assert sum(row.charge for row in charges if row.res == "SEP") == pytest.approx(
        -2, abs=1e-3
    )
    _assert_all_source_connections(pose, array)
    for start in np.flatnonzero(
        np.r_[True, observed.res_id[1:] != observed.res_id[:-1]]
    ):
        resid = observed.res_id[start]
        block = np.flatnonzero(pose.pdb_info.residue_labels[0] == resid)[0]
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        selected = observed.res_id == resid
        offset = int(pose.block_coord_offset[0, block])
        actual = pose.coords[
            0, [offset + bt.atom_to_idx[name] for name in observed.atom_name[selected]]
        ]
        np.testing.assert_array_equal(
            actual.detach().cpu().numpy(), observed.coord[selected]
        )
    _score_and_minimize(pose, context, max_iter=20)


def test_known_and_unknown_chemistry_share_file_contract(
    tmp_path, monkeypatch, forbid_ccd_access, torch_device
):
    """MOL2 preparation makes atom-only PDB/CIF sufficient for scoring."""
    from biotite.structure.io import pdb, pdbx
    from tmol.io import (
        atom_array_from_file,
        pose_stack_from_file,
        pose_stack_from_biotite,
    )
    from tmol.ligand import prepare_ligand_from_mol2, write_params_from_mol2
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2
    from tmol.ligand._preparation import LigandPreparationError
    from tmol.tests.io.test_atomworks_corpus_regressions import _score_and_minimize
    import tmol.ligand._preparation as preparation

    mol2 = DATA / "protein_ligand_test/ace.lig.mol2"
    source = nonstandard_residue_info_from_mol2(mol2, res_name="ZZQ").atom_array
    params_path = tmp_path / "ligand.tmol"
    param_db, _ = prepare_ligand_from_mol2(mol2, res_name="ZZQ", seed=17)
    write_params_from_mol2(mol2, params_path, res_name="ZZQ", seed=17)
    declared_bonds = source.bonds
    source.bonds = None

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Known chemistry must not regenerate or consult disabled CCD"
        )

    for extension in ("pdb", "cif"):
        file = pdb.PDBFile() if extension == "pdb" else pdbx.CIFFile()
        if extension == "pdb":
            file.set_structure(source)
        else:
            pdbx.set_structure(file, source)
            for category in list(file.block):
                if category != "atom_site":
                    del file.block[category]
        path = tmp_path / f"coordinates.{extension}"
        file.write(path)
        if extension == "pdb":
            connected = source.copy()
            connected.bonds = declared_bonds
            file.set_structure(connected)
            connected_path = tmp_path / "connectivity.pdb"
            file.write(connected_path)
            for use_ccd in (False, True):
                untyped = atom_array_from_file(connected_path, use_ccd=use_ccd)
                assert untyped.bonds.get_bond_count() > 0
                assert np.all(untyped.bonds.as_array()[:, 2] == 0)
                with pytest.raises(
                    LigandPreparationError, match="ZZQ.*chemical bond orders"
                ):
                    pose_stack_from_biotite(
                        untyped, torch_device, prepare_ligands=True, use_ccd=False
                    )
        for use_ccd in (False, True):
            ccd_guard = nullcontext() if use_ccd else forbid_ccd_access()
            with monkeypatch.context() as patch, ccd_guard:
                patch.setattr(preparation, "_prepare_ligand_via_smiles", unexpected)
                array = atom_array_from_file(path, use_ccd=use_ccd)
                np.testing.assert_allclose(array.coord, source.coord, atol=0.001)
                assert list(array.atom_name) == list(source.atom_name)
                pose, context = pose_stack_from_file(
                    path,
                    torch_device,
                    use_ccd=use_ccd,
                    prepare_ligands=True,
                    param_db=param_db,
                    no_optH=True,
                    return_context=True,
                )
                _score_and_minimize(pose, context, max_iter=10)
                reloaded, reloaded_context = pose_stack_from_file(
                    path,
                    torch_device,
                    use_ccd=use_ccd,
                    ligand_params_files=[str(params_path)],
                    no_optH=True,
                    return_context=True,
                )
                _score_and_minimize(reloaded, reloaded_context, max_iter=1)
            with pytest.raises(
                LigandPreparationError, match="ZZQ.*chemical bond orders"
            ):
                pose_stack_from_biotite(
                    array, torch_device, prepare_ligands=True, use_ccd=False
                )
