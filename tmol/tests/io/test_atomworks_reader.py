"""Exercise shared parsing through chemical preparation and minimization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.score import beta2016_score_function

pytest.importorskip("atomworks")

DATA = Path(__file__).parents[1] / "data"


@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_file_hydrogen_policy_through_scoring(tmp_path, ubq_pdb, reader):
    from biotite.structure.io import pdbx
    from tmol.io import (
        biotite_from_pose_stack,
        pose_stack_from_pdb,
        pose_stack_from_biotite,
    )

    device = torch.device("cpu")
    source = biotite_from_pose_stack(pose_stack_from_pdb(ubq_pdb, device))
    index = np.flatnonzero((source.res_name == "ALA") & (source.element == "H"))[0]
    source.coord[index] += [0.2, 0.1, -0.1]
    file = pdbx.CIFFile()
    pdbx.set_structure(file, source)
    path = tmp_path / "hydrogens.cif"
    file.write(path)
    for policy in ("preserve", "rebuild"):
        array = atom_array_from_cif(path, reader=reader, hydrogen_policy=policy)
        selected = (
            (array.res_id == source.res_id[index])
            & (array.atom_name == source.atom_name[index])
            & (array.chain_id == source.chain_id[index])
        )
        if policy == "preserve":
            np.testing.assert_allclose(
                array.coord[selected], source.coord[index][None], atol=0.001
            )
        else:
            assert not np.isfinite(array.coord[selected]).all(axis=-1).any()
        pose = pose_stack_from_biotite(array, device, no_optH=True)
        coords = pose.coords.detach().clone().requires_grad_()
        energy = beta2016_score_function(device).render_whole_pose_scoring_module(pose)(
            coords
        )
        energy.sum().backward()
        assert torch.isfinite(energy).all()
        assert torch.isfinite(coords.grad).all()


def test_label_template_substitution_cannot_silently_erase_unknown_atom():
    with pytest.raises(ValueError, match="XYZ"):
        atom_array_from_cif(
            DATA / "atomworks_regressions/unknown_heavy_atom_1a8o.cif",
            reader="atomworks",
        )


@pytest.mark.parametrize(
    "fixture",
    [
        "ncaa_fixtures/capped_peptide_ace_nh2.cif",
        "ncaa_fixtures/beta_peptide_3c3g.cif",
        "ncaa_fixtures/na_dna_8og_183d.cif",
        "ncaa_fixtures/na_dna_5mc_1d17.cif",
        "ncaa_fixtures/na_rna_2ome_310d.cif",
        "atomworks_regressions/hydrolase_intermediate_1tqh.cif.gz",
        "atomworks_regressions/phosphate_charge_4js1.cif.gz",
    ],
)
@pytest.mark.parametrize("reader", ["tmol", "atomworks"])
def test_shared_parser_builds_and_scores_general_chemistry(
    fixture, reader, torch_device
):
    import biotite.structure as struc
    from tmol.tests.io.test_atomworks_corpus_regressions import (
        _assert_all_source_connections,
        _score_and_minimize,
    )

    pose, context = pose_stack_from_cif(
        DATA / fixture,
        torch_device,
        reader=reader,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    array = atom_array_from_cif(DATA / fixture, reader=reader)
    array = array[array.res_name != "HOH"]
    if "1tqh" in fixture:
        residues = list(struc.residue_iter(array))
        unresolved = np.array([not np.isfinite(r.coord).any() for r in residues])
        # AtomWorks also restores five wholly unresolved protein residues.
        # Their absence from the constructed pose must not hide observed atoms.
        assert int(unresolved.sum()) == (5 if reader == "atomworks" else 0)
        keep = np.repeat(~unresolved, [len(r) for r in residues])
        assert not array.hetero[~keep].any()
        array = array[keep]
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
    if "4js1" in fixture:
        from tmol.tests.ligand.test_local_conjugate_params import _charges

        for bi, residue in enumerate(struc.residue_iter(array)):
            if residue.res_name[0] != "PO4":
                continue
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, bi])
            ]
            assert set(bt.atom_to_idx) == {"P", "O1", "O2", "O3", "O4"}
            assert sum(
                _charges(context.parameter_database, bt).values()
            ) == pytest.approx(-3, abs=1e-8)
            offset = int(pose.block_coord_offset[0, bi])
            indices = [offset + bt.atom_to_idx[str(n)] for n in residue.atom_name]
            np.testing.assert_array_equal(
                pose.coords[0, indices].detach().cpu(), residue.coord
            )
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
    parsed = atom_array_from_cif(path, reader="atomworks")
    for name in ("HD1", "HE2"):
        observed = source.atom_name == name
        retained = parsed.atom_name == name
        assert bool(retained.any()) == bool(observed.any())
        if observed.any():
            np.testing.assert_allclose(
                parsed.coord[retained], source.coord[observed], atol=0.001
            )
