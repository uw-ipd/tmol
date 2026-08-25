"""Tests for topology-aware generated-component routing."""

import attr
import biotite.structure as struc
import biotite.structure.info as struc_info
import numpy as np
import torch

from tmol import beta2016_score_function
from tmol.database import ParameterDatabase
from tmol.ligand._detect import NonStandardResidueInfo
from tmol.ligand._polymer import (
    ComponentKind,
    classify_component,
    specialize_component_preparation,
)
from tmol.ligand._registry import LigandPreparation, _build_cartbonded_params


def _info(name, ccd_type, atoms, bonds):
    array = struc.AtomArray(len(atoms))
    array.atom_name = np.asarray(atoms)
    array.element = np.asarray(["C"] * len(atoms))
    array.res_name[:] = name
    array.coord = np.arange(len(atoms) * 3, dtype=float).reshape((-1, 3))
    array.bonds = struc.BondList(
        len(atoms),
        np.asarray(
            [
                (atoms.index(a), atoms.index(b), int(struc.BondType.SINGLE))
                for a, b in bonds
            ],
            dtype=np.int32,
        ),
    )
    return NonStandardResidueInfo(
        res_name=name,
        ccd_type=ccd_type,
        atom_names=tuple(atoms),
        elements=tuple(array.element),
        coords=array.coord.copy(),
        atom_array=array,
    )


def test_classification_requires_backbone_topology():
    peptide = _info(
        "SEP", "L-PEPTIDE LINKING", ["N", "CA", "C", "P"], [("N", "CA"), ("CA", "C")]
    )
    misleading = _info(
        "SEP", "L-PEPTIDE LINKING", ["N", "CA", "C", "P"], [("N", "C"), ("CA", "P")]
    )
    sugar = _info(
        "NAG", "D-SACCHARIDE, BETA LINKING", ["C1", "O5", "C5"], [("C1", "O5")]
    )
    assert classify_component(peptide).kind is ComponentKind.PROTEIN
    assert classify_component(misleading).kind is ComponentKind.GENERAL
    assert classify_component(sugar).kind is ComponentKind.CARBOHYDRATE


def test_generated_ptm_inherits_parent_polymer_contract():
    db = ParameterDatabase.get_default()
    ser = next(residue for residue in db.chemical.residues if residue.name == "SER")
    generated = attr.evolve(
        ser,
        name="SEP",
        base_name="SEP",
        name3="SEP",
        io_equiv_class="SEP",
        connections=(),
        torsions=(),
        icoors=tuple(i for i in ser.icoors if i.name not in ("down", "up")),
        properties=attr.evolve(
            ser.properties,
            is_canonical=False,
            polymer=attr.evolve(
                ser.properties.polymer,
                is_polymer=False,
                polymer_type=None,
                backbone_type=None,
                mainchain_atoms=None,
            ),
        ),
        one_letter_code=None,
    )
    prep = LigandPreparation(
        residue_type=generated,
        partial_charges={atom.name: 0.0 for atom in generated.atoms},
        cartbonded_params=_build_cartbonded_params(generated),
        atom_type_elements=None,
    )
    info = _info(
        "SEP",
        "L-PEPTIDE LINKING",
        [atom.name for atom in generated.atoms],
        [("N", "CA"), ("CA", "C")],
    )
    specialized, profile = specialize_component_preparation(prep, info, db)
    residue = specialized.residue_type
    assert profile.parent_name == "SER"
    assert residue.base_name == "SER"
    assert residue.properties.polymer.is_polymer
    assert residue.properties.polymer.mainchain_atoms == ("N", "CA", "C")
    assert {connection.name for connection in residue.connections} == {"down", "up"}
    assert {torsion.name for torsion in residue.torsions} >= {
        "phi",
        "psi",
        "omega",
        "chi1",
    }


def test_generated_modified_nucleotide_inherits_parent_polymer_contract():
    db = ParameterDatabase.get_default()
    adenosine = next(
        residue for residue in db.chemical.residues if residue.name == "RA"
    )
    generated = attr.evolve(
        adenosine,
        name="1MA",
        base_name="1MA",
        name3="1MA",
        io_equiv_class="1MA",
        connections=(),
        torsions=(),
        icoors=tuple(i for i in adenosine.icoors if i.name not in ("down", "up")),
        properties=attr.evolve(
            adenosine.properties,
            is_canonical=False,
            polymer=attr.evolve(
                adenosine.properties.polymer,
                is_polymer=False,
                polymer_type=None,
                backbone_type=None,
                mainchain_atoms=None,
            ),
        ),
        one_letter_code=None,
    )
    prep = LigandPreparation(
        residue_type=generated,
        partial_charges={atom.name: 0.0 for atom in generated.atoms},
        cartbonded_params=_build_cartbonded_params(generated),
        atom_type_elements=None,
    )
    backbone = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")
    info = _info(
        "1MA", "RNA LINKING", list(backbone), list(zip(backbone, backbone[1:]))
    )
    specialized, profile = specialize_component_preparation(prep, info, db)
    residue = specialized.residue_type
    assert profile.kind is ComponentKind.NUCLEIC_ACID
    assert residue.base_name == "RA"
    assert residue.properties.polymer.backbone_type == "rna"
    assert residue.properties.polymer.mainchain_atoms == backbone
    assert {connection.name for connection in residue.connections} == {"down", "up"}


def test_modified_residues_build_at_peptide_termini(torch_device):
    """CCD PTMs are generated, normalized, and patched as ordinary termini."""

    residues = []
    previous_carbon = None
    for index, name in enumerate(("SEP", "ALA", "PTR"), start=1):
        residue = struc_info.residue(name).copy()
        keep = residue.element != "H"
        if index != 3:
            keep &= residue.atom_name != "OXT"
        residue = residue[keep]
        residue.chain_id[:] = "A"
        residue.res_id[:] = index
        if previous_carbon is not None:
            nitrogen = residue.coord[residue.atom_name == "N"][0]
            residue.coord += previous_carbon + np.array([1.33, 0.0, 0.0]) - nitrogen
        previous_carbon = residue.coord[residue.atom_name == "C"][0].copy()
        residues.append(residue)
    peptide = struc.concatenate(residues)

    from tmol.io import pose_stack_from_biotite

    pose, context = pose_stack_from_biotite(
        peptide,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    block_types = [
        pose.packed_block_types.active_block_types[int(block_type)]
        for block_type in pose.block_type_ind64[0]
        if block_type >= 0
    ]
    sep = next(block_type for block_type in block_types if block_type.name3 == "SEP")
    assert sep.base_name == "SER"
    atom_types = {atom.name: atom.atom_type for atom in sep.atoms}
    assert {name: atom_types[name] for name in ("N", "CA", "C", "O")} == {
        "N": "Nlys",
        "CA": "CAbb",
        "C": "CObb",
        "O": "OCbb",
    }
    assert atom_types["P"] == "PG3"
    assert {atom_types[name] for name in ("O1P", "O2P", "O3P")} == {"OG2"}
    assert {"H1", "H2", "H3"} <= atom_types.keys()
    assert {"HN2", "HN3"}.isdisjoint(atom_types)
    assert block_types[0].name3 == "SEP"
    assert block_types[-1].name3 == "PTR"
    assert "nterm" in block_types[0].name
    assert "cterm" in block_types[-1].name
    for block_index, connection_name in (
        (0, "up"),
        (1, "down"),
        (1, "up"),
        (2, "down"),
    ):
        block_type = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind64[0, block_index])
        ]
        connection_index = block_type.connection_to_cidx[connection_name]
        assert (
            pose.inter_residue_connections64[0, block_index, connection_index, 0] >= 0
        )
    assert torch.isfinite(pose.coords).all()
    score_function = beta2016_score_function(
        torch_device, param_db=context.parameter_database
    )
    coords = pose.coords.detach().clone().requires_grad_(True)
    score = score_function.render_whole_pose_scoring_module(pose)(coords).sum()
    score.backward()
    assert torch.isfinite(score)
    assert torch.isfinite(coords.grad).all()
