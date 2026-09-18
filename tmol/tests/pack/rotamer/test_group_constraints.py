"""Conjugated groups with chemical cycles and multiple polymer anchors.

Synthetic propylsuccinyl-linked lysines exercise topology independently of a PDB
fixture. Coordinates are embedded once from the assembled covalent molecule;
these examples are geometry/packing regressions, not fitted parameter data.
"""

import itertools
import numpy
import pytest
import torch
from biotite.structure import AtomArray, BondList
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.io import pose_stack_from_biotite
from tmol.pack import SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.pack.rotamer._conjugated_groups import find_conjugated_groups
from tmol.numeric import coord_dihedrals
from tmol.pose._util import _resolve_uaid
from tmol.tests.pack.test_conjugated_group_packing import _task, _pack_and_check_score


def crosslinked_lysines(topology):
    lys = "[NH2:1][C@@H:2]([CH2:5][CH2:6][CH2:7][CH2:8][NH2:9])[C:3](=[O:4])[OH:10]"
    ala = "[NH2:1][C@@H:2]([CH3:5])[C:3](=[O:4])[OH:10]"
    linker = (
        "[OH:5][C:1](=[O:2])[CH:3]([CH2:9][CH2:10][CH3:11])[CH2:4][C:6](=[O:7])[OH:8]"
    )
    aa_names = {
        1: "N",
        2: "CA",
        3: "C",
        4: "O",
        5: "CB",
        6: "CG",
        7: "CD",
        8: "CE",
        9: "NZ",
        10: "OXT",
    }
    parts = [("LYS", lys), ("SUC", linker), ("LYS", lys)]
    if topology == "external":
        parts += [("ALA", ala), ("ALA", ala)]
    molecule = None
    for residue, (name, smiles) in enumerate(parts):
        part = Chem.MolFromSmiles(smiles)
        for atom in part.GetAtoms():
            atom.SetIntProp("residue", residue)
            atom.SetProp(
                "name",
                (
                    aa_names[atom.GetAtomMapNum()]
                    if name != "SUC"
                    else f"X{atom.GetAtomMapNum()}"
                ),
            )
        molecule = part if molecule is None else Chem.CombineMols(molecule, part)
    rw = Chem.RWMol(molecule)

    def atom_index(residue, atom_map):
        return next(
            a.GetIdx()
            for a in rw.GetAtoms()
            if a.GetIntProp("residue") == residue and a.GetAtomMapNum() == atom_map
        )

    def amide(carboxyl_residue, carbon, leaving, nitrogen_residue, nitrogen):
        c = atom_index(carboxyl_residue, carbon)
        n = atom_index(nitrogen_residue, nitrogen)
        rw.AddBond(c, n, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(n).SetNumExplicitHs(1)
        rw.RemoveAtom(atom_index(carboxyl_residue, leaving))

    amide(1, 1, 5, 0, 9)
    amide(1, 6, 8, 2, 9)
    if topology == "cycle":
        amide(0, 3, 10, 2, 1)
    elif topology == "external":
        amide(3, 3, 10, 0, 1)
        amide(4, 3, 10, 2, 1)
    molecule = rw.GetMol()
    Chem.SanitizeMol(molecule)
    full = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = 503
    assert AllChem.EmbedMolecule(full, params) == 0
    AllChem.MMFFOptimizeMolecule(full, maxIters=500)
    n = molecule.GetNumAtoms()
    arr = AtomArray(n)
    arr.coord = full.GetConformer().GetPositions()[:n]
    arr.element = numpy.array([a.GetSymbol() for a in molecule.GetAtoms()])
    arr.atom_name = numpy.array([a.GetProp("name") for a in molecule.GetAtoms()])
    residues = numpy.array([a.GetIntProp("residue") for a in molecule.GetAtoms()])
    arr.res_name = numpy.array([parts[r][0] for r in residues])
    numbering = (
        {0: 2, 1: 1, 2: 2, 3: 1, 4: 1}
        if topology == "external"
        else {0: 1, 1: 1, 2: 2 if topology == "cycle" else 1}
    )
    arr.res_id = numpy.array([numbering[r] for r in residues])
    # Explicit peptide links determine polymer continuity; separate chain IDs
    # prevent tmol from inferring an extra bond between free lysines.
    chains = {0: "A", 1: "L", 2: "A" if topology == "cycle" else "B", 3: "A", 4: "B"}
    arr.chain_id = numpy.array([chains[r] for r in residues])
    arr.hetero = arr.res_name == "SUC"
    arr.set_annotation("is_polymer", ~arr.hetero)
    arr.set_annotation(
        "chem_comp_type", numpy.where(arr.hetero, "NON-POLYMER", "L-PEPTIDE LINKING")
    )
    arr.set_annotation(
        "charge", numpy.array([a.GetFormalCharge() for a in molecule.GetAtoms()])
    )
    arr.bonds = BondList(
        n,
        numpy.array(
            [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()))
                for b in molecule.GetBonds()
            ]
        ),
    )
    # Place polymer neighbors in sequence order, retaining atom-level bonds.
    order = [3, 0, 4, 2, 1] if topology == "external" else [0, 2, 1]
    indices = numpy.concatenate([numpy.flatnonzero(residues == r) for r in order])
    return arr[indices]


def pose_bonds(pose):
    bonds = set()
    starts = pose.block_coord_offset[0].tolist()
    types = [
        pose.packed_block_types.active_block_types[t]
        for t in pose.block_type_ind[0].tolist()
    ]
    for owner, bt in enumerate(types):
        bonds.update(
            tuple(sorted((starts[owner] + int(a), starts[owner] + int(b))))
            for a, b in bt.bond_indices
        )
        for conn, (partner, partner_conn) in enumerate(
            pose.inter_residue_connections[0, owner].tolist()
        ):
            if partner < 0:
                continue
            a = starts[owner] + int(bt.ordered_connection_atoms[conn])
            b = starts[partner] + int(
                types[partner].ordered_connection_atoms[partner_conn]
            )
            bonds.add(tuple(sorted((a, b))))
    return bonds


def assert_geometry(pose, coordinates, owners=None):
    source = pose.coords[0].double()
    coordinates = coordinates.double()
    bonds = pose_bonds(pose)
    focus = set(range(len(source)))
    if owners is not None:
        focus = set()
        for block in owners:
            start = int(pose.block_coord_offset[0, block])
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, block])
            ]
            focus.update(range(start, start + bt.n_atoms))
    pairs = torch.tensor(
        sorted((a, b) for a, b in bonds if a in focus or b in focus), device=pose.device
    )
    expected = (source[pairs[:, 0]] - source[pairs[:, 1]]).norm(dim=-1)
    actual = (coordinates[:, pairs[:, 0]] - coordinates[:, pairs[:, 1]]).norm(dim=-1)
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=1e-4, rtol=0)
    neighbors = [set() for _ in range(len(source))]
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    triples = torch.tensor(
        [
            (a, center, b)
            for center, near in enumerate(neighbors)
            for a, b in itertools.combinations(sorted(near), 2)
            if center in focus or a in focus or b in focus
        ],
        device=pose.device,
    )

    def cosines(xyz):
        a, c, b = (xyz[..., triples[:, j], :] for j in range(3))
        ac, bc = a - c, b - c
        return (ac * bc).sum(dim=-1) / (ac.norm(dim=-1) * bc.norm(dim=-1))

    actual = cosines(coordinates)
    torch.testing.assert_close(
        actual, cosines(source).expand_as(actual), atol=1e-4, rtol=0
    )
    tetra = torch.tensor(
        [
            sorted(near)
            for center, near in enumerate(neighbors)
            if len(near) == 4 and (center in focus or near & focus)
        ],
        device=pose.device,
    )

    def volumes(xyz):
        a, b, c, d = (xyz[..., tetra[:, j], :] for j in range(4))
        return ((a - d) * torch.linalg.cross(b - d, c - d)).sum(dim=-1)

    expected = volumes(source)
    valid = expected.abs() > 1e-3
    actual = volumes(coordinates)[:, valid].sign()
    torch.testing.assert_close(
        actual, expected[valid].sign().expand_as(actual), atol=0, rtol=0
    )


@pytest.mark.parametrize("topology", ["free", "cycle", "external"])
def test_constrained_group_builds_rotamers(topology, torch_device):
    arr = crosslinked_lysines(topology)
    pose, context = pose_stack_from_biotite(
        arr,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        ligand_seed=503,
        return_context=True,
    )
    groups = find_conjugated_groups(pose)
    assert len(groups) == 1
    group = groups[0]
    assert len(group.blocks) == 3
    if topology == "cycle":
        assert len(group.links) == 3
    if topology == "external":
        assert len(group.external_links) == 2
    task, sampler = _task(pose, context.parameter_database, torch_device)
    concrete = SetPackerTask.from_packer_task(task)
    pose, rotamers = build_rotamers(pose, concrete, context.parameter_database.chemical)
    counts = [int(rotamers.n_rots_for_block[0, b]) for b in group.blocks]
    assert len(set(counts)) == 1
    assert (
        counts[0] > 1
    ), "The pendant branch must still sample when the core is constrained"
    assert torch.isfinite(rotamers.coords).all()
    all_coords = pose.coords.expand(counts[0], -1, -1).clone()
    for block in group.blocks:
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        first = int(rotamers.rot_offset_for_block[0, block])
        start = int(pose.block_coord_offset[0, block])
        indices = rotamers.coord_offset_for_rot[first : first + counts[0]].long()[
            :, None
        ] + torch.arange(bt.n_atoms, device=torch_device)
        all_coords[:, start : start + bt.n_atoms] = rotamers.coords[indices]
    assert_geometry(pose, all_coords)
    library = sampler.anchor_library_chi(pose, concrete, groups)
    _, columns, targets = sampler.group_conformers(pose, library)[0]
    assert len(columns) == len({frozenset(c[2:]) for c in columns})
    for col, (owner, name, _b, _c) in enumerate(columns):
        block = group.blocks[owner]
        bt = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block])
        ]
        atoms = [_resolve_uaid(pose, 0, block, u) for u in bt.torsion_to_uaids[name]]
        actual = coord_dihedrals(*(all_coords[:, a].double() for a in atoms))
        expected = torch.tensor(targets[:, col], device=torch_device)
        difference = (actual - expected + numpy.pi) % (2 * numpy.pi) - numpy.pi
        assert float(difference.abs().max()) < 2e-4
    if topology != "free":
        # Linker-only motions remain; neither polymer member may leave the
        # closure or its external backbone attachment to sample independently.
        for block in (0, 1) if topology == "cycle" else (1, 3):
            start = int(pose.block_coord_offset[0, block])
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, block])
            ]
            expected = pose.coords[0, start : start + bt.n_atoms]
            actual = all_coords[:, start : start + bt.n_atoms]
            torch.testing.assert_close(
                actual, expected.expand_as(actual), atol=1e-4, rtol=0
            )
    assert float((all_coords - pose.coords).abs().max()) > 0.1
    if topology != "free":
        task.set_chi_sample_budget(len(group.blocks), len(group.blocks))
        one_state = SetPackerTask.from_packer_task(task)
        _, minimal = build_rotamers(
            pose, one_state, context.parameter_database.chemical
        )
        assert [int(minimal.n_rots_for_block[0, b]) for b in group.blocks] == [1] * len(
            group.blocks
        )
        for block in group.blocks:
            bt = pose.packed_block_types.active_block_types[
                int(pose.block_type_ind[0, block])
            ]
            first = int(minimal.rot_offset_for_block[0, block])
            start = int(minimal.coord_offset_for_rot[first])
            source_start = int(pose.block_coord_offset[0, block])
            torch.testing.assert_close(
                minimal.coords[start : start + bt.n_atoms],
                pose.coords[0, source_start : source_start + bt.n_atoms],
                atol=1e-4,
                rtol=0,
            )


@pytest.mark.parametrize("topology", ["free", "cycle", "external"])
def test_constrained_group_packing_preserves_geometry_and_energy(
    topology, torch_device
):
    pose, context = pose_stack_from_biotite(
        crosslinked_lysines(topology),
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        ligand_seed=503,
        return_context=True,
    )
    packed = _pack_and_check_score(pose, context.parameter_database, torch_device)
    # Ordinary samplers may idealize bonds elsewhere (e.g. ALA CA-CB). Check
    # every group bond/angle and every boundary bond/angle against the input.
    assert_geometry(pose, packed.coords, find_conjugated_groups(pose)[0].blocks)
