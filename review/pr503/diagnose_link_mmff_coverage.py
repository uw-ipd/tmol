"""Explore coordinate-independent MMFF attachment parameter coverage.

This is a source-CIF diagnostic, not an installed tmol energy model. It removes
peptide edges to compare a complete covalent group with individual residue pairs.
All coordinates are set to NaN before deriving chemistry. MMFF coefficients keep
their native units and functional form; they cannot be copied unconverted into
tmol's harmonic potentials. Hydrogens are identified by their parent atom here,
not mapped to generated tmol hydrogen names.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import biotite.structure as struc
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from tmol.io import atom_array_from_cif
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES
from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array
from tmol.ligand._detect import _dimorphite_protonate_smiles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = []
for fixture, stem in FIXTURES.items():
    aa = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
    aa = aa[aa.element != "H"]
    starts = struc.get_residue_starts(aa, add_exclusive_stop=True)
    residx = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    cross = []
    graph = nx.Graph()
    for a, b, order in aa.bonds.as_array():
        if residx[a] != residx[b]:
            if set(aa.atom_name[[a, b]]) == {"C", "N"}:
                continue
            cross.append((int(a), int(b), int(order)))
        graph.add_edge(int(a), int(b))
    for a, b, order in cross:
        row = dict(
            fixture=fixture,
            atoms=[str(aa.atom_name[a]), str(aa.atom_name[b])],
            residues=[str(aa.res_name[a]), str(aa.res_name[b])],
            residue_indices=[int(residx[a]), int(residx[b])],
            source_order=order,
        )
        for mode in ("pair", "group"):
            selected = (
                np.flatnonzero(np.isin(residx, [residx[a], residx[b]]))
                if mode == "pair"
                else np.array(sorted(nx.node_connected_component(graph, a)))
            )
            arr = aa[selected].copy()
            arr.coord[:] = np.nan
            try:
                smi = ligand_smiles_from_atom_array(arr, with_atom_map=True)
                smi = _dimorphite_protonate_smiles(smi, ph=7.4)
                mol = Chem.AddHs(Chem.MolFromSmiles(smi))
                props = AllChem.MMFFGetMoleculeProperties(mol)
                amap = {
                    at.GetAtomMapNum(): at.GetIdx()
                    for at in mol.GetAtoms()
                    if at.GetAtomicNum() > 1
                }
                assert set(amap) == set(range(1, len(arr) + 1))
                ia = amap[int(np.flatnonzero(selected == a)[0]) + 1]
                ib = amap[int(np.flatnonzero(selected == b)[0]) + 1]

                def label(idx):
                    atom = mol.GetAtomWithIdx(idx)
                    if atom.GetAtomicNum() == 1:
                        return "H@" + label(atom.GetNeighbors()[0].GetIdx())
                    gi = int(selected[atom.GetAtomMapNum() - 1])
                    return f"{residx[gi]}:{aa.res_name[gi]}:{aa.atom_name[gi]}"

                angles = {}
                for center, other in ((ia, ib), (ib, ia)):
                    for n in mol.GetAtomWithIdx(center).GetNeighbors():
                        ni = n.GetIdx()
                        if ni != other:
                            angles[" / ".join(map(label, [ni, center, other]))] = (
                                props.GetMMFFAngleBendParams(mol, ni, center, other)
                            )
                row[mode] = dict(
                    smiles=smi,
                    link_params=props.GetMMFFBondStretchParams(mol, ia, ib),
                    angles=angles,
                    types=[props.GetMMFFAtomType(i) for i in (ia, ib)],
                    hydrogens=[
                        sum(
                            n.GetAtomicNum() == 1
                            for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        )
                        for i in (ia, ib)
                    ],
                )
            except Exception as err:
                row[mode] = {"error": repr(err)}
        rows.append(row)
        print(row, flush=True)
with args.output.open("w") as out:
    json.dump(rows, out, indent=2)
