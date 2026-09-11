"""Compare connected and disconnected capped parameter models.

The disconnected structures are valence-completed reference fragments, not
physical reactants. This audits MMFF charge/type changes; it does not install
a charge model, predict reaction energies, or independently validate a fit.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from rdkit import Chem, rdBase

from tmol.io import atom_array_from_cif
from tmol.ligand import prepare_ligands
from tmol.ligand._atom_typing import assign_tmol_atom_types
from tmol.ligand._conjugate_model import iter_capped_conjugate_models
from tmol.ligand._connection_params import _parameterized_model
from tmol.tests.data import data_path
from tmol.tests.pack.test_conjugated_group_packing import FIXTURES


def atom_state(mol, props, index, typing):
    atom = mol.GetAtomWithIdx(index)
    hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
    return dict(
        generic_type=typing[index],
        mmff_type=props.GetMMFFAtomType(index),
        formal_charge=atom.GetFormalCharge(),
        partial_charge=props.GetMMFFPartialCharge(index),
        hydrogen_types=[typing[i] for i in hydrogens],
        hydrogen_charges=[props.GetMMFFPartialCharge(i) for i in hydrogens],
    )


def diagnose(array, database):
    rows = []
    for model in iter_capped_conjugate_models(array, database.chemical):
        disconnected = model.atom_array.copy()
        remap = {
            int(source): i
            for i, source in enumerate(model.source_atom_indices)
            if source >= 0
        }
        for a, b, _ in model.connections:
            disconnected.bonds.remove_bond(remap[a], remap[b])
        states = [
            _parameterized_model(m, 7.4)
            for m in (replace(model, atom_array=disconnected), model)
        ]
        types = [
            {a.index: a.atom_type for a in assign_tmol_atom_types(Chem.Mol(mol))}
            for mol, _, _ in states
        ]
        formal = [Chem.GetFormalCharge(mol) for mol, _, _ in states]
        total = [
            sum(props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms()))
            for mol, props, _ in states
        ]
        assert all(abs(a - b) < 1e-8 for a, b in zip(formal, total))
        changes, by_residue, caps_delta = [], {}, 0.0
        for i, source in enumerate(model.source_atom_indices):
            before, after = [
                atom_state(mol, props, mapping[i], typing)
                for (mol, props, mapping), typing in zip(states, types)
            ]
            delta = (
                after["partial_charge"]
                + sum(after["hydrogen_charges"])
                - before["partial_charge"]
                - sum(before["hydrogen_charges"])
            )
            ri = int(model.source_residue_indices[i])
            by_residue[ri] = by_residue.get(ri, 0.0) + delta
            if source < 0:
                caps_delta += delta
            elif before != after:
                changes.append(
                    dict(
                        source_atom=int(source),
                        source_residue=ri,
                        residue=str(array.res_name[source]),
                        atom=str(array.atom_name[source]),
                        before=before,
                        after=after,
                        charge_delta_with_hydrogens=delta,
                    )
                )
        assert abs(sum(by_residue.values()) - (formal[1] - formal[0])) < 1e-8
        rows.append(
            dict(
                formal_charge_before_after=formal,
                total_partial_charge_before_after=total,
                charge_delta_by_source_residue=by_residue,
                cap_charge_delta=caps_delta,
                changed_atoms=changes,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for fixture, stem in FIXTURES.items():
        array = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
        database, _ = prepare_ligands(array, seed=20250828)
        rows[fixture] = diagnose(array, database)
        print(
            fixture,
            [
                (r["formal_charge_before_after"], r["charge_delta_by_source_residue"])
                for r in rows[fixture]
            ],
            flush=True,
        )
    args.output.write_text(
        json.dumps(
            dict(rdkit=rdBase.rdkitVersion, ph=7.4, scope=__doc__, fixtures=rows),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
