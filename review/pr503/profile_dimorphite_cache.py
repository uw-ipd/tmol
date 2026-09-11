"""Paired rule/protonation timing and Python retention across a pH sweep.

Snapshot each old engine beside its site_substructures.smarts file. This loads
modules directly and does not time package imports or the full preparation stack.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from rdkit import Chem, rdBase

from check_dimorphite_contract import load, inventory


def reset(engine):
    funcs = engine.ProtSubstructFuncs
    for name in (
        "_compiled_substructures",
        "load_protonation_substructs_calc_state_for_ph",
    ):
        method = getattr(funcs, name, None)
        if hasattr(method, "cache_clear"):
            method.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engines = {
        label: load(path, label + "_rules")
        for label, path in (("before", args.before), ("after", args.after))
    }
    inputs = [
        Chem.MolFromSmiles(smi)
        for smi in (
            "CC(=O)[O-]",
            "CC[NH3+]",
            "CCN=[N+]=[N-]",
            "CC[N+](=O)[O-]",
            "N[C@@H](CS)C(=O)O",
            "N[C@@H](Cc1cnc[nH]1)C(=O)O",
            "N[C@@H](COP(=O)(O)O)C(=O)O",
        )
    ]
    for mol in inputs:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 17)
    cases = [(mol, ph) for mol in inputs for ph in (2.0, 7.4, 12.0)]

    def products(engine):
        return [
            [
                inventory(product)
                for product in engine.protonate_mol_variants(
                    mol, min_ph=ph, max_ph=ph, pka_precision=0.1
                )
            ]
            for mol, ph in cases
        ]

    inventories = {label: products(engine) for label, engine in engines.items()}
    assert inventories["before"] == inventories["after"]
    # Compare every rule's state, order and SMARTS, independently of whether
    # the seven example molecules happen to exercise that rule.
    for ph in (0, 2, 7.4, 12, 14):
        for precision in (0.1, 1.0, 3.0):
            states = []
            for engine in engines.values():
                subs = engine.ProtSubstructFuncs.load_protonation_substructs_calc_state_for_ph(
                    ph - 1, ph + 1, precision
                )
                states.append(
                    [
                        (
                            s["name"],
                            s["smart"],
                            Chem.MolToSmarts(s["mol"]),
                            s["prot_states_for_pH"],
                        )
                        for s in subs
                    ]
                )
            assert states[0] == states[1]

    timings = {label: [] for label in engines}
    for sample in range(7):
        for label in (("before", "after") if sample % 2 == 0 else ("after", "before")):
            start = time.perf_counter()
            for _ in range(5):
                engine = engines[label]
                for mol, ph in cases:
                    engine.protonate_mol_variants(
                        mol, min_ph=ph, max_ph=ph, pka_precision=0.1
                    )
            timings[label].append((time.perf_counter() - start) / (5 * len(cases)))

    memory = {}
    for label, engine in engines.items():
        reset(engine)
        gc.collect()
        tracemalloc.start()
        for index in range(500):
            ph = index / 30
            # Exercise the direct-molecule API so only engine-owned objects
            # remain alive; none of the returned products is retained here.
            engine.protonate_mol_variants(
                inputs[0], min_ph=ph, max_ph=ph, pka_precision=0.1
            )
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[label] = dict(retained_python_bytes=current, peak_python_bytes=peak)
    result = dict(
        rdkit=rdBase.rdkitVersion,
        source_sha256={
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in (("before", args.before), ("after", args.after))
        },
        ordered_inventory_matches=21,
        all_rule_state_configurations_matched=15,
        seconds_per_molecule=timings,
        median_speed_ratio=statistics.median(timings["before"])
        / statistics.median(timings["after"]),
        ph_sweep_calls=500,
        memory=memory,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
