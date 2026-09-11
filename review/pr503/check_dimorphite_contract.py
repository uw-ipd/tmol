"""Compare vendored rule-engine identity and state contracts before consolidation.

Loads both files directly, so this check does not depend on either package's
parser or native scoring extensions. Results are a compatibility diagnostic,
not independent validation of the pKa model.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from rdkit import Chem


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(mol):
    plain = Chem.Mol(mol)
    for atom in plain.GetAtoms():
        atom.SetAtomMapNum(0)
    return {
        "smiles": Chem.MolToSmiles(plain),
        "mapped_smiles": Chem.MolToSmiles(mol),
        "heavy_maps": sorted(
            a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomworks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    engines = {
        "tmol": load(root / "tmol/ligand/_dimorphite_dl.py", "tmol_rules"),
        "atomworks": load(
            args.atomworks / "src/atomworks/external/dimorphite_dl/dimorphite_dl.py",
            "atomworks_rules",
        ),
    }
    inputs = {
        "acetate": "CC(=O)[O-]",
        "ethylammonium": "CC[NH3+]",
        "azidoethane": "CCN=[N+]=[N-]",
        "nitroethane": "CC[N+](=O)[O-]",
        "cysteine": "N[C@@H](CS)C(=O)O",
        "histidine": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
        "phosphoserine": "N[C@@H](COP(=O)(O)O)C(=O)O",
    }
    rows = []
    for name, smiles in inputs.items():
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        for ph in (2.0, 7.4, 12.0):
            row = dict(name=name, ph=ph, source=inventory(mol))
            for label, engine in engines.items():
                try:
                    products = engine.protonate_mol_variants(
                        mol, min_ph=ph, max_ph=ph, pka_precision=0.1
                    )
                    row[label] = [inventory(product) for product in products]
                except Exception as error:
                    row[label] = {"error": repr(error)}
            rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for label in engines:
        failures = sum(
            not isinstance(row[label], list)
            or not row[label]
            or any(
                product["heavy_maps"] != row["source"]["heavy_maps"]
                for product in row[label]
            )
            for row in rows
        )
        print(label, "identity failures", failures, "of", len(rows))


if __name__ == "__main__":
    main()
