"""Run isolated native exports and diagnostic controls on CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
AUDITED = ["5uoi", "cox1", "1s78", "1kx5", "1ysa", "6q1h", "4lup", "p38"]
LIGANDS = ["hsp90", "p38", "src", "ace", "cdk2", "ada", "ache", "cox1"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--datasets", nargs="+", default=AUDITED)
parser.add_argument("--tmol-python", default=sys.executable)
parser.add_argument("--pyrosetta-python", default=sys.executable)
parser.add_argument("--output", type=Path, default=ROOT / "results")
parser.add_argument("--topology", action="store_true", help="Also reproduce the eight-ligand connectivity comparison")
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update({name: "1" for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
env["TMOL_SCORE_GAP_OUTPUT"] = str(output)
commands = []


def run(python, script, *arguments):
    command = [python, str(ROOT / "scripts" / script), *map(str, arguments)]
    commands.append(command)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print("Running", script, *arguments, flush=True)
    subprocess.run(command, env=env, check=True)


datasets = list(dict.fromkeys(args.datasets + (LIGANDS if args.topology else [])))
for dataset in datasets:
    for engine, python in [("pyrosetta", args.pyrosetta_python), ("tmol", args.tmol_python)]:
        run(python, "export_score_inputs.py", engine, dataset, output / f"exports/{engine}-{dataset}.json")
    run(args.tmol_python, "control_score_inputs.py", dataset)
    run(args.tmol_python, "control_dunbrack.py", dataset)
    run(args.pyrosetta_python, "export_pyro_components.py", dataset)
    if args.topology and dataset in LIGANDS:
        run(args.tmol_python, "audit_ligand_topology.py", dataset, output / f"topology/topology-{dataset}.json")
run(args.tmol_python, "summarize_score_diagnostics.py")
if args.topology:
    run(args.tmol_python, "summarize_topology.py")
print(f"Results: {output}")
