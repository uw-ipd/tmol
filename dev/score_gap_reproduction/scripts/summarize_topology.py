"""Compare native and connectivity-controlled scores on the same eight inputs."""
import csv
import json
import numpy as np
from repro_paths import ROOT, RESULTS

groups = json.loads((ROOT / "metadata/term_groups.json").read_text())
rows = []
for file in sorted((RESULTS / "topology").glob("topology-*.json")):
    controlled = json.loads(file.read_text())
    dataset = controlled["dataset_id"]
    native = json.loads((RESULTS / f"exports/tmol-{dataset}.json").read_text())
    pyro = json.loads((RESULTS / f"exports/pyrosetta-{dataset}.json").read_text())
    assert controlled["input_sha256"] == native["input_sha256"] == pyro["input_sha256"]
    for term, (tmol_names, pyro_names) in groups.items():
        rows.append(dict(dataset_id=dataset, term=term,
            pyrosetta=sum(pyro["weights"].get(n, 0) * pyro["unweighted_terms"].get(n, 0) for n in pyro_names),
            native=sum(native["weights"].get(n, 0) * native["unweighted_terms"].get(n, 0) for n in tmol_names),
            declared_connectivity=sum(controlled["score_terms"].get(n, 0) for n in tmol_names)))
with (RESULTS / "topology_diagnostic.csv").open("w") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
stats = []
for term in groups:
    data = [r for r in rows if r["term"] == term]
    x = np.array([r["pyrosetta"] for r in data])
    for mode in ["native", "declared_connectivity"]:
        y = np.array([r[mode] for r in data])
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
        stats.append(dict(term=term, mode=mode, n=len(x), pearson_r=r, mean_absolute_error=float(np.mean(abs(x-y)))))
(RESULTS / "topology_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
print(json.dumps([r for r in stats if r["term"] == "cart_bonded"], indent=2))
