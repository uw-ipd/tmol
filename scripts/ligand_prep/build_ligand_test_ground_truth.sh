#!/usr/bin/env bash
# Build the ligand-test GROUND TRUTH from an error-free dud80.smi using the
# legacy pipeline. The input SMILES are assumed already correct (the [O]->[O-]
# carboxylate fix is done once when preparing dud80.smi, NOT here) — this script
# does not patch input errors.
#
# Legacy pipeline (per run_obabel.sh / run_params.sh):
#   dimorphite-dl (pH 7.4)
#     -> openbabel per-molecule smi->pdb->mol2 (-h, MMFF94; conformer-search
#        fallback only where mol2genparams fails)
#     -> mol2genparams.py (resname LG1, --rename_atoms)
#
# Output: tmol/tests/data/ligand_test/ligand_ground_truth/{dud80.prot.smi,
#         mol2/, params/, manifest.json}. Run with the openvs conda env.
set -euo pipefail

PY=/home/gzhou/anaconda3/envs/openvs/bin/python
OBABEL=/home/gzhou/anaconda3/envs/openvs/bin/obabel
DIMO=/home/gzhou/git/dimorphite_dl/dimorphite_dl.py
GENP=/home/gzhou/git/Rosetta/rosetta_rosettavs/source/scripts/python/public/generic_potential/mol2genparams.py

OUT=${1:-/home/gzhou/git/tmol/tmol/tests/data/ligand_test/ligand_ground_truth}
cd "$OUT"

if [[ ! -f dud80.smi ]]; then
  echo "ERROR: $OUT/dud80.smi not found (need the error-free input set)" >&2
  exit 1
fi
echo "input molecules: $(wc -l < dud80.smi)  (bare [O] remaining: $(grep -c '\[O\]' dud80.smi || true))"

echo "== 1. dimorphite protonate (pH 7.4) =="
$PY "$DIMO" --smiles_file dud80.smi --min_ph 7.4 --max_ph 7.4 \
    --output_file dud80.prot.raw.smi --pka_precision 0.1 > dimo.log 2>&1
awk '!seen[$2]++' dud80.prot.raw.smi > dud80.prot.smi
echo "protonated: $(wc -l < dud80.prot.smi)"

echo "== 2. per-molecule openbabel -> 3D + MMFF94 mol2 (legacy -h protocol) =="
rm -rf mol2 params
mkdir -p mol2 params
: > dud80.prot.mmff94.mol2
: > obabel.log
n_ok=0
while read -r smi name; do
  [[ -z "${name:-}" ]] && continue
  printf '%s\t%s\n' "$smi" "$name" > _one.smi
  if ! $OBABEL _one.smi -O _one.pdb --gen3d -xl >> obabel.log 2>&1; then
    echo "FAIL gen3d: $name" >> obabel.log; continue
  fi
  if ! $OBABEL _one.pdb -O "mol2/$name.mol2" -h --minimize --steps 2000 --sd \
        --partialcharge mmff94 --title "$name" -xl >> obabel.log 2>&1; then
    echo "FAIL mol2: $name" >> obabel.log; continue
  fi
  cat "mol2/$name.mol2" >> dud80.prot.mmff94.mol2
  n_ok=$((n_ok + 1))
done < dud80.prot.smi
rm -f _one.smi _one.pdb
echo "mol2 molecules: $n_ok"

echo "== 3. mol2genparams -> Rosetta params =="
( cd "$(dirname "$GENP")" && \
  $PY "$GENP" -s "$OUT/dud80.prot.mmff94.mol2" --outdir "$OUT/params" \
      --resname LG1 --typenm automol2name --rename_atoms --multimol2 \
      --infer_atomtypes --no_pdb > "$OUT/genp.log" 2>&1 )
echo "params produced: $(ls -1 params/*.params 2>/dev/null | wc -l)"

echo "== 3b. conformer-search fallback where mol2genparams failed on gen3d =="
missing=$(comm -23 \
  <(ls mol2/*.mol2 2>/dev/null | xargs -r -n1 basename | sed 's/\.mol2$//' | sort) \
  <(ls params/*.params 2>/dev/null | xargs -r -n1 basename | sed 's/\.params$//' | sort))
if [[ -n "$missing" ]]; then
  echo "retrying with conformer search:" $missing
  for name in $missing; do
    smi=$(awk -v n="$name" '$2==n{print $1; exit}' dud80.prot.smi)
    [[ -z "$smi" ]] && continue
    printf '%s\t%s\n' "$smi" "$name" > _one.smi
    $OBABEL _one.smi -O _one.pdb -e --gen3d --conformer -nconf 500 --score energy \
        --partialcharge mmff94 -T 10 -xl >> obabel.log 2>&1 || continue
    $OBABEL _one.pdb -O "mol2/$name.mol2" -e -h --minimize --steps 2000 --sd \
        --partialcharge mmff94 --title "$name" -xl >> obabel.log 2>&1 || continue
  done
  rm -f _one.smi _one.pdb
  : > dud80.prot.mmff94.mol2
  while read -r smi name; do
    [[ -f "mol2/$name.mol2" ]] && cat "mol2/$name.mol2" >> dud80.prot.mmff94.mol2
  done < dud80.prot.smi
  rm -rf params; mkdir -p params
  ( cd "$(dirname "$GENP")" && \
    $PY "$GENP" -s "$OUT/dud80.prot.mmff94.mol2" --outdir "$OUT/params" \
        --resname LG1 --typenm automol2name --rename_atoms --multimol2 \
        --infer_atomtypes --no_pdb > "$OUT/genp.log" 2>&1 )
  echo "params after fallback: $(ls -1 params/*.params 2>/dev/null | wc -l)"
fi

echo "== 4. write manifest.json =="
$PY - <<'PYEOF'
import json, os
def load(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                d[p[1]] = p[0]
    return d
clean = load("dud80.smi")
prot = load("dud80.prot.smi")
mols = []
for name in sorted(clean):
    mol2 = os.path.join("mol2", f"{name}.mol2")
    params = os.path.join("params", f"{name}.params")
    if not (os.path.exists(mol2) and os.path.exists(params)):
        continue
    mols.append({
        "name": name,
        "input_smiles": clean[name],
        "expected_prot_smiles": prot.get(name, clean[name]),
        "mol2": mol2,
        "params": params,
        "charge_mode": "auto",
        "sample_proton_chi": True,
        "obabel_protocol": "gen3d",
    })
manifest = {
    "description": (
        "ligand-test GROUND TRUTH from an error-free dud80.smi (the [O]->[O-] "
        "fix is applied once when preparing dud80.smi, not in this pipeline). "
        "Legacy pipeline: dimorphite-dl (pH 7.4) -> openbabel per-molecule "
        "smi->pdb->mol2 -h MMFF94 (conformer-search fallback where mol2genparams "
        "fails) -> mol2genparams.py (resname LG1, --rename_atoms)."
    ),
    "count": len(mols),
    "molecules": mols,
}
with open("manifest.json", "w") as w:
    json.dump(manifest, w, indent=2)
print(f"manifest molecules: {len(mols)}")
PYEOF

rm -f dud80.prot.raw.smi
echo "DONE -> $OUT"
