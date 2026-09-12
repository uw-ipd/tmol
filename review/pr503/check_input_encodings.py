import argparse
import json
import importlib.util

import torch
import numpy as np
from pathlib import Path
from tmol.io import canonical_form_from_biotite, canonical_ordering_for_biotite
from tmol.io._pose_stack_from_atomworks import ATOMWORKS_ATOM37_NAMES, ATOMWORKS_NAME3S
from tmol.extern.openfold.residue_constants import (
    atom_types,
    restype_name_to_atom14_names,
    restypes,
)
from tmol.extern.rosettafold2.chemical import num2aa, aa2long
import biotite.structure as s

parser = argparse.ArgumentParser(
    description="Audit master input encodings and a PR #380 module. Canonical mapping only; no minimization."
)
parser.add_argument("--pr380-module", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("backbone380", args.pr380_module)
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)
result = {}
result["standard_aw_atom37_slots_match_af2"] = all(
    all(not n or atom_types[i] == n for i, n in enumerate(ATOMWORKS_ATOM37_NAMES[r]))
    for r in ATOMWORKS_NAME3S[1:21]
)
result["atomworks_protein_tokens"] = dict(enumerate(ATOMWORKS_NAME3S[1:21], 1))
result["af2_residue_codes"] = restypes
result["openfold_ALA_atom14"] = restype_name_to_atom14_names["ALA"]
result["aw_ALA_atom37_nonempty"] = {
    i: n for i, n in enumerate(ATOMWORKS_ATOM37_NAMES["ALA"]) if n
}
result["rf2_ALA_atoms"] = [
    str(n).strip() if n else None for n in aa2long[num2aa.index("ALA")]
]
bb = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3).requires_grad_()
seq = torch.zeros((1, 2), dtype=torch.int64)
chain = seq.clone()
a = s.AtomArray(8)
a.res_name[:] = "ALA"
a.atom_name = np.tile(["N", "CA", "C", "O"], 2)
a.element = np.tile(["N", "C", "C", "O"], 2)
a.res_id = np.repeat([1, 2], 4)
a.chain_id[:] = "A"
a.coord = bb.detach().numpy().reshape(8, 3).copy()
a.set_annotation("token_id", np.repeat([0, 1], 4))
a.set_annotation("atom37_slot", np.tile([0, 1, 2, 4], 2))
x = torch.full((1, 2, 37, 3), float("nan"))
x[:, :, [0, 1, 2, 4]] = bb
cf = canonical_form_from_biotite(
    a, torch.device("cpu"), atom37_coords=x, missing_density_distance_threshold=0.0
)
co = canonical_ordering_for_biotite()
inds = [co.restypes_atom_index_mapping["ALA"][n] for n in ["N", "CA", "C", "O"]]
result["backbone_via_biotite_atom37_exact"] = torch.equal(cf.coords[:, :, inds], bb)
torch.nan_to_num(cf.coords).sum().backward()
result["backbone_via_biotite_atom37_gradient_exact"] = bool(
    torch.equal(bb.grad, torch.ones_like(bb))
)
# Pure AtomArray route starts from numpy and cannot retain the Torch tape.
result["pure_atomarray_retains_autograd"] = canonical_form_from_biotite(
    a, torch.device("cpu")
).coords.requires_grad
# A nonfinite overlay retains a finite reference, rather than declaring absence.
x2 = x.detach().clone()
x2[0, 0, 0] = float("nan")
cf2 = canonical_form_from_biotite(a, torch.device("cpu"), atom37_coords=x2)
result["overlay_nan_uses_reference"] = torch.equal(
    cf2.coords[0, 0, inds[0]], torch.as_tensor(a.coord[0])
)
for code in [-2, -1]:
    try:
        z = pr.canonical_form_from_backbone_coords(
            bb.detach(), torch.full_like(seq, code), chain
        )
        result[f"pr380_token_{code}"] = {
            "accepted": True,
            "res_types": z.res_types.tolist(),
        }
    except Exception as e:
        result[f"pr380_token_{code}"] = {"accepted": False, "error": str(e)}
try:
    pr._atomworks_tokens_for_aa_order("X", torch.device("cpu"))
    result["pr380_X_accepted"] = True
except Exception as e:
    result["pr380_X_accepted"] = False
    result["pr380_X_error"] = str(e)
bad = bb.detach().clone()
bad[0, 0, 0, 0] = float("inf")
z = pr.canonical_form_from_backbone_coords(bad, seq, chain)
result["pr380_preserves_infinity"] = bool(torch.isinf(z.coords).any())
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
