import json
import torch
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType

cif = pdbx.CIFFile.read("tmol/tests/data/ncaa_fixtures/6dmz_mod_l.cif")
aa = pdbx.get_structure(cif, model=1, include_bonds=True)
a = aa[aa.res_id == 3].copy()
b = aa[aa.res_id == 47].copy()
a.res_name[:] = "DCY"
a.chain_id[:] = "A"
b.chain_id[:] = "B"
sfxn = ScoreFunction(ParameterDatabase.get_default(), torch.device("cpu"))
sfxn.set_weight(ScoreType.disulfide, 1.0)
results = []
for order in ([a, b], [b, a]):
    ps = pose_stack_from_biotite(
        struc.concatenate(order), torch.device("cpu"), no_optH=True
    )
    scores = sfxn.render_whole_pose_scoring_module(ps)(ps.coords)
    results.append(
        {"order": [str(x.res_name[0]) for x in order], "score": float(scores.sum())}
    )
print(json.dumps(results, indent=2))
