"""Check whether the legacy benchmark's native loaders disagree on gap connectivity.

Preserve the declared PDB chain topology explicitly, as the compatibility PDB
loader does. Do not alter coordinates, energy parameters, weights, or primary
throughput records. This is a separate diagnostic comparison.
"""
from pathlib import Path
import sys,json,time
import torch,biotite.structure.io
from common import sha256,machine_metadata
import benchmark_tmol as bt
from tmol.io import pose_stack_from_biotite
from tmol.database import ParameterDatabase
from tmol.score import beta2016_score_function
row=bt.select(sys.argv[1]);device=torch.device('cpu');torch.set_num_threads(1)
a=biotite.structure.io.load_structure(row['structure_path'],model=1,include_bonds=True)
fn=pose_stack_from_biotite if callable(pose_stack_from_biotite) else pose_stack_from_biotite.pose_stack_from_biotite
pose,ctx=fn(a,device,prepare_ligands=True,ligand_params_files=[row['ligand_tmol_params']],
 no_optH=True,param_db=ParameterDatabase.get_default(),return_context=True,
 missing_density_distance_threshold=0.0)
sfxn=beta2016_score_function(device,param_db=ctx.parameter_database);scorer=sfxn.render_whole_pose_scoring_module(pose)
x=pose.coords.detach().clone().requires_grad_(True);v=scorer(x);g,=torch.autograd.grad(v.sum(),x)
assert torch.isfinite(v).all() and torch.isfinite(g).all()
result=dict(dataset_id=row['dataset_id'],input_sha256=sha256(Path(row['structure_path'])),
 mode='retain_declared_polymer_connectivity',changed_loader_option={'missing_density_distance_threshold':0.0},
 prepared_atoms=int(pose.n_ats_per_block[0].sum()),score=float(v.detach().sum()),
 score_terms=bt.score_terms(sfxn,scorer,x.detach()),machine=machine_metadata(),
 warning='Declared long backbone connections are retained for an engine-comparison diagnostic, not repaired into a physically valid structure.')
p=Path(sys.argv[2]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['dataset_id','score','prepared_atoms']}))
