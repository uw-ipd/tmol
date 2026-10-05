"""Test Rosetta's missing-well fallback on a private tmol parameter database."""
from pathlib import Path
import json,sys,traceback,itertools,attr,torch
from repro_paths import ROOT as source, RESULTS as audit, pyrosetta_path
import benchmark_tmol as bt
from tmol.score import beta2016_score_function
dataset=sys.argv[1];out={'dataset_id':dataset,'status':'running'}
try:
 torch.set_num_threads(1)
 pose,db=bt.replicated_pose(bt.select(dataset),torch.device('cpu'),1)
 pyro=json.loads((audit/f'exports/pyrosetta-{dataset}.json').read_text())
 libs=[];changes={}
 for lib in db.scoring.dun.rotameric_libraries:
  if lib.table_name not in ['arg','lys']:
   libs.append(lib);continue
  data=lib.rotameric_data
  present={tuple(x) for x in data.rotamers.tolist()};n=data.rotamers.shape[1]
  aliases=data.rotamer_alias.tolist();new=[]
  for well in itertools.product([1,2,3],repeat=n):
   if well in present:continue
   found=None
   for i in range(n-1,-1,-1):
    for j in [1,2,3]:
     trial=list(well);trial[i]=j
     if tuple(trial) in present:found=trial;break
    if found:break
   if found:new.append(list(well)+found)
  changes[lib.table_name]={'existing_aliases':list(aliases),'new_aliases':new,'n_rotamers':len(present)}
  aliases.extend(new)
  libs.append(attr.evolve(lib,rotameric_data=attr.evolve(data,rotamer_alias=torch.tensor(aliases,dtype=data.rotamer_alias.dtype))))
 out['table_audit']=changes
 newdb=attr.evolve(db,scoring=attr.evolve(db.scoring,dun=attr.evolve(db.scoring.dun,rotameric_libraries=tuple(libs))))
 out['controls']={}
 for label,pdb in [('native',db),('missing_well_fallback',newdb)]:
  sfxn=beta2016_score_function(torch.device('cpu'),param_db=pdb)
  scorer=sfxn.render_block_pair_scoring_module(pose)
  with torch.no_grad():energies=scorer.unweighted_scores(pose.coords)[:,0].sum(-1)*sfxn.weights_tensor()[:,None]
  names=[s.name for s in sfxn.all_score_types()]
  out['controls'][label]={'totals':dict(zip(names,energies.sum(-1).tolist())),'rotdev_per_residue':energies[names.index('dunbrack_rotdev')].tolist()}
 out['status']='ok'
except Exception as e:out.update(status='failed',error=str(e),traceback=traceback.format_exc())
(audit/'dunbrack').mkdir(exist_ok=True)
(audit/f'dunbrack/{dataset}.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['status'],out.get('traceback',''),flush=True)
if out['status']!='ok':raise SystemExit(2)
