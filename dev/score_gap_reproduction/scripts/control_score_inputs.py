"""One-variable score diagnostics; never used as primary benchmark results."""
from pathlib import Path
import json, sys, traceback
import attr
import torch
from repro_paths import ROOT as source, RESULTS as audit, pyrosetta_path
import benchmark_tmol as bt
from tmol.score import beta2016_score_function
dataset=sys.argv[1]
out={'dataset_id':dataset,'status':'running','purpose':'diagnostic controls, not primary scores'}
try:
    torch.set_num_threads(1)
    pose,db=bt.replicated_pose(bt.select(dataset),torch.device('cpu'),1)
    pyro=json.loads((audit/f'exports/pyrosetta-{dataset}.json').read_text())
    assert len(pyro['residues'])==int((pose.block_type_ind64[0]>=0).sum())
    fields=['lj_radius','lj_wdepth','lk_dgfree','lk_lambda','lk_volume']
    candidates={}; matched=[]; unmatched=[]; mismatched=[]
    coords=pose.coords.clone()
    for i,ti in enumerate(pose.block_type_ind64[0].tolist()):
        if ti<0:continue
        rt=pose.packed_block_types.active_block_types[ti]
        pr=pyro['residues'][i]
        pa={a['name']:a for a in pr['atoms'] if not a['virtual']}
        aliases={a.name:a.alt_name for a in rt.atom_aliases}
        offset=int(pose.block_coord_offset[0,i])
        for j,a in enumerate(rt.atoms):
            # Prefer explicit Rosetta aliases, including stereospecific hydrogen names.
            p=pa.get(aliases.get(a.name,a.name)) or pa.get(a.name)
            if p is None:
                unmatched.append([i+1,rt.name,a.name]);continue
            xyz=torch.tensor(p['coords'],dtype=coords.dtype)
            delta=float(torch.linalg.vector_norm(coords[0,offset+j]-xyz))
            matched.append(delta)
            if delta>0.01:mismatched.append([i+1,rt.name,a.name,p['name'],delta])
            coords[0,offset+j]=xyz
            candidates.setdefault(a.atom_type,set()).add(tuple(p['parameters'][f] for f in fields))
    out['coordinate_match']={'matched_atoms':len(matched),'unmatched':unmatched,
        'rmsd_angstrom':(sum(x*x for x in matched)/len(matched))**.5,
        'max_delta_angstrom':max(matched),'differences_over_0.01':mismatched}
    natypes={'Oet2','Oet3','Nglyc','Nbacc','Nbamn','Pdna','OOP','Obacc','ObaccG','Nbdon'}
    changes={}
    params=[]
    for p in db.scoring.ljlk.atom_type_parameters:
        v=candidates.get(p.name,set())
        if p.name in natypes and len(v)==1:
            values=dict(zip(fields,next(iter(v))))
            changes[p.name]={'before':{f:getattr(p,f) for f in fields},'after':values}
            params.append(attr.evolve(p,**values))
        else:params.append(p)
    out['parameter_changes']=changes
    out['ambiguous_na_parameter_maps']={k:list(v) for k,v in candidates.items() if k in natypes and len(v)>1}
    control_db=attr.evolve(db,scoring=attr.evolve(db.scoring,
        ljlk=attr.evolve(db.scoring.ljlk,atom_type_parameters=tuple(params))))
    out['controls']={}
    for name,parameter_db,xyz in [('native',db,pose.coords),('pyro_coordinates',db,coords),
        ('pyro_na_ljlk',control_db,pose.coords),('pyro_coordinates_and_na_ljlk',control_db,coords)]:
        sfxn=beta2016_score_function(torch.device('cpu'),param_db=parameter_db)
        scorer=sfxn.render_whole_pose_scoring_module(pose)
        with torch.no_grad():raw=scorer.unweighted_scores(xyz)[:,0]
        terms=[t.name for t in sfxn.all_score_types()]
        out['controls'][name]={'unweighted_terms':dict(zip(terms,raw.tolist())),
            'weighted_terms':dict(zip(terms,(raw*sfxn.weights_tensor()).tolist()))}
    out['status']='ok'
except Exception as exc:
    out.update(status='failed',error=str(exc),traceback=traceback.format_exc())
(audit/'controls').mkdir(exist_ok=True)
(audit/f'controls/{dataset}.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['dataset_id','status']}),flush=True)
if out['status']!='ok':print(out['traceback']);raise SystemExit(2)
