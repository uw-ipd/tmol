"""Diagnostic export of native coordinates, atom parameters and per-residue energies."""
from pathlib import Path
import json
import sys
import traceback

from repro_paths import ROOT as source, RESULTS as audit, pyrosetta_path
engine, dataset, destination = sys.argv[1], sys.argv[2], Path(sys.argv[3])
from common import sha256, machine_metadata
fields = ['lj_radius','lj_wdepth','lk_dgfree','lk_lambda','lk_volume']
result = dict(engine=engine,dataset_id=dataset,machine=machine_metadata(),status='running')
try:
    if engine == 'tmol':
        import benchmark_tmol as bt
        import torch
        row = bt.select(dataset)
        torch.set_num_threads(1)
        pose, db = bt.replicated_pose(row, torch.device('cpu'), 1)
        from tmol.score import beta2016_score_function
        sfxn = beta2016_score_function(torch.device('cpu'), param_db=db)
        scorer = sfxn.render_whole_pose_scoring_module(pose)
        pair_scorer = sfxn.render_block_pair_scoring_module(pose)
        with torch.no_grad():
            whole = scorer.unweighted_scores(pose.coords)[:,0]
            weights = sfxn.weights_tensor()
            pairs = pair_scorer.unweighted_scores(pose.coords)[:,0]
            per_residue = pairs.sum(dim=-1)
        names = [t.name for t in sfxn.all_score_types()]
        import tmol
        result['runtime']={'tmol_version':tmol.__version__,'tmol_import_path':tmol.__file__,'torch_version':torch.__version__}
        result['weights'] = dict(zip(names,weights.tolist()))
        result['unweighted_terms'] = dict(zip(names,whole.tolist()))
        result['pair_sum_minus_whole'] = dict(zip(names,(pairs.sum(dim=(-1,-2))-whole).tolist()))
        params = {p.name:{k:getattr(p,k) for k in fields} for p in db.scoring.ljlk.atom_type_parameters}
        residues=[]
        for i,ti in enumerate(pose.block_type_ind64[0].tolist()):
            if ti<0:continue
            rt=pose.packed_block_types.active_block_types[ti]
            offset=int(pose.block_coord_offset[0,i])
            atoms=[]
            for a,atom in enumerate(rt.atoms):
                atoms.append(dict(name=atom.name,atom_type=atom.atom_type,
                                  coords=pose.coords[0,offset+a].tolist(),parameters=params.get(atom.atom_type)))
            residues.append(dict(index=i+1,name=rt.name,atoms=atoms,
                                 unweighted_terms=dict(zip(names,per_residue[:,i].tolist())),
                                 connections=pose.inter_residue_connections[0,i].tolist()))
        result['residues']=residues
    else:
        import benchmark_pyrosetta as bp
        row=bp.select(dataset)
        py=bp.initialize(row,pyrosetta_path())
        pose=py.pose_from_file(row['structure_path'])
        sfxn=py.create_score_function('beta_genpot_cart' if row['modality']=='protein_ligand' else 'beta_nov16_cart')
        result['runtime']={'pyrosetta_version':py.version()}
        result['total_score']=float(sfxn(pose))
        score_types=list(sfxn.get_nonzero_weighted_scoretypes())
        names=[py.rosetta.core.scoring.name_from_score_type(t) for t in score_types]
        result['weights']={n:float(sfxn.weights()[t]) for n,t in zip(names,score_types)}
        result['unweighted_terms']={n:float(pose.energies().total_energies()[t]) for n,t in zip(names,score_types)}
        residues=[]
        for i in range(1,pose.total_residue()+1):
            res=pose.residue(i)
            atoms=[]
            for a in range(1,res.natoms()+1):
                at=res.atom_type(a);xyz=res.xyz(a)
                atoms.append(dict(name=res.atom_name(a).strip(),atom_type=at.name(),
                                  coords=[xyz.x,xyz.y,xyz.z],virtual=at.is_virtual(),
                                  parameters={k:float(getattr(at,k)()) for k in fields}))
            residues.append(dict(index=i,name=res.name(),atoms=atoms,
                                 pdb_chain=pose.pdb_info().chain(i),pdb_number=pose.pdb_info().number(i),
                                 chi=[res.chi(j) for j in range(1,res.nchi()+1)],
                                 unweighted_terms={n:float(pose.energies().residue_total_energies(i)[t]) for n,t in zip(names,score_types)}))
        result['residues']=residues
    result.update(status='ok',modality=row['modality'],input_sha256=sha256(Path(row['structure_path'])))
except Exception as exc:
    result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(result)+'\n')
print(json.dumps({k:result.get(k) for k in ['engine','dataset_id','status','error']}),flush=True)
if result['status']!='ok':raise SystemExit(2)
