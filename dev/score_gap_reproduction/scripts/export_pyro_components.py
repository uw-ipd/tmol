"""Export cart_bonded components separately, retaining native parameters."""
from pathlib import Path
import json,sys,traceback
from repro_paths import ROOT as source, RESULTS as audit, pyrosetta_path
import benchmark_pyrosetta as bp
dataset=sys.argv[1];out={'dataset_id':dataset,'status':'running'}
try:
 row=bp.select(dataset);py=bp.initialize(row,pyrosetta_path())
 pose=py.pose_from_file(row['structure_path'])
 sfxn=py.create_score_function('beta_genpot_cart' if row['modality']=='protein_ligand' else 'beta_nov16_cart')
 scoring=py.rosetta.core.scoring
 names=['cart_bonded_length','cart_bonded_angle','cart_bonded_torsion','cart_bonded_improper','cart_bonded_ring']
 sfxn.set_weight(scoring.cart_bonded,0)
 for n in names:sfxn.set_weight(getattr(scoring,n),0.5)
 sfxn(pose)
 out['totals']={n:float(pose.energies().total_energies()[getattr(scoring,n)])*.5 for n in names}
 out['residues']=[{'index':i,'name':pose.residue(i).name(),'terms':{n:float(pose.energies().residue_total_energies(i)[getattr(scoring,n)])*.5 for n in names}} for i in range(1,pose.total_residue()+1)]
 out['status']='ok'
except Exception as e:out.update(status='failed',error=str(e),traceback=traceback.format_exc())
(audit/'pyro_components').mkdir(exist_ok=True)
(audit/f'pyro_components/{dataset}.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['status'],out.get('traceback',''),flush=True)
if out['status']!='ok':raise SystemExit(2)
