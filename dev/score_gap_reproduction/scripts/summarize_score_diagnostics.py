"""Summarize diagnostic interventions without replacing primary measurements."""
from pathlib import Path
import json,csv

from repro_paths import ROOT, RESULTS
root=RESULTS
groups=json.loads((ROOT/'metadata/term_groups.json').read_text())
rows=[];outliers=[];audit=[]
for file in sorted((root/'controls').glob('*.json')):
 d=json.loads(file.read_text());assert d['status']=='ok'
 name=d['dataset_id'];py=json.loads((root/f'exports/pyrosetta-{name}.json').read_text())
 for term,(tn,pn) in groups.items():
  target=sum(py['weights'].get(k,0)*py['unweighted_terms'].get(k,0) for k in pn)
  for control,values in d['controls'].items():
   measured=sum(values['weighted_terms'].get(k,0) for k in tn)
   rows.append(dict(dataset_id=name,term=term,control=control,tmol=measured,pyrosetta=target,difference=measured-target))
 dun=json.loads((root/f'dunbrack/{name}.json').read_text());assert dun['status']=='ok'
 a=dun['controls']['native']['rotdev_per_residue'];b=dun['controls']['missing_well_fallback']['rotdev_per_residue']
 for i,(native,corrected) in enumerate(zip(a,b)):
  if abs(native-corrected)<.01:continue
  r=py['residues'][i];target=r['unweighted_terms']['fa_dun_dev']*py['weights']['fa_dun_dev']
  outliers.append(dict(dataset_id=name,pose_index=i+1,residue=r['name'],chain=r['pdb_chain'],pdb_number=r['pdb_number'],native_tmol=native,fallback_control_tmol=corrected,pyrosetta=target))
 cm=d['coordinate_match']
 audit.append(dict(dataset_id=name,matched_atoms=cm['matched_atoms'],unmatched_atoms=len(cm['unmatched']),coordinate_rmsd_angstrom=cm['rmsd_angstrom'],max_coordinate_delta_angstrom=cm['max_delta_angstrom'],empty_native_lys_arg_aliases=all(not v["existing_aliases"] for v in dun["table_audit"].values())))
for filename,data in [('controlled_term_differences.csv',rows),('missing_well_outliers.csv',outliers),('coordinate_audit.csv',audit)]:
 with (root/filename).open('w') as out:
  w=csv.DictWriter(out,fieldnames=list(data[0]) if data else ['dataset_id']);w.writeheader();w.writerows(data)
summary={'controls_are_diagnostic_only':True,'datasets':len(audit),'primary_figure_scores_changed':False,
 'missing_well_control':{'lys_fallback_entries':8,'arg_fallback_entries':6,'affected_residues_in_audit':len(outliers)},
 'na_solvation_controls':[],
 'remaining_unresolved':['Residual LJ repulsion differences','Residual RNA-complex bonded terms, concentrated in PyRosetta improper torsions','Proline and terminal-residue Dunbrack residuals'],
 'component_accounting':'PyRosetta cart_bonded_torsion includes cart_bonded_improper; do not add both. Per-residue allocation of interresidue terms differs across engines.',
 'source_links':{'rosetta_fallback':'https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/pack/dunbrack/RotamericSingleResidueDunbrackLibrary.tmpl.hh','tmol_revision':'f92d40d12a8670dd2654db32f37ca885714d2f4a'}}
for dataset in ['1kx5','1ysa','4lup','6q1h']:
 if dataset not in {r['dataset_id'] for r in rows}:continue
 native=next(r for r in rows if r['dataset_id']==dataset and r['term']=='fa_sol' and r['control']=='native')
 control=next(r for r in rows if r['dataset_id']==dataset and r['term']=='fa_sol' and r['control']=='pyro_na_ljlk')
 summary['na_solvation_controls'].append(dict(dataset_id=dataset,pyrosetta=native['pyrosetta'],native_tmol=native['tmol'],parameter_control_tmol=control['tmol'],absolute_error_reduction_fraction=1-abs(control['difference'])/abs(native['difference'])))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
