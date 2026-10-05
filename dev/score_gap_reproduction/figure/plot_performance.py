"""Figure 1c,d from archived CoreWeave measurements; no fitted speed curves."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter

ROOT=Path(__file__).resolve().parent
D=ROOT/'data/performance'
C=json.loads((ROOT/'style/theme.json').read_text())['colors']
font=Path('/System/Library/Fonts/Helvetica.ttc')
if font.exists():font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':'Helvetica','font.size':7,'svg.fonttype':'none',
 'svg.hashsalt':'tmol-throughput-20261004','axes.labelcolor':C['ink'],'axes.edgecolor':C['ink'],
 'text.color':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],'axes.linewidth':.5,
 'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':2,
 'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':5.5,'ytick.labelsize':5.5,'legend.fontsize':5.5})
rows=list(csv.DictReader((D/'throughput.csv').open()))
relax_rows=list(csv.DictReader((ROOT/'data/fastrelax_latest/timing_summary.csv').open()))
relax_status=json.loads((ROOT/'data/fastrelax_latest/status.json').read_text())
stats=list(csv.DictReader((D/'agreement_statistics.csv').open()))
SERIES=[('pyrosetta','cpu',1,'PyRosetta CPU','target_gray','s','--'),
 ('tmol','cpu',1,'tmol CPU','amber','D','--'),
 ('tmol','cuda',1,'tmol GPU B1','blue','o','-'),
 ('tmol','cuda',10,'tmol GPU B10','teal','^','-'),
 ('tmol','cuda',100,'tmol GPU B100','mint','v','-'),
 ('tmol','cuda',1000,'tmol GPU B1000','purple','P','-')]
fig=plt.figure(figsize=(183/25.4,76/25.4),facecolor='white')
handles=[Line2D([],[],color=C[c],marker=m,linestyle=l,linewidth=.8,markersize=2.5,label=lab) for _,_,_,lab,c,m,l in SERIES]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.365,.995),ncol=3,
 frameon=False,handlelength=2.0,columnspacing=1.7,handletextpad=.5,labelspacing=.7)

for protocol,title,left in [('score_gradient','Score + gradient',13),('fastrelax','FastRelax',75)]:
 ax=fig.add_axes([left/183,13/76,44/183,44/76])
 for engine,device,batch,label,color,marker,ls in SERIES:
  source=relax_rows if protocol=='fastrelax' else rows
  data=[r for r in source if r['engine']==engine and r['device']==device and int(r['batch_size'])==batch and r['protocol']==protocol and r['status']=='ok' and (protocol=='fastrelax' or r['modality']=='protein')]
  atom_key='atoms' if protocol=='fastrelax' else 'prepared_atoms'
  data.sort(key=lambda r:float(r[atom_key]))
  if not data:continue
  x=np.array([float(r[atom_key]) for r in data]);y=np.array([float(r['structures_per_second']) for r in data])
  if protocol=='fastrelax':
   # Native FastRelax has n=1; no timing uncertainty is estimated.
   ax.plot(x,y,color=C[color],marker=marker,linestyle=ls,linewidth=.8,markersize=2.5,
           markeredgewidth=.4,zorder=3)
  else:
   lo=np.array([float(r['q25']) for r in data]);hi=np.array([float(r['q75']) for r in data])
   ax.errorbar(x,y,yerr=[y-lo,hi-y],color=C[color],marker=marker,linestyle=ls,
       linewidth=.8,markersize=2.5,markeredgewidth=.4,elinewidth=.5,capsize=1,zorder=3)
 if protocol=='fastrelax' and not relax_status['complete']:
  ax.text(.98,.98,f"{relax_status['successful']}/{relax_status['expected']} completed\nLatest rerun in progress",transform=ax.transAxes,ha='right',va='top',fontsize=5.5,color=C['target_gray'])
 ax.set_xscale('log');ax.set_yscale('log')
 ax.set_xlim(600,30000)
 ax.set_xticks([1000,10000]);ax.set_xticklabels(['1,000','10,000'])
 ax.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,5]));ax.xaxis.set_minor_formatter(NullFormatter())
 ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
 ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}' if v<1000 else f'{v:,.0f}'))
 ax.yaxis.set_minor_formatter(NullFormatter());ax.tick_params(which='minor',length=1,width=.25)
 ax.set_xlabel('Input atoms per structure' if protocol=='fastrelax' else 'Atoms per structure',labelpad=3);ax.set_ylabel('Speed (structures/s)',labelpad=3)
 ax.set_title(title,pad=5);
 fig.text((left+22)/183,65/76,'tmol 0.1.58 · CPU 32 threads' if protocol=='fastrelax' else 'tmol 0.1.58 · CPU 1 thread',ha='center',va='center',fontsize=5.5,color=C['target_gray']);ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='y',which='major',color=C['light_gray'],linewidth=.25,zorder=0)

# One pooled Pearson estimate per term. Bootstrap matched structure pairs
# within chemical strata, retaining the benchmark's 10/8/8 composition.
paired=list(csv.DictReader((D/'term_agreement.csv').open()))
terms=list(dict.fromkeys(r['term'] for r in stats if r['term']!='ALL'))
n_bootstrap=10000
rng=np.random.default_rng(20261005)
pooled=[]
for term in terms:
 data=[r for r in paired if r['term']==term]
 x=np.array([float(r['pyrosetta_energy']) for r in data])
 y=np.array([float(r['tmol_energy']) for r in data])
 assert len({r['dataset_id'] for r in data})==len(data)
 groups=[np.array([i for i,r in enumerate(data) if r['modality']==m]) for m in sorted({r['modality'] for r in data})]
 indices=np.concatenate([rng.choice(g,size=(n_bootstrap,len(g)),replace=True) for g in groups],axis=1)
 xb=x[indices];yb=y[indices]
 xb-=xb.mean(axis=1,keepdims=True);yb-=yb.mean(axis=1,keepdims=True)
 denominator=np.sqrt(np.sum(xb**2,axis=1)*np.sum(yb**2,axis=1))
 valid=denominator>0
 bootstrap=np.clip(np.sum(xb[valid]*yb[valid],axis=1)/denominator[valid],-1,1)
 estimate=float(np.corrcoef(x,y)[0,1])
 low,high=np.quantile(bootstrap,[.025,.975])
 pooled.append(dict(term=term,n=len(data),pearson_r=estimate,ci95_low=float(low),ci95_high=float(high),
     bootstrap_draws=n_bootstrap,valid_bootstrap_draws=int(valid.sum()),seed=20261005,
     method='paired structure bootstrap, stratified by modality; percentile 95% CI'))
with (D/'pooled_pearson_bootstrap.csv').open('w') as out:
 writer=csv.DictWriter(out,fieldnames=list(pooled[0]));writer.writeheader();writer.writerows(pooled)
ax=fig.add_axes([156/183,13/76,25/183,47/76])
labels={'fa_atr':'LJ attractive','fa_rep':'LJ repulsive','fa_sol':'LK solvation','fa_elec':'Electrostatics',
 'hbond':'H bonds','lk_ball':'lk_ball','lk_ball_iso':'lk_ball_iso','lk_bridge':'lk_bridge',
 'lk_bridge_uncpl':'lk_bridge_uncpl','dunbrack_rot':'Dunbrack rot.', 'dunbrack_rotdev':'Dunbrack dev.',
 'dunbrack_semirot':'Dunbrack semi.','cart_bonded':'Cartesian bonded','hxl_tors':'Hydroxyl torsion',
 'omega':'Omega','rama':'Ramachandran','ref':'Reference','disulfide':'Disulfide'}
labels['gen_bonded']='Generic bonded'

labels['gen_bonded']='Generic bonded*'
for k,row in enumerate(pooled):
 # Draw the interval explicitly: percentile intervals need not enclose the estimate.
 ax.hlines(k,row['ci95_low'],row['ci95_high'],color=C['teal'],lw=.8,zorder=2)
 ax.vlines([row['ci95_low'],row['ci95_high']],k-.10,k+.10,color=C['teal'],lw=.6,zorder=2)
 ax.plot(row['pearson_r'],k,'o',color=C['teal'],ms=2.5,mew=.4,zorder=3)
ax.set_yticks(np.arange(len(terms)),[labels.get(t,t) for t in terms]);ax.tick_params(axis='y',length=0,pad=3)
ax.set_ylim(len(terms)-.4,-.6)
# The displayed range contains every computed confidence interval.
assert min(r['ci95_low'] for r in pooled) >= .45
ax.set_xlim(.45,1.04);ax.set_xticks([.5,.75,1]);ax.set_xticklabels(['0.5','0.75','1.0'])
ax.axvline(1,color=C['interface_gray'],ls='--',lw=.5,zorder=0)
ax.spines[['top','right','left']].set_visible(False)
ax.set_xlabel('Pearson r',labelpad=3)
fig.text(155/183,73/76,'Score-term agreement',ha='center',va='top',fontsize=7)
fig.text(155/183,67/76,'26 structures · 95% bootstrap CI',ha='center',va='center',fontsize=5.5,color=C['target_gray'])
fig.text(155/183,3/76,'*8 ligand systems; 1 input failed',ha='center',va='center',fontsize=5.5,color=C['target_gray'])
fig.savefig(ROOT/'main/performance.svg',metadata={'Date':None})
plt.close(fig)

# Full identity-line plots are supplied as editable source-data figures.
# Separate modalities avoid hiding chemistry-dependent deviations in pooled r.
paired=list(csv.DictReader((D/'term_agreement.csv').open()))
for modality in sorted({r['modality'] for r in paired}):
 terms=list(dict.fromkeys(r['term'] for r in paired if r['modality']==modality))
 nrows=int(np.ceil(len(terms)/6))
 f,axes=plt.subplots(nrows,6,figsize=(183/25.4,(nrows*37+13)/25.4),squeeze=False)
 for a,term in zip(axes.flat,terms):
  p=[r for r in paired if r['modality']==modality and r['term']==term]
  x=np.array([float(r['pyrosetta_energy']) for r in p]);y=np.array([float(r['tmol_energy']) for r in p])
  lo=min(x.min(),y.min());hi=max(x.max(),y.max());pad=max((hi-lo)*.08,.05)
  a.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color=C['interface_gray'],lw=.5,ls='--')
  a.scatter(x,y,s=8,c=C['teal'],edgecolors='none')
  a.set_xlim(lo-pad,hi+pad);a.set_ylim(lo-pad,hi+pad)
  a.set_title(term,fontsize=5.5,pad=4);a.spines[['top','right']].set_visible(False)
  a.tick_params(labelsize=5.5);a.locator_params(axis='both',nbins=3)
 for a in list(axes.flat)[len(terms):]:a.set_visible(False)
 f.supxlabel('PyRosetta weighted energy',fontsize=7,y=.012)
 f.supylabel('tmol weighted energy',fontsize=7,x=.005)
 f.subplots_adjust(left=.065,right=.995,bottom=.11,top=.95,wspace=.75,hspace=.65)
 f.savefig(D/f'identity_{modality}.svg',metadata={'Date':None})
 f.savefig(D/f'identity_{modality}.png',dpi=300)
 plt.close(f)
print('Rendered measured throughput, all-chemistry per-term Pearson r and full identity-line comparisons.')
