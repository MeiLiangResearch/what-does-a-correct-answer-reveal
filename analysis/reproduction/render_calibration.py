"""Source-backed publication figure; output is overwritten in place.
Python/seaborn, 180-mm width; intervals and denominators are defined in caption.
QA products are temporary so the project contains one current figure per result.
"""
from pathlib import Path
import json, sys, tempfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--results-dir',type=Path,required=True)
parser.add_argument('--outdir',type=Path,required=True)
args=parser.parse_args()
OUT=args.outdir.resolve();OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import apply_style, AXIS_WIDTH, ERROR_WIDTH, save_figure
RESULTS=args.results_dir.resolve()
QA=OUT/'qa'
QA.mkdir(exist_ok=True)
SKILL=Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL))
INK = '#24333F'
COLORS = {'Control':'#687780', 'GPT Base':'#416F91', 'GPT Tutor':'#A36B45'}
ORDER = list(COLORS)
apply_style()
def finish(fig):
    fig.canvas.draw()
    save_figure(fig, OUT)
# Core question: can added evidence improve out-of-class calibration and discrimination?
# Calibration bins are descriptive summaries of held-out predictions, never training inputs.
r=json.loads((RESULTS/'bastani_results.json').read_text(encoding='utf-8'))
rows=r['cv']['metrics']
def get(model,subset):
    return next(q for q in rows if q['model']==model and q['subset']==subset)['binary']
fig=plt.figure(figsize=(180/25.4,158/25.4))
gs=fig.add_gridspec(2,6,left=.10,right=.99,bottom=.18,top=.90,wspace=1.25,hspace=.50)
top=[fig.add_subplot(gs[0,i:i+2]) for i in (0,2,4)]
max_bin_n=max(b['n'] for m in ['M1','M4'] for c in ORDER for b in get(m,c)['curve'])
for j,(ax,condition) in enumerate(zip(top,ORDER)):
    ax.plot([0,1],[0,1],ls='--',color='#BAC3CA',lw=AXIS_WIDTH,zorder=0)
    for model,color,marker,label in [('M1','#416F91','o','M1: context + practice'),('M4','#A36B45','^','M4: full model')]:
        d=pd.DataFrame(get(model,condition)['curve'])
        # Every nonempty bin is retained. Point area scales with bin count; a
        # 7-point-square visibility floor keeps sparse bins visible without joins.
        sizes=7+65*d['n'].to_numpy()/max_bin_n
        ax.scatter(d.predicted,d.observed,s=sizes,color=color,marker=marker,
                   linewidth=0,edgecolor='none',label=label if j==0 else None)
    ax.set(xlim=(-.04,1.04),ylim=(-.04,1.04),xticks=[0,.5,1],yticks=[0,.5,1],
           xlabel='Predicted probability',ylabel='Observed full-credit rate' if j==0 else '')
    ax.set_title(f'{chr(65+j)}  '+['Textbook + notes','General-purpose AI','Tutoring-oriented AI'][j],loc='left',fontweight='bold',pad=8,fontdict={'fontsize':10.5})
    sns.despine(ax=ax)
handles,labels=top[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.10,.985),frameon=False,ncol=2,columnspacing=2)
models=['M1','M2','M3','M4']
for ax,key,title in [(fig.add_subplot(gs[1,0:2]),'brier','D  Probability error'),
                     (fig.add_subplot(gs[1,2:4]),'auc','E  Discrimination')]:
    vals=[get(m,'All')[key] for m in models]
    sns.lineplot(x=models,y=vals,ax=ax,color=INK,marker='o',markersize=5,
                 markeredgewidth=0,markeredgecolor='none',errorbar=None)
    if key=='brier':
        cis=np.array([get(m,'All')['brier_ci'] for m in models])
        ax.errorbar(np.arange(4),vals,yerr=[np.array(vals)-cis[:,0],cis[:,1]-np.array(vals)],fmt='none',ecolor=INK,capsize=3,elinewidth=ERROR_WIDTH,capthick=AXIS_WIDTH)
    ax.set(xlabel='Prediction model',ylabel='Brier score' if key=='brier' else 'AUC')
    ax.set_title(title,loc='left',fontweight='bold',pad=8)
    ax.tick_params(axis='x',labelsize=9.5)
    if key=='auc': ax.yaxis.labelpad=0
    sns.despine(ax=ax)
# Continuous calibration retains partial-credit independent scores and negative predictions.
ax=fig.add_subplot(gs[1,4:6])
curves={m:pd.DataFrame(next(q for q in rows if q['model']==m and q['subset']=='All')['continuous']['curve'])
        for m in ['M1','M4']}
max_cont_n=max(q['n'].max() for q in curves.values())
min_x=min(0,min(q.predicted.min() for q in curves.values()))-.03
max_x=max(1,max(q.predicted.max() for q in curves.values()))+.03
ax.plot([min_x,max_x],[min_x,max_x],ls='--',color='#BAC3CA',lw=AXIS_WIDTH,zorder=0)
for model,color,marker in [('M1','#416F91','o'),('M4','#A36B45','^')]:
    d=curves[model]
    ax.scatter(d.predicted,d.observed,s=7+65*d['n'].to_numpy()/max_cont_n,
               color=color,marker=marker,linewidth=0,edgecolor='none')
ax.set(xlim=(min_x,max_x),ylim=(-.03,1.03),xticks=[0,.5,1],yticks=[0,.5,1],
       xlabel='Predicted score',ylabel='Observed score')
ax.set_title('F  Continuous score',loc='left',fontweight='bold',pad=8)
sns.despine(ax=ax)
fig.text(.10,.085,'M1: context + practice   M2: + condition   M3: + practice × condition',fontsize=9.5,color=INK)
fig.text(.10,.045,'M4: + previous GPA. All predictions hold out whole classes.',fontsize=9.5,color=INK)
finish(fig)
