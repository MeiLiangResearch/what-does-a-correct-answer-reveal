"""Render the four main figures and Figure S1 from reproduced results.
Figure drawing functions are preserved from the manuscript figure sources.
"""
from pathlib import Path
import argparse,json,csv,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from figure_style import apply_style,save_figure,AXIS_WIDTH,DATA_WIDTH,ERROR_WIDTH,INK,MUTED
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
parser=argparse.ArgumentParser()
parser.add_argument('--results-dir',type=Path,required=True)
parser.add_argument('--diagnostics-dir',type=Path,required=True)
parser.add_argument('--outdir',type=Path,required=True)
args=parser.parse_args()
RESULTS=args.results_dir.resolve();OUT=args.outdir.resolve();OUT.mkdir(parents=True,exist_ok=True)
HERE=args.diagnostics_dir.resolve();QA=OUT/'qa';QA.mkdir(exist_ok=True)
WIDTH=180/25.4
COLORS=['#64747C','#2E7392','#B87942'];ARMS=['Control','GPT Base','GPT Tutor']
NAMES=['Textbook + notes\n(no AI)','General-purpose\nAI','Tutoring-oriented\nAI']
DISPLAY=['Textbook + notes','General-purpose AI','Tutoring-oriented AI']
BLUE='#325F7D';ORANGE='#B15B3B';GRAY='#75828B';LIGHT='#D9E2E7'

def export(fig,folder,rows=None,axes=None):
    folder=OUT/folder;folder.mkdir(parents=True,exist_ok=True);fig.canvas.draw()
    fig.savefig(folder/'figure.pdf');fig.savefig(folder/'figure.svg')
    fig.savefig(folder/'figure.png',dpi=300);fig.savefig(folder/'figure.jpg',dpi=600)
    if rows:
        with (folder/'source_data.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    plt.close(fig)

def fig1():
    fig,ax=plt.subplots(figsize=(WIDTH,78/25.4));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,edge='#CFD8DE'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',facecolor='#F3F6F7',edgecolor=edge,linewidth=1.1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,linespacing=1.2)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#7B8A92',linewidth=1.2))
    ax.text(.015,.98,'Mathematics practice and immediate independent assessment',fontsize=10.5,fontweight='bold',va='top')
    ax.text(.015,.885,'839 students  |  44 classes  |  11,596 matched response pairs',fontsize=9.5,color='#52636D')
    box(.015,.46,.18,.22,'Teacher\ninstruction\nand examples')
    for y,name,col in zip([.70,.50,.30],['Textbook + notes\nNo AI','Textbook + notes\n+ general-purpose AI','Textbook + notes\n+ tutoring-oriented AI'],COLORS):
        box(.275,y,.285,.15,name,col);arrow((.207,.57),(.262,y+.075));arrow((.572,y+.075),(.627,.57))
    box(.64,.46,.14,.22,'Teacher\nreviews\nsolutions')
    box(.845,.46,.14,.22,'Matched test\nClosed book\nNo AI')
    arrow((.793,.57),(.832,.57))
    ax.text(.416,.225,'Practice: correct or incorrect',ha='center',fontsize=9)
    ax.text(.81,.155,'Test scores retain partial credit',ha='center',fontsize=9)
    ax.text(.015,.035,'All stages occur within the same lesson.',fontsize=9.5,color='#52636D')
    export(fig,'fig01_design')
def figure2():
    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(180 / 25.4, 128 / 25.4))
    fig.subplots_adjust(left=.085, right=.98, bottom=.235, top=.78, wspace=.42)
    xx = np.linspace(-4.2, 4.2, 300)
    for ax in axs:
        ax.set(xlim=(-4.3, 4.3), ylim=(-4.8, 6.2), xticks=[-4, 0, 4],
               yticks=[-4, 0, 4], xlabel='x', ylabel='y')
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(width=AXIS_WIDTH)
        sns.despine(ax=ax)
    for ax, slope, intercept in [(axs[0], 2/3, 5/3), (axs[1], -3/2, .5)]:
        sns.lineplot(x=xx, y=slope*xx+intercept, ax=ax, color='#839AA7',
                     linestyle='--', linewidth=DATA_WIDTH, errorbar=None)
    sns.lineplot(x=xx, y=(2/3)*xx+13/3, ax=axs[0], color='#287CA3', errorbar=None)
    sns.lineplot(x=xx, y=(-3/2)*xx+3, ax=axs[1], color='#287CA3', errorbar=None)
    sns.scatterplot(x=[-2], y=[3], ax=axs[0], color='#287CA3', s=24, edgecolor='none', zorder=4)
    sns.scatterplot(x=[0], y=[3], ax=axs[1], color='#287CA3', s=24, edgecolor='none', zorder=4)
    sns.scatterplot(x=[-1, 3], y=[2, -4], ax=axs[1], marker='s', color='#839AA7',
                    s=22, edgecolor='none', zorder=4)
    annotation = dict(fontsize=9.5, arrowprops=dict(arrowstyle='-', lw=.7, color='#839AA7'))
    axs[0].annotate('A (−2, 3)', xy=(-2, 3), xytext=(-3.8, 4.1), **annotation)
    axs[1].annotate('B (0, 3)', xy=(0, 3), xytext=(.7, 4.2), **annotation)
    axs[1].annotate('C (−1, 2)', xy=(-1, 2), xytext=(-3.9, 1.9), **annotation)
    axs[1].annotate('D (3, −4)', xy=(3, -4), xytext=(.2, -3.2), **annotation)
    for left, title, task, answer in [
        (.018, 'A  Practice with assigned resources',
         'Given:  2x − 3y + 5 = 0\nFind the line through A (−2, 3)\nparallel to the given line.',
         'Reference slope = 2/3\nParallel line:  y = (2/3)x + 13/3'),
        (.528, 'B  Independent test',
         'Given:  C (−1, 2) and D (3, −4)\nFind the line through B (0, 3)\nparallel to the line through C and D.',
         'Reference slope = −3/2\nParallel line:  y = −(3/2)x + 3'),
    ]:
        fig.text(left, .985, title, fontsize=10.5, weight='bold', va='top')
        fig.text(left, .930, task, fontsize=9.5, linespacing=1.25, va='top')
        fig.text(left, .170, answer, fontsize=9.5, linespacing=1.25, va='top')
    fig.text(.018, .093,
             'Same geometric idea; the independent item also requires calculating the reference slope.',
             fontsize=9.5, va='top', color=MUTED)
    save_figure(fig, OUT / 'fig02_task_example')

def figure3():
    apply_style()
    b = json.loads((RESULTS / 'bastani_results.json').read_text(encoding='utf-8'))
    fig, axs = plt.subplots(2, 2, figsize=(180 / 25.4, 138 / 25.4))
    fig.subplots_adjust(left=.18, right=.985, bottom=.19, top=.86, wspace=.85, hspace=1.14)
    headers = [
        ('A  Correct practice answers', 'Among all matched response pairs'),
        ('B  Full credit on the independent test', 'Among correct practice responses'),
        ('C  Independent scores by practice outcome', 'Adjusted for assessment context and GPA'),
        ('D  Average independent scores', 'Replication: AI versus textbook + notes'),
    ]
    for i, (title, subtitle) in enumerate(headers):
        left = .018 + .50*(i % 2)
        top = .98 if i < 2 else .53
        fig.text(left, top, title, fontsize=10.5, weight='bold', va='top')
        fig.text(left, top-.043, subtitle, fontsize=9.5, color=MUTED, va='top')
    for ax, key in zip(axs[0], ['coverage', 'confirmation']):
        for j, arm in enumerate(ARMS):
            m = next(v for v in b['decisions'] if v['threshold'] == 1 and v['condition'] == arm)['metrics']
            if key == 'confirmation':
                z = m['unconfirmed_positive']
                e, lo, hi = 1-z['estimate'], 1-z['ci_high'], 1-z['ci_low']
            else:
                z = m[key]
                e, lo, hi = (z[k] for k in ['estimate', 'ci_low', 'ci_high'])
            e, lo, hi = [100*v for v in (e, lo, hi)]
            ax.errorbar(e, 2-j, xerr=[[e-lo], [hi-e]], fmt='none', color=COLORS[j],
                        capsize=3, elinewidth=ERROR_WIDTH, capthick=AXIS_WIDTH)
            sns.scatterplot(x=[e], y=[2-j], ax=ax, marker=['o', 's', '^'][j],
                            color=COLORS[j], s=30, edgecolor='none', zorder=3)
            ax.text(hi+4, 2-j, f'{e:.1f}%', va='center', fontsize=9.5, color=COLORS[j])
        ax.set(yticks=[2, 1, 0], yticklabels=NAMES, xlim=(0, 100), ylim=(-.5, 2.5),
               xticks=[0, 25, 50, 75, 100])
        ax.set_xlabel('Practice responses (%)' if key == 'coverage' else 'Correct practice responses (%)')
    ax = axs[1, 0]
    for j, arm in enumerate(ARMS):
        v = [r for r in b['continuous']['adjusted_means'] if r['condition'] == arm]
        es = np.array([r['estimate'] for r in v])*100
        lo = np.array([r['ci_low'] for r in v])*100
        hi = np.array([r['ci_high'] for r in v])*100
        xs = np.array([0, 1]) + (j-1)*.045
        sns.lineplot(x=xs, y=es, ax=ax, color=COLORS[j], marker=['o', 's', '^'][j],
                     markersize=4.5, markeredgewidth=0, label=DISPLAY[j], errorbar=None,
                     sort=False, legend=False)
        ax.errorbar(xs, es, yerr=[es-lo, hi-es], fmt='none', color=COLORS[j],
                    capsize=3, elinewidth=ERROR_WIDTH, capthick=AXIS_WIDTH)
    ax.set(xlim=(-.15, 1.15), xticks=[0, 1], xticklabels=['Incorrect', 'Correct'],
           ylim=(0, 60), yticks=[0, 20, 40, 60], xlabel='Practice answer', ylabel='Adjusted score (%)')
    ax = axs[1, 1]
    for j, r in enumerate(b['replication']['Part3Tot']['effects']):
        e, lo, hi = [100*r[k] for k in ['estimate', 'ci_low', 'ci_high']]
        ax.errorbar(e, 1-j, xerr=[[e-lo], [hi-e]], fmt='none', color=COLORS[j+1],
                    capsize=3, elinewidth=ERROR_WIDTH, capthick=AXIS_WIDTH)
        sns.scatterplot(x=[e], y=[1-j], ax=ax, color=COLORS[j+1], marker=['s', '^'][j],
                        s=30, edgecolor='none', zorder=3)
        ax.text(e-1.3, 1-j+.25, f'{e:+.2f}', ha='center', fontsize=9.5, color=COLORS[j+1])
    ax.axvline(0, ls='--', color='#9FAAB0', lw=AXIS_WIDTH, zorder=0)
    ax.set(xlim=(-12, 4), ylim=(-.5, 1.5), yticks=[1, 0], yticklabels=NAMES[1:], xticks=[-10, -5, 0],
           xlabel='Difference from textbook + notes\n(score percentage points)')
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.50, .015), ncol=3,
               frameon=False, columnspacing=1, handlelength=1.5)
    for ax in axs.flat:
        ax.tick_params(width=AXIS_WIDTH)
        sns.despine(ax=ax)
    save_figure(fig, OUT / 'fig02_performance')

def figure4():
    apply_style()
    # These are the displayed AND plotted records in the accepted 9 October SVG.
    # Keeping them frozen preserves the primary/exploratory distinction and all intervals.
    records = json.loads((Path(__file__).resolve().parent / 'figure4_aggregate_data.json').read_text(encoding='utf-8'))
    # The course estimate is retained in Supplementary Table S12.
    records = [r for r in records if r['panel'] == 0]
    fig, ax = plt.subplots(figsize=(180 / 25.4, 69 / 25.4))
    fig.subplots_adjust(left=.43, right=.975, bottom=.23, top=.88)
    ax.set(ylim=(-.35,1.35), yticks=[], xlim=(-5.1,.25), xticks=[-5,-4,-3,-2,-1,0],
           xlabel='Change in MAE (percentage points)')
    ax.axvline(0, color='#A0ABB3', ls='--', lw=AXIS_WIDTH, zorder=0)
    sns.despine(ax=ax,left=True)
    fig.text(.02,.98,'Predictive information added by previous GPA',fontsize=10.5,weight='bold',va='top')
    for record in records:
        e, lo, hi = record['estimate'], record['ci_low'], record['ci_high']
        y = 1 if record['label'].startswith('All') else 0
        ax.errorbar(e, y, xerr=[[e-lo], [hi-e]], fmt='none', color=record['color'],
                    capsize=3, elinewidth=ERROR_WIDTH, capthick=AXIS_WIDTH)
        sns.scatterplot(x=[e], y=[y], ax=ax, color=record['color'], marker=record['marker'],
                        s=38, edgecolor='none', zorder=3)
        label = 'All matched responses\n(primary)' if y else 'Correct practice only\n(exploratory)'
        ax.text(-.05,y,label,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=9.5)
        ax.text(e,y+.25,record['display'],ha='center',va='bottom',fontsize=9,color=record['color'])
    save_figure(fig, OUT / 'fig03_priorachievement')
def save(fig,name,exclude_axes=()):
    fig.canvas.draw()
    fig.savefig(OUT/f'{name}.pdf',dpi=350,facecolor='white')
    fig.savefig(OUT/f'{name}.svg',dpi=350,facecolor='white')
    fig.savefig(OUT/f'{name}.png',dpi=350,facecolor='white')
    plt.close(fig)

def smooth(ax,x,y):
    xy=lowess(y,x,frac=.25,it=1,delta=.01*np.ptp(x),return_sorted=True)
    ax.plot(xy[:,0],xy[:,1],color=ORANGE,lw=1.4)

def title(ax,letter,text):
    ax.set_title(text,loc='left',pad=10)
    ax.text(-.16,1.075,letter,transform=ax.transAxes,weight='bold',fontsize=9,va='bottom')

def residual_figure():
    d=pd.read_csv(HERE/'tables/ols_diagnostic_source.csv')
    f=d.fitted.to_numpy()*100;e=d.residual.to_numpy()*100;r=d.studentized_residual.to_numpy()
    h=d.leverage.to_numpy();cook=d.cooks_distance.to_numpy();n=len(d)
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,145/25.4))
    fig.subplots_adjust(left=.105,right=.975,bottom=.12,top=.92,wspace=.37,hspace=.50)
    ax=axs[0,0];ax.scatter(f,e,s=4,alpha=.09,color=BLUE,linewidths=0,rasterized=True)
    smooth(ax,f,e);ax.axhline(0,color=GRAY,lw=.7,ls='--')
    ax.set(xlabel='Fitted score (%)',ylabel='Residual (percentage points)',xticks=[-40,0,40,80,120],xlim=(-42,112))
    title(ax,'a','Residuals vs fitted')
    ax=axs[0,1];theory=stats.norm.ppf((np.arange(n)+.5)/n);ordered=np.sort(r)
    ax.scatter(theory,ordered,s=5,alpha=.18,color=BLUE,linewidths=0,rasterized=True)
    q=np.quantile(r,[.25,.75]);slope=(q[1]-q[0])/(stats.norm.ppf(.75)-stats.norm.ppf(.25));intercept=q[0]-slope*stats.norm.ppf(.25)
    ax.plot([-4,4],intercept+slope*np.array([-4,4]),color=ORANGE,lw=1.2)
    ax.set(xlabel='Theoretical normal quantile',ylabel='Studentized residual',xticks=[-4,-2,0,2,4])
    title(ax,'b','Normal Q-Q')
    ax=axs[1,0];scale=np.sqrt(abs(r))
    ax.scatter(f,scale,s=4,alpha=.09,color=BLUE,linewidths=0,rasterized=True);smooth(ax,f,scale)
    ax.set(xlabel='Fitted score (%)',ylabel='Square root of |studentized residual|',xticks=[-40,0,40,80,120],xlim=(-42,112))
    title(ax,'c','Scale-location')
    ax=axs[1,1];flag=cook>4/n
    ax.scatter(h[~flag],r[~flag],s=4,alpha=.10,color=BLUE,linewidths=0,rasterized=True)
    ax.scatter(h[flag],r[flag],s=7,alpha=.35,color=ORANGE,linewidths=0,rasterized=True)
    ax.axvline(2*80/n,color=GRAY,lw=.7,ls='--')
    for y in [-3,3]:ax.axhline(y,color=GRAY,lw=.7,ls=':')
    ax.set(xlabel='Leverage',ylabel='Studentized residual',xticks=[0,.005,.010,.015,.020],xlim=(0,.024))
    title(ax,'d','Influence diagnostics')
    fig.text(.105,.025,f'All {n:,} complete pairs shown; 825 students in 44 classes. Orange points in d: Cook distance > 4/n.',fontsize=7)
    save(fig,'Figure_S1_OLS_diagnostics')
apply_style()
fig1();figure2();figure3();figure4()
plt.rcParams.update({"font.size":8,"axes.labelsize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5})
residual_figure()
print("Rendered main Figures 1-4 and Supplementary Figure S1")
