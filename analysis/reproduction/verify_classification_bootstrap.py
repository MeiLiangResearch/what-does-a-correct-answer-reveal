#!/usr/bin/env python3
"""Resample school classes to check full-credit classification and score-threshold SI.

Full-credit contrasts use one eight-test Holm family:
2 contrasts x (practice success, nonconfirmation, burden, missed success).
CI of rates uses condition-stratified percentile bootstrap. The p-value
procedure matches the specified normal test based on bootstrap SD.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from reproduce_correct_practice_analysis import get_data,matched_pairs

CONDS=['Control','Base','Tutor']
STAT_KEYS=['practice_success','nonconfirmation','burden','missed']
THRESHOLDS=[1.0,.9,.8,.7,.6]


def counts(df,threshold):
    d=df.copy()
    d['d']=d.practice.eq(1).astype(int)
    d['y']=d.independent.ge(threshold-1e-9).astype(int)
    d['burden']=d.d*(1-d.y)
    d['missed']=(1-d.d)*d.y
    d['confirmed']=d.d*d.y
    tab=d.groupby(['condition','Class'],sort=True).agg(
        n=('d','size'),practice=('d','sum'),confirmed=('confirmed','sum'),
        burden=('burden','sum'),missed=('missed','sum'),independent=('y','sum'))
    return {c:tab.loc[c].to_numpy(float) for c in CONDS}


def measures(a):
    z=a.sum(axis=0) # n, practice, confirmed, burden, missed, independent
    return dict(practice_success=z[1]/z[0], nonconfirmation=z[3]/z[1],
                burden=z[3]/z[0],missed=z[4]/z[0],
                independent_success=z[5]/z[0],
                confirmation=z[2]/z[1])


def eval_one(df,cutoff,B=2000,seed=20260922):
    strata=counts(df,cutoff)
    point={c:measures(a) for c,a in strata.items()}
    rng=np.random.default_rng(seed) # restarted for each threshold as stated in SI
    draws={c:{m:np.empty(B) for m in point[c]} for c in CONDS}
    # Preserve the original release's PRNG stream order:
    # complete all bootstrap draws for Control, then Base, then Tutor.
    # Swapping these two loops changes finite-B percentile CIs and Holm p-values.
    for c in CONDS:
        for b in range(B):
            a=strata[c];m=measures(a[rng.integers(len(a),size=len(a))])
            for k,v in m.items():draws[c][k][b]=v
    cis={c:{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in d.items()}
         for c,d in draws.items()}
    contrasts={}
    if cutoff==1.0:
        for k in STAT_KEYS:
            for a,b in [('Base','Control'),('Tutor','Base')]:
                diff=point[a][k]-point[b][k]
                s=np.std(draws[a][k]-draws[b][k],ddof=1)
                pval=2*norm.sf(abs(diff/s))
                contrasts[f'{k}:{a}-{b}']={'difference':diff,'bootstrap_se':float(s),
                                               'raw_p':float(pval)}
        keys=list(contrasts)
        adj=multipletests([contrasts[k]['raw_p'] for k in keys],method='holm')[1]
        for k,p in zip(keys,adj):contrasts[k]['holm_p']=float(p)
    return {'cutoff':cutoff,'point':point,'ci95':cis,'contrasts':contrasts}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data-dir',type=Path,default=Path('source_data'))
    a.add_argument('--outdir',type=Path,default=Path('analysis_outputs'))
    a.add_argument('--download',action='store_true')
    a.add_argument('--assert-expected',action='store_true')
    z=a.parse_args();get_data(z.data_dir,z.download)
    df=matched_pairs(z.data_dir)
    result=[eval_one(df,x) for x in THRESHOLDS]
    if z.assert_expected:
        p=result[0]['point'];assert abs(p['Base']['confirmation']-.16193)<.0001
        assert abs(p['Control']['confirmation']-.36807)<.0001
        assert abs(p['Tutor']['confirmation']-.21621)<.0001
        assert abs(p['Base']['practice_success']-.48270)<.0001
    z.outdir.mkdir(parents=True,exist_ok=True)
    (z.outdir/'response_classifications_bootstrap.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'full_credit':result[0], 'other_cutoffs':[(x['cutoff'],x['point']) for x in result[1:]]},indent=2))
if __name__=='__main__':main()
