#!/usr/bin/env python3
"""Independent checks of class-clustered CR2 and equal-student weighted CR1.

CR2 calculation is based on a bias-reduced linearization, with an identity
working covariance and contrast-specific Satterthwaite degrees of freedom.
The eigendecomposition is done in the small model-rank space rather than
forming large n_class x n_class matrices.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import qr as pivoted_qr
from scipy.stats import t as tdist
import statsmodels.formula.api as smf
import patsy
from reproduce_correct_practice_analysis import get_data, matched_pairs

FORMULA = 'score ~ practice*C(condition, Treatment(reference="Control")) + gpa_prev + C(part3) + C(teacher) + C(grader)'


def cr2_model(frame):
    y_df, x_df = patsy.dmatrices(FORMULA, frame, return_type='dataframe')
    y=y_df.iloc[:,0].to_numpy(dtype=float)
    X=x_df.to_numpy(dtype=float)
    # Pivoted QR avoids a singular numerical system in purely synthetic smoke tests.
    # No columns are removed when the real-study design has full rank.
    Q0,R0,perm=pivoted_qr(X,mode='economic',pivoting=True)
    rank=int(np.sum(np.abs(np.diag(R0)) > (np.max(np.abs(np.diag(R0))) * 1e-11)))
    selected=perm[:rank]
    X=X[:,selected]
    Q,R=np.linalg.qr(X,mode='reduced')
    beta=np.linalg.solve(R,Q.T@y)
    e=y-X@beta
    cols=[list(x_df.columns)[int(j)] for j in selected]
    base=next(i for i,k in enumerate(cols) if 'practice:C(condition' in k and '[T.Base]' in k)
    tutor=next(i for i,k in enumerate(cols) if 'practice:C(condition' in k and '[T.Tutor]' in k)
    ids=frame.loc[x_df.index,'Class'].to_numpy()
    group_labels=np.unique(ids)
    N=len(y);G=len(group_labels);K=X.shape[1]
    if G < 3: raise ValueError("Too few clusters for CR2")
    B=[];v=[];inverse_root=[]
    for cl in group_labels:
        sub=Q[ids==cl,:]
        Bj=sub.T@sub
        lam,V=np.linalg.eigh(np.eye(K)-Bj)
        # Moore-Penrose convention for non-identifiable within-cluster directions.
        diag=np.where(lam>1e-10,1/np.sqrt(np.maximum(lam,1e-30)),0)
        op=(V*diag)@V.T
        B.append(Bj)
        v.append(sub.T@e[ids==cl])
        inverse_root.append(op)
    def contrast(items):
        c=np.zeros(K)
        for i,z in items:c[i]=z
        estimate=float(c@beta)
        t=np.linalg.solve(R.T,c)
        q=[];tw=[];vdiag=[]
        for Bj,z,op in zip(B,v,inverse_root):
            w=op@t
            q.append(float(z@w))
            tw.append(Bj@w)
            vdiag.append(float(w@Bj@w))
        se=float(np.sqrt(np.sum(np.square(q))))
        Kcov=-np.stack(tw)@np.stack(tw).T
        Kcov[np.diag_indices(G)]+=vdiag
        df=float(np.trace(Kcov)**2/np.sum(Kcov**2))
        ci=[estimate - tdist.ppf(.975,df)*se, estimate+tdist.ppf(.975,df)*se]
        p=float(2*tdist.sf(abs(estimate/se),df))
        return {'estimate':estimate,'se_CR2':se,'Satterthwaite_df':df,'CI95':ci,
                'raw_p':p,'variance_expected_under_identity':float(np.trace(Kcov)),
                'OLS_homoskedastic_variance':float(t@t)}
    return {'n_pairs':N,'classes':G,'parameters':K,
        'base_minus_control':contrast([(base,1)]),
        'tutor_minus_base':contrast([(tutor,1),(base,-1)])}


def weighted_cr1(frame):
    tmp=frame.copy()
    tmp['weight']=1/tmp.groupby('student').student.transform('size')
    fit=smf.wls(FORMULA,tmp,weights=tmp.weight).fit(
        cov_type='cluster',cov_kwds={'groups':tmp.Class,'use_correction':True,'df_correction':True},use_t=True)
    names=list(fit.params.index)
    base=next(i for i,x in enumerate(names) if 'practice:C(condition' in x and '[T.Base]' in x)
    tutor=next(i for i,x in enumerate(names) if 'practice:C(condition' in x and '[T.Tutor]' in x)
    def contrast(index):
        v=np.zeros(len(names))
        for k,n in index:v[k]=n
        a=float(v@fit.params);se=float(np.sqrt(v@fit.cov_params()@v))
        q=tdist.ppf(.975,tmp.Class.nunique()-1)
        return {'estimate':a,'CI95':[a-q*se,a+q*se],
                'raw_p':float(2*tdist.sf(abs(a/se),43))}
    return {'base_minus_control':contrast([(base,1)]),
            'tutor_minus_base':contrast([(tutor,1),(base,-1)])}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('source_data'))
    p.add_argument('--outdir',type=Path,default=Path('analysis_outputs'))
    p.add_argument('--download',action='store_true')
    p.add_argument('--assert-expected',action='store_true')
    args=p.parse_args()
    get_data(args.data_dir,args.download)
    full=matched_pairs(args.data_dir)
    frame=full.loc[full.gpa_prev.notna() & full.grader.ne('Missing')].copy()
    result={'CR2':cr2_model(frame),'equal_student_weight_CR1':weighted_cr1(frame)}
    if args.assert_expected:
        C=result['CR2']
        assert abs(C['base_minus_control']['estimate']+12.94)<.05
        assert abs(C['tutor_minus_base']['estimate']-2.92)<.05
        assert abs(C['base_minus_control']['Satterthwaite_df']-24.18)<.2, C
        assert abs(C['tutor_minus_base']['Satterthwaite_df']-23.77)<.2, C
        W=result['equal_student_weight_CR1']
        assert abs(W['base_minus_control']['estimate']+12.53)<.05
        assert abs(W['tutor_minus_base']['estimate']-2.46)<.05
    args.outdir.mkdir(parents=True,exist_ok=True)
    (args.outdir/'cr2_equal_student_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
