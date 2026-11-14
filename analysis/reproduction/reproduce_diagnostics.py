"""Read-only source audit and reviewer-requested statistical sensitivities.

Run with Python 3: python run_analysis.py
All outputs are placed beside this script, never in the original analysis tree.
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import ast
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import patsy
import scipy
from scipy import linalg, stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT/'data/Bastani/main_regressions'
RESULTS = ROOT/'analysis/results'
TABLES = HERE/'tables'
ARMS = ['Control', 'GPT Base', 'GPT Tutor']
COMPS = ['GPT Base - Control', 'GPT Tutor - GPT Base']
FORMULA = 'practice*C(condition) + gpa_prev + C(part3) + C(teacher) + C(Grader)'
TARGETS = ['practice', 'practice:C(condition)[T.GPT Base]', 'practice:C(condition)[T.GPT Tutor]']
RESULT = {'created_utc':datetime.now(timezone.utc).isoformat(), 'models':{}, 'checks':{}, 'notes':[]}

def log(s):
    print(s, flush=True)

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,np.bool_)): return x.item()
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def dump(name, value):
    (HERE/name).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csv(name, rows):
    pd.DataFrame(rows).to_csv(TABLES/name,index=False,encoding='utf-8-sig')

def design(df, formula=FORMULA):
    return patsy.dmatrix(formula,df,return_type='dataframe')

def contrasts(names, slope_names=TARGETS, separate=False):
    out={}
    for label,weights in zip(COMPS, ([-1,1,0] if separate else [0,1,0], [0,-1,1])):
        c=np.zeros(len(names))
        for n,w in zip(slope_names,weights): c[list(names).index(n)] = w
        out[label]=c
    return out

def reduce_nuisance(X, targets):
    """Retain a full-rank nuisance basis; never silently remove target columns."""
    nuisance=[c for c in X if c not in targets]
    N=X[nuisance].to_numpy(float)
    q,r,piv=linalg.qr(N,mode='economic',pivoting=True)
    rank=int(np.linalg.matrix_rank(N))
    keep_n=sorted(piv[:rank].tolist())
    dropped=[n for i,n in enumerate(nuisance) if i not in keep_n]
    residual=X[targets].to_numpy(float)-q[:,:rank]@(q[:,:rank].T@X[targets].to_numpy(float))
    target_rank=int(np.linalg.matrix_rank(residual))
    if target_rank != len(targets): raise ValueError(f'Targets not identifiable: residualized rank {target_rank}/{len(targets)}')
    names=[nuisance[i] for i in keep_n]+list(targets)
    xx=X[names]
    assert np.linalg.matrix_rank(xx)==xx.shape[1]
    return xx, {'candidate_columns':X.shape[1], 'candidate_rank':int(np.linalg.matrix_rank(X)),
                'effective_columns':len(names), 'dropped_exact_nuisance_aliases':dropped,
                'target_residualized_rank':target_rank,
                'target_residualized_singular_values':np.linalg.svd(residual,compute_uv=False)}

def fit_linear(X, d, name, weights=None, targets=TARGETS, separate=False):
    xx,ident=reduce_nuisance(X,targets)
    y=d.independent.to_numpy(float)
    base=sm.OLS(y,xx).fit() if weights is None else sm.WLS(y,xx,weights=weights).fit()
    fit=base.get_robustcov_results(cov_type='cluster',groups=d.Class,use_correction=True,df_correction=True,use_t=True)
    G=d.Class.nunique()
    assert abs(fit.df_resid_inference-(G-1))<1e-9
    cs=contrasts(xx.columns,targets,separate)
    rows=[]
    for label,c in cs.items():
        tt=fit.t_test(c); ci=np.asarray(tt.conf_int()).ravel()
        rows.append({'contrast':label,'estimate':float(np.asarray(tt.effect).item()),'se':float(np.asarray(tt.sd).item()),
                     'ci_low':float(ci[0]),'ci_high':float(ci[1]),'p_value':float(np.asarray(tt.pvalue).item()),'df':G-1})
    for row,p in zip(rows,multipletests([r['p_value'] for r in rows],method='holm')[1]): row['p_holm_within_model']=float(p)
    info={'n':len(d),'students':d.Student_ID.nunique(),'classes':G,'teachers':d.teacher.nunique(),
          'covariance':'Class-clustered CR1; G/(G-1)*(N-1)/(N-rank), t(G-1)',
          'identifiability':ident,'contrasts':rows}
    RESULT['models'][name]=info
    log(f'{name}: n={len(d)}, rank={xx.shape[1]}, '+', '.join(f'{r["estimate"]*100:.4f}' for r in rows)+' pp')
    return base,fit,xx,cs

def sample_info(d):
    return {'pairs':len(d),'students':d.Student_ID.nunique(),'classes':d.Class.nunique(),'teachers':d.teacher.nunique()}

def reconstruct():
    raw=RAW
    def read(n): return pd.read_csv(raw/n).rename(columns={'Student ID':'Student_ID','Treatment arm':'Treatment_arm'})
    p2,p3,mp=read('problem_part2.csv'),read('problem_part3.csv'),read('problem_mapping.csv')
    assert mp.part2.is_unique and mp.part3.dropna().is_unique
    assert not p2.duplicated(['Student_ID','Problem']).any()
    assert not p3.duplicated(['Student_ID','Problem']).any()
    a=p2.rename(columns={'Problem':'part2','Score':'practice'}).merge(mp.dropna(subset=['part3']),on='part2',validate='many_to_one')
    a=a.merge(p3[['Student_ID','Problem','Score','Class','Session','Year']].rename(columns={'Problem':'part3','Score':'independent'}),
              on=['Student_ID','part3','Class','Session','Year'],validate='one_to_one')
    a['condition']=a.Treatment_arm.map({'control':'Control','vanilla':'GPT Base','augmented':'GPT Tutor'})
    a=a[a.Honors.eq(0)].copy()
    saved=pd.read_csv(RESULTS/'bastani_matched.csv')
    keys=['Student_ID','Class','Session','Year','part2','part3']
    a=a.sort_values(keys).reset_index(drop=True); saved=saved.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[saved.columns],saved,check_dtype=False,atol=1e-12,rtol=1e-10)
    assert saved.practice.isin([0,1]).all() and saved.independent.between(0,1).all()
    assert saved.groupby('Class').condition.nunique().eq(1).all()
    assert saved.groupby('Class').teacher.nunique().eq(1).all()
    assert saved.groupby('Student_ID').Class.nunique().eq(1).all()
    RESULT['checks']['raw_reconstruction_matches_saved']=True
    return saved

def teacher_tables(d, tag):
    rows=[]
    for teacher in sorted(d.teacher.unique(),key=lambda x:int(x.split('_')[1])):
        for arm in ARMS:
            z=d[(d.teacher==teacher)&(d.condition==arm)]
            rows.append({'sample':tag,'teacher':teacher,'condition':arm,'classes':z.Class.nunique(),'students':z.Student_ID.nunique(),'pairs':len(z),
                         'correct_pairs':int(z.practice.sum()),'incorrect_pairs':int((z.practice==0).sum())})
    t=pd.DataFrame(rows); csv(f'teacher_condition_{tag}.csv',t)
    p=t.pivot(index='teacher',columns='condition',values='classes').reindex(columns=ARMS)
    return {'teachers':len(p),'teachers_in_multiple_conditions':int(p.gt(0).sum(axis=1).ge(2).sum()),
            'teachers_in_all_conditions':int(p.gt(0).all(axis=1).sum()),'class_totals':p.sum().to_dict()}

def matrix_diagnostics(X, term_slices):
    a=X.to_numpy(float); s=np.linalg.svd(a,compute_uv=False)
    names=list(X.columns); Z=X.drop(columns='Intercept')
    centered=Z.to_numpy()-Z.to_numpy().mean(axis=0)
    standardized=centered/np.sqrt((centered**2).sum(axis=0))
    U,sv,Vt=np.linalg.svd(standardized,full_matrices=False)
    corr=standardized.T@standardized; invcorr=np.linalg.inv(corr)
    vifs=np.diag(invcorr)
    csv('column_vif.csv',[{'term':n,'VIF':v,'scope':'Treatment-coded column; intercept excluded'} for n,v in zip(Z.columns,vifs)])
    # Independent auxiliary-regression VIF validation for the two interaction coefficients.
    checks={}
    for name in TARGETS[1:]:
        expected=variance_inflation_factor(a,names.index(name))
        checks[name]=abs(expected-vifs[list(Z.columns).index(name)])
    assert max(checks.values())<1e-7
    ci=sv.max()/sv; proportions=(Vt.T**2)/(sv**2)
    proportions/=proportions.sum(axis=1,keepdims=True)
    crows=[]
    for i,v in enumerate(ci):
        high=np.flatnonzero(proportions[:,i]>=.5)
        crows.append({'component':i+1,'condition_index':v,'singular_value':sv[i],
                      'terms_with_variance_proportion_ge_0_5':'; '.join(Z.columns[j] for j in high)})
    csv('condition_indices.csv',crows)
    csv('variance_decomposition.csv',[{'term':name,**{f'component_{i+1}':v for i,v in enumerate(proportions[j])}} for j,name in enumerate(Z.columns)])
    # Fox-Monette-style coefficient-correlation GVIF, invariant to coding within each term subspace.
    cov=np.linalg.inv(a.T@a)[1:,1:]; se=np.sqrt(np.diag(cov)); R=cov/np.outer(se,se)
    _,logdet=np.linalg.slogdet(R); grouped=[]
    for term,sl in term_slices.items():
        if term=='Intercept':continue
        ix=np.arange(sl.start,sl.stop)-1; jx=np.setdiff1d(np.arange(len(R)),ix)
        sign1,ld1=np.linalg.slogdet(R[np.ix_(ix,ix)]); sign2,ld2=np.linalg.slogdet(R[np.ix_(jx,jx)])
        assert sign1>0 and sign2>0
        gvif=np.exp(ld1+ld2-logdet)
        grouped.append({'term':term,'df':len(ix),'GVIF':gvif,'GVIF_power_1_over_2df':gvif**(1/(2*len(ix)))})
    csv('grouped_gvif.csv',grouped)
    return {'n':len(X),'columns':X.shape[1],'rank':int(np.linalg.matrix_rank(a)),
            'raw_condition_number':s.max()/s.min(),'centered_scaled_condition_number':ci.max(),
            'minimum_singular_value_raw':s.min(),'max_column_vif':vifs.max(),
            'max_column_vif_term':str(Z.columns[vifs.argmax()]),
            'key_vifs':{k:float(vifs[list(Z.columns).index(k)]) for k in TARGETS},
            'vif_independent_check_errors':checks,
            'coding_note':'Centered/scaled condition indices exclude intercept. VIFs use treatment coding. Interaction/dummy VIFs and grouped GVIFs are descriptive, not a standalone confounding test.'}

def load_cr2():
    # Reuse only two audited pure functions; never execute the old write-producing main().
    tree=ast.parse((ROOT/'analysis/bastani.py').read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('audit_cr2','audit_cr2_validation')]
    scope={'np':np,'pd':pd,'st':stats,'sm':sm}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_cr2_functions','exec'),scope)
    RESULT['checks']['cr2_algebra_tests']=scope['audit_cr2_validation']()
    return scope['audit_cr2']

def cluster_auxiliary(y, X, groups, tested):
    f=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True},use_t=True)
    C=np.eye(X.shape[1])[tested]
    t=f.wald_test(C,use_f=True,scalar=True)
    return {'F':float(t.statistic),'p_value':float(t.pvalue),'df_num':float(t.df_num),'df_denom':float(t.df_denom)}

def diagnostics(base, X, d):
    e=np.asarray(base.resid); mu=np.asarray(base.fittedvalues); inf=base.get_influence()
    h=inf.hat_matrix_diag; r=inf.resid_studentized_internal; cook=inf.cooks_distance[0]
    n,p=X.shape; jb=stats.jarque_bera(e); bp=het_breuschpagan(e,X,robust=True)
    aux=patsy.dmatrix('1 + fitted + I(fitted**2) + C(condition)',d.assign(fitted=mu),return_type='dataframe')
    hetero=cluster_auxiliary(e**2,aux,d.Class,list(range(1,aux.shape[1])))
    reset=X.copy(); reset['fitted_squared']=mu**2;reset['fitted_cubed']=mu**3
    resettest=cluster_auxiliary(d.independent,reset,d.Class,[reset.shape[1]-2,reset.shape[1]-1])
    out=d[['Student_ID','Class','teacher','condition','part2','part3']].copy()
    out['fitted']=mu;out['residual']=e;out['studentized_residual']=r;out['leverage']=h;out['cooks_distance']=cook
    csv('ols_diagnostic_source.csv',out)
    csv('residual_summary_by_condition.csv',d.assign(residual=e,fitted=mu).groupby('condition',observed=True).agg(
        pairs=('residual','size'),mean_residual=('residual','mean'),residual_sd=('residual','std'),fitted_min=('fitted','min'),fitted_max=('fitted','max')).reset_index())
    return {'residual_mean':e.mean(),'residual_skewness':stats.skew(e),'residual_excess_kurtosis':stats.kurtosis(e),
            'jarque_bera_iid_reference':{'statistic':jb.statistic,'p_value':jb.pvalue},
            'breusch_pagan_iid_reference':{'LM':bp[0],'LM_p':bp[1],'F':bp[2],'F_p':bp[3]},
            'clustered_variance_auxiliary':hetero,'clustered_RESET':resettest,
            'fitted_below_zero':int((mu<0).sum()),'fitted_above_one':int((mu>1).sum()),'fitted_min':mu.min(),'fitted_max':mu.max(),
            'studentized_abs_gt_3':int((abs(r)>3).sum()),'max_abs_studentized':abs(r).max(),
            'max_leverage':h.max(),'mean_leverage':h.mean(),'leverage_gt_2p_over_n':int((h>2*p/n).sum()),
            'max_cooks_distance':cook.max(),'cooks_gt_4_over_n':int((cook>4/n).sum()),
            'threshold_note':'2p/n and 4/n are descriptive flags, not deletion criteria. Studentized residuals and Cook distance use classical OLS scaling; cluster deletions assess dependence-aware influence. iid-reference p values do not account for repeated/nested observations.'}

def fractional_logit(X,d):
    f=sm.GLM(d.independent,X,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10,
          cov_type='cluster',cov_kwds={'groups':d.Class,'use_correction':True})
    assert f.converged
    mu=np.asarray(f.fittedvalues); xx=np.asarray(X)
    bread=np.linalg.inv(xx.T@(xx*(mu*(1-mu))[:,None]))
    scores=np.vstack([xx[d.Class.eq(g)].T@(d.independent.to_numpy()[d.Class.eq(g)]-mu[d.Class.eq(g)]) for g in d.Class.unique()])
    G=d.Class.nunique();N,K=xx.shape
    sandwich=bread@scores.T@scores@bread*(G/(G-1))*((N-1)/(N-K))
    cov_error=float(np.max(abs(sandwich-np.asarray(f.cov_params()))))
    assert cov_error<1e-8
    effects={}; gradients={}
    for arm in ARMS:
        vals=[];grads=[]
        for practice in [0.,1.]:
            h=d.copy(); h['condition']=pd.Categorical([arm]*len(h),categories=ARMS);h['practice']=practice
            xx=np.asarray(patsy.build_design_matrices([X.design_info],h)[0]);p=1/(1+np.exp(-(xx@f.params)))
            vals.append(p.mean());grads.append((xx*(p*(1-p))[:,None]).mean(axis=0))
        effects[arm]=vals[1]-vals[0];gradients[arm]=grads[1]-grads[0]
    rows=[];df=d.Class.nunique()-1;crit=stats.t.ppf(.975,df)
    for label,(a,b) in zip(COMPS,[('GPT Base','Control'),('GPT Tutor','GPT Base')]):
        est=effects[a]-effects[b];grad=gradients[a]-gradients[b];se=np.sqrt(grad@f.cov_params()@grad)
        rows.append({'contrast':label,'estimate':est,'se':se,'ci_low':est-crit*se,'ci_high':est+crit*se,'p_value':2*stats.t.sf(abs(est/se),df),'df':df})
    for row,p in zip(rows,multipletests([r['p_value'] for r in rows],method='holm')[1]):row['p_holm_within_model']=p
    # Check the analytic delta-method gradients against central differences.
    eps=1e-5; maxerr=0.
    for arm in ARMS:
        mats=[]
        for practice in [0.,1.]:
            h=d.copy();h['condition']=pd.Categorical([arm]*len(h),categories=ARMS);h['practice']=practice
            mats.append(np.asarray(patsy.build_design_matrices([X.design_info],h)[0]))
        for j in [0,list(X.columns).index('practice'),list(X.columns).index(TARGETS[1])]:
            b=f.params.to_numpy();delta=np.eye(len(b))[j]*eps
            def fun(beta):return np.mean(1/(1+np.exp(-mats[1]@beta))-1/(1+np.exp(-mats[0]@beta)))
            maxerr=max(maxerr,abs((fun(b+delta)-fun(b-delta))/(2*eps)-gradients[arm][j]))
    assert maxerr<1e-7
    RESULT['models']['fractional_logit']={'n':len(d),'students':d.Student_ID.nunique(),'classes':d.Class.nunique(),'teachers':d.teacher.nunique(),
        'contrasts':rows,'converged':bool(f.converged),'iterations':f.fit_history['iteration'],'average_practice_changes':effects,
        'gradient_validation_max_error':maxerr,'covariance':'Class-clustered QMLE sandwich with finite-sample multiplier; delta method; t(G-1) approximation',
        'independent_sandwich_max_error':cov_error,
        'estimand':'Difference between average discrete practice changes on common observed complete-case covariate distribution; additive extrapolation remains where condition-context cells are absent.'}
    log('fractional_logit: '+', '.join(f'{r["estimate"]*100:.4f}' for r in rows)+' pp')

def main():
    inputs=[ROOT/'analysis/bastani.py',RESULTS/'bastani_results.json',RESULTS/'bastani_matched.csv',
            *[RAW/n for n in ['problem_part2.csv','problem_part3.csv','problem_mapping.csv','final_data.csv']]]
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    dump('input_hashes_before.json',before)
    old=json.loads((RESULTS/'bastani_results.json').read_text(encoding='utf-8'))
    log('Reconstructing source data')
    all_d=reconstruct();d=all_d.dropna(subset=['practice','independent','condition','gpa_prev','part3','teacher','Grader']).copy().reset_index(drop=True)
    d['condition']=pd.Categorical(d.condition,categories=ARMS)
    assert (len(all_d),len(d),d.Student_ID.nunique(),d.Class.nunique())==(11596,11392,825,44)
    RESULT['sample']={'all_matched':sample_info(all_d),'complete_case':sample_info(d),'excluded_pairs':len(all_d)-len(d)}
    RESULT['teacher_support']={'all_matched':teacher_tables(all_d,'all_matched'),'complete_case':teacher_tables(d,'complete_case')}
    class_summary=d.groupby(['Class','teacher','condition'],observed=True).agg(pairs=('practice','size'),correct_pairs=('practice','sum'),practice_levels=('practice','nunique')).reset_index()
    csv('class_practice_support.csv',class_summary)
    RESULT['teacher_support']['classes_with_both_practice_states']=int(class_summary.practice_levels.eq(2).sum())
    X=design(d);base,fit,XX,cs=fit_linear(X,d,'original_OLS_CR1')
    diffs=[]
    for row,orig in zip(RESULT['models']['original_OLS_CR1']['contrasts'],old['continuous']['slope_contrasts']):
        for k in ['estimate','se','ci_low','ci_high','p_value']:diffs.append(abs(row[k]-orig[k]))
    assert max(diffs)<1e-8
    RESULT['checks']['original_model_max_abs_error']=max(diffs)
    RESULT['design_diagnostics']=matrix_diagnostics(X,X.design_info.term_name_slices)
    RESULT['ols_diagnostics']=diagnostics(base,XX,d)
    csv('baseline_coefficients.csv',[{'term':n,'coefficient':b,'cluster_CR1_se':s} for n,b,s in zip(XX.columns,fit.params,fit.bse)])
    for label,formula in [('no_teacher_FE',FORMULA.replace(' + C(teacher)','')),
                          ('no_grader_FE',FORMULA.replace(' + C(Grader)','')),
                          ('no_item_FE',FORMULA.replace(' + C(part3)',''))]:
        fit_linear(design(d,formula),d,label)
    fit_linear(X,d,'equal_student_weight',weights=1/d.groupby('Student_ID').Student_ID.transform('size'))
    fit_linear(design(d,FORMULA+' + I(gpa_prev**2)'),d,'quadratic_GPA')
    class_targets=[f'practice:C(condition)[{a}]' for a in ARMS]
    classX=design(d,'C(Class) + C(part3) + C(Grader) + gpa_prev + practice:C(condition)')
    cb,cf,cX,ccs=fit_linear(classX,d,'class_FE',targets=class_targets,separate=True)
    # Equivalent full dummy pseudoinverse fit must preserve fitted values and identifiable slopes.
    fullbeta=np.linalg.lstsq(np.asarray(classX),d.independent,rcond=None)[0]
    err=float(np.max(abs(np.asarray(classX)@fullbeta-cb.fittedvalues)))
    assert err<1e-8
    RESULT['checks']['class_FE_equivalent_fitted_values_max_error']=err
    N=classX.drop(columns=class_targets).to_numpy()
    S=classX[class_targets].to_numpy();yy=d.independent.to_numpy()
    Sres=S-N@np.linalg.lstsq(N,S,rcond=None)[0]
    yres=yy-N@np.linalg.lstsq(N,yy,rcond=None)[0]
    fwl=np.linalg.lstsq(Sres,yres,rcond=None)[0]
    fwl_error=float(np.max(abs(fwl-cb.params[class_targets].to_numpy())))
    assert fwl_error<1e-8
    RESULT['checks']['class_FE_FWL_slope_max_error']=fwl_error
    cr2=load_cr2()
    for label,xx,cspec in [('original_OLS_CR2',XX,cs),('class_FE_CR2',cX,ccs)]:
        log('Computing '+label)
        cr=cr2(xx,d.independent,d.Class,cspec)
        cr['contrasts']=cr.pop('results');cr.update({'n':len(d),'students':d.Student_ID.nunique(),'classes':d.Class.nunique(),'teachers':d.teacher.nunique(),'covariance':'CR2 identity working covariance; contrast-specific Satterthwaite t'})
        for row in cr['contrasts']:row['df']=row['df_satterthwaite']
        for row,p in zip(cr['contrasts'],multipletests([r['p_value'] for r in cr['contrasts']],method='holm')[1]):row['p_holm_within_model']=p
        RESULT['models'][label]=cr
    archived=old['submission_audit']['small_sample']
    dump('archived_CR2_reference.json',archived)
    cr2err=max(abs(a[k]-b[k]) for a,b in zip(RESULT['models']['original_OLS_CR2']['contrasts'],archived['results'])
               for k in ['estimate','se','ci_low','ci_high','df_satterthwaite'])
    assert cr2err<1e-8
    RESULT['checks']['CR2_archive_max_error']=cr2err
    eqold=old['submission_audit']['equal_student']['continuous']['slope_contrasts']
    eqerr=max(abs(a[k]-b[k]) for a,b in zip(RESULT['models']['equal_student_weight']['contrasts'],eqold) for k in ['estimate','se','ci_low','ci_high'])
    assert eqerr<1e-8; RESULT['checks']['equal_student_archive_max_error']=eqerr
    support=d.groupby(['teacher','condition'],observed=True).size().unstack(fill_value=0)
    restricted=[]
    for i,(a,b) in enumerate([('GPT Base','Control'),('GPT Tutor','GPT Base')]):
        teachers=support.index[(support[a]>0)&(support[b]>0)]
        sub=d[d.teacher.isin(teachers)&d.condition.isin([a,b])].copy()
        sub['arm_binary']=(sub.condition==a).astype(int)
        sx=design(sub,'practice*arm_binary + gpa_prev + C(part3) + C(teacher) + C(Grader)')
        targets=['practice','practice:arm_binary'];sx,ident=reduce_nuisance(sx,targets)
        sf=sm.OLS(sub.independent,sx).fit(cov_type='cluster',cov_kwds={'groups':sub.Class,'use_correction':True},use_t=True)
        c=np.zeros(sx.shape[1]);c[list(sx.columns).index('practice:arm_binary')]=1;tt=sf.t_test(c);ci=tt.conf_int().ravel()
        row={'contrast':COMPS[i],**sample_info(sub),'teacher_ids':list(teachers),'estimate':float(np.asarray(tt.effect).item()),'se':float(np.asarray(tt.sd).item()),'ci_low':ci[0],'ci_high':ci[1],'p_value':float(np.asarray(tt.pvalue).item()),'df':sub.Class.nunique()-1,'identifiability':ident}
        row['CR2']=cr2(sx,sub.independent,sub.Class,{COMPS[i]:c})['results'][0]
        restricted.append(row)
    RESULT['overlap_restricted']=restricted
    csv('overlap_restricted_contrasts.csv',[{k:v for k,v in r.items() if k!='identifiability'} for r in restricted])
    deletion=[]
    for unit in ['Class','teacher']:
        log('Deleting each '+unit+' in turn')
        for value in sorted(d[unit].unique()):
            mask=d[unit]!=value;sub=d.loc[mask]
            try:
                dx,ident=reduce_nuisance(X.loc[mask],TARGETS)
                df=sm.OLS(sub.independent,dx).fit(cov_type='cluster',cov_kwds={'groups':sub.Class,'use_correction':True},use_t=True)
                for label,c in contrasts(dx.columns).items():
                    tt=df.t_test(c);ci=tt.conf_int().ravel()
                    deletion.append({'unit':unit,'omitted':value,'contrast':label,'status':'estimable','n':len(sub),'classes':sub.Class.nunique(),
                         'estimate':float(np.asarray(tt.effect).item()),'se':float(np.asarray(tt.sd).item()),'ci_low':ci[0],'ci_high':ci[1]})
            except ValueError as e:
                for label in COMPS:deletion.append({'unit':unit,'omitted':value,'contrast':label,'status':str(e)})
    csv('leave_one_cluster_out.csv',deletion)
    RESULT['deletion_summary']=[]
    for (unit,contrast),g in pd.DataFrame(deletion).groupby(['unit','contrast']):
        valid=g[g.status=='estimable'];idx=COMPS.index(contrast);orig=RESULT['models']['original_OLS_CR1']['contrasts'][idx]['estimate']
        deviations=abs(valid.estimate-orig);largest=valid.loc[deviations.idxmax()]
        RESULT['deletion_summary'].append({'unit':unit,'contrast':contrast,'successful_fits':len(valid),'failed_fits':len(g)-len(valid),
           'estimate_min':valid.estimate.min(),'estimate_max':valid.estimate.max(),'max_abs_change':deviations.max(),'most_influential_omitted':largest.omitted,
           'intervals_excluding_zero':int(((valid.ci_low>0)|(valid.ci_high<0)).sum())})
    dg=RESULT['ols_diagnostics']
    trigger=(dg['fitted_below_zero']+dg['fitted_above_one']>0 or dg['clustered_RESET']['p_value']<.05 or dg['clustered_variance_auxiliary']['p_value']<.05)
    RESULT['fractional_logit_trigger']={'triggered':trigger,'rule':'Any out-of-range fitted mean, clustered RESET p<.05, or clustered variance auxiliary p<.05, as specified before fitting.'}
    if trigger:fractional_logit(X,d)
    rows=[]
    for model,info in RESULT['models'].items():
        for row in info['contrasts']:
            rows.append({'model':model,'n':info['n'],'students':info['students'],'classes':info['classes'],
                         'contrast':row['contrast'],**{k+'_pp':row[k]*100 for k in ['estimate','se','ci_low','ci_high']},
                         'df':row['df'],'p_value':row['p_value'],'p_holm_within_model':row.get('p_holm_within_model'),
                         'covariance':info['covariance']})
    csv('Table_S7_model_contrasts.csv',rows)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    assert before==after
    RESULT['checks']['all_original_input_hashes_unchanged']=True
    RESULT['environment']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
                           'statsmodels':statsmodels.__version__,'patsy':patsy.__version__}
    RESULT['notes']=['Full column rank establishes estimability in the specified model, not absence of confounding.',
       'Class intercepts absorb class-invariant means but not teacher-specific practice slopes or unmeasured within-class differences.',
       'Pairwise overlap restrictions change the population and can leave few independent classes.',
       'Observed teacher/class identifiers are pseudonymous. No data were excluded because of a residual diagnostic.',
       'The local submitted SI tables inspected before fitting did not contain a no-teacher-FE row; its stated -13.97 value is independently checked here.']
    dump('input_hashes_after.json',after);dump('analysis_results.json',RESULT)
    log('Analysis complete; original files unchanged.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    RAW=args.data_dir.resolve(); RESULTS=args.results_dir.resolve()
    HERE=args.outdir.resolve(); HERE.mkdir(parents=True,exist_ok=True)
    TABLES=HERE/'tables'; TABLES.mkdir(exist_ok=True)
    main()
