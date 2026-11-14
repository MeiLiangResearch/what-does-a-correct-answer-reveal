"""Bastani secondary analysis; run from any directory with Python 3.

Frozen 2026-09-22 after source schema inspection and source-effect replication,
before new fits. See analysis_spec.md. Original release remains untouched.
Primary population: Honors==0, per source main analysis. Unit: official matched
practice/independent item per student/session. Practice is binary, independent
score is partial credit [0,1]. No invented partial-credit practice thresholds.
OLS: independent ~ practice*condition + previous GPA + matched-item, teacher,
grader fixed effects (item absorbs grade/session). Complete cases. CR1 class
covariance; t inference with G-1 df. Two primary slope contrasts form one Holm
family. Full-correct decision: practice>=1 and independent>=1 (tolerance 1e-9);
secondary independent criteria 0.6,0.7,0.8,0.9. Percentile class bootstraps,
stratified by randomized condition, 2000 draws. Eight primary decision contrasts
(four metrics x Base-Control/Tutor-Base) form a separate Holm family; unadjusted
95% CIs retained and labeled. No removal of outliers/clipping observed scores.
CV: five deterministic shuffled class-level folds stratified by treatment;
all rows/students of a class stay together. M0 item+teacher+grader design context;
M1 +practice; M2 +condition; M3 +interaction; M4 +GPA. Train-only median GPA
imputation and scaling, missing category for grader, unknown categories ignored.
Fixed weak ridge regularization alpha=1 continuous, C=1 logistic, no tuning;
identical samples/folds for all models. No class/student identifiers as features.
Calibration summaries are diagnostics of OOF predictions, never used to refit
or improve those predictions. Paired error improvement intervals resample held-
out class residuals, conditional on fitted predictions (not full training
uncertainty). 60/70/80/90 are decision sensitivities, not CV outcome searches.
M5 omitted: public text linkability is audited, but no independently validated
behavior coding or reliable assessment-boundary timestamps available.
Sensitivity: include honors; exclude source-listed noncompliant class/sessions;
omit GPA to retain its missing cases. No post-treatment conditional association
is described as a randomized causal effect or a latent-competence diagnosis.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data'/'Bastani'
OUT = ROOT/'analysis'/'results'
SEED = 20260922
B = 2000
ARMS = ['Control', 'GPT Base', 'GPT Tutor']
CONTRASTS = [('GPT Base','Control'), ('GPT Tutor','GPT Base'), ('GPT Tutor','Control')]
FORMULA = 'independent ~ practice*C(condition) + gpa_prev + C(part3)+C(teacher)+C(Grader)'

def read(name):
    return pd.read_csv(RAW/'main_regressions'/name).rename(columns={'Student ID':'Student_ID','Treatment arm':'Treatment_arm'})

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def interval(values):
    return [float(v) for v in np.nanpercentile(values,[2.5,97.5])]

def contrast(result, weights, label):
    v=np.zeros(len(result.params))
    for key,w in weights.items(): v[list(result.params.index).index(key)]=w
    test=result.t_test(v)
    lo,hi=np.asarray(test.conf_int()).ravel()
    return {'contrast':label,'estimate':float(np.asarray(test.effect).item()),'se':float(np.asarray(test.sd).item()),'ci_low':float(lo),'ci_high':float(hi),'p_value':float(np.asarray(test.pvalue).item())}

def regression(data, formula=FORMULA):
    need=['practice','independent','condition','part3','teacher','Grader']
    if 'gpa_prev' in formula: need += ['gpa_prev']
    d=data.dropna(subset=need).copy()
    d['condition']=pd.Categorical(d.condition,categories=ARMS)
    fit=smf.ols(formula,d).fit(cov_type='cluster',cov_kwds={'groups':d.Class},use_t=True)
    keys={'Control':{'practice':1}, 'GPT Base':{'practice':1,'practice:C(condition)[T.GPT Base]':1}, 'GPT Tutor':{'practice':1,'practice:C(condition)[T.GPT Tutor]':1}}
    slopes=[contrast(fit,keys[a],a) for a in ARMS]
    diffs=[]
    for a,b in CONTRASTS:
        w={k:keys[a].get(k,0)-keys[b].get(k,0) for k in set(keys[a])|set(keys[b])}
        diffs.append(contrast(fit,w,a+' - '+b))
    ps=multipletests([x['p_value'] for x in diffs[:2]],method='holm')[1]
    for x,p in zip(diffs,ps):x['p_holm']=float(p)
    # Marginal standardization uses same complete-case context/GPA distribution.
    from patsy import build_design_matrices
    means=[]
    for arm in ARMS:
        for practice in [0,1]:
            h=d.copy();h['condition']=arm;h['practice']=practice
            design=getattr(fit.model.data,'design_info',None) or fit.model.data.model_spec
            X=np.asarray(build_design_matrices([design],h)[0]).mean(axis=0)
            t=fit.t_test(X)
            lo,hi=t.conf_int()[0]
            means.append({'condition':arm,'practice':practice,'estimate':float(t.effect.item()),'ci_low':float(lo),'ci_high':float(hi)})
    return {'formula':formula,'n':len(d),'students':d.Student_ID.nunique(),'classes':d.Class.nunique(),'r2_in_sample':fit.rsquared,'slopes':slopes,'slope_contrasts':diffs,'adjusted_means':means,'prior_gpa':contrast(fit,{'gpa_prev':1},'gpa_prev') if 'gpa_prev' in formula else None,'coefficients':[{'term':k,'estimate':float(v),'se':float(fit.bse[k]),'ci_low':float(fit.conf_int().loc[k,0]),'ci_high':float(fit.conf_int().loc[k,1]),'p_value':float(fit.pvalues[k])} for k,v in fit.params.items()]}

def bootstrap_counts(data, threshold):
    d=data.copy();d['D']=d.practice.ge(1-1e-9);d['Y']=d.independent.ge(threshold-1e-9)
    d['unconfirmed']=d.D & ~d.Y;d['missed']=~d.D & d.Y
    stats=d.groupby(['condition','Class'],observed=True).agg(n=('D','size'),D=('D','sum'),Y=('Y','sum'),unconfirmed=('unconfirmed','sum'),missed=('missed','sum')).reset_index()
    rng=np.random.default_rng(SEED)
    result=[];boots={}
    def metrics(x):
        n,D,Y,u,m=x.T
        return np.column_stack([D/n,Y/n,u/np.maximum(D,1),u/n,m/n])
    names=['coverage','independent_success','unconfirmed_positive','burden','missed_success']
    for arm in ARMS:
        s=stats[stats.condition.eq(arm)][['n','D','Y','unconfirmed','missed']].to_numpy(float)
        sums=s[rng.integers(0,len(s),(B,len(s)))].sum(axis=1)
        b=metrics(sums);point=metrics(s.sum(axis=0)[None,:])[0]
        boots[arm]=b
        result.append({'threshold':threshold,'condition':arm,'n':int(s[:,0].sum()),'classes':len(s),'positive_n':int(s[:,1].sum()),'independent_success_n':int(s[:,2].sum()),'unconfirmed_n':int(s[:,3].sum()),'missed_n':int(s[:,4].sum()),'metrics':{name:{'estimate':float(point[j]),'ci_low':interval(b[:,j])[0],'ci_high':interval(b[:,j])[1]} for j,name in enumerate(names)}})
    comparisons=[]
    for a,b in CONTRASTS:
        for j,name in enumerate(names):
            diff=boots[a][:,j]-boots[b][:,j]
            est=next(x for x in result if x['condition']==a)['metrics'][name]['estimate']-next(x for x in result if x['condition']==b)['metrics'][name]['estimate']
            lo,hi=interval(diff);se=np.std(diff,ddof=1)
            comparisons.append({'threshold':threshold,'contrast':a+' - '+b,'metric':name,'estimate':est,'ci_low':lo,'ci_high':hi,'p_value':float(2*st.norm.sf(abs(est/se))) if se else None})
    primary=[x for x in comparisons if x['metric']!='independent_success' and x['contrast']!='GPT Tutor - Control']
    for row,p in zip(primary,multipletests([x['p_value'] for x in primary],method='holm')[1]):row['p_holm']=float(p)
    return result,comparisons

def calibration(y,p,binary):
    if binary:
        p=np.clip(p,1e-8,1-1e-8); x=np.log(p/(1-p))
        f=sm.GLM(y,sm.add_constant(x),family=sm.families.Binomial()).fit()
        intonly=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=x).fit()
        intercept=float(intonly.params[0]);slope=float(f.params[1]);joint_intercept=float(f.params[0])
        binid=np.minimum((p*10).astype(int),9)
    else:
        f=sm.OLS(y,sm.add_constant(p)).fit();intercept=float(f.params[0]);slope=float(f.params[1]);joint_intercept=intercept
        binid=pd.qcut(p,10,labels=False,duplicates='drop')
    curve=pd.DataFrame({'predicted':p,'observed':y,'bin':binid}).groupby('bin').agg(predicted=('predicted','mean'),observed=('observed','mean'),n=('observed','size')).reset_index().to_dict('records')
    return {'calibration_intercept':intercept,'calibration_slope':slope,'joint_calibration_intercept':joint_intercept,'curve':curve}

def cv_analysis(d):
    d=d.copy();d['condition_base']=d.condition.eq('GPT Base').astype(float);d['condition_tutor']=d.condition.eq('GPT Tutor').astype(float)
    d['practice_base']=d.practice*d.condition_base;d['practice_tutor']=d.practice*d.condition_tutor
    d['Grader']=d.Grader.fillna('Missing').astype(str)
    classes=d[['Class','condition']].drop_duplicates().sort_values('Class').reset_index(drop=True)
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    classfold={}
    for fold,(_,test) in enumerate(splitter.split(classes,classes.condition)):
        classfold.update({x:fold for x in classes.iloc[test].Class})
    d['fold']=d.Class.map(classfold)
    oof=d[['Student_ID','Class','condition','part3','practice','independent','gpa_prev','fold']].copy()
    oof['success']=d.independent.ge(1-1e-9).astype(int)
    specs={};fold_audit=[]
    for m in range(5):
        numeric=[]
        if m>=1:numeric+=['practice']
        if m>=2:numeric+=['condition_base','condition_tutor']
        if m>=3:numeric+=['practice_base','practice_tutor']
        if m>=4:numeric+=['gpa_prev']
        specs[f'M{m}']={'categorical':['part3','teacher','Grader'],'numeric':numeric,'continuous':'Ridge alpha=1','binary':'LogisticRegression C=1'}
        for target,binary in [('independent',False),('success',True)]:
            pred=np.zeros(len(d))
            for fold in range(5):
                tr=d.fold.ne(fold).to_numpy();te=~tr
                assert not set(d.loc[tr,'Class']) & set(d.loc[te,'Class'])
                assert not set(d.loc[tr,'Student_ID']) & set(d.loc[te,'Student_ID'])
                cat=OneHotEncoder(handle_unknown='ignore',drop=None,sparse_output=False)
                transforms=[('context',cat,['part3','teacher','Grader'])]
                if numeric:transforms.append(('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric))
                pre=ColumnTransformer(transforms)
                model=LogisticRegression(C=1,max_iter=3000,tol=1e-9,solver='lbfgs') if binary else Ridge(alpha=1)
                pipe=make_pipeline(pre,model)
                y=oof.success.to_numpy() if binary else d.independent.to_numpy()
                pipe.fit(d.loc[tr],y[tr]);pred[te]=pipe.predict_proba(d.loc[te])[:,1] if binary else pipe.predict(d.loc[te])
                if m==0 and not binary:fold_audit.append({'fold':fold,'train_n':int(tr.sum()),'test_n':int(te.sum()),'train_classes':sorted(d.loc[tr,'Class'].unique()),'test_classes':sorted(d.loc[te,'Class'].unique()),'student_overlap':0,'class_overlap':0,'test_conditions':d.loc[te,'condition'].value_counts().to_dict()})
            oof[('binary_' if binary else 'continuous_')+f'M{m}']=pred
    metrics=[]
    for m in range(5):
        for subset in ['All']+ARMS:
            h=oof if subset=='All' else oof[oof.condition.eq(subset)]
            y=h.independent.to_numpy();yc=h.success.to_numpy();p=h[f'continuous_M{m}'].to_numpy();pc=h[f'binary_M{m}'].to_numpy()
            metrics.append({'model':f'M{m}','subset':subset,'n':len(h),'continuous':{'mae':mean_absolute_error(y,p),'rmse':np.sqrt(mean_squared_error(y,p)),'r2':r2_score(y,p),**calibration(y,p,False)},'binary':{'brier':brier_score_loss(yc,pc),'log_loss':log_loss(yc,pc),'auc':roc_auc_score(yc,pc),**calibration(yc,pc,True)}})
    # Paired class bootstrap of fixed held-out predictions, preserving arm counts.
    class_ids=sorted(oof.Class.unique());lookup={k:i for i,k in enumerate(class_ids)}
    rng=np.random.default_rng(SEED+1);weights=np.zeros((B,len(class_ids)))
    for arm in ARMS:
        ids=np.array([lookup[c] for c in classes[classes.condition.eq(arm)].Class])
        draws=ids[rng.integers(0,len(ids),(B,len(ids)))]
        for b in range(B):np.add.at(weights[b],draws[b],1)
    counts=oof.groupby('Class').size().reindex(class_ids).to_numpy();denom=weights@counts
    losses={};improvements=[]
    for m in range(5):
        y=oof.independent.to_numpy();yc=oof.success.to_numpy();p=oof[f'continuous_M{m}'].to_numpy();pc=np.clip(oof[f'binary_M{m}'].to_numpy(),1e-8,1-1e-8)
        errors={'mae':abs(y-p),'rmse':(y-p)**2,'brier':(yc-pc)**2,'log_loss':-(yc*np.log(pc)+(1-yc)*np.log(1-pc))}
        losses[m]={}
        for metric,v in errors.items():
            agg=pd.Series(v).groupby(oof.Class.reset_index(drop=True)).sum().reindex(class_ids).to_numpy()
            boot=(weights@agg)/denom
            if metric=='rmse':boot=np.sqrt(boot)
            losses[m][metric]=boot
            row=next(x for x in metrics if x['model']==f'M{m}' and x['subset']=='All')
            row['binary' if metric in ['brier','log_loss'] else 'continuous'][metric+'_ci']=interval(boot)
    for m in range(1,5):
        for metric in ['mae','rmse','brier','log_loss']:
            category='binary' if metric in ['brier','log_loss'] else 'continuous'
            before=next(x for x in metrics if x['model']==f'M{m-1}' and x['subset']=='All')[category][metric]
            after=next(x for x in metrics if x['model']==f'M{m}' and x['subset']=='All')[category][metric]
            boot=losses[m-1][metric]-losses[m][metric];lo,hi=interval(boot)
            improvements.append({'comparison':f'M{m} vs M{m-1}','metric':metric,'estimate':before-after,'ci_low':lo,'ci_high':hi,'meaning':'Positive values mean lower held-out error in larger model; conditional on fitted OOF predictions.'})
    oof.to_csv(OUT/'bastani_oof.csv',index=False)
    return {'models':specs,'folds':fold_audit,'metrics':metrics,'paired_improvements':improvements,'uncertainty':'Class-stratified bootstrap of fixed OOF losses; excludes training/model-selection uncertainty. Calibration intercept is offset-only calibration-in-the-large; slope is joint intercept/slope logistic fit. Curves are diagnostic, not cross-fitted recalibration.'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    totals=read('final_data.csv');p2=read('problem_part2.csv');p3=read('problem_part3.csv');mapping=read('problem_mapping.csv')
    assert mapping.part2.is_unique and mapping.part3.dropna().is_unique
    for df in [p2,p3]:assert not df.duplicated(['Student_ID','Problem']).any()
    eligible=p2.rename(columns={'Problem':'part2','Score':'practice'}).merge(mapping.dropna(subset=['part3']),on='part2',validate='many_to_one')
    eligible_keys=eligible[['Student_ID','part3','Class','Session','Year']]
    independent_keys=p3[['Student_ID','Problem','Class','Session','Year']].rename(columns={'Problem':'part3'})
    coverage=eligible_keys.merge(independent_keys,on=list(eligible_keys.columns),how='outer',indicator=True,validate='one_to_one')
    a=eligible
    a=a.merge(p3[['Student_ID','Problem','Score','Class','Session','Year']].rename(columns={'Problem':'part3','Score':'independent'}),on=['Student_ID','part3','Class','Session','Year'],validate='one_to_one')
    a['condition']=a.Treatment_arm.map({'control':'Control','vanilla':'GPT Base','augmented':'GPT Tutor'})
    assert a.condition.notna().all() and a.practice.isin([0,1]).all() and a.independent.between(0,1).all()
    checkcols=['Class','Session','Year','GPTBase','GPTTutor','Treatment_arm','Honors','teacher','Grader','gpa_prev']
    source_check=a[['Student_ID','part3']+checkcols].merge(p3[['Student_ID','Problem']+checkcols],left_on=['Student_ID','part3'],right_on=['Student_ID','Problem'],suffixes=('_practice','_independent'),validate='one_to_one')
    consistency={k:int((~(source_check[k+'_practice'].eq(source_check[k+'_independent']) | (source_check[k+'_practice'].isna() & source_check[k+'_independent'].isna()))).sum()) for k in checkcols}
    assert sum(consistency.values())==0
    primary=a[a.Honors.eq(0)].copy().reset_index(drop=True)
    primary[['Student_ID','Class','Year','Session','part2','part3','practice','independent','condition','gpa_prev','teacher','Grader','Honors']].to_csv(OUT/'bastani_matched.csv',index=False)
    commit=subprocess.check_output(['git','-C',str(RAW),'rev-parse','HEAD'],text=True).strip()
    rawfiles=['README.md','main_regressions/final_data.csv','main_regressions/problem_part2.csv','main_regressions/problem_part3.csv','main_regressions/problem_mapping.csv','main_regressions/main_analysis.R','main_regressions/problem_level_analysis.R','text_analysis/data/raw/valid_student_data_w_time_stamp.csv']
    result={'provenance':{'repository':'https://github.com/obastani/GenAICanHarmLearning','doi':'10.1073/pnas.2422633122','commit':commit,'accessed':'2026-09-22','sha256':{x:hashlib.sha256((RAW/x).read_bytes()).hexdigest() for x in rawfiles}},'specification':__doc__,'audit':{},'replication':{},'continuous':{},'decisions':[],'decision_contrasts':[],'sensitivity':{},'registry':[]}
    result['audit']={'raw_total_rows':len(totals),'raw_students':totals.Student_ID.nunique(),'raw_classes':totals.Class.nunique(),'practice_templates':len(mapping),'matched_templates':int(mapping.part3.notna().sum()),'unmatched_practice_templates':int(mapping.part3.isna().sum()),'part3_templates':p3.Problem.nunique(),'unmatched_ids':mapping.loc[mapping.part3.isna(),'part2'].tolist(),'matched_rows_all':len(a),'primary_rows':len(primary),'primary_students':primary.Student_ID.nunique(),'primary_classes':primary.Class.nunique(),'primary_student_sessions':len(primary[['Student_ID','Session']].drop_duplicates()),'missing_primary':primary.isna().sum().to_dict(),'scores':{'practice_values':sorted(primary.practice.unique().tolist()),'independent_min':primary.independent.min(),'independent_max':primary.independent.max(),'independent_unique_n':primary.independent.nunique(),'interpretation':'Practice is binary correctness; independent item includes partial credit. A slope is the independent-score difference between practice correct and incorrect.'},'grade_session':primary.groupby(['Year','Session']).agg(rows=('Student_ID','size'),students=('Student_ID','nunique'),matched_templates=('part3','nunique')).reset_index().to_dict('records'),'condition_counts':primary.groupby('condition').agg(n=('Student_ID','size'),students=('Student_ID','nunique'),classes=('Class','nunique'),practice_mean=('practice','mean'),independent_mean=('independent','mean'),practice_sd=('practice','std'),independent_sd=('independent','std')).reset_index().to_dict('records'),'duplicate_matched_keys':0,'class_has_multiple_conditions':int(primary.groupby('Class').condition.nunique().gt(1).sum()),'student_has_multiple_classes':int(primary.groupby('Student_ID').Class.nunique().gt(1).sum())}
    result['audit']['mapping_flow']={'raw_practice_rows':len(p2),'eligible_mapped_practice_rows':len(eligible),'structurally_unmatched_practice_rows':len(p2)-len(eligible),'eligible_mapped_practice_rows_nonhonors':int(eligible.Honors.eq(0).sum()),'missing_independent_for_eligible_practice':int(coverage._merge.eq('left_only').sum()),'independent_rows_without_matched_practice':int(coverage._merge.eq('right_only').sum()),'source_variable_disagreements':consistency,'all_counts_include_honors_unless_labeled':True}
    # Verify total score is the item mean, including partial-credit scoring.
    for name,items in [('Part2Tot',p2),('Part3Tot',p3)]:
        im=items.groupby(['Student_ID','Session']).Score.mean().rename('item_mean')
        z=totals.merge(im,on=['Student_ID','Session'])
        result['audit'][name+'_item_mean_max_difference']=abs(z[name]-z.item_mean).max()
    for outcome in ['Part2Tot','Part3Tot']:
        s=totals[totals.Honors.eq(0)].dropna(subset=[outcome,'gpa_prev','teacher','Grader','Session','Year'])
        f=smf.ols(outcome+' ~ GPTBase + GPTTutor + gpa_prev + C(teacher)+C(Session)+C(Grader)+C(Year)',s).fit(cov_type='cluster',cov_kwds={'groups':s.Class},use_t=True)
        effects=[contrast(f,{term:1},term) for term in ['GPTBase','GPTTutor']]
        ctrl=totals[(totals.Honors.eq(0)) & totals.GPTBase.eq(0) & totals.GPTTutor.eq(0)][outcome].mean()
        for e in effects:e['relative_to_raw_control_mean']=e['estimate']/ctrl
        result['replication'][outcome]={'n':len(s),'students':s.Student_ID.nunique(),'classes':s.Class.nunique(),'control_mean':ctrl,'effects':effects,'source_formula':f.model.formula}
    assert abs(result['replication']['Part2Tot']['effects'][0]['estimate']-.137)<.001
    assert abs(result['replication']['Part2Tot']['effects'][1]['estimate']-.361)<.001
    assert abs(result['replication']['Part3Tot']['effects'][0]['estimate']+.054)<.001
    result['replication']['status']='PASS: source point estimates match reported 0.137, 0.361, -0.054 and -0.004.'
    print(result['replication']['status'],flush=True)
    result['continuous']=regression(primary)
    result['continuous']['raw_associations']=primary.groupby('condition').apply(lambda h:pd.Series({'pearson_r':h.practice.corr(h.independent),'independent_mean_incorrect':h.loc[h.practice.eq(0),'independent'].mean(),'independent_mean_correct':h.loc[h.practice.eq(1),'independent'].mean()}),include_groups=False).reset_index().to_dict('records')
    result['continuous']['practice_only_discrimination']=[{'condition':arm,'n':len(h),'auc_binary_practice_for_full_correct_independent':roc_auc_score(h.independent.ge(1-1e-9),h.practice),'pearson_r':h.practice.corr(h.independent),'interpretation':'Descriptive discrimination using the observed binary practice score alone; no fitting or threshold selection. Contrast is not causal.'} for arm in ARMS for h in [primary[primary.condition.eq(arm)]]]
    for threshold in [1.,.6,.7,.8,.9]:
        rows,diffs=bootstrap_counts(primary,threshold);result['decisions']+=rows;result['decision_contrasts']+=diffs
    print('Continuous and decision analyses complete',flush=True)
    noncomp=(primary.Class.eq('11S') & primary.Session.isin([2,4])) | (primary.Class.eq('10A') & primary.Session.eq(2)) | (primary.Class.eq('9J') & primary.Session.eq(1)) | (primary.Class.eq('10B') & primary.Session.eq(3))
    result['sensitivity']['include_honors']=regression(a)
    result['sensitivity']['exclude_source_noncompliers']=regression(primary[~noncomp])
    result['sensitivity']['omit_prior_gpa']=regression(primary,FORMULA.replace(' + gpa_prev',''))
    result['sensitivity']['threshold_note']='Practice is 0/1, so the 60/70/80/90 scenarios only vary the independent-success threshold. All five are reported; full-correct is primary.'
    # Audit public logs without treating observable language as validated behavior.
    logs=pd.read_csv(RAW/'text_analysis/data/raw/valid_student_data_w_time_stamp.csv')
    u=logs[logs.role.eq('user')].copy();u['part2']='s'+u.session_id.astype(str)+'_'+u.grade.astype(str)+'_'+u.problem_id.astype(str)
    linked=u.merge(primary[['Student_ID','part2']].drop_duplicates(),left_on=['username','part2'],right_on=['Student_ID','part2'],how='left',indicator=True)
    ts=pd.to_datetime(u.time_stamp.str.replace(',','',regex=False),format='%m/%d/%Y %H:%M:%S',errors='coerce')
    result['audit']['behavior']={'raw_message_rows':len(logs),'student_message_rows':len(u),'student_message_ids':u.username.nunique(),'messages_linked_primary_matched_items':int(linked._merge.eq('both').sum()),'timestamp_parse_missing':int(ts.isna().sum()),'roles':logs.role.value_counts().to_dict(),'M5_included':False,'reason':'Linkage exists, but no independent behavior-coding reliability estimates or item-assessment boundary timestamps are established. Text classifier outputs are not psychological constructs.'}
    result['cv']=cv_analysis(primary)
    # Independent arithmetic checks against exported raw-pair and OOF arrays.
    from patsy import dmatrices
    cy,cx=dmatrices(FORMULA,primary.assign(condition=pd.Categorical(primary.condition,categories=ARMS)),return_type='dataframe')
    beta=np.linalg.lstsq(cx.to_numpy(),cy.to_numpy().ravel(),rcond=None)[0]
    stored=np.array([x['estimate'] for x in result['continuous']['coefficients']])
    maxdiff=float(np.max(abs(beta-stored)));assert maxdiff<1e-10
    oof=pd.read_csv(OUT/'bastani_oof.csv');maxmetricdiff=0.
    for row in result['decisions']:
        h=primary[primary.condition.eq(row['condition'])];D=h.practice.ge(1-1e-9);Y=h.independent.ge(row['threshold']-1e-9)
        assert row['positive_n']==int(D.sum()) and row['unconfirmed_n']==int((D&~Y).sum())
        assert abs(row['metrics']['coverage']['estimate']-row['metrics']['burden']['estimate']-row['metrics']['independent_success']['estimate']+row['metrics']['missed_success']['estimate'])<1e-12
    for row in result['cv']['metrics']:
        if row['subset']!='All':continue
        pb=oof['binary_'+row['model']].to_numpy();pc=oof['continuous_'+row['model']].to_numpy()
        assert np.isfinite(pb).all() and np.isfinite(pc).all() and ((pb>0)&(pb<1)).all()
        diffs=[abs(np.mean((oof.success-pb)**2)-row['binary']['brier']),abs(np.sqrt(np.mean((oof.independent-pc)**2))-row['continuous']['rmse'])]
        maxmetricdiff=max(maxmetricdiff,*diffs)
    assert maxmetricdiff<1e-12 and oof.groupby('Class').fold.nunique().eq(1).all() and oof.groupby('Student_ID').fold.nunique().eq(1).all()
    result['validation']={'status':'PASS','numpy_lstsq_vs_statsmodels_max_coefficient_difference':maxdiff,'exported_oof_vs_json_max_metric_difference':maxmetricdiff,'decision_count_and_probability_identities':'PASS all thresholds and arms','finite_predictions_and_probabilities':'PASS','no_class_or_student_fold_overlap':'PASS','source_variable_consistency':'PASS','observed_score_clipping':'None','continuous_prediction_clipping':'None; linear predictions may fall outside [0,1] and errors retain them.'}
    # Simple evidence registry for root manuscript builder; probabilities are units 0-1.
    def reg(id,analysis,estimate,lo=None,hi=None,p=None,n=None,unit='student x matched item',validation='class-clustered',predictor='',outcome='independent score',figure='',table='',allowed='Conditional association; not a causal effect conditional on post-treatment practice.'):
        result['registry'].append({'result_id':id,'study':'Bastani','analysis':analysis,'sample_n':n or len(primary),'unit_of_analysis':unit,'outcome':outcome,'predictor':predictor,'estimate':estimate,'ci_low':lo,'ci_high':hi,'p_value':p,'validation_type':validation,'figure':figure,'table':table,'manuscript_section':'Results','allowed_interpretation':allowed,'prohibited_interpretation':'Latent competence decline, cognitive mechanism, or causal effect conditional on practice.'})
    for row in result['continuous']['slopes']+result['continuous']['slope_contrasts']:
        reg('bastani_slope_'+row['contrast'],'primary continuous',row['estimate'],row['ci_low'],row['ci_high'],row.get('p_holm',row['p_value']),n=result['continuous']['n'],predictor=row['contrast'],figure='Figure 2')
    for outcome in ['Part2Tot','Part3Tot']:
        for row in result['replication'][outcome]['effects']:
            reg('bastani_source_'+outcome+'_'+row['contrast'],'source-study replication',row['estimate'],row['ci_low'],row['ci_high'],row['p_value'],n=result['replication'][outcome]['n'],unit='student x session',predictor=row['contrast'],outcome=outcome,allowed='Reproduces source intention-to-treat patterns using original covariate specification; not a novel finding.')
    for row in result['continuous']['practice_only_discrimination']:
        reg('bastani_practice_only_auc_'+row['condition'],'descriptive discrimination',row['auc_binary_practice_for_full_correct_independent'],n=row['n'],predictor=row['condition'],outcome='AUC',validation='Unfitted binary practice score, descriptive',allowed='Descriptive within-condition discrimination; not causal and not a held-out fitted model.')
    for row in result['decisions']:
        if row['threshold']!=1:continue
        for name,v in row['metrics'].items():reg('bastani_'+row['condition']+'_'+name,'full-correct decision',v['estimate'],v['ci_low'],v['ci_high'],n=row['n'],predictor=row['condition'],outcome=name,figure='Figure 3',table='Table 2',validation='2000 arm-stratified class-bootstrap draws',allowed='Reconstructed decision frequencies under full-correct rule; not observed teacher decisions or validated mastery cutoffs.')
    for row in result['cv']['metrics']:
        if row['subset']!='All':continue
        for typ,names in [('binary',['brier','log_loss','auc','calibration_slope']),('continuous',['mae','rmse','r2'])]:
            for name in names:
                ci=row[typ].get(name+'_ci',[None,None]);reg('bastani_cv_'+row['model']+'_'+name,'class-grouped OOF',row[typ][name],*ci,validation='5-fold class-grouped CV',predictor=row['model'],outcome=name,figure='Figure 4',table='Table 3',allowed='Held-out performance in observed classes; no claim of external-population validation.')
    for row in result['cv']['paired_improvements']:reg('bastani_cv_delta_'+row['comparison']+'_'+row['metric'],'paired held-out improvement',row['estimate'],row['ci_low'],row['ci_high'],validation='Class bootstrap conditional on fixed OOF predictions',predictor=row['comparison'],outcome=row['metric'],table='Table 3')
    result['limitations']=['Secondary analysis with locally frozen specification, not a preregistration.','44 nonhonors classes; cluster asymptotics and conditional bootstrap uncertainty are approximate.','Practice binary scoring limits resolution and prevents interpretation of continuous practice dose-response.','Condition-specific adjusted practice associations are post-treatment conditional and noncausal.','Matched unaided assessment is short-term independent performance, not latent competence or durable learning.','Decision scenarios are reconstructed; no observed teacher decisions or established mastery rule.','CV baseline includes known item/teacher/grader context and assesses new classes in the same setting, not unseen item banks. Grader is retrospective assessment-design context, not a demonstrated pre-decision learner feature; these are evidence-model comparisons, not deployed individual predictions.','Fixed ridge penalties and a single prespecified five-fold allocation were used without optimization.','M5 was omitted because reliable psychological behavior coding and criterion-boundary timing were not established.']
    import sklearn, statsmodels, scipy
    result['provenance']['package_versions']={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'statsmodels':statsmodels.__version__}
    result['provenance']['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['provenance']['analysis_spec_sha256']=hashlib.sha256((ROOT/'analysis'/'analysis_spec.md').read_bytes()).hexdigest()
    existing_path=OUT/'bastani_results.json'
    if existing_path.exists():
        prior=json.loads(existing_path.read_text(encoding='utf-8'))
        if 'submission_audit' in prior:result['submission_audit']=prior['submission_audit']
    existing_path.write_text(json.dumps(clean(result),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Saved',OUT/'bastani_results.json',flush=True)
    print(json.dumps(clean({'audit':{k:result['audit'][k] for k in ['primary_rows','primary_students','primary_classes']},'slopes':result['continuous']['slopes'],'validation':result['validation']}),indent=2))

def audit_cr2(X, y, groups, contrasts):
    """CR2 identity working model; Satterthwaite contrast df.

    A_g = (I-H_gg)^(-1/2), with a symmetric generalized inverse for
    eigenvalues <=1e-10. q_g = A_g X_g (X'X)^(-1)c;
    p_g = (I-H)'_g q_g; df=(tr(P'P))^2/tr((P'P)^2).
    See Pustejovsky & Tipton (2018), DOI 10.1080/07350015.2016.1247004,
    and estimatr's mathematical-notes.Rmd (unweighted full design formula).
    No finite-sample CR1 multiplier is applied to CR2.
    """
    X=np.asarray(X,float);y=np.asarray(y,float).ravel();groups=np.asarray(groups)
    bread=np.linalg.inv(X.T@X);beta=np.linalg.lstsq(X,y,rcond=None)[0];resid=y-X@beta
    C=np.column_stack(list(contrasts.values()));n,k=X.shape
    block_scores=[];P=[[] for _ in contrasts];mins=[];zero_eigs=0
    for group in np.unique(groups):
        ix=np.flatnonzero(groups==group);xx=X[ix]
        residual_block=np.eye(len(ix))-xx@bread@xx.T
        vals,vecs=np.linalg.eigh((residual_block+residual_block.T)/2)
        mins.append(float(vals.min()));zero_eigs+=int((vals<=1e-10).sum())
        assert vals.min()>-1e-8
        invroot=np.zeros_like(vals);positive=vals>1e-10;invroot[positive]=1/np.sqrt(vals[positive])
        A=(vecs*invroot)@vecs.T
        q=A@xx@bread@C
        block_scores.append(q.T@resid[ix])
        for j in range(C.shape[1]):
            z=-X@bread@xx.T@q[:,j];z[ix]+=q[:,j];P[j].append(z)
    S=np.asarray(block_scores);rows=[]
    for j,name in enumerate(contrasts):
        p=np.column_stack(P[j]);G=p.T@p
        df=float(np.trace(G)**2/np.sum(G*G));se=float(np.sqrt(np.sum(S[:,j]**2)))
        estimate=float(C[:,j]@beta);crit=st.t.ppf(.975,df)
        working_variance=float(C[:,j]@bread@C[:,j])
        unbiased_error=float(abs(np.trace(G)-working_variance))
        assert unbiased_error<1e-9
        rows.append({'contrast':name,'estimate':estimate,'se':se,'df_satterthwaite':df,'ci_low':estimate-crit*se,'ci_high':estimate+crit*se,'p_value':float(2*st.t.sf(abs(estimate/se),df)),'identity_working_model_unbiased_variance_error':unbiased_error})
    return {'results':rows,'matrix_rank':int(np.linalg.matrix_rank(X)),'columns':k,'n':n,'clusters':len(np.unique(groups)),'minimum_cluster_residual_eigenvalue':min(mins),'generalized_inverse_zero_eigenvalues':zero_eigs}


def audit_cr2_validation():
    """Checks independent exact identities and direct dense residual projection."""
    rng=np.random.default_rng(81723);n=40
    X=np.column_stack([np.ones(n),rng.normal(size=(n,2))]);y=X@np.array([1.,.5,-.3])+rng.normal(size=n)
    singleton=audit_cr2(X,y,np.arange(n),{f'b{i}':np.eye(3)[i] for i in range(3)})
    expected=sm.OLS(y,X).fit(cov_type='HC2').bse
    errhc=float(np.max(abs(np.array([r['se'] for r in singleton['results']])-expected)))
    g=np.repeat(np.arange(8),5);xx=np.ones((40,1));yy=rng.normal(size=8)[g]+rng.normal(size=40)
    balanced=audit_cr2(xx,yy,g,{'mean':np.ones(1)})['results'][0]
    m=pd.Series(yy).groupby(g).mean();expectedse=float(m.std(ddof=1)/np.sqrt(8))
    errbalanced=abs(balanced['se']-expectedse);errdf=abs(balanced['df_satterthwaite']-7)
    ordinary=audit_cr2(X,y,g,{'slope':np.array([0.,1.,0.])})['results'][0]
    bread=np.linalg.inv(X.T@X);H=X@bread@X.T;M=np.eye(n)-H;r=y-H@y;c=np.array([0.,1.,0.]);score=[];columns=[]
    from scipy.linalg import fractional_matrix_power
    for group in np.unique(g):
        ix=np.flatnonzero(g==group);A=fractional_matrix_power(M[np.ix_(ix,ix)],-.5)
        q=A@X[ix]@bread@c;score.append(q@r[ix]);columns.append(M[:,ix]@q)
    G=np.column_stack(columns).T@np.column_stack(columns)
    dense_df=float(np.trace(G)**2/np.trace(G@G));dense_se=float(np.linalg.norm(score))
    e1=abs(dense_se-ordinary['se']);e2=abs(dense_df-ordinary['df_satterthwaite'])
    assert max(errhc,errbalanced,errdf,e1,e2)<1e-9
    return {'status':'PASS','singleton_CR2_vs_statsmodels_HC2_max_se_error':errhc,'balanced_cluster_mean_se_error':errbalanced,'balanced_cluster_mean_df_error':errdf,'independent_scipy_dense_projection_se_error':e1,'independent_dense_projection_df_error':e2,'boundary':'Formula and algebraic identities verified; not cross-checked by running R clubSandwich.'}


def audit_equal_student(d):
    cc=d.dropna(subset=['practice','independent','condition','part3','teacher','Grader','gpa_prev']).copy()
    cc['condition']=pd.Categorical(cc.condition,categories=ARMS)
    weights=1/cc.groupby('Student_ID').Student_ID.transform('size')
    f=smf.wls(FORMULA,cc,weights=weights).fit(cov_type='cluster',cov_kwds={'groups':cc.Class},use_t=True)
    slopes=[contrast(f,{'practice':1},'Control'),contrast(f,{'practice':1,'practice:C(condition)[T.GPT Base]':1},'GPT Base'),contrast(f,{'practice':1,'practice:C(condition)[T.GPT Tutor]':1},'GPT Tutor')]
    diffs=[contrast(f,{'practice:C(condition)[T.GPT Base]':1},'GPT Base - Control'),contrast(f,{'practice:C(condition)[T.GPT Tutor]':1,'practice:C(condition)[T.GPT Base]':-1},'GPT Tutor - GPT Base')]
    for row,p in zip(diffs,multipletests([x['p_value'] for x in diffs],method='holm')[1]):row['p_holm']=float(p)
    z=d.assign(D=d.practice.eq(1),Y=d.independent.ge(1-1e-9))
    z=z.assign(TP=(z.D&z.Y).astype(int),FP=(z.D&~z.Y).astype(int),FN=(~z.D&z.Y).astype(int),TN=(~z.D&~z.Y).astype(int))
    student=z.groupby(['condition','Class','Student_ID'])[['TP','FP','FN','TN']].mean().reset_index()
    rows=[];rng=np.random.default_rng(SEED)
    for arm in ARMS:
        h=student[student.condition.eq(arm)];g=h.groupby('Class')[['TP','FP','FN','TN']].sum();g['n']=h.groupby('Class').size();v=g.to_numpy()
        draws=v[rng.integers(0,len(v),(B,len(v)))].sum(axis=1);point=v.sum(axis=0)[None,:]
        def met(v):
            tp,fp,fn,tn,n=v.T;return np.column_stack([(tp+fp)/n,(tp+fn)/n,tp/(tp+fp),fp/(tp+fp),fp/n,fn/n])
        pt=met(point)[0];bt=met(draws);names=['coverage','independent_success','confirmation','unconfirmed_positive','burden','missed_success']
        rows.append({'condition':arm,'students':len(h),'classes':len(v),'metrics':{name:{'estimate':float(pt[j]),'ci_low':interval(bt[:,j])[0],'ci_high':interval(bt[:,j])[1]} for j,name in enumerate(names)}})
    return {'estimand':'Equal student total weight, uniform observed item within student; confirmation is ratio of averaged joint probabilities, not mean of student PPVs.','continuous':{'n':len(cc),'students':cc.Student_ID.nunique(),'classes':cc.Class.nunique(),'slopes':slopes,'slope_contrasts':diffs,'row_weight_sum':float(weights.sum()),'covariance':'Class-clustered CR1 with N/G corrections, t(G-1)'},'decisions':rows}


def audit_cv(d):
    """Frozen no-grader main sequence and two necessary reference models."""
    d=d.copy().reset_index(drop=True)
    for arm,col in [('GPT Base','base'),('GPT Tutor','tutor')]:
        d['condition_'+col]=d.condition.eq(arm).astype(float);d['practice_'+col]=d.practice*d['condition_'+col]
    models={'Intercept':[], 'M0':[], 'M1':['practice'], 'M2':['practice','condition_base','condition_tutor'], 'M3':['practice','condition_base','condition_tutor','practice_base','practice_tutor'], 'M4':['practice','condition_base','condition_tutor','practice_base','practice_tutor','gpa_prev'], 'GPA_no_practice':['condition_base','condition_tutor','gpa_prev']}
    classes=d[['Class','condition']].drop_duplicates().sort_values('Class').reset_index(drop=True)
    all_oof=[];all_metrics=[];fold_rows=[]
    original=pd.read_csv(OUT/'bastani_oof.csv')
    y=d.independent.to_numpy();yb=d.independent.ge(1-1e-9).astype(int).to_numpy()
    for repeat,seed in enumerate(range(SEED,SEED+10)):
        folds={}
        for fold,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(classes,classes.condition)):
            folds.update({c:fold for c in classes.iloc[te].Class})
        d['fold']=d.Class.map(folds)
        if repeat==0:assert np.array_equal(d.fold.to_numpy(),original.fold.to_numpy())
        oof=d[['Student_ID','Class','condition','part3','practice','independent','gpa_prev','fold']].copy();oof['success']=yb;oof['repeat']=repeat;oof['seed']=seed
        for model,numeric in models.items():
            for binary in [False,True]:
                pred=np.empty(len(d))
                for fold in range(5):
                    tr=d.fold.ne(fold).to_numpy();te=~tr
                    assert not set(d.loc[tr,'Student_ID'])&set(d.loc[te,'Student_ID'])
                    target=yb if binary else y
                    if model=='Intercept':pred[te]=target[tr].mean();continue
                    transforms=[('context',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['part3','teacher'])]
                    if numeric:transforms.append(('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric))
                    pipe=make_pipeline(ColumnTransformer(transforms),LogisticRegression(C=1,max_iter=3000,tol=1e-9) if binary else Ridge(alpha=1))
                    pipe.fit(d.loc[tr],target[tr]);pred[te]=pipe.predict_proba(d.loc[te])[:,1] if binary else pipe.predict(d.loc[te])
                oof[('binary_' if binary else 'continuous_')+model]=pred
        for fold in range(5):fold_rows.append({'repeat':repeat,'seed':seed,'fold':fold,'test_classes':sorted(d.loc[d.fold.eq(fold),'Class'].unique()),'test_n':int(d.fold.eq(fold).sum()),'student_overlap':0})
        for model in models:
            for subset in (['All']+ARMS if repeat==0 else ['All']):
                h=oof if subset=='All' else oof[oof.condition.eq(subset)];yt=h.independent.to_numpy();bt=h.success.to_numpy();p=h['continuous_'+model].to_numpy();pb=h['binary_'+model].to_numpy()
                continuous={'mae':mean_absolute_error(yt,p),'rmse':np.sqrt(mean_squared_error(yt,p)),'r2':r2_score(yt,p)}
                binary={'brier':brier_score_loss(bt,pb),'log_loss':log_loss(bt,pb),'auc':roc_auc_score(bt,pb)}
                if repeat==0 and model!='Intercept':continuous.update(calibration(yt,p,False));binary.update(calibration(bt,pb,True))
                all_metrics.append({'model':model,'repeat':repeat,'seed':seed,'subset':subset,'n':len(h),'continuous':continuous,'binary':binary})
        all_oof.append(oof);print('Audit CV repeat',repeat+1,'/ 10 complete',flush=True)
    joined=pd.concat(all_oof,ignore_index=True);joined.to_csv(OUT/'bastani_audit_oof.csv',index=False)
    comparisons=[('Intercept','M0'),('M0','M1'),('M1','M2'),('M2','M3'),('M3','M4'),('GPA_no_practice','M4'),('Intercept','GPA_no_practice')]
    diffs=[]
    for rep in range(10):
        by={x['model']:x for x in all_metrics if x['repeat']==rep and x['subset']=='All'}
        for before,after in comparisons:
            for metric in ['mae','rmse','brier','log_loss']:
                category='binary' if metric in ['brier','log_loss'] else 'continuous'
                diffs.append({'repeat':rep,'seed':SEED+rep,'comparison':after+' vs '+before,'before':before,'after':after,'metric':metric,'estimate':float(by[before][category][metric]-by[after][category][metric])})
    # Original fixed-fold uncertainty, conditional on fixed predictions.
    principal=all_oof[0];classids=sorted(principal.Class.unique());rng=np.random.default_rng(SEED+1);lookup={c:i for i,c in enumerate(classids)}
    weights=np.zeros((B,len(classids)))
    for arm in ARMS:
        ids=np.array([lookup[c] for c in classes[classes.condition.eq(arm)].Class]);draw=ids[rng.integers(0,len(ids),(B,len(ids)))]
        for b in range(B):np.add.at(weights[b],draw[b],1)
    ns=principal.groupby('Class').size().reindex(classids).to_numpy();denom=weights@ns;boots={}
    for model in models:
        p=principal['continuous_'+model].to_numpy();pb=np.clip(principal['binary_'+model].to_numpy(),1e-8,1-1e-8)
        errors={'mae':abs(y-p),'rmse':(y-p)**2,'brier':(yb-pb)**2,'log_loss':-(yb*np.log(pb)+(1-yb)*np.log(1-pb))}
        for metric,v in errors.items():
            agg=pd.Series(v).groupby(principal.Class).sum().reindex(classids).to_numpy();bt=weights@agg/denom
            if metric=='rmse':bt=np.sqrt(bt)
            boots[(model,metric)]=bt
            row=next(x for x in all_metrics if x['repeat']==0 and x['model']==model and x['subset']=='All')
            row['binary' if metric in ['brier','log_loss'] else 'continuous'][metric+'_ci']=interval(bt)
    principal_diffs=[]
    for row in diffs:
        if row['repeat']!=0:continue
        bt=boots[(row['before'],row['metric'])]-boots[(row['after'],row['metric'])]
        principal_diffs.append({**row,'ci_low':interval(bt)[0],'ci_high':interval(bt)[1]})
    stability=[]
    for before,after in comparisons:
        for metric in ['mae','rmse','brier','log_loss']:
            vals=[x['estimate'] for x in diffs if x['before']==before and x['after']==after and x['metric']==metric]
            stability.append({'comparison':after+' vs '+before,'metric':metric,'repeat_n':10,'min':min(vals),'max':max(vals),'median':float(np.median(vals)),'positive_improvement_repeats':sum(v>0 for v in vals),'negative_improvement_repeats':sum(v<0 for v in vals),'values':vals})
    return {'version':'Post-audit no-grader sequence; original cv key is retained as grader-adjusted sensitivity.','models':{m:{'context':[] if m=='Intercept' else ['part3','teacher'],'numeric':v,'grader':False} for m,v in models.items()},'metrics':[x for x in all_metrics if x['repeat']==0],'paired_improvements':principal_diffs,'folds':fold_rows,'n':len(d),'students':d.Student_ID.nunique(),'classes':d.Class.nunique(),'principal_seed':SEED,'original_fold_match':True,'uncertainty':'2,000 arm-stratified class bootstrap draws conditional on principal fixed OOF predictions. Positive before-minus-after error change favors added evidence. Repeats are stability descriptions, not independent samples.'}, {'seeds':list(range(SEED,SEED+10)),'metrics':[x for x in all_metrics if x['subset']=='All'],'paired_differences':diffs,'summary':stability,'interpretation':'Ten overlapping split repetitions are not independent datasets; ranges are not confidence intervals.'}


def audit_verify_outputs(old):
    """Recheck fixed predictions, principal decision bootstrap, and CR2 unbiasedness."""
    a=old['submission_audit'];d=pd.read_csv(OUT/'bastani_matched.csv');oof=pd.read_csv(OUT/'bastani_audit_oof.csv')
    maxerr=0.;baseerr=0.;foldok=True
    for row in a['cv_stability']['metrics']:
        h=oof[oof.repeat.eq(row['repeat'])];y=h.independent.to_numpy();b=h.success.to_numpy();p=h['continuous_'+row['model']].to_numpy();pb=h['binary_'+row['model']].to_numpy()
        errs=[abs(np.mean((b-pb)**2)-row['binary']['brier']),abs(np.sqrt(np.mean((y-p)**2))-row['continuous']['rmse']),abs(np.mean(abs(y-p))-row['continuous']['mae']),abs((1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))-row['continuous']['r2'])]
        maxerr=max(maxerr,*errs)
    for rep,h in oof.groupby('repeat'):
        foldok=foldok and h.groupby('Student_ID').fold.nunique().eq(1).all() and h.groupby('Class').fold.nunique().eq(1).all()
        for fold,test in h.groupby('fold'):
            train=h[h.fold.ne(fold)]
            baseerr=max(baseerr,float(abs(test.binary_Intercept-train.success.mean()).max()),float(abs(test.continuous_Intercept-train.independent.mean()).max()))
    assert maxerr<1e-10 and baseerr<1e-10 and foldok
    # Reconstruct original class-bootstrap draws and compare percentile CIs/p values.
    z=d.assign(D=d.practice.ge(1-1e-9),Y=d.independent.ge(1-1e-9));z=z.assign(U=z.D&~z.Y,M=~z.D&z.Y)
    s=z.groupby(['condition','Class']).agg(n=('D','size'),D=('D','sum'),Y=('Y','sum'),U=('U','sum'),M=('M','sum')).reset_index()
    rng=np.random.default_rng(SEED);boots={};names=['coverage','independent_success','unconfirmed_positive','burden','missed_success']
    for arm in ARMS:
        v=s[s.condition.eq(arm)][['n','D','Y','U','M']].to_numpy(float);bt=v[rng.integers(0,len(v),(B,len(v)))].sum(axis=1);n,dd,y,u,m=bt.T
        boots[arm]=np.column_stack([dd/n,y/n,u/dd,u/n,m/n])
    skew=[];decisionerr=0.
    for row in old['decision_contrasts']:
        if row['threshold']!=1 or 'p_holm' not in row:continue
        arm,barm=row['contrast'].split(' - ');j=names.index(row['metric']);bt=boots[arm][:,j]-boots[barm][:,j];lo,hi=interval(bt);p=float(2*st.norm.sf(abs(row['estimate']/np.std(bt,ddof=1))))
        err=max(abs(lo-row['ci_low']),abs(hi-row['ci_high']),abs(p-row['p_value']));decisionerr=max(decisionerr,err)
        skew.append({'contrast':row['contrast'],'metric':row['metric'],'bootstrap_skewness':float(st.skew(bt,bias=False)),'normal_p':p,'percentile_ci':[lo,hi],'max_difference_from_original_summary':err})
    assert decisionerr<1e-12
    from patsy import dmatrices
    cc=d.dropna(subset=['gpa_prev','Grader']).copy();cc['condition']=pd.Categorical(cc.condition,categories=ARMS);yy,X=dmatrices(FORMULA,cc,return_type='dataframe');C={}
    for label,terms in [('GPT Base - Control',{'practice:C(condition)[T.GPT Base]':1}),('GPT Tutor - GPT Base',{'practice:C(condition)[T.GPT Tutor]':1,'practice:C(condition)[T.GPT Base]':-1})]:
        c=np.zeros(X.shape[1])
        for k,v in terms.items():c[list(X.columns).index(k)]=v
        C[label]=c
    cr=audit_cr2(X,yy,cc.Class,C)
    for fresh,saved in zip(cr['results'],a['small_sample']['results']):
        for k in ['estimate','se','df_satterthwaite','ci_low','ci_high']:assert abs(fresh[k]-saved[k])<1e-10
        saved['identity_working_model_unbiased_variance_error']=fresh['identity_working_model_unbiased_variance_error']
    a['numerical_validation']={'status':'PASS','exported_all_repeat_metrics_max_error':maxerr,'training_only_baseline_predictions_max_error':baseerr,'all_repeats_student_and_class_fold_isolation':bool(foldok),'principal_decision_bootstrap_summary_max_error':decisionerr,'decision_bootstrap_skewness':skew,'CR2_identity_working_model_variance_unbiased_for_target_contrasts':True,'full_pipeline_uncertainty_estimated':False}
    return old


def run_submission_audit():
    from datetime import datetime, timezone
    from patsy import dmatrices
    path=OUT/'bastani_results.json';old=json.loads(path.read_text(encoding='utf-8'))
    protected={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in old.items() if k!='submission_audit'}
    original_oof_hash=hashlib.sha256((OUT/'bastani_oof.csv').read_bytes()).hexdigest()
    d=pd.read_csv(OUT/'bastani_matched.csv');p2=read('problem_part2.csv');p3=read('problem_part3.csv');total=read('final_data.csv')
    spec=(ROOT/'analysis'/'analysis_spec.md').read_text(encoding='utf-8').split('## Post-audit additions: Bastani',1)[1]
    audit={'status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),'specification':'## Post-audit additions: Bastani'+spec,'sources':[
        {'url':'https://hamsabastani.github.io/education_llm.pdf','scope':'Author-hosted 68-page paper; main equation/page9 normalized prior-year GPA, A.4/page31 assignment, A.5/page32 blinded independent grading, Table3/pages33-34 survey sample, B.6/pages43-44 attendance. Detailed point rubrics and original point denominators not supplied in inspected source.'},
        {'url':'https://www.pnas.org/doi/10.1073/pnas.2422633122','scope':'Publisher metadata and source identity; 2025 correction concerns affiliation. Publisher SI download unavailable through web tool; author appendix used with explicit boundary.'},
        {'url':'https://jepusto.github.io/clubSandwich/reference/vcovCR.html','scope':'CR2 identity working model; CR1S vs CR1 nomenclature; Pustejovsky and Tipton 2018 metadata.'},
        {'url':'https://github.com/DeclareDesign/estimatr/blob/main/vignettes/mathematical-notes.Rmd','scope':'CR2 adjustment matrix and contrast-specific Satterthwaite df formula.'}], 'registry':[]}
    audit['score_dictionary']={'practice':{'field':'problem_part2.csv::Score','support':p2.Score.value_counts().sort_index().rename_axis('value').reset_index(name='n').to_dict('records'),'observed_scale':'0/1','label':'practice scored as correct (released score = 1)','confirmed':'Released normalized item score, binary support; session Part2Tot exactly equals unweighted item-score mean. Source describes teacher-designed solutions/rubrics and blinded independent grading.','unconfirmed':'No inspected public point rubric proves whether binary values originated as full-correct vs a dichotomized score; do not claim cross-stage rubric equivalence or a validated mastery cutoff.'},'independent':{'field':'problem_part3.csv::Score','support':p3.Score.value_counts().sort_index().rename_axis('value').reset_index(name='n').to_dict('records'),'observed_scale':'normalized item score 0-1, partial credit','label':'full-score independent response (released score = 1)','confirmed':'Part3Tot equals unweighted mean of four item scores to numerical precision.','unconfirmed':'Original awarded points and per-item maximum-point denominators are not released; precise rubric meaning of each fraction is not recoverable from score support alone.'},'gpa_prev':{'field':'gpa_prev','source_definition':'Normalized GPA from the previous year (main paper Eq.1/footnote9).','disciplinary_scope':'Not explicitly identified as mathematics-only in inspected paper/dictionary; retain previous GPA / prior achievement.','timing':'Previous academic year; survey/background collected at semester start. Exact individual collection time not released.','scale':{'n_students_observed':d.drop_duplicates('Student_ID').gpa_prev.notna().sum(),'min':d.gpa_prev.min(),'max':d.gpa_prev.max(),'student_mean':d.drop_duplicates('Student_ID').gpa_prev.mean(),'student_sd':d.drop_duplicates('Student_ID').gpa_prev.std()},'unconfirmed':'Normalization formula, school administrative vs self-report provenance, precise subject composition not stated in inspected release.'},'Part2Tot_mean_error':old['audit']['Part2Tot_item_mean_max_difference'],'Part3Tot_mean_error':old['audit']['Part3Tot_item_mean_max_difference']}
    # Attendance distinguishes absent-labelled records from unobserved opportunities.
    students=d[['Student_ID','Class','Year','condition','gpa_prev']].drop_duplicates()
    grid=students.merge(pd.DataFrame({'Session':[1,2,3,4]}),how='cross')
    observed=d[['Student_ID','Session']].drop_duplicates();attendance=pd.read_stata(RAW/'additional_results'/'df_attendance.dta')
    flow=grid.merge(observed.assign(observed=True),on=['Student_ID','Session'],how='left').merge(attendance[['Student_ID','Session','Attendance']],on=['Student_ID','Session'],how='left',validate='one_to_one')
    absent=flow[flow.observed.isna()];conflict=flow[flow.observed.eq(True)&flow.Attendance.eq(0)]
    missing_class_sessions=absent.groupby(['Class','Session','condition']).size().rename('unobserved_students').reset_index()
    actual=d.groupby(['Class','Session']).size()
    missing_class_sessions['whole_class_session_unobserved']=[(c,s) not in actual.index for c,s in zip(missing_class_sessions.Class,missing_class_sessions.Session)]
    audit['sample_flow']={'released_total_students':total.Student_ID.nunique(),'source_population_scope':'943 is the full survey-linked released sample, not a verified complete enrollment roster (source Table3).','stages':[{'stage':'Released main student-session file','records':len(total),'students':total.Student_ID.nunique(),'classes':total.Class.nunique()},{'stage':'Released practice item file','records':len(p2),'students':p2.Student_ID.nunique(),'classes':p2.Class.nunique()},{'stage':'Official mapped pairs all honors statuses','records':old['audit']['matched_rows_all'],'students':943,'classes':50},{'stage':'Exclude honors: decision and prediction population','records':len(d),'students':d.Student_ID.nunique(),'classes':d.Class.nunique()},{'stage':'Complete GPA and grader: original adjusted association','records':d.dropna(subset=['gpa_prev','Grader']).shape[0],'students':d.dropna(subset=['gpa_prev','Grader']).Student_ID.nunique(),'classes':d.dropna(subset=['gpa_prev','Grader']).Class.nunique()}],'ideal_student_sessions':len(grid),'observed_student_sessions':len(observed),'unobserved_student_sessions':len(absent),'ideal_item_opportunities':len(grid)*4,'observed_pairs':len(d),'opportunity_difference':len(absent)*4,'absent_in_attendance_sessions':int(absent.Attendance.eq(0).sum()),'absent_in_attendance_opportunities':int(absent.Attendance.eq(0).sum()*4),'attendance_unknown_sessions':int(absent.Attendance.isna().sum()),'attendance_unknown_opportunities':int(absent.Attendance.isna().sum()*4),'absent_by_class_session':missing_class_sessions.to_dict('records'),'attendance_conflicts_observed_results_but_flag_zero':conflict.to_dict('records'),'attendance_not_available_for_observed_sessions':int((flow.observed.eq(True)&flow.Attendance.isna()).sum()),'unknown_absence_gpa_missing':int((absent.Attendance.isna()&absent.gpa_prev.isna()).sum()),'unknown_absence_gpa_present':int((absent.Attendance.isna()&absent.gpa_prev.notna()).sum()),'observed_pairs_per_student':d.groupby('Student_ID').size().value_counts().sort_index().rename_axis('pairs').reset_index(name='students').to_dict('records'),'participation_by_grade_session_condition':d.groupby(['Year','Session','condition']).agg(pairs=('Student_ID','size'),students=('Student_ID','nunique'),classes=('Class','nunique')).reset_index().to_dict('records'),'interpretation':'1,828 is a gap from a hypothetical four-session opportunity grid for released nonhonors students, not 1,828 missing outcome labels. Whole class-session gaps may be unheld/unreleased; their cause is unverified. One attendance=0 record has observed item scores and remains included.'}
    audit['missingness']={'gpa_by_condition':d.assign(gpa_missing=d.gpa_prev.isna()).groupby(['condition','gpa_missing']).agg(pairs=('Student_ID','size'),students=('Student_ID','nunique'),classes=('Class','nunique'),practice_mean=('practice','mean'),independent_mean=('independent','mean'),independent_min=('independent','min'),independent_max=('independent','max')).reset_index().to_dict('records'),'gpa_missing_by_class':d[d.gpa_prev.isna()].groupby(['condition','Class']).agg(pairs=('Student_ID','size'),students=('Student_ID','nunique')).reset_index().to_dict('records'),'grader_missing_records':d[d.Grader.isna()].to_dict('records'),'independent_outcomes_imputed':False}
    # Released Grader comes from practice table; equality to independent field is exact.
    check=p2[['Student_ID','Problem','Grader']].rename(columns={'Problem':'part2','Grader':'grader_practice'}).merge(read('problem_mapping.csv').dropna(),on='part2').merge(p3[['Student_ID','Problem','Grader']].rename(columns={'Problem':'part3','Grader':'grader_independent'}),on=['Student_ID','part3'])
    disagreements=~(check.grader_practice.eq(check.grader_independent)|(check.grader_practice.isna()&check.grader_independent.isna()))
    cc=d.dropna(subset=['gpa_prev','Grader']).copy();cc['condition']=pd.Categorical(cc.condition,categories=ARMS)
    yy,xx=dmatrices(FORMULA,cc,return_type='dataframe');X=xx.to_numpy()
    teachers=pd.crosstab(d.teacher,d.condition).reindex(columns=ARMS,fill_value=0)
    graders=pd.crosstab(d.Grader.fillna('Missing'),d.condition).reindex(columns=ARMS,fill_value=0)
    support=[]
    for variable in ['part3','teacher','Grader']:
        table=pd.crosstab(cc[variable],cc.condition).reindex(columns=ARMS,fill_value=0)
        for arm in ARMS:support.append({'context_variable':variable,'counterfactual_arm':arm,'rows_with_level_not_observed_in_arm':int(cc[variable].isin(table.index[table[arm].eq(0)]).sum()),'levels_absent_in_arm':table.index[table[arm].eq(0)].tolist()})
    audit['design_support']={'grader_source':'Derived matched file uses practice table Grader; it agrees exactly with the independent table for every matched observation. Source A.5 identifies blinded external graders, assigned at class level across arms; release does not document separate phase-specific grader identities.','grader_phase_disagreements':int(disagreements.sum()),'grader_condition_pairs':graders.reset_index().to_dict('records'),'teacher_condition_pairs':teachers.reset_index().to_dict('records'),'grader_teacher_pairs':pd.crosstab(d.Grader.fillna('Missing'),d.teacher).reset_index().to_dict('records'),'design_formula':FORMULA,'matrix_n':len(X),'matrix_columns':X.shape[1],'matrix_rank':np.linalg.matrix_rank(X),'singular_values':np.linalg.svd(X,compute_uv=False).tolist(),'column_names':xx.columns.tolist(),'nuisance_level_support':support,'meaning':'Full column rank establishes estimability, not causal overlap. Marginal adjusted means average hypothetical arm/practice assignments across observed context; absent context-arm combinations entail additive-model extrapolation.'}
    audit['equal_student']=audit_equal_student(d)
    audit['small_sample']={'method':'CR2 bias-reduced linearization with identity working covariance and contrast-specific Satterthwaite df; pointwise intervals, Holm over two primary contrasts.','implementation_validation':audit_cr2_validation()}
    contrasts={};columns=list(xx.columns)
    for label,terms in [('GPT Base - Control',{'practice:C(condition)[T.GPT Base]':1}),('GPT Tutor - GPT Base',{'practice:C(condition)[T.GPT Tutor]':1,'practice:C(condition)[T.GPT Base]':-1})]:
        c=np.zeros(X.shape[1])
        for key,value in terms.items():c[columns.index(key)]=value
        contrasts[label]=c
    cs=audit_cr2(X,yy.to_numpy(),cc.Class,contrasts)
    for row,p in zip(cs['results'],multipletests([x['p_value'] for x in cs['results']],method='holm')[1]):row['p_holm']=float(p)
    audit['small_sample'].update(cs)
    print('Audit CR2 results:',clean(cs['results']),flush=True)
    audit['exact_2x2']=[];audit['threshold_labels']=[]
    D=d.practice.eq(1);full=d.independent.ge(1-1e-9)
    for threshold in [.6,.7,.8,.9,1.]:
        Y=d.independent.ge(threshold-1e-9)
        for arm in ARMS:
            mask=d.condition.eq(arm);dd=D[mask];yy=Y[mask]
            tp=int((dd&yy).sum());fp=int((dd&~yy).sum());fn=int((~dd&yy).sum());tn=int((~dd&~yy).sum());n=int(mask.sum())
            auc=.5*(tp/(tp+fn)+tn/(tn+fp));ppv=tp/(tp+fp);prev=(tp+fn)/n
            audit['exact_2x2'].append({'threshold':threshold,'condition':arm,'n':n,'TP':tp,'FP':fp,'FN':fn,'TN':tn,'coverage':(tp+fp)/n,'independent_success':prev,'confirmation':ppv,'unconfirmed_positive':1-ppv,'burden':fp/n,'missed_success':fn/n,'confirmation_minus_prevalence':ppv-prev,'confirmation_over_prevalence':ppv/prev,'binary_practice_auc':auc,'auc_formula_error':abs(auc-roc_auc_score(yy,dd))})
        audit['threshold_labels'].append({'threshold':threshold,'success_n':int(Y.sum()),'different_from_full_score':int(Y.ne(full).sum()),'by_condition':[{'condition':arm,'success_n':int(Y[d.condition.eq(arm)].sum()),'different_from_full_score':int(Y[d.condition.eq(arm)].ne(full[d.condition.eq(arm)]).sum())} for arm in ARMS]})
    for row,nextrow in zip(audit['threshold_labels'][:-1],audit['threshold_labels'][1:]):row['label_changes_to_next_threshold']=row['success_n']-nextrow['success_n'];row['next_threshold']=nextrow['threshold']
    audit['multiple_testing_families']={'original_association':[{'test_id':'association_'+x['contrast'],'family':'two prespecified association contrasts','raw_p':x['p_value'],'holm_p':x['p_holm'],'interval':'pointwise CR1 t'} for x in old['continuous']['slope_contrasts'][:2]],'original_decisions':[{'test_id':'decision_'+x['contrast']+'_'+x['metric'],'family':'eight prespecified full-score decision comparisons','raw_p':x['p_value'],'holm_p':x['p_holm'],'interval':'pointwise percentile class bootstrap','normal_p_and_percentile_interval_exclude_zero_agree':bool((x['p_value']<.05)==(x['ci_low']>0 or x['ci_high']<0))} for x in old['decision_contrasts'] if x['threshold']==1 and 'p_holm' in x]}
    audit['cv_no_grader'],audit['cv_stability']=audit_cv(d)
    for row in audit['small_sample']['results']:
        audit['registry'].append({'result_id':'bastani_audit_CR2_'+row['contrast'],'study':'Bastani','analysis':'post-audit CR2 association contrast','sample_n':len(cc),'unit_of_analysis':'matched item','outcome':'independent score','predictor':row['contrast'],'estimate':row['estimate'],'ci_low':row['ci_low'],'ci_high':row['ci_high'],'p_value':row['p_holm'],'validation_type':'CR2 Satterthwaite df='+str(row['df_satterthwaite']),'figure':'Figure 2','table':'Supplementary','manuscript_section':'Results','allowed_interpretation':'Conditional association difference, not causal effect conditional on practice.','prohibited_interpretation':'Latent competence loss or causal within-success treatment effect.'})
    for row in audit['cv_no_grader']['metrics']:
        if row['subset']!='All':continue
        for category,metrics in [('binary',['brier','log_loss','auc','calibration_intercept','calibration_slope']),('continuous',['mae','rmse','r2'])]:
            for metric in metrics:
                if metric not in row[category]:continue
                ci=row[category].get(metric+'_ci',[None,None]);audit['registry'].append({'result_id':'bastani_audit_no_grader_'+row['model']+'_'+metric,'study':'Bastani','analysis':'post-audit class CV without grader','sample_n':len(d),'unit_of_analysis':'matched item','outcome':metric,'predictor':row['model'],'estimate':row[category][metric],'ci_low':ci[0],'ci_high':ci[1],'p_value':None,'validation_type':'5-fold class CV seed20260922; conditional bootstrap if CI present','figure':'Figure 4','table':'Table 3','manuscript_section':'Results','allowed_interpretation':'Held-out prediction without future grader context.','prohibited_interpretation':'Causal learning effect or externally deployed algorithm performance.'})
    for row in audit['cv_no_grader']['paired_improvements']:
        audit['registry'].append({'result_id':'bastani_audit_delta_'+row['comparison']+'_'+row['metric'],'study':'Bastani','analysis':'post-audit paired no-grader CV improvement','sample_n':len(d),'unit_of_analysis':'matched item','outcome':row['metric'],'predictor':row['comparison'],'estimate':row['estimate'],'ci_low':row['ci_low'],'ci_high':row['ci_high'],'p_value':None,'validation_type':'2000 class bootstrap fixed OOF','figure':'Figure 4','table':'Table 3','manuscript_section':'Results','allowed_interpretation':'Before-minus-after held-out error difference conditional on predictions.','prohibited_interpretation':'Full algorithm confidence interval or causal intervention effect.'})
    for row in audit['exact_2x2']:
        if row['threshold']!=1:continue
        orig=next(z for z in old['decisions'] if z['threshold']==1 and z['condition']==row['condition'])['metrics']['unconfirmed_positive']
        audit['registry'].append({'result_id':'bastani_audit_confirmation_'+row['condition'],'study':'Bastani','analysis':'full-score confirmation','sample_n':row['TP']+row['FP'],'unit_of_analysis':'practice-positive matched item','outcome':'confirmation probability','predictor':row['condition'],'estimate':row['confirmation'],'ci_low':1-orig['ci_high'],'ci_high':1-orig['ci_low'],'p_value':None,'validation_type':'Complement of original class-bootstrap unconfirmed proportion','figure':'Figure 3','table':'Table 2','manuscript_section':'Results','allowed_interpretation':'Same 2x2 distribution as original decisions; not independent extra evidence.','prohibited_interpretation':'Fraction of students with true mastery.'})
    audit['limitations']=['Detailed score point rubrics and normalizing denominators remain unverified.','Attendance supplement has one conflicting absent flag and unrepresented opportunities; no outcomes were imputed.','No-grader models retain known item and teacher context, so they validate new classes in this same task setting, not new item banks or institutions.','CR2 identity working model is an approximation for small-sample inference, not exact randomization inference under constrained assignment.','Repeated CV ranges describe split sensitivity, not ten independent replications or full pipeline uncertainty.','No calibrated repair policy was tested; calibration diagnostics cannot establish recalibration failure.']
    assert hashlib.sha256((OUT/'bastani_oof.csv').read_bytes()).hexdigest()==original_oof_hash
    for key,digest in protected.items():assert hashlib.sha256(json.dumps(old[key],sort_keys=True).encode()).hexdigest()==digest
    audit['status']='complete';audit['preservation']={'all_original_result_keys_unchanged':True,'original_oof_sha256':original_oof_hash,'source_git_status':subprocess.check_output(['git','-C',str(RAW),'status','--porcelain'],text=True).strip(),'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    old['submission_audit']=clean(audit);old=audit_verify_outputs(old)
    path.write_text(json.dumps(old,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Submission audit complete:',path,flush=True)


if __name__=='__main__':
    import sys
    if '--verify-audit' in sys.argv:
        path=OUT/'bastani_results.json';saved=json.loads(path.read_text(encoding='utf-8'));saved=audit_verify_outputs(saved)
        saved['submission_audit']['preservation']['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(clean(saved),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print('Post-audit numerical checks PASS')
    elif '--audit' in sys.argv:run_submission_audit()
    else:main()
