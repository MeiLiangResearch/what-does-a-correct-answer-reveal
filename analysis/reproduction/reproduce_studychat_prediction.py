"""Reproduce only the StudyChat prediction analyses reported in Supplement S3.

Algorithms are retained from analysis/studychat.py; legacy unreported dialogue
models are not run. Source score files remain local and are never redistributed.
"""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,required=True,help='StudyChat directory containing scores/')
parser.add_argument('--outdir',type=Path,required=True)
args=parser.parse_args()
RAW=args.data_dir.resolve(); OUT=args.outdir.resolve(); OUT.mkdir(parents=True,exist_ok=True)
def save(df,name): df.to_csv(OUT/('studychat_'+name+'.csv'),index=False)
grades=pd.concat([pd.read_csv(RAW/'scores'/f'{s}_grades_released_normalized.csv').assign(semester=s) for s in ['f24','s25']],ignore_index=True)
grades['student']=grades.userId
cross_semester_ids=len(set(grades.loc[grades.semester=='f24','student']) & set(grades.loc[grades.semester=='s25','student']))
initial_hashes={s:hashlib.sha256((RAW/'scores'/f'{s}_grades_released_normalized.csv').read_bytes()).hexdigest() for s in ['f24','s25']}

EXTENSION_PATH = OUT / 'studychat_extension.json'
WINDOWS = {'e2': ['a3', 'a4'], 'e3': ['a3', 'a4', 'a5', 'a6', 'a7']}
POINTS = {'f24': dict(zip([f'a{i}' for i in range(1,8)], [30,100,55,61,65,113,101])),
          's25': dict(zip([f'a{i}' for i in range(1,8)], [31,100,55,90,105,113,135]))}
SEED = 20260922
BOOTSTRAPS = 2000
specification = {
    'status': 'fixed before extension outcome fitting; not a preregistration',
    'windows': WINDOWS,
    'window_rationale': 'Mean of available post-e1 assignments; a1/a2 already precede e1. All-prior-assignment sensitivity adds a1/a2.',
    'outcomes': ['e2', 'e3'],
    'unit': 'one student-examination record; two records per student in primary analysis',
    'models': {
        'baseline': 'exam_pct = intercept',
        'assignment': 'exam_pct = intercept + b1 * earlier_assignment_mean_pct',
        'assignment_e1': 'exam_pct = intercept + b1 * earlier_assignment_mean_pct + b2 * e1_pct'},
    'estimation': 'Ordinary least squares separately in each semester-by-target-exam cell; raw percentage-point units; no tuning, imputation, scaling or clipping predictions.',
    'within_semester_cv': 'Five folds from seeded permutation of sorted student IDs within semester; identical fold assignment for e2 and e3; fit exclusively on four training folds.',
    'temporal_validation': 'Fit each target-exam model on all Fall students; apply unchanged to Spring; no Spring-label recalibration or model selection.',
    'metrics': {
        'mae': 'mean(abs(y - prediction))',
        'rmse': 'sqrt(mean((y - prediction)^2))',
        'r2': '1 - sum((y - prediction)^2) / sum((y - mean(y within evaluation semester-by-exam cell))^2); same as standard test-set R2 within a cell',
        'r2_vs_training_mean': '1 - sum((y - prediction)^2) / sum((y - prediction_from_training_intercept_only_model)^2)',
        'calibration': 'OLS observed percentage score on held-out predicted percentage score: intercept 0 and slope 1 denote ideal calibration. Descriptive only, not used to recalibrate predictions. Intercept-only baseline calibration is omitted because fold means create artificial between-fold slopes. Interpret semester-by-exam calibration rather than pooled means across different exams.',
        'curve': 'Up to five equal-frequency bins of held-out predictions within semester-by-exam; bin means of predictions and outcomes. No calibration bins are used for model fitting.'},
    'uncertainty': '2,000 paired bootstrap resamples of students within evaluation semester, preserving both examination records and all model predictions; percentile 95% intervals for model B minus model A metrics. Conditional on fitted held-out predictions and this fixed fold split; does not include training/fold variability. No confirmatory p-values or multiple-testing claims.',
    'comparisons': [['baseline','assignment'], ['assignment','assignment_e1']],
    'primary_handling': 'Retain all released zero and extra-credit scores; complete released grades require no imputation. Consent selection and meaning of zero scores remain unknown.',
    'sensitivities': {
        'cap_extra_credit': 'Explicitly cap e2/e3 target scores at 100 in a separate analysis; never alter raw files or primary outcomes.',
        'exclude_zero': 'Exclude a target record if e1, any assignment used in its predictor, or its target exam equals zero; apply same sample to all models.',
        'points_weighted': 'Weight normalized assignment scores by published assignment maximum points, separately by semester.',
        'all_prior_assignments': 'Use a1-a4 for e2 and a1-a7 for e3; all are due before the target examination.'},
    'ceiling_audit': 'Report score mass at/above 100 and primary held-out errors stratified by earlier assignment mean >=90 versus <90 (fixed descriptive threshold, not a mastery cutpoint). Original below-full-mark selected slope sensitivity remains in studychat_sensitivity.csv.',
    'behavior_model': 'Not fit: task-tagged logs cross course phases, exact exam cutoffs are unavailable, and label spot checks provide no independent human agreement. Original exploratory concurrent behavior results remain reproducible but are not predictive evidence.',
    'prohibited': ['No a5-a7 predictor for e2', 'No e1-containing no_llm aggregate outcome', 'No future examinations in predictors', 'No verified-AI-use, causal, latent-competence or deployable-grade-availability claim']}
extension = {'analysis_date': datetime.now(timezone.utc).isoformat(),
             'specification': specification, 'status': 'specification frozen; extension not yet fitted'}
# Persist the declared design before the first extension prediction is fitted.
EXTENSION_PATH.write_text(json.dumps(extension, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

score_readme = RAW / 'scores' / 'README.md'
assert '(a1 → a2 → e1 → a3 → a4 → e2 → a5 → a6 → a7 → e3)' in score_readme.read_text(encoding='utf-8')
assert cross_semester_ids == 0, 'Temporal validation would require excluding repeated students.'
fold_map = {}
fold_rng = np.random.default_rng(SEED)
for sem, g in grades.groupby('semester', sort=True):
    shuffled = fold_rng.permutation(sorted(g.student.tolist()))
    fold_map.update({sid: int(i % 5) for i, sid in enumerate(shuffled)})

def prediction_frame(scenario):
    rows = []
    for _, row in grades.iterrows():
        for exam, focal in WINDOWS.items():
            cols = ([f'a{i}' for i in range(1, int(focal[-1][1:])+1)]
                    if scenario == 'all_prior_assignments' else focal)
            scores = row[cols].to_numpy(dtype=float) * 100
            e1, y = float(row.e1*100), float(row[exam]*100)
            if scenario == 'exclude_zero' and (e1 == 0 or y == 0 or np.any(scores == 0)):
                continue
            weights = [POINTS[row.semester][a] for a in cols] if scenario == 'points_weighted' else None
            assignment = float(np.average(scores, weights=weights))
            rows.append(dict(semester=row.semester, student=row.student, exam=exam,
                             fold=fold_map[row.student], assignment_pct=assignment,
                             e1_pct=e1, observed_pct=min(y,100) if scenario=='cap_extra_credit' else y,
                             scenario=scenario))
    return pd.DataFrame(rows)

def design(frame, model):
    columns = [np.ones(len(frame))]
    if model != 'baseline': columns.append(frame.assignment_pct.to_numpy())
    if model == 'assignment_e1': columns.append(frame.e1_pct.to_numpy())
    return np.column_stack(columns)

coefficient_audit = []
split_audit = []
def predict_split(train, test, validation, scenario, fold):
    overlap = set(train.student) & set(test.student)
    assert not overlap
    pred = test.copy()
    pred['validation'] = validation
    pred['fold'] = fold
    split_audit.append(dict(scenario=scenario, validation=validation,
                            semester=str(test.semester.iloc[0]), exam=str(test.exam.iloc[0]), fold=fold,
                            train_n=len(train), test_n=len(test), overlapping_students=len(overlap),
                            training_assignment_min=float(train.assignment_pct.min()),
                            training_assignment_max=float(train.assignment_pct.max()),
                            test_assignment_below_training_range=int((test.assignment_pct<train.assignment_pct.min()).sum()),
                            test_assignment_above_training_range=int((test.assignment_pct>train.assignment_pct.max()).sum()),
                            training_e1_min=float(train.e1_pct.min()),training_e1_max=float(train.e1_pct.max()),
                            test_e1_outside_training_range=int(((test.e1_pct<train.e1_pct.min())|(test.e1_pct>train.e1_pct.max())).sum())))
    for model in specification['models']:
        x = design(train,model)
        coef, _, rank, _ = np.linalg.lstsq(x,train.observed_pct.to_numpy(),rcond=None)
        assert rank == x.shape[1]
        pred[model] = design(test,model) @ coef
        coefficient_audit.append(dict(scenario=scenario,validation=validation,
                                      semester=str(test.semester.iloc[0]),exam=str(test.exam.iloc[0]),fold=fold,
                                      model=model,intercept=float(coef[0]),
                                      assignment_slope=float(coef[1]) if len(coef)>1 else None,
                                      e1_slope=float(coef[2]) if len(coef)>2 else None))
    return pred

pred_frames = []
for scenario in ['primary', *specification['sensitivities']]:
    data = prediction_frame(scenario)
    for (sem, exam), cell in data.groupby(['semester','exam'],sort=True):
        for fold in range(5):
            pred_frames.append(predict_split(cell[cell.fold!=fold],cell[cell.fold==fold],
                                            'within_semester_cv',scenario,fold))
    for exam in WINDOWS:
        pred_frames.append(predict_split(data[(data.semester=='f24')&(data.exam==exam)],
                                        data[(data.semester=='s25')&(data.exam==exam)],
                                        'fall_to_spring',scenario,-1))
oof = pd.concat(pred_frames, ignore_index=True)
assert not oof.duplicated(['scenario','validation','student','exam']).any()
save(oof, 'oof')

def metrics(frame, model):
    y, p = frame.observed_pct.to_numpy(), frame[model].to_numpy()
    err = y-p
    center = frame.groupby(['semester','exam']).observed_pct.transform('mean').to_numpy()
    sst = float(np.sum((y-center)**2))
    base_sse = float(np.sum((y-frame.baseline.to_numpy())**2))
    cal = np.linalg.lstsq(np.column_stack([np.ones(len(p)), p]), y, rcond=None)[0]
    return dict(n_students=int(frame.student.nunique()),n_records=len(frame),
                mae=float(np.mean(np.abs(err))),rmse=float(np.sqrt(np.mean(err**2))),
                r2=float(1-np.sum(err**2)/sst) if sst>0 else None,
                r2_vs_training_mean=float(1-np.sum(err**2)/base_sse) if base_sse>0 else None,
                calibration_intercept=float(cal[0]) if model!='baseline' and np.std(p)>1e-10 else None,
                calibration_slope=float(cal[1]) if model!='baseline' and np.std(p)>1e-10 else None,
                observed_mean=float(np.mean(y)),predicted_mean=float(np.mean(p)),
                prediction_min=float(np.min(p)),prediction_max=float(np.max(p)))

metric_rows, calibration_rows, ceiling_rows = [], [], []
for (scenario, validation), sample in oof.groupby(['scenario','validation'],sort=True):
    groups = [('all',sample)] + [(f'{s}_{e}',g) for (s,e),g in sample.groupby(['semester','exam'],sort=True)]
    for group, frame in groups:
        for model in specification['models']:
            metric_rows.append(dict(scenario=scenario,validation=validation,group=group,model=model,**metrics(frame,model)))
            if scenario=='primary' and group!='all' and model!='baseline':
                bins = pd.qcut(frame[model],q=5,duplicates='drop')
                for b, cell in frame.groupby(bins,observed=True,sort=True):
                    calibration_rows.append(dict(validation=validation,group=group,model=model,
                                                 bin_low=float(b.left),bin_high=float(b.right),n=len(cell),
                                                 mean_predicted=float(cell[model].mean()),
                                                 mean_observed=float(cell.observed_pct.mean()),
                                                 sd_observed=float(cell.observed_pct.std())))
            if scenario=='primary' and model!='baseline':
                for label, cell in [('assignment_below_90',frame[frame.assignment_pct<90]),
                                    ('assignment_90_or_above',frame[frame.assignment_pct>=90])]:
                    if len(cell)>3:
                        ceiling_rows.append(dict(validation=validation,group=group,model=model,
                                                 assignment_stratum=label,**metrics(cell,model)))

def paired_bootstrap(frame, before, after):
    """Resample student loss vectors, preserving exam dependence and pairing."""
    # Only primary models receive intervals; all sensitivities remain fully shown.
    student_losses = []
    for (semester, sid), g in frame.groupby(['semester','student'],sort=True):
        y = g.observed_pct.to_numpy()
        a, b = g[before].to_numpy(), g[after].to_numpy()
        student_losses.append([semester,sid,len(g),np.abs(y-a).sum(),np.abs(y-b).sum(),
                               ((y-a)**2).sum(),((y-b)**2).sum()])
    loss = pd.DataFrame(student_losses,columns=['semester','student','n','abs_a','abs_b','sq_a','sq_b'])
    rng = np.random.default_rng(SEED)
    sums = np.zeros((BOOTSTRAPS,5))
    for _, sl in loss.groupby('semester',sort=True):
        values = sl[['n','abs_a','abs_b','sq_a','sq_b']].to_numpy(dtype=float)
        ix = rng.integers(0,len(values),size=(BOOTSTRAPS,len(values)))
        sums += values[ix].sum(axis=1)
    differences = {
        'mae': sums[:,2]/sums[:,0]-sums[:,1]/sums[:,0],
        'rmse': np.sqrt(sums[:,4]/sums[:,0])-np.sqrt(sums[:,3]/sums[:,0])}
    ma, mb = metrics(frame,before),metrics(frame,after)
    return {metric:dict(estimate=mb[metric]-ma[metric],
                        ci_low=float(np.quantile(v,.025)),ci_high=float(np.quantile(v,.975)))
            for metric,v in differences.items()}

paired_rows=[]
primary_oof=oof[oof.scenario=='primary']
for validation,sample in primary_oof.groupby('validation',sort=True):
    groups=[('all',sample)]+[(f'{s}_{e}',g) for (s,e),g in sample.groupby(['semester','exam'],sort=True)]
    for group, frame in groups:
        for before,after in specification['comparisons']:
            paired_rows.append(dict(validation=validation,group=group,before=before,after=after,
                                    n_students=int(frame.student.nunique()),n_records=len(frame),
                                    difference_direction='after minus before; negative error difference favors added evidence',
                                    differences=paired_bootstrap(frame,before,after)))

timing_evidence = {
    'source': 'data/StudyChat/scores/README.md',
    'source_url': 'https://huggingface.co/datasets/wmcnicho/StudyChat/blob/24d7987d9fbb30d9da12acc53455a10f1cdd2d7f/scores/README.md',
    'source_sha256': hashlib.sha256(score_readme.read_bytes()).hexdigest(),
    'source_git_blob_oid': '3daf8626b55b65bbedebd3d9d9df1a679a3e50e6',
    'official_metadata_verification': 'On 2026-09-22 the unauthenticated official Hugging Face tree API at the pinned revision supplied README Git blob OID matching the local file exactly. Score CSVs are LFS objects with redacted content hashes; their local hashes are recorded without claiming fresh official content-hash verification.',
    'order': ['a1','a2','e1','a3','a4','e2','a5','a6','a7','e3'],
    'verified_scope': 'Local official release metadata specifies assignment due-date and examination order for both semesters.',
    'unverified': ['absolute examination dates/times','individual submission timestamps aligned to grades','dates on which graded scores became available'],
    'limitation': 'Earlier due dates establish ordering, not actual score availability to an instructor before the exam; early completion of later assignments is explicitly possible. Retrospective inference only.'}
score_fields = [f'a{i}' for i in range(1,8)]+['e1','e2','e3']
readme_bytes=score_readme.read_bytes()
assert hashlib.sha1(f'blob {len(readme_bytes)}\0'.encode()+readme_bytes).hexdigest()==timing_evidence['source_git_blob_oid']
audit_rows=[]
for semester, g in grades.groupby('semester',sort=True):
    for field in score_fields:
        x=g[field]*100
        audit_rows.append(dict(semester=semester,assessment=field,n=len(x),missing=int(x.isna().sum()),
                               zeros=int((x==0).sum()),at_full=int(np.isclose(x,100).sum()),
                               above_full=int((x>100+1e-8).sum()),min=float(x.min()),max=float(x.max()),
                               mean=float(x.mean()),sd=float(x.std())))
extension.update({
    'status':'complete', 'timing_audit':timing_evidence,
    'input_sha256': {f'{s}_grades': hashlib.sha256((RAW/'scores'/f'{s}_grades_released_normalized.csv').read_bytes()).hexdigest() for s in ['f24','s25']},
    'sample':dict(n_students=len(grades),semester_n={k:int(v) for k,v in grades.semester.value_counts().items()},
                  n_primary_records=int(len(primary_oof[primary_oof.validation=='within_semester_cv'])),
                  cross_semester_repeated_ids=cross_semester_ids,missing_numeric_fields=int(grades[score_fields].isna().sum().sum()),
                  selection_boundary='Consented released-grade sample only; no-log students remain in score prediction. Consent selection is not statistically corrected.'),
    'score_audit':audit_rows,'split_audit':split_audit,'metrics':metric_rows,
    'paired_improvements':paired_rows,'calibration_bins':calibration_rows,
    'high_assignment_score_audit':ceiling_rows,'fitted_coefficients':coefficient_audit,
    'validation_checks': {'all_train_test_student_overlaps_zero':all(x['overlapping_students']==0 for x in split_audit),
                           'same_student_fold_across_exams':bool(primary_oof[primary_oof.validation=='within_semester_cv'].groupby('student').fold.nunique().eq(1).all()),
                           'primary_record_count_362':bool(len(primary_oof[primary_oof.validation=='within_semester_cv'])==362),
                           'raw_score_hashes_unchanged':all(hashlib.sha256((RAW/'scores'/f'{s}_grades_released_normalized.csv').read_bytes()).hexdigest()==initial_hashes[s] for s in ['f24','s25'])},
    'interpretation_boundaries': [
        'AI-permitted assignments do not certify assignment-level AI use.',
        'e1 is an early-achievement proxy after a1/a2, not a pure AI-naive pretest.',
        'e2/e3 measure independent examination performance, not complete latent competence.',
        'Shared examination format can contribute to incremental prediction by e1.',
        'Calibration slopes and observed-versus-predicted bins assess held-out predictions, not causal mechanisms.',
        'Fall-to-Spring evaluation includes changes in cohorts, grading and assignment content; negative test R2 may indicate distribution shift.',
        'Fixed-fold paired uncertainty does not include training-set or fold-split instability.',
        'Zero exclusion changes selection; grade completeness does not resolve absence/non-submission or consent missingness.',
        'High-score subgroup errors are descriptive and do not identify a validated mastery threshold.']})
independent_prediction_error = 0.0
validation_formulas = {'baseline':'observed_pct ~ 1',
                       'assignment':'observed_pct ~ assignment_pct',
                       'assignment_e1':'observed_pct ~ assignment_pct + e1_pct'}
for (sem, exam, fold), test in primary_oof[primary_oof.validation=='within_semester_cv'].groupby(['semester','exam','fold']):
    train = primary_oof[(primary_oof.validation=='within_semester_cv') &
                        (primary_oof.semester==sem) & (primary_oof.exam==exam) &
                        (primary_oof.fold!=fold)]
    for model, formula in validation_formulas.items():
        check_fit = smf.ols(formula, data=train).fit()
        independent_prediction_error = max(independent_prediction_error,
                                           float(np.max(np.abs(check_fit.predict(test)-test[model]))))
extension['numerical_validation'] = {
    'method':'Independent statsmodels formula OLS reconstruction of every primary within-semester held-out model; production uses numpy.linalg.lstsq.',
    'max_prediction_absolute_error':independent_prediction_error,
    'passed':bool(independent_prediction_error<1e-9)}
assert extension['numerical_validation']['passed']
assert all(extension['validation_checks'].values())

# Targeted post-submission-audit diagnostics (D06/S11/S12). These summarize the
# existing released scores and unchanged held-out predictions, not new models.
post_audit_spec = {
    'status':'post-audit addition; method fixed before diagnostic computation',
    'tasks':['D06','S11','S12'],
    'exam_descriptives':'Within semester and target e2/e3: n, mean, sample SD, range, n and proportion above nominal 100.',
    'transport':'Original Fall-to-Spring predictions, separately by exam and model: MAE, RMSE, prediction-minus-outcome mean bias; observed-on-predicted OLS intercept/slope; Pearson and Spearman correlation. Constant predictions have undefined correlation and joint intercept/slope.',
    'no_recalibration':'Do not fit Spring centering corrections, recalibration models for prediction, or new model specifications.',
    'sensitivity_reporting':'State evaluation students/records, inclusion rule, outcome scale and predictor definition for every existing sensitivity. Differences across populations or transformed targets are not improvements on the same task.',
    'uncertainty':'Existing fixed-prediction paired intervals remain unchanged; the new correlations and calibration coefficients are descriptive.'}
extension['post_audit_diagnostics']={'specification':post_audit_spec,'status':'specification fixed; diagnostics not yet computed'}
EXTENSION_PATH.write_text(json.dumps(extension,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

exam_descriptives=[]
for a in audit_rows:
    if a['assessment'] in WINDOWS:
        exam_descriptives.append(dict(a,extra_credit_proportion=a['above_full']/a['n'],
                                      extra_credit_percent=100*a['above_full']/a['n'],
                                      units='percentage points; released scores unchanged'))
transport=[]
for exam, sample in primary_oof[primary_oof.validation=='fall_to_spring'].groupby('exam',sort=True):
    for model in specification['models']:
        y=sample.observed_pct.to_numpy()
        pred=sample[model].to_numpy()
        varying=bool(np.ptp(pred)>1e-10)
        row=dict(exam=exam,model=model,train_semester='f24',test_semester='s25',
                 **metrics(sample,model),mean_prediction_minus_observed=float(np.mean(pred-y)),
                 pearson_r=float(pearsonr(pred,y).statistic) if varying else None,
                 spearman_rho=float(spearmanr(pred,y).statistic) if varying else None,
                 correlation_status='defined' if varying else 'undefined: constant Fall training-mean prediction',
                 calibration_definition='Observed Spring score = intercept + slope * unadjusted Fall-model prediction; diagnostic fit only.',
                 calibration_used_for_prediction=False)
        if model=='baseline':
            fall_mean=float(100*grades.loc[grades.semester=='f24',exam].mean())
            assert np.max(np.abs(pred-fall_mean))<1e-10
            row['verified_fall_training_mean']=fall_mean
        transport.append(row)

scenario_descriptions={
    'primary':('All released records with complete selected grades; retain zeros and extra credit.',
               'Original released examination scores multiplied by 100; e2 may exceed 100.',
               'Equal-weight mean a3-a4 for e2 or a3-a7 for e3.'),
    'cap_extra_credit':('Same students and records as primary.',
                        'Target examination percentage scores capped at 100; transformed outcome differs from primary.',
                        'Same assignment predictors as primary.'),
    'exclude_zero':('Drop a target record if its e1, any included assignment, or target examination is zero; models refit on retained records.',
                    'Original released percentage scores; different selected evaluation population from primary.',
                    'Same assignment definition as primary; zero-containing records excluded.'),
    'points_weighted':('Same students, records and target outcomes as primary.',
                       'Original released examination percentage scores, with extra credit retained.',
                       'Assignment mean weighted by published maximum points within semester.'),
    'all_prior_assignments':('Same students, records and target outcomes as primary.',
                            'Original released examination percentage scores, with extra credit retained.',
                            'Equal-weight mean a1-a4 for e2 or a1-a7 for e3.')}
sensitivity_samples=[]
for (scenario,validation), sample in oof.groupby(['scenario','validation'],sort=True):
    groups=[('all',sample)]+[(f'{s}_{e}',g) for (s,e),g in sample.groupby(['semester','exam'],sort=True)]
    for group, frame in groups:
        rule,scale,predictor=scenario_descriptions[scenario]
        matched_metrics=[m for m in metric_rows if m['scenario']==scenario and m['validation']==validation and m['group']==group]
        for m in matched_metrics:
            assert m['n_students']==frame.student.nunique() and m['n_records']==len(frame)
        sensitivity_samples.append(dict(scenario=scenario,validation=validation,group=group,
                                         n_students=int(frame.student.nunique()),n_records=len(frame),
                                         inclusion_rule=rule,outcome_scale=scale,predictor_definition=predictor,
                                         observed_min=float(frame.observed_pct.min()),observed_max=float(frame.observed_pct.max()),
                                         target_modified=scenario=='cap_extra_credit',
                                         population_selected=scenario=='exclude_zero',
                                         metrics={m['model']:{k:m[k] for k in ['mae','rmse','r2']} for m in matched_metrics}))
extension['post_audit_diagnostics'].update({
    'status':'complete',
    'timing_dimensions':[
        {'dimension':'Assignment due dates/order','evidence':'Official scores README gives a1,a2,e1,a3,a4,e2,a5,a6,a7,e3 for both semesters.','verified':True},
        {'dimension':'Individual assignment completion timestamps','evidence':'Not established by grade tables; the release explicitly allows early completion. Prompt times are not completion times.','verified':False},
        {'dimension':'Grade creation/release timestamps','evidence':'Not provided in analyzed grade tables or score documentation.','verified':False},
        {'dimension':'Absolute examination dates/times','evidence':'Only ordering relative to assignment due dates is established; no verified absolute cutoff.','verified':False},
        {'dimension':'Final grade-data export time','evidence':'Not established. Repository retrieval/version timestamps are not grade export or availability times.','verified':False}],
    'exam_descriptives':exam_descriptives,'unadjusted_transport_by_exam':transport,
    'cell_centered_r2':{
        'formula':'R2_cell = 1 - sum_i (y_i - yhat_i)^2 / sum_c sum_(i in c) (y_i - ybar_c,test)^2',
        'c':'Evaluation semester-by-examination cell; in Fall-to-Spring evaluation, Spring e2 and Spring e3 are separate cells.',
        'denominator_role':'Evaluation only. Test-cell outcome means do not enter predictions, preprocessing, estimation or recalibration.',
        'within_cell':'For a single exam and semester, this equals conventional test-set R2.',
        'baseline':'Unchanged Fall mean for each exam in temporal validation; within CV, each fold training mean.'},
    'sensitivity_samples_and_scales':sensitivity_samples,
    'interpretation':'Absolute error combines location/scale shifts and individual prediction error. Rank correlations describe ordering separately; good ordering can coexist with poor absolute calibration. No post-hoc Spring alignment replaces unadjusted transport.',
    'checks':{'constant_baseline_verified':True,'all_sensitivity_sample_counts_match_original_metrics':True,
              'no_new_prediction_models':True,'no_Spring_recalibration_for_prediction':True}})

EXTENSION_PATH.write_text(json.dumps(extension,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':'complete','validation_checks':extension['validation_checks'],'numerical_validation':extension['numerical_validation']}))
