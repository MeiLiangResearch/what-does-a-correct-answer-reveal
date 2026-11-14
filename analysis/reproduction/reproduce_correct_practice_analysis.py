#!/usr/bin/env python3
"""Independent reconstruction of the matched mathematics responses and exploratory GPA checks.

Requirements: numpy, pandas, scikit-learn, scipy, requests.
Sources: Bastani et al. (2025) public GitHub repository at the manuscript's fixed commit.
No StudyChat data are downloaded or processed by this script.

Usage:
  python reproduce_correct_practice_analysis.py --download --outdir analysis_outputs
  python reproduce_correct_practice_analysis.py --data-dir ./data --outdir analysis_outputs

Important: The correct-practice analysis was developed after examining the original
results; it is exploratory, not preregistered. Confidence intervals condition on fixed
cross-validated predictions; they are not prospective-outcome intervals.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COMMIT = '2f63dae1a01d51453826fe07ef5cf6678e339588'
BASE = f'https://raw.githubusercontent.com/obastani/GenAICanHarmLearning/{COMMIT}/main_regressions/'
NAMES = ['problem_part2.csv', 'problem_part3.csv', 'problem_mapping.csv']
SEED_SPLIT = 20260922
SEED_BOOT = 20260923


def get_data(data_dir: Path, download: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if download:
        import requests
        for name in NAMES:
            response = requests.get(BASE + name, timeout=90)
            response.raise_for_status()
            (data_dir/name).write_bytes(response.content)
    missing = [n for n in NAMES if not (data_dir/n).is_file()]
    if missing:
        raise FileNotFoundError('Missing original source files: ' + ', '.join(missing))


def matched_pairs(data_dir: Path) -> pd.DataFrame:
    p2 = pd.read_csv(data_dir/NAMES[0]).rename(columns={'Student ID': 'student', 'Problem': 'part2', 'Score': 'practice'})
    p3 = pd.read_csv(data_dir/NAMES[1]).rename(columns={'Student ID': 'student', 'Problem': 'part3', 'Score': 'independent'})
    maps = pd.read_csv(data_dir/NAMES[2]).dropna(subset=['part3'])
    p2 = p2.merge(maps[['part2', 'part3']], on='part2', how='inner', validate='many_to_one')
    keys = ['student', 'Class', 'Year', 'Session', 'part3']
    m = p2.merge(p3[keys+['independent']], on=keys, how='inner', validate='one_to_one')
    m = m.loc[m.Honors.eq(0)].copy()
    m['condition'] = m['Treatment arm'].replace({'control': 'Control', 'vanilla': 'Base', 'augmented': 'Tutor'})
    m['grader'] = m['Grader'].astype('string').fillna('Missing').astype(str)
    m['teacher'] = m['teacher'].astype(str)
    m['gpa_prev'] = pd.to_numeric(m.gpa_prev, errors='coerce')
    m['practice'] = pd.to_numeric(m.practice, errors='raise')
    m['independent'] = pd.to_numeric(m.independent, errors='raise')
    m['full_credit'] = (m.independent >= (1.0 - 1e-9)).astype(int)
    m['score'] = m.independent * 100
    assert len(m) == 11596, f'Unexpected matched sample: {len(m)}'
    assert m.student.nunique() == 839 and m.Class.nunique() == 44
    assert int(m.practice.eq(1).sum()) == 5468
    return m


def folds_for_classes(m: pd.DataFrame) -> dict:
    classes = m[['Class', 'condition']].drop_duplicates().sort_values('Class')
    assert len(classes) == 44 and not classes.Class.duplicated().any()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_SPLIT)
    class_to_fold = {}
    for fold, (_, held_out) in enumerate(skf.split(classes[['Class']], classes['condition'])):
        for cl in classes.iloc[held_out]['Class']:
            class_to_fold[cl] = fold
    return class_to_fold


def pipeline(use_gpa: bool, regression: bool, include_grader: bool=False):
    cats = ['part3', 'teacher', 'condition'] + (['grader'] if include_grader else [])
    blocks = [('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)]
    if use_gpa:
        blocks.append(('gpa', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), ['gpa_prev']))
    pre = ColumnTransformer(blocks, sparse_threshold=0)
    model = Ridge(alpha=1) if regression else LogisticRegression(C=1,solver='lbfgs',max_iter=3000,tol=1e-9)
    return Pipeline([('pre', pre),('model',model)])


def predictions(m: pd.DataFrame, include_grader: bool=False) -> pd.DataFrame:
    subset = m.loc[m.practice.eq(1)].copy()
    subset['fold'] = subset.Class.map(folds_for_classes(m))
    subset['pred_score_without_gpa'] = np.nan
    subset['pred_score_with_gpa'] = np.nan
    subset['prob_without_gpa'] = np.nan
    subset['prob_with_gpa'] = np.nan
    features = ['part3','teacher','condition','gpa_prev'] + (['grader'] if include_grader else [])
    for fold in range(5):
        tr = subset[subset.fold.ne(fold)]
        te = subset[subset.fold.eq(fold)]
        for add_gpa in [False,True]:
            suffix = 'with_gpa' if add_gpa else 'without_gpa'
            reg = pipeline(add_gpa,regression=True,include_grader=include_grader)
            reg.fit(tr[features],tr['score'])
            subset.loc[te.index,'pred_score_'+suffix] = reg.predict(te[features])
            clf = pipeline(add_gpa,regression=False,include_grader=include_grader)
            clf.fit(tr[features],tr['full_credit'])
            subset.loc[te.index,'prob_'+suffix] = clf.predict_proba(te[features])[:,1]
    assert not subset[['pred_score_without_gpa','pred_score_with_gpa','prob_without_gpa','prob_with_gpa']].isna().any().any()
    return subset


def calc_metrics(df: pd.DataFrame):
    mae = [float(np.mean(np.abs(df['score']-df['pred_score_'+x]))) for x in ['without_gpa','with_gpa']]
    rmse = [float(np.sqrt(np.mean(np.square(df['score']-df['pred_score_'+x])))) for x in ['without_gpa','with_gpa']]
    brier = [float(np.mean(np.square(df.full_credit-df['prob_'+x]))) for x in ['without_gpa','with_gpa']]
    return {'n':len(df),'MAE_without':mae[0],'MAE_with':mae[1],'MAE_change':mae[1]-mae[0],
            'RMSE_without':rmse[0],'RMSE_with':rmse[1],
            'Brier_without':brier[0],'Brier_with':brier[1],'Brier_change':brier[1]-brier[0]}


def bootstrap_class_CI(df: pd.DataFrame, B=2000):
    """Resample classes within Control, Base and Tutor, with each class sorted; predictions fixed."""
    by_condition = {}
    for condition in ['Control', 'Base', 'Tutor']:
        dg = df.loc[df.condition.eq(condition)]
        cl = []
        for cls,dd in dg.groupby('Class',sort=True):
            diff=np.abs(dd.score-dd.pred_score_with_gpa)-np.abs(dd.score-dd.pred_score_without_gpa)
            cl.append((cls,len(dd),float(diff.sum())))
        by_condition[condition] = cl
    gen=np.random.default_rng(SEED_BOOT)
    draws=np.empty(B)
    for i in range(B):
        total_n=0;total_loss=0
        for condition,clusters in by_condition.items():
            indexes=gen.integers(0,len(clusters),size=len(clusters))
            for j in indexes:
                _,n,l=clusters[j]
                total_n+=n;total_loss+=l
        draws[i]=total_loss/total_n
    return [float(x) for x in np.quantile(draws,[.025,.975])]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('./source_data'))
    ap.add_argument('--outdir',type=Path,default=Path('./analysis_outputs'))
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--with-grader-sensitivity',action='store_true', help='Also fit correct-practice models including grader identity')
    ap.add_argument('--assert-expected',action='store_true',help='Assert manuscript exploratory MAE and Brier estimates')
    args=ap.parse_args()
    get_data(args.data_dir,args.download)
    m=matched_pairs(args.data_dir)
    preds=predictions(m)
    args.outdir.mkdir(parents=True,exist_ok=True)
    report={'source_repository':BASE,'source_commit':COMMIT,'analysis_status':'exploratory; not preregistered',
      'overall':calc_metrics(preds),'per_condition':{k:calc_metrics(d) for k,d in preds.groupby('condition')},
      'MAE_change_95CI_conditional':bootstrap_class_CI(preds)}
    if args.with_grader_sensitivity:
        report['grader_sensitivity'] = calc_metrics(predictions(m,include_grader=True))
    if args.assert_expected:
        assert abs(report['overall']['MAE_without']-29.97185487)<0.02
        assert abs(report['overall']['MAE_with']-26.64871054)<0.02
        assert abs(report['overall']['Brier_without']-0.15718411)<0.0002
        assert abs(report['overall']['Brier_with']-0.13818530)<0.0002
    (args.outdir/'correct_practice_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    preds.to_csv(args.outdir/'correct_practice_fold_predictions.csv',index=False)
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
