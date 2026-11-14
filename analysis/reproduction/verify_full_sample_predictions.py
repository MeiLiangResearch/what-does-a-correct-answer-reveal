#!/usr/bin/env python3
"""Reconstruct M0-M4 full-sample held-class predictions from Bastani's released data.

Source files: the three original CSV files at commit
2f63dae1a01d51453826fe07ef5cf6678e339588
All transformations and imputations are trained within each class-level fold.

NOTE: Running this script on a mock dataset tests execution only, not publication values.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             brier_score_loss, roc_auc_score, log_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from reproduce_correct_practice_analysis import get_data, matched_pairs, folds_for_classes

CATS = ['part3', 'teacher', 'grader']
NUMERIC = {0: [], 1: ['practice'], 2: ['practice', 'Base', 'Tutor'],
           3: ['practice', 'Base', 'Tutor', 'practice_Base', 'practice_Tutor'],
           4: ['practice', 'Base', 'Tutor', 'practice_Base', 'practice_Tutor', 'gpa_prev']}


def make_model(m: int, binary: bool) -> Pipeline:
    cols = NUMERIC[m]
    blocks = [('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATS)]
    if cols:
        blocks.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), cols))
    pre = ColumnTransformer(blocks, sparse_threshold=0)
    estimator = (LogisticRegression(C=1, solver='lbfgs', max_iter=3000, tol=1e-9)
                 if binary else Ridge(alpha=1))
    return Pipeline([('pre', pre), ('model', estimator)])


def resample_classes(frame: pd.DataFrame, loss0, loss1, seed=20260923, B=2000):
    """Paired fixed-prediction resamples; conditional rather than retraining intervals."""
    a = frame[['Class', 'condition']].copy()
    a['l0'], a['l1'] = np.asarray(loss0), np.asarray(loss1)
    stats = (a.groupby(['condition', 'Class'], sort=True)
              .agg(n=('l0', 'size'), l0=('l0', 'sum'), l1=('l1', 'sum')).reset_index())
    groups = [stats.loc[stats.condition.eq(k), ['n','l0','l1']].to_numpy()
              for k in ['Control','Base','Tutor']]
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        sample = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        out[b] = (sample[:, 2].sum() - sample[:, 1].sum()) / sample[:, 0].sum()
    return np.quantile(out, [.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('./source_data'))
    p.add_argument('--outdir', type=Path, default=Path('./analysis_outputs'))
    p.add_argument('--download', action='store_true')
    p.add_argument('--assert-expected', action='store_true')
    args=p.parse_args()
    get_data(args.data_dir, args.download)
    df=matched_pairs(args.data_dir)
    df['fold']=df.Class.map(folds_for_classes(df))
    df['Base']=(df.condition=='Base').astype(int)
    df['Tutor']=(df.condition=='Tutor').astype(int)
    df['practice_Base']=df.practice*df.Base
    df['practice_Tutor']=df.practice*df.Tutor
    cols=CATS+NUMERIC[4]
    predictions=pd.DataFrame(index=df.index)
    for m in range(5):
        pred=np.full(len(df), np.nan)
        prob=np.full(len(df), np.nan)
        for k in range(5):
            train=df.loc[df.fold.ne(k)]
            test=df.loc[df.fold.eq(k)]
            reg=make_model(m, False)
            clf=make_model(m, True)
            reg.fit(train[cols], train.score)
            clf.fit(train[cols], train.full_credit)
            pred[df.index.get_indexer(test.index)]=reg.predict(test[cols])
            prob[df.index.get_indexer(test.index)]=clf.predict_proba(test[cols])[:,1]
        if not np.isfinite(pred).all() or not np.isfinite(prob).all():
            raise ValueError('Nonfinite predictions')
        predictions[f'M{m}_score']=pred
        predictions[f'M{m}_prob']=prob
    results={}
    for m in range(5):
        ps=predictions[f'M{m}_score'].to_numpy()
        pb=predictions[f'M{m}_prob'].to_numpy()
        results[f'M{m}']={'MAE':float(mean_absolute_error(df.score,ps)),
            'RMSE':float(np.sqrt(mean_squared_error(df.score,ps))),
            'R2':float(r2_score(df.score,ps)),
            'Brier':float(brier_score_loss(df.full_credit,pb)),
            'AUC':float(roc_auc_score(df.full_credit,pb)),
            'log_loss':float(log_loss(df.full_credit,pb))}
    m3=np.abs(df.score.to_numpy()-predictions['M3_score'].to_numpy())
    m4=np.abs(df.score.to_numpy()-predictions['M4_score'].to_numpy())
    b3=(df.full_credit.to_numpy()-predictions['M3_prob'].to_numpy())**2
    b4=(df.full_credit.to_numpy()-predictions['M4_prob'].to_numpy())**2
    results['M3_to_M4']={
        'MAE_after_minus_before':float(m4.mean()-m3.mean()),
        'MAE_CI95_conditional':resample_classes(df,m3,m4),
        'Brier_after_minus_before':float(b4.mean()-b3.mean()),
        'Brier_CI95_conditional':resample_classes(df,b3,b4),
    }
    if args.assert_expected:
        expect=[(28.01,35.33,.113,.1205,.723),(27.22,34.43,.157,.1161,.753),
                (27.21,34.37,.161,.1164,.754),(26.95,34.11,.173,.1155,.759),
                (24.29,30.73,.329,.1013,.834)]
        for i,(mae,rmse,r2,brier,auc) in enumerate(expect):
            got=results[f'M{i}']
            for key, value, tol in [('MAE',mae,.03),('RMSE',rmse,.03),('R2',r2,.004),
                                    ('Brier',brier,.0004),('AUC',auc,.004)]:
                assert abs(got[key]-value)<tol, (i,key,got[key],value)
    args.outdir.mkdir(parents=True,exist_ok=True)
    (args.outdir/'all_models_M0_M4.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    predictions.insert(0,'Class',df.Class.to_numpy())
    predictions.insert(0,'student',df.student.to_numpy())
    predictions.to_csv(args.outdir/'full_sample_fold_predictions.csv',index=False)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
