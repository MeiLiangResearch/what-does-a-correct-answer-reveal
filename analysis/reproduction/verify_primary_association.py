#!/usr/bin/env python3
"""Reconstruct the primary mathematics practice-test association contrasts.

Source: Bastani et al. (2025) official CSVs, repository commit stated in the manuscript.
The two contrasts are observational comparisons *conditional* on practice correctness,
not randomization-based effects within a common subgroup of learners.

Usage:
    python verify_primary_association.py --data-dir ./source_data --outdir ./analysis_outputs
The source_data directory requires problem_part2.csv, problem_part3.csv,
and problem_mapping.csv. To download them, first run
    python reproduce_correct_practice_analysis.py --download
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from reproduce_correct_practice_analysis import get_data, matched_pairs

EXPECTED = {
    "primary": [-12.936144446913474, 2.9197683017440195],
    "omit_teacher": [-13.96740413163476, 3.2305959797528523],
    "omit_grader": [-13.110161752984537, 3.4316812547762865],
}


def fit_contrasts(frame, label: str, *, teacher: bool, grader: bool):
    cond = 'C(condition, Treatment(reference="Control"))'
    formula = f'score ~ practice*{cond} + gpa_prev + C(part3)'
    if teacher:
        formula += ' + C(teacher)'
    if grader:
        formula += ' + C(grader)'
    fit = smf.ols(formula, data=frame).fit(
        cov_type='cluster',
        cov_kwds={'groups': frame['Class'], 'use_correction': True,
                   'df_correction': True},
        use_t=True,
    )
    names = list(fit.params.index)
    ib = next(i for i, k in enumerate(names) if 'practice:C(condition' in k and '[T.Base]' in k)
    it = next(i for i, k in enumerate(names) if 'practice:C(condition' in k and '[T.Tutor]' in k)
    def contrast(mapping):
        v = np.zeros(len(names))
        for idx, value in mapping.items():
            v[idx] = value
        estimate = float(v @ fit.params)
        se = float(np.sqrt(v @ fit.cov_params() @ v))
        tcrit = stats.t.ppf(0.975, frame['Class'].nunique() - 1)
        ci = [estimate - tcrit * se, estimate + tcrit * se]
        return {'estimate': estimate, 'se_CR1': se, 'ci95': ci,
                'p_two_sided_raw': float(2 * stats.t.sf(abs(estimate / se),
                    frame['Class'].nunique() - 1))}
    return {'model': label, 'n_pairs': int(fit.nobs), 'k_parameters': len(names),
            'classes': int(frame['Class'].nunique()),
            'base_minus_control': contrast({ib: 1}),
            'tutor_minus_base': contrast({it: 1, ib: -1})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('./source_data'))
    ap.add_argument('--outdir', type=Path, default=Path('./analysis_outputs'))
    ap.add_argument('--assert-expected', action='store_true')
    args = ap.parse_args()
    get_data(args.data_dir, False)
    all_pairs = matched_pairs(args.data_dir)
    # The matched_pairs function encodes missing grader identity as 'Missing'
    # for prediction; the original adjusted association *excludes* these rows.
    a = all_pairs.loc[all_pairs.gpa_prev.notna() & all_pairs.grader.ne('Missing')].copy()
    if args.assert_expected:
        assert len(a) == 11392, (len(a), 'expected 11,392 complete-case pairs')
    checks = [fit_contrasts(a, 'primary', teacher=True, grader=True),
              fit_contrasts(a, 'omit_teacher', teacher=False, grader=True),
              fit_contrasts(a, 'omit_grader', teacher=True, grader=False)]
    for result in checks:
        if args.assert_expected:
            expected = EXPECTED[result['model']]
            observed = [result['base_minus_control']['estimate'],
                        result['tutor_minus_base']['estimate']]
            assert np.allclose(observed, expected, atol=1e-3), (observed, expected)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {'analysis': 'primary OLS, cluster-robust CR1, 43 df; no Holm applied to raw p values here',
              'all_pairs': len(all_pairs), 'complete_case_pairs': len(a),
              'complete_case_students': int(a.student.nunique()),
              'models': checks}
    (args.outdir / 'primary_association_verification.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
