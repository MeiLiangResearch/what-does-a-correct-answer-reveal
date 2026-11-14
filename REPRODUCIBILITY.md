# Code for the GenAI assessment revision

Version: revision-1. Manuscript: *What Does a Correct Answer Reveal? AI-Assisted Practice and Performance Without AI*. Author: Mei Liang. This is the code developed for the present secondary analysis, rather than a copy of the source-study repositories.

## Environment and sources

Tested with Python 3.13.5 on Windows; Git is also required. Install the pinned scientific dependencies with `python -m pip install -r requirements.txt`. Use a separate environment if needed. No credentials are included or needed for the public mathematics source.

Obtain the mathematics repository separately:

```text
git clone https://github.com/obastani/GenAICanHarmLearning.git source_bastani
git -C source_bastani checkout 2f63dae1a01d51453826fe07ef5cf6678e339588
```

StudyChat is optional for the mathematics analysis. Obtain the authorized v1.1 release from https://huggingface.co/datasets/wmcnicho/StudyChat under its access conditions. The pinned repository revision is `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f`. Supply a directory containing `scores/README.md`, `scores/f24_grades_released_normalized.csv`, and `scores/s25_grades_released_normalized.csv`. The program does not request, store, or distribute access tokens.

## Run

From this archive's root, run:

```text
python analysis/reproduction/run_all.py --bastani-dir source_bastani --studychat-dir source_studychat --outdir local_outputs --figures
```

Omit `--studychat-dir source_studychat` when those authorized records are unavailable. That step is then explicitly marked not run; mathematics and its figures remain reproducible. Omit `--figures` to run numerical analyses only. An Arial installation gives the same manuscript typography; numerical results do not depend on the plotting font.

## What runs and what to check

- `analysis/bastani.py`: matched-pair reconstruction, source-study aggregate comparisons, response associations, threshold analyses, existing sensitivity checks, class-held-out prediction, CR2 validation, weighting, and the ten existing fold-split checks.
- `verify_*.py` and `reproduce_correct_practice_analysis.py`: independent numerical checks with assertions against the manuscript estimates. Grader sensitivity is retained for the correct-practice analysis.
- `reproduce_diagnostics.py`: teacher overlap, matrix rank/VIFs, residual checks, class fixed effects, overlap-restricted contrasts, deletion influence, and fractional logit. These reproduce the existing S8 analyses; they do not introduce new model choices.
- `reproduce_studychat_prediction.py`: the existing S3 prediction analyses, with student-grouped folds and unchanged Fall-to-Spring transport. Unreported legacy dialogue models are not run.
- `render_figures.py` and `render_calibration.py`: main Figures 1–4 and supplementary Figures S1–S2, reusing the manuscript drawing code. Figure 4 uses the two manuscript aggregate records in `figure4_aggregate_data.json`; those values are checked by the prediction scripts. Figure 1 is a design schematic and Figure 2 a source-task illustration.

Inspect `run_status.json` and each step's log. Assertions verify 11,596 matched pairs, 11,392 complete association pairs, primary association contrasts, CR2/weighting results, confirmation-rate bootstrap results, and full-sample/correct-practice prediction results. Diagnostic output is `diagnostics/analysis_results.json` and `diagnostics/tables/`. The StudyChat JSON reports 181 students, 362 examination records, zero training/test student overlap, unchanged input hashes, and agreement with independent statsmodels reconstruction.

The primary seed is 20260922, with 2,000 bootstrap draws. Existing repeated-fold sensitivity seeds and model settings are recorded in `analysis_spec.md` and the output JSON. Uncertainty from fixed held-out predictions is conditional on the fitted models and folds. Cross-environment numerical drift in logistic prediction can reach approximately 0.00002 in reported metrics; the verified revision rerun agrees at three decimal places. Association, teacher, and OLS diagnostic checks agree within 1e-7.

## Data handling

Source records are not redistributed. Generated local outputs may contain de-identified student records and predictions and remain subject to the source access conditions. Do not publish those outputs automatically. This repository contains analysis and figure code plus two aggregate Figure 4 records. A Zenodo archive has not yet been created.
