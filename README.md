# What Does a Correct Answer Reveal?

Analysis code for **What Does a Correct Answer Reveal? AI-Assisted Practice and Performance Without AI**, by Mei Liang.

This repository reproduces the study's analyses of how correct mathematics practice responses relate to subsequent performance without AI. It includes matched-data construction, statistical models, teacher-allocation checks, OLS diagnostics, sensitivity analyses, prediction analyses, and figure generation. The separate exploratory StudyChat analysis is reported in Supplementary Section S9.

## Reproduce the mathematics analyses

The workflow was tested with Python 3.13.5 on Windows. Git is also required. Run the following commands in a terminal, preferably in a separate Python environment:

```bash
git clone https://github.com/MeiLiangResearch/what-does-a-correct-answer-reveal.git
cd what-does-a-correct-answer-reveal
python -m pip install -r requirements.txt
git clone https://github.com/obastani/GenAICanHarmLearning.git source_bastani
git -C source_bastani checkout 2f63dae1a01d51453826fe07ef5cf6678e339588
python analysis/reproduction/run_all.py --bastani-dir source_bastani --outdir local_outputs --figures
```

These commands download the original mathematics data separately, select the version used in the paper, and run the analyses and figure scripts. Omit `--figures` to run the numerical analyses alone.

## Find the results

Outputs are written to `local_outputs/`:

| Location | Contents |
| --- | --- |
| `mathematics/` and `checks/` | Reconstructed responses, model results, and numerical checks |
| `diagnostics/` | Teacher allocation, OLS diagnostics, and sensitivity results for Supplementary Section S1 |
| `figures/` | Main Figures 1–4 and Figure S1; prediction diagnostics for Figure S2 are in `figures/figure_S2/` |
| `run_status.json` and `*.log` | Completion status and the detailed log for each step |

A successful run records an `exit_code` of `0` for each executed step. Without StudyChat data, the StudyChat step is explicitly marked as not run. The scripts check the principal manuscript estimates. The completed revision rerun agreed with the manuscript at the reported precision; numerical tolerances and validation details are given in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Optional StudyChat analysis

Obtain the StudyChat v1.1 grade release from the [original Hugging Face repository](https://huggingface.co/datasets/wmcnicho/StudyChat), following its access conditions. Use revision `24d7987d9fbb30d9da12acc53455a10f1cdd2d7f` and retain this layout:

```text
source_studychat/
  scores/
    README.md
    f24_grades_released_normalized.csv
    s25_grades_released_normalized.csv
```

Add `--studychat-dir source_studychat` to the final command above. StudyChat results will appear in `local_outputs/studychat/`. This step reproduces the exploratory course-grade analysis in Section S9; it is not required to reproduce the mathematics findings.

## Code and data sources

- [analysis/bastani.py](analysis/bastani.py): mathematics data construction and analyses.
- [analysis/reproduction/](analysis/reproduction/): numerical checks, reviewer-requested diagnostics, StudyChat analysis, and figure scripts.
- [analysis/analysis_spec.md](analysis/analysis_spec.md): recorded analysis specifications and their timing.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): source versions, software settings, seeds, and detailed output checks.

This repository provides the code for the present study. Student-level source records remain in their original repositories and are not redistributed here. Generated outputs may contain student-level records or predictions and should be handled under the source data's access conditions.

Please cite the manuscript when using this work. Citation metadata are provided in [CITATION.cff](CITATION.cff).
