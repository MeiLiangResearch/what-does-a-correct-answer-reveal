# What Does a Correct Answer Reveal?

Analysis code for **What Does a Correct Answer Reveal? AI-Assisted Practice and Performance Without AI**, by Mei Liang.

The code reconstructs matched mathematics practice–test responses and reproduces the statistical models, teacher-allocation checks, OLS diagnostics, sensitivity analyses, prediction results, and figures. The separate StudyChat prediction analysis supports Supplementary Section S3.

## Quick start

Tested with Python 3.13.5. Git is required.

```bash
git clone https://github.com/MeiLiangResearch/what-does-a-correct-answer-reveal.git
cd what-does-a-correct-answer-reveal
pip install -r requirements.txt
git clone https://github.com/obastani/GenAICanHarmLearning.git source_bastani
git -C source_bastani checkout 2f63dae1a01d51453826fe07ef5cf6678e339588
python analysis/reproduction/run_all.py --bastani-dir source_bastani --outdir local_outputs --figures
```

To also reproduce StudyChat, obtain the authorized grade files from the [original repository](https://huggingface.co/datasets/wmcnicho/StudyChat) and add `--studychat-dir source_studychat` to the last command. No source student records or access credentials are included here.

## Files

- `analysis/bastani.py`: mathematics reconstruction and analyses.
- `analysis/reproduction/`: numerical verification, reviewer-requested diagnostics, StudyChat prediction, and figure scripts.
- `analysis/analysis_spec.md`: recorded analysis specification.
- `REPRODUCIBILITY.md`: source versions, detailed commands, seeds, and output checks.

Results are written to the chosen output directory. The scripts check the principal manuscript estimates and report execution status in `run_status.json`. The full revision workflow has been run on the source data; the reproduced results agree with the manuscript at the reported precision.

Please cite the manuscript when using this work. Citation metadata are provided in `CITATION.cff`.
