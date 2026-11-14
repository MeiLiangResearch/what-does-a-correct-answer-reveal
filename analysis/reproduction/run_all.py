"""Run the existing revision analyses from separately obtained source data.

Outputs can contain de-identified student records. Keep outputs local; the
review code archive contains code and aggregate Figure 4 values only.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bastani-dir', type=Path, required=True,
                        help='Git checkout of GenAICanHarmLearning at the documented commit')
    parser.add_argument('--studychat-dir', type=Path,
                        help='Authorized StudyChat release containing scores/; omit to skip Section S9')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.bastani_dir.resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '2f63dae1a01d51453826fe07ef5cf6678e339588', commit
    status = []

    def run(script, *arguments):
        started = time.time()
        command = [sys.executable, str(here / script), *map(str, arguments)]
        with (out / (Path(script).stem + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        status.append({'step': script, 'exit_code': process.returncode,
                       'seconds': round(time.time() - started, 2)})
        (out / 'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'{script} failed; see its output log')
        print(f'PASS {script}', flush=True)

    module_path = here.parent / 'bastani.py'
    spec = importlib.util.spec_from_file_location('mathematics_analysis', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RAW = source
    module.OUT = out / 'mathematics'
    module.OUT.mkdir(exist_ok=True)
    previous_stdout = sys.stdout
    with (out / 'mathematics.log').open('w', encoding='utf-8') as log:
        try:
            sys.stdout = log
            module.main()
            module.run_submission_audit()
        finally:
            sys.stdout = previous_stdout
    status.append({'step': 'mathematics reconstruction and existing audit', 'exit_code': 0})
    print('PASS mathematics reconstruction and existing audit', flush=True)
    for script in ['verify_primary_association.py', 'verify_cr2_and_weighting.py',
                   'verify_classification_bootstrap.py', 'verify_full_sample_predictions.py',
                   'reproduce_correct_practice_analysis.py']:
        extra = ['--with-grader-sensitivity'] if script.startswith('reproduce_correct') else []
        run(script, '--data-dir', source / 'main_regressions', '--outdir', out / 'checks',
            '--assert-expected', *extra)
    run('reproduce_diagnostics.py', '--data-dir', source / 'main_regressions',
        '--results-dir', out / 'mathematics', '--outdir', out / 'diagnostics')
    if args.studychat_dir:
        run('reproduce_studychat_prediction.py', '--data-dir', args.studychat_dir.resolve(),
            '--outdir', out / 'studychat')
    else:
        status.append({'step': 'StudyChat', 'status': 'not run: authorized scores not supplied'})
        (out / 'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    if args.figures:
        run('render_figures.py', '--results-dir', out / 'mathematics',
            '--diagnostics-dir', out / 'diagnostics', '--outdir', out / 'figures')
        run('render_calibration.py', '--results-dir', out / 'mathematics',
            '--outdir', out / 'figures' / 'figure_S2')

if __name__ == '__main__':
    main()
