"""The workspace: a scratch directory with the layout the experiment modules expect.

The modules in ``pulseflap.spike_steer`` read and write paths relative to the current directory:

    artifacts/spike_steer/...   frozen controllers and plan summaries   <- weights/ and data/summaries/
    runs/spike_steer/...        per-episode rows                         <- data/rows/ (decompressed)
    runs/neuro_gates/scratch/source   relocated robot model, written by FlapSim's prepare_source on first use

``prepare()`` builds this layout under ``work/`` (or ``$PULSEFLAP_WORK``) and ``enter()`` also changes into it. Files
already in the workspace are never overwritten, so results of new runs stay there until you delete the folder.
With ``rows=False`` (``--fresh``) only the stored case lists (``cases.json``) are copied, not the result rows; then
every ``run`` stage flies all arms again instead of skipping the ones whose rows exist. Compare such a re-run with
the stored rows with ``python experiments/compare_rows.py <workspace>``.
"""
import gzip
import os
import shutil
from pathlib import Path
from . import paths

# Files the modules read from runs/ that are inputs rather than results.
RUN_INPUTS = {'runs/spike_steer/v7/baseline.json': 'plan_v7/baseline.json'}


def _copy_tree(src, dst, only=None):
    for f in sorted(src.rglob('*')):
        if f.is_dir() or f.name == 'README.md' or (only and f.name not in only):
            continue
        rel = f.relative_to(src)
        if f.suffix == '.gz':
            out = dst/rel.with_suffix('')
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(f, 'rb') as a, open(out, 'wb') as b:
                    shutil.copyfileobj(a, b)
        else:
            out = dst/rel
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, out)


def check_flapsim():
    ok = (paths.FLAPSIM/'flapsim'/'__init__.py').exists() and (paths.FLAPSIM/'experiments'/'algorithms').is_dir()
    if not ok:
        raise SystemExit(f'FlapSim not found at {paths.FLAPSIM}. Run scripts/setup_flapsim.sh first '
                         '(or set FLAPSIM_ROOT to a FlapSim checkout at commit deab6c8).')


def prepare(work=None, rows=True):
    check_flapsim()
    work = Path(work or paths.WORK)
    art = work/'artifacts'/'spike_steer'
    _copy_tree(paths.WEIGHTS, art)
    _copy_tree(paths.SUMMARIES, art)
    _copy_tree(paths.ROWS, work/'runs'/'spike_steer', None if rows else ('cases.json.gz',))
    for dst, src in RUN_INPUTS.items():
        out = work/dst
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(paths.WEIGHTS/src, out)
    return work


def enter(work=None, rows=True):
    work = prepare(work, rows)
    os.chdir(work)
    return work
