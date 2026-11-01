"""Run one of the paper's experiment modules inside the workspace, with its original command line.

    python -m pulseflap.run blind12 run --test kick --workers 16
    python -m pulseflap.run blind12 report --test kick
    python -m pulseflap.run --fresh ablate14 run --workers 16      # workspace without the stored rows

Options before the module name: --work DIR (workspace folder, default work/), --fresh (do not copy stored rows).
"""
import runpy
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    work, rows = None, True
    while argv and argv[0].startswith('--'):
        opt = argv.pop(0)
        if opt == '--work':
            work = argv.pop(0)
        elif opt == '--fresh':
            rows = False
        else:
            raise SystemExit(__doc__)
    if not argv:
        raise SystemExit(__doc__)
    from . import workspace
    workspace.enter(work, rows)
    module = argv.pop(0)
    sys.argv = [module]+argv
    runpy.run_module(f'pulseflap.spike_steer.{module}', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
