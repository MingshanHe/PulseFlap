"""Fly one episode with each controller of the paper and compare it with the stored row of the same case.

    python experiments/quickstart.py                       # first evade case of the 3 mN·m hit set
    python experiments/quickstart.py --task dash --case 5  # another case of the same set
    python experiments/quickstart.py --set nominal

One episode takes about 15 s of CPU; the script runs the controllers one after another in a single process.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulseflap import workspace  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--set', default='kick', choices=['kick', 'nominal'])
    p.add_argument('--task', default='evade', choices=['evade', 'redirect', 'dash'])
    p.add_argument('--case', type=int, default=0, help='index of the case within the task (0-63)')
    a = p.parse_args()
    workspace.enter()

    from pulseflap import controllers as C, tasks as K
    import refly                                                          # stored-row lookup (experiments/refly.py)
    from pulseflap.spike_steer import ppo_blind14 as P14

    K.load_signs()
    if not (K.SOURCE/'robot_upper7_lower8.xml').exists():
        K.prepare_source(K.SOURCE)
    K.worker_init(str(K.SOURCE), .15)
    env = K.env()

    arms, law = C.load_arms()
    pay = {n: x for n, _, x in arms}
    cases, _, ref, stored_arms = refly.set_info(a.set)
    c = cases[a.task][a.case]
    sched = lambda opp: opp[a.task][K.klass(c)]
    bc1 = json.loads(Path('artifacts/spike_steer/plan_v11/snn/s1/bc.json').read_text())
    pols = [('classical', 'Classical', C.Classical(law, sched(pay['classical']))),
            ('cls_es_s1', 'Classical + ES (seed 1)', C.Classical(law, sched(pay['cls_es_s1']))),
            ('probe', 'Classical + pulse law', C.ClassicalPulseLaw(law, sched(pay['classical']))),
            ('snn11_s1', 'SNN (seed 1)', C.SNN(pay['snn11_s1'])),
            ('snn11_s1_nopulse', 'SNN, pulses suppressed', C.SNNNoPulse(pay['snn11_s1'])),
            ('snn11_s1_noref', 'SNN, set-points zeroed', C.SNNNoRef(pay['snn11_s1'])),
            ('snn_bc_s1', 'SNN, imitation only', C.SNN(bc1)),
            (None, 'SNN + pulse law (reconstruction)', C.SNNPlusPulseLaw(pay['snn11_s1']))]
    print(f"set {a.set}, task {a.task}, case seed {c['seed']}, class {K.klass(c)}"
          + (f", hit at {c['upset']['hits'][0][0]-c['t0']:.3f} s after the command" if 'upset' in c else ''))
    print(f"{'controller':34s} {'flown':>5s} {'score':>8s} {'pulses':>6s}   stored row (flown, score, pulses)")

    def show(key, label, r, dt):
        st = ''
        if key in stored_arms:
            rows = refly.load_rows(ref(key))
            s = next((x for x in rows if x['task'] == a.task and x['seed'] == c['seed']), None)
            if s is not None:
                same = (s['flown'], s['score'], s['pulses']) == (r['flown'], r['score'], r['pulses'])
                st = f"{s['flown']!s:>5s} {s['score']:8.4f} {s['pulses']:6d}   {'identical' if same else 'differs'}"
        print(f"{label:34s} {r['flown']!s:>5s} {r['score']:8.4f} {r['pulses']:6d}   {st}   ({dt:.0f} s)", flush=True)

    for key, label, pol in pols:
        t0 = time.time()
        r = K.run_episode(env, a.task, c, pol)
        show(key, label, r, time.time()-t0)

    # MLP + PPO: decisions every 10 ms through its gym wrapper (ppo_blind14.job), on the same environment
    P14._init(str(K.SOURCE))
    g = json.loads((P14.PDIR/'reselect_s1.json').read_text())['selected']['generation']
    t0 = time.time()
    r = P14.job((1, str(P14.PDIR/f's1/centres/ckpt_g{g}.zip'), a.task, c))
    show('mlp_ppo_s1', 'MLP + PPO (seed 1)', r, time.time()-t0)
    print('\nRows are bit-identical only on the machine that produced them; see README.md, "Reproducibility".')


if __name__ == '__main__':
    main()
