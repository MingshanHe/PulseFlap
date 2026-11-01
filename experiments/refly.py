"""Re-fly stored episodes and compare them with the stored rows, field by field.

Any arm of the paper on any stored workstation set can be flown again with the code path that produced it:
classical arms and classical + pulse law through ablate14.job, SNN arms through bench10.run_grouped (shared
pre-command prefix, as in blind12/13/15/16) or ablate14.job (the v14 ablation rows), MLP + PPO through
ppo_blind14.job. New rows go to work/runs/refly/<set>/rows_<arm>.json; the stored rows are not touched.

    python experiments/refly.py --set kick --arms classical,probe,snn11_s1 --n 4 --workers 12
    python experiments/refly.py --set nominal --arms snn11_s1 --n 64 --workers 18
    python experiments/refly.py --set lag150 --arms classical,snn11_s2 --n 8
    python experiments/refly.py --list

--n is the number of cases per task (the first n of each task; 64 = the whole set of 192 episodes).
Arms: classical, cls_es_s1, cls_es_s2, probe (= classical + pulse law), snn11_s1..5, snn9_s1..5, snn_bc_s1..5,
snn11_s<k>_nopulse, snn11_s<k>_noref, mlp_ppo_s1..5 (not every arm exists on every set; see --list).

Single episodes of the chaotic dynamics depend on the compiler and libm, so rows are bit-identical only on the
machine that produced them. The stored rows come from two workstations (see data/README.md); most of them
reproduce exactly on either. The PPO rows do not: for the main hit set the comparison therefore uses the PPO rows
that the workstation flew itself (data/rows/q1/dt5/kick), while Table I quotes the second machine's rows (v14).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulseflap import workspace  # noqa: E402

FIELDS = ('flown', 'score', 'pulses', 'termination')


def set_info(name):
    """(cases per task, plant change or None, function arm -> stored rows path or None, arms available)."""
    from pulseflap.spike_steer import bench9 as T, descend9 as D9
    from pulseflap.spike_steer.blind12 import cases12, TESTS as T12
    from pulseflap.spike_steer.blind13 import cases13, TESTS as T13
    from pulseflap.spike_steer.blind15 import cases15, TESTS as T15
    from pulseflap.spike_steer.blind16 import cases16, TESTS as T16
    V14, R = Path('runs/spike_steer/v14'), Path('runs/spike_steer')
    snn = [f'snn11_s{s}' for s in range(1, 6)]
    ppo = [f'mlp_ppo_s{s}' for s in range(1, 6)]
    plant = None
    if name == 'nominal':
        cs = {t: T.cases9(t, 2_099_100_000, 64) for t in D9.TASKS}
        d = R/'v11'

        def ref(arm):
            f = {'classical': d/'test_opponent.json', 'probe': V14/'nominal__cls_unsup.json'}
            for s in range(1, 6):
                f[f'snn11_s{s}'] = d/f'evaluate_snn_es_s{s}_v11.json'
                f[f'snn_bc_s{s}'] = d/f'evaluate_snn_bc_s{s}_v11.json'
                f[f'mlp_ppo_s{s}'] = V14/f'nominal__mlp_ppo_s{s}.json'
            for s in (1, 2):
                f[f'cls_es_s{s}'] = d/f'evaluate_cls_es_s{s}_v11.json'
            return f.get(arm)
    elif name in T12 or name in T13:
        if name in T12:
            K = T12[name]
            cs = {t: cases12(t, K['base'], 64, K['condition'], K['kick'], K.get('mag'), K.get('kick_t')) for t in D9.TASKS}
            d = R/'v12'/name
        else:
            cs = {t: cases13(t, T13[name]) for t in D9.TASKS}
            d = R/'v13'/name

        def ref(arm):
            if arm == 'probe':
                return V14/f'{name}__cls_unsup.json'
            if arm.startswith('mlp_ppo'):
                q1 = R/'q1'/'dt5'/name/f'rows_{arm}.json'        # kick: the workstation's own PPO rows (Q1 reference run)
                return q1 if q1.exists() else V14/f'{name}__{arm}.json'
            if (V14/f'{name}__{arm}.json').exists():                      # v14 ablations and imitation starts
                return V14/f'{name}__{arm}.json'
            return d/f'rows_{arm}.json'
    elif name in T15:
        cs = {t: cases15(t, name) for t in D9.TASKS}
        plant, d = name, R/'v15'/name

        def ref(arm):
            if arm.startswith('mlp_ppo'):
                return V14/f'v15_{name}__{arm}.json'
            return d/f'rows_{arm}.json'
    elif name in T16:
        cs = {t: cases16(t, name) for t in D9.TASKS}
        d = R/'v16'/name

        def ref(arm):
            if arm == 'probe':
                return d/'rows_cls_unsup.json'
            if arm.startswith('mlp_ppo'):
                return V14/f'v16_{name}__{arm}.json'
            return d/f'rows_{arm}.json'
    else:
        raise SystemExit(f'unknown set {name}; see --list')
    cand = ['classical', 'cls_es_s1', 'cls_es_s2', 'probe']+snn+[f'snn9_s{s}' for s in range(1, 6)] + \
        [f'snn_bc_s{s}' for s in range(1, 6)]+[f'{a}_{k}' for a in snn for k in ('nopulse', 'noref')]+ppo
    avail = [a for a in cand if ref(a) is not None and Path(ref(a)).exists()]
    return cs, plant, ref, avail


def all_sets():
    from pulseflap.spike_steer.blind12 import TESTS as T12
    from pulseflap.spike_steer.blind13 import TESTS as T13
    from pulseflap.spike_steer.blind15 import TESTS as T15
    from pulseflap.spike_steer.blind16 import TESTS as T16
    return ['nominal']+list(T12)+list(T13)+list(T15)+list(T16)


def load_rows(f):
    d = json.loads(Path(f).read_text())
    return d if isinstance(d, list) else d['rows']


# ------------------------------------------------------------------------------------------------- workers
_P = {}


def _init_main(plant):
    from pulseflap.spike_steer import bench as B, bench8 as E
    E._init(str(E.SOURCE), .15, dict(E.SIGNS))
    if plant:
        from pulseflap.spike_steer.blind15 import apply_plant, TESTS
        apply_plant(B._ENV, TESTS[plant])


def _init_ppo(plant):
    from pulseflap.spike_steer import ppo_blind14 as P14, bench8 as E
    P14._init(str(E.SOURCE), plant)


def fly(arms, cs, plant, workers, direct=(), log=print):
    """{arm: rows} in case order (task-major). Arms listed in `direct` fly without the shared-prefix cache."""
    from pulseflap.spike_steer import bench8 as E, bench9 as T, bench10 as T10, descend9 as D9, ppo_blind14 as P14
    from pulseflap.spike_steer import cache8 as _C8  # noqa: F401  (as in blind12.stage_run)
    from pulseflap.spike_steer.ablate14 import job as job14
    from pulseflap.spike_steer.blind12 import load_arms
    E.load_signs()                     # attitude -> motion direction signs, before the workers fork (as every driver does)
    al, law = load_arms()
    pay = {n: p for n, k, p in al}
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        E.prepare_source(E.SOURCE)
    out = {}
    main = [a for a in arms if not a.startswith('mlp')]
    if main:
        with ProcessPoolExecutor(workers, initializer=_init_main, initargs=(plant,)) as ex:
            t0 = time.time()
            for a in main:
                if a == 'probe' or a in ('classical', 'cls_es_s1', 'cls_es_s2'):
                    opp = pay['classical' if a == 'probe' else a]
                    kind = 'classical_unsup' if a == 'probe' else 'classical'
                    out[a] = list(ex.map(job14, [(a, t, c, kind, dict(law=law, schedule=opp[t][T.klass(c)]))
                                                for t in D9.TASKS for c in cs[t]], chunksize=1))
                    log(a, 'crashes', sum(not r['flown'] for r in out[a]), '%.0fs' % (time.time()-t0))
            learned = [a for a in main if a not in out]
            specs = {}
            for a in learned:
                base, kind = (a.rsplit('_', 1) if a.endswith(('_nopulse', '_noref')) else (a, 'learned'))
                if base.startswith('snn_bc_s'):
                    s = int(base[-1])
                    f = Path('artifacts/spike_steer/plan_v11')/(f'snn/s{s}/bc.json' if s <= 3 else f'homepc_snn/s{s}/bc.json')
                    specs[a] = (kind, json.loads(f.read_text()))
                else:
                    specs[a] = (kind, pay[base])
            grouped = [a for a in learned if specs[a][0] == 'learned' and a not in direct]
            if grouped:
                rows = T10.run_grouped(ex, [(a, t, c, specs[a][1]) for a in grouped for t in D9.TASKS for c in cs[t]],
                                       chunk=len(grouped))
                for a in grouped:
                    out[a] = [r for r in rows if r['controller'] == a]
                    log(a, 'crashes', sum(not r['flown'] for r in out[a]), '%.0fs' % (time.time()-t0))
            for a in learned:
                if a in out:
                    continue
                kind, spec = specs[a]
                out[a] = list(ex.map(job14, [(a, t, c, kind, spec) for t in D9.TASKS for c in cs[t]], chunksize=1))
                log(a, 'crashes', sum(not r['flown'] for r in out[a]), '%.0fs' % (time.time()-t0))
    ppo = [a for a in arms if a.startswith('mlp')]
    if ppo:
        ck = {}
        for s in range(1, 6):
            g = json.loads((P14.PDIR/f'reselect_s{s}.json').read_text())['selected']['generation']
            ck[s] = str(P14.PDIR/f's{s}/centres/ckpt_g{g}.zip')
        with ProcessPoolExecutor(workers, initializer=_init_ppo, initargs=(plant,)) as ex:
            t0 = time.time()
            seeds = [int(a[-1]) for a in ppo]
            rows = list(ex.map(P14.job, [(s, ck[s], t, c) for s in seeds for t in D9.TASKS for c in cs[t]], chunksize=4))
            for a, s in zip(ppo, seeds):
                out[a] = [r for r in rows if r['controller'] == f'mlp_ppo_s{s}']
                log(a, 'crashes', sum(not r['flown'] for r in out[a]), '%.0fs' % (time.time()-t0))
    return out


def canon(r):
    r = {k: v for k, v in r.items() if k != 'controller'}
    return json.dumps(json.loads(json.dumps(r)), sort_keys=True)


def compare(rows, stored):
    ref = {(r['task'], r['seed']): r for r in stored}
    full = [canon(r) == canon(ref[(r['task'], r['seed'])]) for r in rows]
    key = [all(json.dumps(r.get(k)) == json.dumps(ref[(r['task'], r['seed'])].get(k)) for k in FIELDS) for r in rows]
    bad = [(r, ref[(r['task'], r['seed'])]) for r, e in zip(rows, full) if not e]
    diff = None
    if bad:
        a, b = bad[0]
        diff = dict(task=a['task'], seed=a['seed'],
                    **{k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k != 'controller' and json.dumps(a.get(k)) != json.dumps(b.get(k))})
    return dict(n=len(rows), identical=int(sum(full)), same_flown_score_pulses=int(sum(key)), first_difference=diff)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--set', default='kick')
    p.add_argument('--arms', default='classical,probe,snn11_s1')
    p.add_argument('--n', type=int, default=4, help='cases per task (64 = full set)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--direct', default='', help='learned arms to fly without the shared-prefix cache')
    p.add_argument('--list', action='store_true', help='list sets and the arms stored for each')
    p.add_argument('--work', default=None, help='workspace folder (default work/)')
    a = p.parse_args()
    workspace.enter(a.work)
    if a.list:
        for s in all_sets():
            print(f'{s:16s}', ' '.join(set_info(s)[3]))
        return
    cs, plant, ref, avail = set_info(a.set)
    arms = [x for x in a.arms.split(',') if x]
    missing = [x for x in arms if x not in avail]
    if missing:
        raise SystemExit(f'no stored rows for {missing} on {a.set}; stored: {avail}')
    cs = {t: v[:a.n] for t, v in cs.items()}
    v14_direct = {x for x in arms if x.startswith('snn_bc_s') and a.set in ('kick', 'turb15') and x != 'snn_bc_s1'}
    direct = set(a.direct.split(',')) | v14_direct if a.direct else v14_direct
    t0 = time.time()
    rows = fly(arms, cs, plant, a.workers, direct)
    out = Path('runs/refly')/a.set
    out.mkdir(parents=True, exist_ok=True)
    res = dict(set=a.set, cases_per_task=a.n, wall_s=round(time.time()-t0, 1), arms={})
    for arm, rr in rows.items():
        (out/f'rows_{arm}.json').write_text(json.dumps(rr))
        res['arms'][arm] = c = compare(rr, load_rows(ref(arm)))
        print(f"{a.set:14s} {arm:20s} identical {c['identical']}/{c['n']}  flown+score+pulses+termination "
              f"{c['same_flown_score_pulses']}/{c['n']}" + (f"  first difference {c['first_difference']}" if c['first_difference'] else ''),
              flush=True)
    (out/'compare.json').write_text(json.dumps(res, indent=1))
    ok = all(c['identical'] == c['n'] for c in res['arms'].values())
    print('ALL IDENTICAL' if ok else 'SOME ROWS DIFFER', f'({res["wall_s"]} s); rows and compare.json in', Path.cwd()/out)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
