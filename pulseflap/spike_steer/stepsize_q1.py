"""Q1: physics step-size / integrator sensitivity of the frozen-controller ranking (artifacts/spike_steer/q1/preregistration.json).

Only the MuJoCo step changes: model.opt.timestep (and opt.integrator for 'rk4') is set once per worker after the env is
built. Sensors, actuators, the 2 ms control tick and every limit are untouched; the tick re-derives its substep count
(pid_airborne/core.py advance: round(DT/opt.timestep)), the IMU re-derives its 1 kHz sample period at every reset
(topology_hover/sensors.py reset_sensor: round(.001/opt.timestep)), and mj_resetData / mj_setConst do not touch opt.
The cases, the arms and the episode code are those of blind12 / blind16 / ablate14 / ppo_blind14; dt5 must reproduce
the stored reference rows bit for bit (selftest).

    python -m pulseflap.spike_steer.stepsize_q1 selftest --workers 18
    python -m pulseflap.spike_steer.stepsize_q1 timing --workers 18
    python -m pulseflap.spike_steer.stepsize_q1 run --variant dt10 --set kick --workers 18
    python -m pulseflap.spike_steer.stepsize_q1 report
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import time
from pathlib import Path
import mujoco
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from . import ppo_blind14 as P14
from .ablate14 import job as job14
from .blind12 import load_arms, cases12, TESTS as T12
from .blind16 import cases16
from .evaluate_v8 import clopper_pearson
from .paired_report import paired

OUT = Path('runs/spike_steer/q1')
ART = Path('artifacts/spike_steer/q1')
VARIANTS = dict(dt5=dict(timestep=5e-6, integrator='implicitfast'),
                dt2p5=dict(timestep=2.5e-6, integrator='implicitfast'),
                dt10=dict(timestep=1e-5, integrator='implicitfast'),
                rk4=dict(timestep=5e-6, integrator='RK4'))
INTEG = dict(implicitfast=mujoco.mjtIntegrator.mjINT_IMPLICITFAST, RK4=mujoco.mjtIntegrator.mjINT_RK4)
SNN = [f'snn11_s{s}' for s in range(1, 6)]
PPO = [f'mlp_ppo_s{s}' for s in range(1, 6)]
ARMS = ['classical', 'probe']+SNN+PPO
V14 = Path('runs/spike_steer/v14')


# ----------------------------------------------------------------------------------------- sets and reference rows
def build_cases(setname):
    if setname == 'kick':
        K = T12['kick']
        cs = {t: cases12(t, K['base'], 64, K['condition'], K['kick'], K.get('mag'), K.get('kick_t')) for t in D9.TASKS}
        ref = Path('runs/spike_steer/v12/kick/cases.json')
    elif setname == 'turbc15':
        cs = {t: cases16(t, 'turbc15') for t in D9.TASKS}
        ref = Path('runs/spike_steer/v16/turbc15/cases.json')
    elif setname == 'nominal':
        return {t: T.cases9(t, 2_099_100_000, 64) for t in D9.TASKS}
    else:
        raise SystemExit(setname)
    assert json.dumps(json.loads(json.dumps(cs)), sort_keys=True) == json.dumps(json.loads(ref.read_text()), sort_keys=True), f'{setname}: cases differ from {ref}'
    return cs


def ref_path(setname, arm):
    if setname == 'kick':
        return {'classical': Path('runs/spike_steer/v12/kick/rows_classical.json'), 'probe': V14/'kick__cls_unsup.json'}.get(
            arm, V14/f'kick__{arm}.json' if arm.startswith('mlp') else Path(f'runs/spike_steer/v12/kick/rows_{arm}.json'))
    if setname == 'turbc15':
        d = Path('runs/spike_steer/v16/turbc15')
        return {'classical': d/'rows_classical.json', 'probe': d/'rows_cls_unsup.json'}.get(
            arm, V14/f'v16_turbc15__{arm}.json' if arm.startswith('mlp') else d/f'rows_{arm}.json')
    if setname == 'nominal':
        d = Path('runs/spike_steer/v11')
        A11 = Path('artifacts/spike_steer/plan_v11')
        f = {'classical': d/'test_opponent.json', 'probe': V14/'nominal__cls_unsup.json'}
        for s in range(1, 6):
            f[f'snn11_s{s}'] = (d/f'evaluate_snn_es_s{s}_v11.json') if s <= 3 else (A11/f'homepc/evaluate_snn_es_s{s}_v11.json')
            f[f'mlp_ppo_s{s}'] = V14/f'nominal__mlp_ppo_s{s}.json'
        return f[arm]


def load_rows(f):
    d = json.loads(Path(f).read_text())
    return d if isinstance(d, list) else d['rows']


def rows_for(variant, setname, arm):
    if variant == 'dt5':
        f = OUT/'dt5'/setname/f'rows_{arm}.json'
        return load_rows(f if f.exists() else ref_path(setname, arm))
    f = OUT/variant/setname/f'rows_{arm}.json'
    return load_rows(f) if f.exists() else None


# ----------------------------------------------------------------------------------------- the only physics change
def apply_step(env, variant):
    m = env.sim.model
    assert m is env.air.sim.model and m is env.air.env.sim.model and env.sim.backend.model is m, 'model not shared'
    if variant == 'dt5':                                # the reference: assert the MJCF values, change nothing
        assert m.opt.timestep == 5e-6 and m.opt.integrator == INTEG['implicitfast'], (m.opt.timestep, m.opt.integrator)
        return
    v = VARIANTS[variant]
    m.opt.timestep = v['timestep']
    m.opt.integrator = INTEG[v['integrator']]


def opt_state(env):
    m = env.sim.model
    return dict(timestep=float(m.opt.timestep), integrator=int(m.opt.integrator), substeps=round(T10.DT/m.opt.timestep) if hasattr(T10, 'DT') else round(.002/m.opt.timestep),
                imu_period=int(env.frontend.sensor.period))


_V = {}


def _init_main(variant):
    E._init(str(E.SOURCE), .15, dict(E.SIGNS))
    apply_step(B._ENV, variant)
    _V['variant'] = variant


def _init_ppo(variant):
    P14._init(str(E.SOURCE))
    assert P14._S['env9'].env is B._ENV
    apply_step(B._ENV, variant)
    _V['variant'] = variant


def check_job(args):
    """Fly one short episode and report opt before and after reset (reset must not restore opt)."""
    task, c = args
    before = opt_state(B._ENV)
    B._ENV.configure(c)
    B._ENV.reset(seed=c['seed'])
    after = opt_state(B._ENV)
    return dict(variant=_V['variant'], before=before, after_reset=after)


def pool(kind, variant, workers):
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        E.prepare_source(E.SOURCE)
    return ProcessPoolExecutor(workers, initializer=_init_main if kind == 'main' else _init_ppo, initargs=(variant,))


def ckpts():
    ck = {}
    for s in range(1, 6):
        g = json.loads((P14.PDIR/f'reselect_s{s}.json').read_text())['selected']['generation']
        ck[s] = str(P14.PDIR/f's{s}/centres/ckpt_g{g}.zip')
    return ck


def run_arms(variant, cs, workers, arms=ARMS, log=print):
    """cs: {task: [cases]} -> {arm: rows} in case order (task-major)."""
    al, law = load_arms()
    opp = al[0][2]
    specs = {n: s for n, k, s in al if k == 'learned'}
    out = {}
    main = [a for a in arms if not a.startswith('mlp')]
    if main:
        with pool('main', variant, workers) as ex:
            chk = list(ex.map(check_job, [(t, cs[t][0]) for t in D9.TASKS for _ in range(max(1, workers//3))]))
            log('opt check', json.dumps(chk[0]), 'all equal' if all(x == chk[0] for x in chk) else 'DIFFER')
            t0 = time.time()
            for name, kind in (('classical', 'classical'), ('probe', 'classical_unsup')):
                if name in main:
                    out[name] = list(ex.map(job14, [(name, t, c, kind, dict(law=law, schedule=opp[t][T.klass(c)])) for t in D9.TASKS for c in cs[t]], chunksize=1))
                    log(variant, name, 'crashes', sum(not r['flown'] for r in out[name]), '%.0fs' % (time.time()-t0))
            learned = [n for n in SNN if n in main]
            if learned:
                rows = T10.run_grouped(ex, [(n, t, c, specs[n]) for n in learned for t in D9.TASKS for c in cs[t]], chunk=len(learned))
                for n in learned:
                    out[n] = [r for r in rows if r['controller'] == n]
                    log(variant, n, 'crashes', sum(not r['flown'] for r in out[n]), '%.0fs' % (time.time()-t0))
    ppo = [a for a in arms if a.startswith('mlp')]
    if ppo:
        ck = ckpts()
        with pool('ppo', variant, workers) as ex:
            chk = list(ex.map(check_job, [(t, cs[t][0]) for t in D9.TASKS]))
            log('ppo opt check', json.dumps(chk[0]), 'all equal' if all(x == chk[0] for x in chk) else 'DIFFER')
            t0 = time.time()
            seeds = [int(a[-1]) for a in ppo]
            jobs = [(s, ck[s], t, c) for s in seeds for t in D9.TASKS for c in cs[t]]
            rows = list(ex.map(P14.job, jobs, chunksize=4))
            for a, s in zip(ppo, seeds):
                out[a] = [r for r in rows if r['controller'] == f'mlp_ppo_s{s}']
                log(variant, a, 'crashes', sum(not r['flown'] for r in out[a]), '%.0fs' % (time.time()-t0))
    return out


# ----------------------------------------------------------------------------------------- stages
def canon(r):
    return json.dumps(json.loads(json.dumps(r)), sort_keys=True)


def stage_selftest(a):
    """dt5 on a subset of every set must equal the stored reference rows field for field; opt must survive reset."""
    res = dict(variant='dt5', per_set={})
    ok_all = True
    for setname in a.sets.split(','):
        full = build_cases(setname)
        cs = {t: full[t][:a.n] for t in D9.TASKS}
        t0 = time.time()
        rows = run_arms('dt5', cs, a.workers)
        el = time.time()-t0
        per = {}
        for arm, rr in rows.items():
            ref = {(r['task'], r['seed']): r for r in load_rows(ref_path(setname, arm))}
            if arm.startswith('mlp'):
                eq = [canon(r) == canon(ref[(r['task'], r['seed'])]) for r in rr]
            else:
                eq = [canon(dict(r, controller=ref[(r['task'], r['seed'])]['controller'])) == canon(ref[(r['task'], r['seed'])]) for r in rr]
            per[arm] = dict(n=len(eq), identical=int(sum(eq)))
            if not all(eq):
                ok_all = False
                bad = [r for r, e in zip(rr, eq) if not e][0]
                rf = ref[(bad['task'], bad['seed'])]
                per[arm]['first_diff'] = {k: (bad.get(k), rf.get(k)) for k in set(bad) | set(rf) if json.dumps(bad.get(k)) != json.dumps(rf.get(k))}
            print(setname, arm, per[arm], flush=True)
        res['per_set'][setname] = dict(cases_per_task=a.n, wall_s=el, arms=per)
    for v in VARIANTS:                                   # opt survives reset in every variant
        with pool('main', v, 2) as ex:
            res.setdefault('opt_after_reset', {})[v] = list(ex.map(check_job, [('dash', build_cases('kick')['dash'][0])]*2))[0]
            print(v, res['opt_after_reset'][v], flush=True)
    res['bit_for_bit'] = ok_all
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'selftest.json').write_text(json.dumps(res, indent=1))
    print('selftest', 'PASS' if ok_all else 'FAIL', flush=True)


def stage_timing(a):
    """Wall time for a small block of each variant (all arms, n cases per task) on the kick set."""
    full = build_cases(a.sets.split(',')[0])
    cs = {t: full[t][:a.n] for t in D9.TASKS}
    res = {}
    for v in a.variants.split(','):
        t0 = time.time()
        run_arms(v, cs, a.workers)
        res[v] = time.time()-t0
        print('timing', v, '%.1f s for %d cases x %d arms' % (res[v], 3*a.n, len(ARMS)), flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'timing.json').write_text(json.dumps(dict(cases_per_task=a.n, workers=a.workers, wall_s=res), indent=1))


def stage_run(a):
    d = OUT/a.variant/a.set
    d.mkdir(parents=True, exist_ok=True)
    cs = build_cases(a.set)
    (d/'cases.json').write_text(json.dumps(cs))
    todo = [x for x in (a.arms.split(',') if a.arms else ARMS) if not (d/f'rows_{x}.json').exists()]
    t0 = time.time()
    rows = run_arms(a.variant, cs, a.workers, todo)
    for arm, rr in rows.items():
        (d/f'rows_{arm}.json').write_text(json.dumps(rr))
    wall = time.time()-t0
    log = json.loads((d/'wall.json').read_text()) if (d/'wall.json').exists() else []
    log.append(dict(arms=todo, wall_s=wall, workers=a.workers))
    (d/'wall.json').write_text(json.dumps(log, indent=1))
    print(a.variant, a.set, 'done in %.0f s' % wall, flush=True)


def mcnemar_exact(x, y):
    """x, y: crash indicators on the same cases. Two-sided exact binomial test on the discordant pairs."""
    b = int(sum(1 for p, q in zip(x, y) if p and not q))
    c = int(sum(1 for p, q in zip(x, y) if q and not p))
    n = b+c
    if n == 0:
        return dict(arm_only=b, ref_only=c, p=1.)
    k = min(b, c)
    p = min(1., 2*sum(math.comb(n, i) for i in range(k+1))/2**n)
    return dict(arm_only=b, ref_only=c, p=float(p))


def aligned_crash(rows, ref):
    key = lambda r: (r['task'], r['seed'])
    R = {key(r): r for r in ref}
    rs = sorted(rows, key=key)
    return [not r['flown'] for r in rs], [not R[key(r)]['flown'] for r in rs]


def stage_report(a):
    S = dict(study='q1 step size / integrator', variants=VARIANTS, sets={})
    lines = []
    for setname in ('kick', 'turbc15', 'nominal'):
        for v in VARIANTS:
            arms = {x: rows_for(v, setname, x) for x in ARMS} if (v == 'dt5' or (OUT/v/setname).exists()) else {}
            if v == 'dt5' and setname == 'nominal' and not (OUT/v/setname).exists():
                arms = {}
            if not arms or arms.get('classical') is None or arms.get('probe') is None:
                continue
            cls, probe = arms['classical'], arms['probe']
            base = {x: rows_for('dt5', setname, x) for x in ARMS}
            D = {}
            lines.append(f'{setname} / {v}')
            for x, rr in arms.items():
                if rr is None:
                    continue
                cr = sum(not r['flown'] for r in rr)
                e = dict(crashes=cr, n=len(rr), ci95=clopper_pearson(cr, len(rr)),
                         drift_vs_dt5=cr-sum(not r['flown'] for r in base[x]) if base.get(x) else None)
                if x != 'classical':
                    p = paired(cls, rr)
                    e['vs_classical'] = dict(delta=p['delta'], ci95=p['delta_ci95'], wins=p['wins'])
                if x != 'probe':
                    p = paired(probe, rr)
                    e['vs_probe'] = dict(delta=p['delta'], ci95=p['delta_ci95'], wins=p['wins'])
                    e['mcnemar_vs_probe'] = mcnemar_exact(*aligned_crash(rr, probe))
                D[x] = e
                s = '  %-11s crashes %3d (drift %s)' % (x, cr, '%+d' % e['drift_vs_dt5'] if e['drift_vs_dt5'] is not None else '-')
                if 'vs_classical' in e:
                    s += '  Δcls %+.3f [%+.3f, %+.3f]' % (e['vs_classical']['delta'], *e['vs_classical']['ci95'])
                if 'vs_probe' in e:
                    m = e['mcnemar_vs_probe']
                    s += '  Δprobe %+.3f [%+.3f, %+.3f]  McNemar %d/%d p=%.3g' % (e['vs_probe']['delta'], *e['vs_probe']['ci95'], m['arm_only'], m['ref_only'], m['p'])
                lines.append(s)
            snn = [x for x in SNN if x in D]
            if setname == 'kick' and len(snn) == 5:
                k = sum(D['probe']['crashes'] < D[x]['crashes'] for x in snn)
                D['criterion_a'] = dict(seeds_probe_fewer_crashes=k, holds=bool(k >= 4))
                lines.append('  (a) probe crashes fewer than SNN in %d / 5 seeds -> %s' % (k, 'HOLDS' if k >= 4 else 'fails'))
            if setname == 'turbc15' and len(snn) == 5:
                k = sum(D[x]['vs_probe']['ci95'][1] < 0 for x in snn)
                D['criterion_b'] = dict(seeds_upper_below_0=k, holds=bool(k >= 3))
                lines.append('  (b) SNN Δ vs probe upper 95%% < 0 in %d / 5 seeds -> %s' % (k, 'HOLDS' if k >= 3 else 'fails'))
            S['sets'].setdefault(setname, {})[v] = D
    verdict = {}
    for v in VARIANTS:
        ka = S['sets'].get('kick', {}).get(v, {}).get('criterion_a')
        kb = S['sets'].get('turbc15', {}).get(v, {}).get('criterion_b')
        if ka and kb:
            verdict[v] = dict(a=ka['holds'], b=kb['holds'], robust=bool(ka['holds'] and kb['holds']))
    S['verdict_per_step'] = verdict
    S['ranking_step_size_robust'] = bool(verdict) and all(x['robust'] for x in verdict.values())
    lines.append('verdict per step: ' + json.dumps(verdict))
    lines.append('ranking step-size-robust over tested steps: %s' % S['ranking_step_size_robust'])
    walls = {}
    for f in OUT.glob('*/*/wall.json'):
        walls[f'{f.parent.parent.name}/{f.parent.name}'] = json.loads(f.read_text())
    S['wall'] = walls
    if (ART/'selftest.json').exists():
        S['selftest'] = json.loads((ART/'selftest.json').read_text())
    txt = '\n'.join(lines)
    print(txt)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'summary.json').write_text(json.dumps(S, indent=1))
    (ART/'report.txt').write_text(txt+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'timing', 'run', 'report'])
    p.add_argument('--variant', choices=list(VARIANTS), default='dt10')
    p.add_argument('--variants', default='dt5,dt10,dt2p5,rk4')
    p.add_argument('--set', choices=['kick', 'turbc15', 'nominal'], default='kick')
    p.add_argument('--sets', default='kick,turbc15')
    p.add_argument('--arms', default='')
    p.add_argument('--n', type=int, default=6, help='cases per task for selftest / timing')
    p.add_argument('--workers', type=int, default=18)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    {'selftest': stage_selftest, 'timing': stage_timing, 'run': stage_run, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
