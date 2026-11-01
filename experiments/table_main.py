"""Table I and the main numbers of the text, recomputed from the stored per-episode rows (no simulation, ~1 min).

    python experiments/table_main.py

Workstation columns come from data/rows; cluster columns from data/cluster (summaries only, the cluster rows are not
included). Each line prints the value computed here next to the value printed in the paper; the script exits with an
error if a workstation number does not match.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulseflap import workspace  # noqa: E402

SNN = [f'snn11_s{s}' for s in range(1, 6)]
PPO = [f'mlp_ppo_s{s}' for s in range(1, 6)]
BAD = []


def rows(f):
    d = json.loads(Path(f).read_text())
    return d if isinstance(d, list) else d['rows']


def crashes(rr):
    return sum(not r['flown'] for r in rr)


def rng(v):
    return str(v[0]) if min(v) == max(v) else f'{min(v)}–{max(v)}'


def check(label, got, paper):
    ok = str(got) == str(paper)
    if not ok:
        BAD.append(label)
    print(f'  {label:58s} {str(got):>14s}   paper {str(paper):>14s}   {"ok" if ok else "MISMATCH"}')


def main():
    workspace.enter()
    from pulseflap.eval import paired, mcnemar_exact, clopper_pearson
    R, V14 = Path('runs/spike_steer'), Path('runs/spike_steer/v14')
    kick = R/'v12/kick'
    K = {a: rows(kick/f'rows_{a}.json') for a in ['classical', 'cls_es_s1', 'cls_es_s2']+SNN}
    K['probe'] = rows(V14/'kick__cls_unsup.json')
    K.update({a: rows(V14/f'kick__{a}.json') for a in PPO})
    for s in range(1, 6):
        K[f'nopulse_s{s}'] = rows(V14/f'kick__snn11_s{s}_nopulse.json')
        K[f'noref_s{s}'] = rows(V14/f'kick__snn11_s{s}_noref.json')
        K[f'bc_s{s}'] = rows(kick/'rows_snn_bc_s1.json') if s == 1 else rows(V14/f'kick__snn_bc_s{s}.json')

    print('Table I, workstation, 3 mN·m torque hit (crashed episodes of 192)')
    check('Classical (1)', crashes(K['classical']), 60)
    check('Classical + ES (2)', rng([crashes(K[f'cls_es_s{s}']) for s in (1, 2)]), '56–58')
    check('SNN, pulses suppressed (5)', rng([crashes(K[f'nopulse_s{s}']) for s in range(1, 6)]), '62–93')
    check('Classical + pulse law (1)', crashes(K['probe']), 5)
    check('SNN (5)', rng([crashes(K[a]) for a in SNN]), '22–40')
    check('MLP + PPO (5)', rng([crashes(K[a]) for a in PPO]), '41–47')
    check('SNN, set-points zeroed (5)', rng([crashes(K[f'noref_s{s}']) for s in range(1, 6)]), '33–54')
    check('SNN, imitation only (5)', rng([crashes(K[f'bc_s{s}']) for s in range(1, 6)]), '23–32')

    def better(fam, opp, sets_rows, need_fewer):
        n = 0
        for a in fam:
            pr = paired(opp, sets_rows[a])
            n += pr['delta_ci95'][1] < 0 and (not need_fewer or pr['crashes'] < pr['classical_crashes'])
        return f'{n}/5'
    check('SNN seeds better than classical + pulse law, hit', better(SNN, K['probe'], K, True), '0/5')
    check('MLP + PPO seeds better than classical + pulse law, hit', better(PPO, K['probe'], K, True), '0/5')

    for test, scale, col in (('turbc15', '×1.5', dict(classical=32, cls_es=34, probe=12, snn='1–7', ppo='5–10', bs='1/5', bp='0/5')),
                             ('turbc175', '×1.75', dict(classical=160, cls_es=157, probe=137, snn='84–90', ppo='86–92', bs='5/5', bp='5/5'))):
        d = R/'v16'/test
        S = {a: rows(d/f'rows_{a}.json') for a in ['classical', 'cls_es_s1', 'cls_unsup']+SNN}
        S.update({a: rows(V14/f'v16_{test}__{a}.json') for a in PPO})
        print(f'Table I, workstation, sustained disturbance {scale} (crashed episodes of 192)')
        check('Classical (1)', crashes(S['classical']), col['classical'])
        check('Classical + ES (1 seed flown)', crashes(S['cls_es_s1']), col['cls_es'])
        check('Classical + pulse law (1)', crashes(S['cls_unsup']), col['probe'])
        check('SNN (5)', rng([crashes(S[a]) for a in SNN]), col['snn'])
        check('MLP + PPO (5)', rng([crashes(S[a]) for a in PPO]), col['ppo'])
        fewer = test == 'turbc15'
        check('SNN seeds better than classical + pulse law', better(SNN, S['cls_unsup'], S, fewer), col['bs'])
        check('MLP + PPO seeds better than classical + pulse law', better(PPO, S['cls_unsup'], S, fewer), col['bp'])

    print('Table I, cluster columns (summaries in data/cluster)')
    p4 = json.loads(Path(workspace.paths.REPO/'data/cluster/p4_hit_replication.json').read_text())['arms']
    p5 = json.loads(Path(workspace.paths.REPO/'data/cluster/p5_sustained_disturbance.json').read_text())
    p6 = json.loads(Path(workspace.paths.REPO/'data/cluster/p6_snn_plus_pulse_law.json').read_text())
    print(f"  hit, of 768: classical {p4['classical']['crashes']}, classical + ES {p4['cls_es_s2']['crashes']}–{p4['cls_es_s1']['crashes']}, "
          f"classical + pulse law {p4['probe']['crashes']}, SNN {rng([p4[f'snn_s{s}']['crashes'] for s in range(1, 6)])}, "
          f"MLP + PPO {rng([p4[f'ppo_s{s}']['crashes'] for s in range(1, 6)])}")
    for k, lab in (('x1_5', '×1.5'), ('x1_75', '×1.75')):
        A = p5[k]['arms']
        print(f"  {lab}, of 576: classical {A['classical']['crashes']}, classical + ES {A['cls_es_s1']['crashes']}, "
              f"classical + pulse law {A['rule']['crashes']}, SNN {rng([A[f'snn_s{s}']['crashes'] for s in range(1, 6)])}, "
              f"MLP + PPO {rng([A[f'ppo_s{s}']['crashes'] for s in range(1, 6)])}")
    print(f"  SNN + pulse law, hit, of 192: {rng(p6['kick']['hybrid_crashes'])} (classical {p6['kick']['classical']}, "
          f"classical + pulse law {p6['kick']['probe']})")

    print('Text, workstation')
    nom = R/'v11'
    opp = rows(nom/'test_opponent.json')
    pr = paired(opp, rows(V14/'nominal__cls_unsup.json'))
    check('nominal: classical + pulse law Δ vs classical [95 % CI]',
          '%.2f [%.2f,%.2f]' % (pr['delta'], *pr['delta_ci95']), '-0.13 [-0.19,-0.09]')
    check('nominal: classical + pulse law crashes', pr['crashes'], 0)
    ds = [paired(opp, rows(nom/f'evaluate_snn_es_s{s}_v11.json'))['delta'] for s in range(1, 6)]
    check('nominal: SNN Δ vs classical, range over seeds', '%+.2f..%+.2f' % (min(ds), max(ds)), '-0.06..+0.06')
    check('nominal: SNN crashes', sum(crashes(rows(nom/f'evaluate_snn_es_s{s}_v11.json')) for s in range(1, 6)), 0)
    lo, hi = clopper_pearson(5, 192)
    check('hit: classical + pulse law crash rate, 95 % CI (%)', '%.1f [%.1f,%.1f]' % (500/192, 100*lo, 100*hi), '2.6 [0.9,6.0]')
    key = lambda r: (r['task'], r['seed'])
    P = {key(r): not r['flown'] for r in K['probe']}

    def mc(a):
        rr = sorted(K[a], key=key)
        return mcnemar_exact([not r['flown'] for r in rr], [P[key(r)] for r in rr])['p']
    check('hit: max McNemar p, every SNN and PPO seed vs pulse law <= 8e-5', max(mc(a) for a in SNN+PPO) <= 8e-5, True)
    h1 = sum(paired(K['classical'], K[a])['delta_ci95'][1] < 0 and crashes(K[a]) <= 60 for a in SNN)
    check('hit: SNN seeds passing H1 (Δ CI < 0, crashes <= classical)', f'{h1}/5', '4/5')
    rep = R/'v12/kick_rep'
    check('hit replication: classical crashes', crashes(rows(rep/'rows_classical.json')), 42)
    check('hit replication: SNN crashes', rng([crashes(rows(rep/f'rows_{a}.json')) for a in SNN]), '16–39')
    k45 = R/'v12/kick45'
    check('4.5 mN·m hit: classical + pulse law crashes', crashes(rows(V14/'kick45__cls_unsup.json')), 30)
    check('4.5 mN·m hit: SNN and PPO crashes', rng([crashes(rows(k45/f'rows_{a}.json')) for a in SNN] +
                                                   [crashes(rows(V14/f'kick45__{a}.json')) for a in PPO]), '46–95')
    print('\nALL WORKSTATION NUMBERS MATCH' if not BAD else f'\nMISMATCH: {BAD}')
    sys.exit(1 if BAD else 0)


if __name__ == '__main__':
    main()
