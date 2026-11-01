"""Check that the figure inputs in data/figures agree with the per-episode rows and onset records.

    python figures/check_figure_data.py

Fig. 4: every crash count in robustness_crashes.json and pulse_law_crashes.json is recounted from data/rows.
Fig. 7: the pulse-phase shares quoted in Sec. V-I are recomputed from the full-precision onsets
(data/summaries/pulse_phase/onsets.json.gz).
"""
import gzip
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO/'data'/'rows'


def rows(p):
    d = json.loads(gzip.open(R/(p+'.gz')).read())
    return d if isinstance(d, list) else d['rows']


def crashes(p):
    return sum(not r['flown'] for r in rows(p))


def main():
    S = json.loads((REPO/'data/figures/robustness_crashes.json').read_text(encoding='utf-8'))['sets']
    PR = json.loads((REPO/'data/figures/pulse_law_crashes.json').read_text(encoding='utf-8'))['per_set']
    bad = 0
    print('Fig. 4: crashes of 192 per set (classical, classical + ES, SNN seeds, PPO seeds, classical + pulse law)')
    for s in S:
        t = PR[s['label']]['set']
        plan = 'v12' if (R/'v12'/t).exists() else 'v13'
        got = dict(classical=crashes(f'{plan}/{t}/rows_classical.json'),
                   cls_es=[crashes(f'{plan}/{t}/rows_cls_es_s{k}.json') for k in (1, 2)],
                   snn=[crashes(f'{plan}/{t}/rows_snn11_s{k}.json') for k in range(1, 6)],
                   ppo=[crashes(f'v14/{t}__mlp_ppo_s{k}.json') for k in range(1, 6)],
                   probe=crashes(f'v14/{t}__cls_unsup.json'))
        exp = dict(classical=s['classical'], cls_es=s['cls_es'], snn=s['snn'], ppo=s['ppo'], probe=PR[s['label']]['probe'])
        ok = got == exp
        bad += not ok
        print(f"  {s['label']:20s} {t:15s} {'ok' if ok else f'MISMATCH rows {got} figure {exp}'}")

    print('Fig. 7: pulse onsets within +-45 deg and +-22.5 deg of 135 / 315 deg (nominal and main hit cases)')
    on = json.loads(gzip.open(REPO/'data/summaries/pulse_phase/onsets.json.gz').read())
    fig = json.loads((REPO/'data/figures/cost_energy_phase.json').read_text())['phase']
    for name, sel, key in (("SNN's own pulses, 5 seeds", lambda x: x['arm'] != 'classical' and x['src'] == 'net', 'snn_own_pulses_deg'),
                           ('pulse law (classical)', lambda x: x['arm'] == 'classical' and x['src'] == 'law', 'classical_law_deg')):
        a = np.array([x['phase_deg'] for x in on if sel(x)])
        d = np.min([np.abs((a-c+180) % 360-180) for c in (135, 315)], 0)
        same = len(a) == len(fig[key]) and np.allclose(np.round(a, 2), fig[key])
        bad += not same
        print(f'  {name:26s} {len(a):5d} pulses  {100*np.mean(d <= 45):5.1f} %  {100*np.mean(d <= 22.5):5.1f} %'
              f'   figure data {"ok" if same else "MISMATCH"}')
    print('all figure data consistent' if not bad else f'{bad} mismatches')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
