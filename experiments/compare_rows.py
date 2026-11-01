"""Compare the rows of a re-run workspace with the stored rows in data/rows.

    python experiments/compare_rows.py work_rerun            # every rows file that has a stored counterpart
    python experiments/compare_rows.py work_rerun v12/kick   # only files under this sub-folder

For each file: episodes compared, episodes identical in every field, and episodes with the same crash outcome,
score and pulse count. Rows are bit-identical only on the machine that produced them (see README.md).
"""
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STORED = REPO/'data'/'rows'


def load(f):
    raw = gzip.open(f).read() if f.suffix == '.gz' else f.read_bytes()
    d = json.loads(raw)
    if isinstance(d, dict):
        d = d.get('rows')
    return d if isinstance(d, list) and d and isinstance(d[0], dict) and 'seed' in d[0] else None


def canon(r):
    return json.dumps({k: v for k, v in r.items() if k != 'controller'}, sort_keys=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    runs = Path(sys.argv[1]).resolve()/'runs'/'spike_steer'
    sub = sys.argv[2] if len(sys.argv) > 2 else ''
    total = [0, 0, 0]
    for f in sorted((runs/sub).rglob('*.json')):
        rel = f.relative_to(runs)
        g = STORED/(str(rel)+'.gz')
        if f.name == 'cases.json' or not g.exists():
            continue
        new, old = load(f), load(g)
        if new is None or old is None:
            continue
        ref = {(r['task'], r['seed']): r for r in old}
        pairs = [(r, ref[(r['task'], r['seed'])]) for r in new if (r['task'], r['seed']) in ref]
        same = sum(canon(a) == canon(b) for a, b in pairs)
        key = sum(all(a.get(k) == b.get(k) for k in ('flown', 'score', 'pulses')) for a, b in pairs)
        total = [total[0]+len(pairs), total[1]+same, total[2]+key]
        print(f'{str(rel):48s} {len(pairs):4d} episodes  identical {same:4d}  same flown/score/pulses {key:4d}')
    print(f'total {total[0]} episodes, identical {total[1]}, same flown/score/pulses {total[2]}')


if __name__ == '__main__':
    main()
