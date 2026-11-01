# Data

```
rows/        per-episode result rows of every workstation run (gzip JSON), mirrors runs/spike_steer/ of the original runs
summaries/   plan summaries, preregistrations, reports (Holm, damage, compute cost, energy, pulse phase, step size)
cluster/     summary tables of the cluster runs (the cluster's per-episode rows are not included)
figures/     the numbers plotted in Figs. 4 and 7, extracted from rows/ and summaries/
```

`pulseflap.workspace` decompresses `rows/` into `work/runs/spike_steer/` and copies `summaries/` next to the weights
in `work/artifacts/spike_steer/`, which is where the experiment modules read them.

## Rows

A rows file is a JSON list with one object per episode (some `evaluate_*.json` files wrap the list as `{"rows": [...]}`).
Episodes are identified by `(task, seed)`; the same key on two arms of one set is the same test case, which is what
makes every comparison paired.

| Field | Meaning |
|---|---|
| `task`, `seed`, `cls` | task (evade, redirect, dash), case seed, direction class (pitch, roll, diag) |
| `dir_deg`, `dir2_deg`, `t0` | commanded direction, second direction (redirect), command time (s) |
| `flown`, `termination` | `false` = crashed; reason (`time_limit` for a completed episode, otherwise tilt, low height, height escape, ...) |
| `score` | manoeuvre score (bench10, s, lower is better); `score_bench9` before the symmetric altitude term |
| `pulses` | pulses started during the episode |
| `t_reach`, `t_settle`, `goal_t`, `cross_m`, `max_along_m`, `height_loss_cm`, `alt_dev_cm`, `tilt_peak_deg` | score components and diagnostics |
| `controller` | arm name |

Arm names: `classical`; `cls_es_s1`, `cls_es_s2` (classical + ES); `cls_unsup` or `probe` (classical + pulse law);
`snn11_s1..5` (SNN, the paper's controller); `snn11_s<k>_nopulse`, `snn11_s<k>_noref` (channel ablations);
`snn_bc_s1..5` (imitation only); `mlp_ppo_s1..5` (MLP + PPO); `snn9_s1..5` (earlier SNN centres, not in the paper).

| Folder | Content | Paper |
|---|---|---|
| `v11/` | nominal blind set (bench9 cases, base 2_099_100_000, 64 per task): classical (`test_opponent.json`), SNN and imitation starts, classical + ES | Sec. V-A |
| `v12/<set>/` | hit sets `kick` (main 3 mN·m hit, base 2_099_300_000), `kick_rep` (replication), `kick2`, `kick45`, `kick_early`, `kick_late`, and `ood` (lower body +12.5 %) | Table I, Fig. 4 |
| `v13/<set>/` | `kick_pre`, `kick_double`, `kick_rollaxis`, `kick_pitchaxis`, `turb125`, `turb15`, `turb2`, `turb3`, `noise5`, `delay3` | Fig. 4 |
| `v14/` | classical + pulse law (`<set>__cls_unsup.json`), channel ablations and imitation starts on `kick`, and every MLP + PPO row (`<set>__mlp_ppo_s<k>.json`, with `v15_`/`v16_` prefixes for the vehicle and sustained sets) | Table I, Fig. 4 |
| `v15/<plant>/` | vehicle variants, 192 stress cases each (base 2_103_000_000) | Sec. V-G |
| `v16/<set>/` | sustained disturbance ×1.25, ×1.5, ×1.75 (`turbc125`, `turbc15`, `turbc175`), including classical + pulse law | Table I, Sec. V-D |
| `q1/<dt>/<set>/` | step size 2.5 and 10 µs; `dt5` holds the workstation's own 5 µs PPO and SNN re-runs | Sec. V-F |
| `damage/` | 8/6/4-bit weights and 10/20/30 % pruning of the SNN and PPO layers, nominal and main hit | Sec. V-J |

`cases.json` in a set folder is the list of test cases of that set (inputs, not results).

## Provenance

The rows were flown on two machines, labelled in the files as `HomePC` (a 20-core workstation, WSL2, g++ 13.3,
glibc 2.39) and `CloudPC-2` (a cloud container). Both ran FlapSim at commit deab6c8. For every arm except MLP + PPO,
the two machines produce bit-identical episodes, so the paper calls both "the workstation". MLP + PPO episodes are not
bit-identical across machines (float32 torch inference): the PPO rows of the nominal and main hit sets
(`v14/nominal__*`, `v14/kick__*`) come from CloudPC-2, every other PPO row from HomePC, and `q1/dt5/kick` holds HomePC's
own PPO rows on the main hit cases.

| Rows | Machine |
|---|---|
| `v11` SNN seeds 1-3 and imitation starts 1, 3, classical + ES seed 1 | CloudPC-2 |
| `v11` classical (`test_opponent.json`) | both, identical |
| `v11` SNN seeds 4-5 and imitation starts 4-5, classical + ES seed 2 | HomePC |
| `v12`, `v13`, `v14` (except PPO on sets other than nominal and kick) | CloudPC-2 |
| `v14` PPO on all other sets, `v15`, `v16`, `q1`, `damage` | HomePC |

## Cluster

`cluster/` holds the cluster results that the paper quotes (Table I cluster columns, Figs. 5 and 6, the tuned pulse
law and SNN + pulse law). The cluster ran the same FlapSim commit on Ubuntu 22.04 with g++ 12.3 and glibc 2.35; its
episodes differ from the workstation's, so its numbers are compared only with each other. Its per-episode rows are not
included in this repository yet. See `cluster/README.md`.
