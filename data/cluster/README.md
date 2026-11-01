# Cluster results (summaries)

These runs used FlapSim at commit deab6c8 on a compute cluster (Ubuntu 22.04, g++ 12.3, glibc 2.35, Intel Xeon),
892 core-hours in total. Single episodes differ between the cluster and the workstation, so cluster numbers are only
compared with each other. The per-episode rows of these runs are not included in this repository yet; the files here
hold every number the paper quotes from them.

| File | Experiment | Paper |
|---|---|---|
| `p4_hit_replication.json` | main 3 mN·m hit on 768 new cases (base 2_104_000_000), all arms | Table I (cluster hit column), Sec. V-B |
| `p5_sustained_disturbance.json` | sustained disturbance ×1.5 (base 2_104_400_000) and ×1.75 (base 2_104_600_000), 576 cases each; SNN-minus-PPO family comparison | Table I (cluster ×1.5, ×1.75), Sec. V-D |
| `p6_snn_plus_pulse_law.json` | SNN + pulse law on 192 nominal and 192 hit cases (the workstation's case seeds) | Table I (last row), Sec. V-H |
| `p3_tuned_pulse_law.json` | pulse law tuned jointly with the schedules at the SNN's ES budget, 3 seeds | Sec. V-B (last paragraph) |
| `q2_pulse_authority.json` | pulse-authority sweep: 9 plants (air density, motor voltage) × {hit, ×1.5}, 192 cases each | Fig. 5, Sec. V-E |
| `p1_vehicle_variants.json` | classical + pulse law and SNN on 11 vehicles, 192 cases each | Fig. 6, Sec. V-G |
| `es_refinement_pilot.json` | 19 short ES refinement runs of the frozen SNN on hit cases | Sec. V-J |
| `core_hours.json` | cost per experiment | Reproduce section of the README |
| `table_crashes.csv` | P4, P5 and P6 as one flat table: crashes, paired Δ against classical + pulse law with its 95 % CI, exact McNemar counts and p | Table I |

Naming in the JSON files: `probe` and `rule` both mean classical + pulse law; `snn_s<k>` is SNN seed k (`snn11_s<k>`
in the workstation rows), `ppo_s<k>` is MLP + PPO seed k. Δ is the paired difference of the manoeuvre ratio from
`pulseflap.eval.paired` (negative = better than the reference), given as `[value, ci_low, ci_high]`; McNemar entries
are `[arm-only crashes, reference-only crashes, exact two-sided p]`.

The code of the SNN + pulse law arm, the tuned-law search and the authority sweep driver ran only on the cluster and
is not part of this repository. `pulseflap.controllers.SNNPlusPulseLaw` is a re-implementation from the
description, not checked against these numbers.
