# Frozen controllers

The folder mirrors `artifacts/spike_steer/` of the original runs; `pulseflap.workspace` copies it into the workspace
unchanged, so the experiment modules find every file at its original path.

| Path | Content |
|---|---|
| `plan_v9/baseline.json` | classical controller: pulse law (`law`: k_theta 8, k_q 0.4, threshold 0.5, width 45°), righting law (`upset_law`), per-class set-point schedules (`opponent`), ratio normalisers. The same file is `plan_v11/baseline.json`. |
| `plan_v11/snn/s1..s3/`, `plan_v11/homepc_snn/s4..s5/` | SNN seeds 1-5: `selected.json` (frozen ES centre used in the paper), `bc.json` (imitation start = "SNN, imitation only"), ES `history.jsonl`, `config.json`, `reselect.json` |
| `plan_v11/cls_es/s1/opp_g15.json`, `plan_v11/homepc/cls_es_s2/opp_g3.json` | classical + ES, seeds 1-2 (selected schedules) and their ES histories |
| `plan_v11/preregistration_v11.json`, `amendment_1.json` | training and selection protocol of the SNN and classical + ES |
| `plan_v9/snn/s*/selected.json` | the earlier plan-v9 SNN centres (secondary arms `snn9_s*` in the stress sets) |
| `ppo_direct/v7full/s*/centres/ckpt_g*.zip`, `reselect_s*.json` | MLP + PPO seeds 1-5: the selected stable-baselines3 checkpoint of each seed (seeds 2, 3, 5 selected their imitation start, g-1) and its selection record |
| `plan_v7/baseline.json` | the classical teacher of the PPO wrapper (copied to `runs/spike_steer/v7/baseline.json` in the workspace) |
| `plan_v6/signs.json` | attitude-to-motion direction signs used by every task |
| `plan_v4/r5/train/imitate_spec_s1.json` | source of the righting law used during SNN training |

An SNN spec holds the input weights over 11 delay taps (`W_in`, 25 inputs), 8 LIF hidden neurons, 2 spiking pulse
outputs and 2 graded set-point outputs (2,564 parameters); `pulseflap.controllers.SNN` runs it.
