#!/usr/bin/env bash
# How the frozen controllers in weights/ were trained (Sec. IV). Listed for reference: this was not re-run when the
# repository was assembled, and ES runs are bit-reproducible only on the machine that produced them.
#
# SNN (plan v11), per seed s = 1..5: imitation of the classical controller over whole episodes (BPTT with a surrogate
# gradient, descend9/descend10 'imitate'), then OpenAI-ES on the manoeuvre score (16 generations, 8 antithetic pairs,
# 24 episodes per candidate, 6,528 episodes), then selection of the best validation centre:
#     python -m pulseflap.run descend12 finetune --seed s --workers 18
#     python -m pulseflap.run descend12 reselect --seed s --n-select 64 --workers 18
#     python -m pulseflap.run descend12 evaluate --seed s --n-test 64 --test-base 2099100000 --workers 18
#   finetune starts from runs/spike_steer/v11/imitate_spec_s<s>.json; the imitation starts are kept as
#   weights/plan_v11/{snn,homepc_snn}/s<s>/bc.json (the same spec with a 'generation' field added), the ES histories
#   as .../history.jsonl and the selected centres as .../selected.json.
#
# Classical + ES (28 schedule parameters, same budget, sigma 0.05), seeds 1-2:
#     python -m pulseflap.run descend_cls finetune --seed s --workers 18
#     python -m pulseflap.run descend_cls reselect --seed s --workers 18
#     python -m pulseflap.run descend_cls evaluate --seed s --workers 18
#
# MLP + PPO (57-64-64-3, 1,020 episodes, imitation start), seeds 1-5; needs the descend9 collect logs:
#     V7PPO_PROFILE=full python -m pulseflap.run v7ppo bc --seed s
#     V7PPO_PROFILE=full python -m pulseflap.run v7ppo finetune --seed s --budget-episodes 1020 --n-envs 18
#     V7PPO_PROFILE=full python -m pulseflap.run v7ppo reselect --seed s --workers 18
echo "training.sh only documents the training commands; read the file."
