#!/usr/bin/env bash
# Sec. V-D, sustained disturbance (Table I, workstation x1.5 and x1.75): the training disturbance scaled x1.25, x1.5,
# x1.75 on 192 new cases each; classical, classical + pulse law, classical + ES, SNN, MLP + PPO. About 7,500 episodes,
# roughly 31 core-hours.
source "$(dirname "$0")/_common.sh"
for t in turbc125 turbc15 turbc175; do
    run blind16 run --test "$t" --workers "$WORKERS"
done
run ppo_blind14 --workers "$WORKERS" --sets v16_turbc125,v16_turbc15,v16_turbc175
done_msg
