#!/usr/bin/env bash
# Sec. V-B and Fig. 4: the main 3 mN·m hit (Table I, workstation), its replication, hit dose / timing / axis sets,
# sustained-disturbance x1.25-x3, sensor and plant stress sets; classical + pulse law and MLP + PPO on every set.
# About 49,000 episodes, roughly 205 core-hours.
source "$(dirname "$0")/_common.sh"
for t in kick kick_rep kick2 kick45 kick_early kick_late ood; do
    arms=$MAIN_ARMS; [ "$t" = kick ] && arms=$arms,snn_bc_s1
    run blind12 run --test "$t" --arms "$arms" --workers "$WORKERS"
done
for t in kick_pre kick_double kick_rollaxis kick_pitchaxis turb125 turb15 turb2 turb3 noise5 delay3; do
    run blind13 run --test "$t" --arms "$MAIN_ARMS" --workers "$WORKERS"
done
run ablate14 run --workers "$WORKERS"          # classical + pulse law on nominal/kick/turb15, channel ablations, imitation starts
run ablate14 unsup_all --workers "$WORKERS"    # classical + pulse law on the other stress sets
run ppo_blind14 --workers "$WORKERS" --sets kick,kick_rep,kick2,kick45,kick_early,kick_late,ood,kick_pre,kick_double,kick_rollaxis,kick_pitchaxis,turb125,turb15,turb2,turb3,noise5,delay3
done_msg
