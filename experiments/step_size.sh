#!/usr/bin/env bash
# Sec. V-F, integration step: the workstation's 192 hit and 192 x1.5 cases at 2.5, 5 and 10 us, classical,
# classical + pulse law, SNN and MLP + PPO. About 97 core-hours. Needs the case lists of v12/kick and v16/turbc15,
# which the fresh workspace keeps.
source "$(dirname "$0")/_common.sh"
run stepsize_q1 selftest --sets kick,turbc15 --n 2 --workers "$WORKERS" || true   # dt5 subset against the stored rows
for v in dt5 dt2p5 dt10; do
    for s in kick turbc15; do
        run stepsize_q1 run --variant "$v" --set "$s" --workers "$WORKERS"
    done
done
done_msg
