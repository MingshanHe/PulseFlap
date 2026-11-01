#!/usr/bin/env bash
# Sec. V-I and V-J: network cost, motor energy, pulse timing (Fig. 7) and quantised / pruned weights.
# compute_cost, energy and pulse_phase fly 16 cases per task per arm (a few core-hours); damage flies 23,040 episodes
# (about 95 core-hours).
source "$(dirname "$0")/_common.sh"
run compute_cost run --per-task 16 --workers "$WORKERS"
run energy selftest
run energy run --per-task 16 --workers "$WORKERS"
run pulse_phase run --per-task 16 --workers "$WORKERS"
run damage run --workers "$WORKERS"
run damage report
done_msg
