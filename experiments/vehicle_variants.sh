#!/usr/bin/env bash
# Sec. V-G (workstation part): the frozen controllers on 13 modified vehicles plus the amendment doses, 192 cases each.
# About 35,000 episodes, roughly 150 core-hours. (Classical + pulse law on the vehicles, Fig. 6, is a cluster result.)
source "$(dirname "$0")/_common.sh"
PLANTS="nominal lower125 lower_com wing_mass130 span95 span105 chord90 asym97 feather80 density90 volt90 kt90 lag150 unit_b lower110"
for t in $PLANTS; do
    run blind15 run --test "$t" --arms "$MAIN_ARMS" --workers "$WORKERS"
done
for t in nominal lower125 lower_com wing_mass130 span95 chord90 asym97 density90 volt90 kt90 lag150 lower110; do
    run ppo_blind14 --workers "$WORKERS" --sets "v15_$t"      # one vehicle per call (the plant is set per worker)
done
done_msg
