#!/usr/bin/env bash
# Sec. V-A, nominal manoeuvres: 192 nominal cases (bench9 cases, base 2_099_100_000), every frozen arm.
# 18 arms x 192 episodes, about 15 core-hours. Rows go to $WORK/runs/refly/nominal and are compared on the fly.
# (The original rows came out of the training pipeline's evaluate stages, descend12 / descend_cls / v7ppo.)
source "$(dirname "$0")/_common.sh"
python -W ignore experiments/refly.py --work "$WORK" --set nominal --n 64 --workers "$WORKERS" \
    --arms classical,cls_es_s1,cls_es_s2,probe,$SNN,snn_bc_s1,snn_bc_s3,snn_bc_s4,snn_bc_s5,mlp_ppo_s1,mlp_ppo_s2,mlp_ppo_s3,mlp_ppo_s4,mlp_ppo_s5 || true
