# Sourced by the experiment scripts. Runs a paper module in a fresh workspace (stored rows not copied, so every arm
# is flown again). Settings: WORKERS (default: all cores), WORK (default: work_rerun).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
WORKERS="${WORKERS:-$(nproc)}"
WORK="${WORK:-work_rerun}"
export MUJOCO_GL=
run() { python -W ignore -m pulseflap.run --fresh --work "$WORK" "$@"; }
SNN=snn11_s1,snn11_s2,snn11_s3,snn11_s4,snn11_s5
MAIN_ARMS=classical,cls_es_s1,cls_es_s2,$SNN
done_msg() { echo "done; compare with the stored rows:  python experiments/compare_rows.py $WORK"; }
