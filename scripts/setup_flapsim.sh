#!/usr/bin/env bash
# Fetch FlapSim at the commit used for the paper into third_party/FlapSim (git-ignored).
#
#   bash scripts/setup_flapsim.sh            # sparse checkout: the simulator, its experiment packages, robot model
#   FULL=1 bash scripts/setup_flapsim.sh     # full checkout of the commit (about 1 GB, mostly FlapSim's own artifacts)
#
# PulseFlap puts third_party/FlapSim on sys.path when the pulseflap package is imported, so no install step is needed.
# To use a FlapSim checkout elsewhere, set FLAPSIM_ROOT instead of running this script.
set -euo pipefail

REPO_URL="${FLAPSIM_URL:-https://github.com/MingshanHe/FlapSim.git}"
COMMIT="deab6c802ffc3b130d013fd0cb03b31773189bf3"   # branch research/original-env-snn-neat
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/FlapSim"

if [ -d "$DEST/.git" ]; then
    echo "FlapSim already present at $DEST"
else
    mkdir -p "$ROOT/third_party"
    git clone --filter=blob:none --no-checkout "$REPO_URL" "$DEST"
fi
cd "$DEST"
if [ -z "${FULL:-}" ]; then
    git sparse-checkout init --no-cone
    git sparse-checkout set '/flapsim/' '/experiments/' '/artifacts/pid_airborne_20261008/' '/pyproject.toml' '/README.md'
fi
git -c advice.detachedHead=false checkout --quiet "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT"
echo "FlapSim checked out at $(git rev-parse --short HEAD) in $DEST"

# Check that the modules PulseFlap needs import (the native C++ backend is compiled on first use).
cd "$ROOT"
python - <<'EOF'
import pulseflap
from pulseflap import paths
import flapsim, experiments.algorithms.neuro_gates.gates  # noqa: F401
assert str(paths.FLAPSIM) in flapsim.__file__, flapsim.__file__
print('ok: flapsim imported from', paths.FLAPSIM)
EOF
