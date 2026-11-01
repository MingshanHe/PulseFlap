"""Repository locations. Override with the environment variables FLAPSIM_ROOT and PULSEFLAP_WORK."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FLAPSIM = Path(os.environ.get('FLAPSIM_ROOT', REPO/'third_party'/'FlapSim')).resolve()
FLAPSIM_COMMIT = 'deab6c802ffc3b130d013fd0cb03b31773189bf3'
WEIGHTS = REPO/'weights'          # frozen controllers, mirrors artifacts/spike_steer/ of the original runs
ROWS = REPO/'data'/'rows'         # per-episode rows (*.json.gz), mirrors runs/spike_steer/
SUMMARIES = REPO/'data'/'summaries'
WORK = Path(os.environ.get('PULSEFLAP_WORK', REPO/'work')).resolve()
