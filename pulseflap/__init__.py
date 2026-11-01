"""PulseFlap: code for "A Phase-Timed Pulse Channel to Robust Flapping-Wing Manoeuvres".

Importing this package puts the FlapSim clone (``third_party/FlapSim``, or ``$FLAPSIM_ROOT``) on ``sys.path`` so that
``flapsim`` and FlapSim's ``experiments.algorithms.*`` packages resolve to it. FlapSim is not part of this repository;
get it with ``scripts/setup_flapsim.sh``.
"""
import sys as _sys
from . import paths as _paths

if _paths.FLAPSIM.is_dir() and str(_paths.FLAPSIM) not in _sys.path:
    _sys.path.insert(0, str(_paths.FLAPSIM))

__version__ = '1.0.0'
