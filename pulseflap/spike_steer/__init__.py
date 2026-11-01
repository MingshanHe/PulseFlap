"""The experiment modules used in the paper, kept close to their original form.

Each module still reads and writes paths relative to the current directory (``artifacts/spike_steer/...`` and
``runs/spike_steer/...``). Run them from a PulseFlap workspace (``python -m pulseflap.run <module> ...``), which builds
that layout from ``weights/`` and ``data/``. FlapSim itself (``flapsim`` and ``experiments.algorithms.*``) is imported
from the separate clone in ``third_party/FlapSim``.
"""
