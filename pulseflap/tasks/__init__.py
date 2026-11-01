"""Tasks, test cases, test conditions, plant variants and scoring (aliases of the experiment modules).

Tasks: evade, redirect, dash (bench9; 64 stress cases per task per set). Scoring: bench10.score10 (bench9 score with a
symmetric altitude term). The manoeuvre ratio divides a score by the classical controller's median for the same task
and direction class (pulseflap.eval.ratios).

    cases_nominal(task, base, n)        nominal blind cases (bench9.cases9), e.g. base 2_099_100_000, n 64
    cases_hit(task, base, n, ...)       cases with a 20 ms torque hit (blind12.cases12); HIT_SETS = blind12.TESTS
    cases_stress(task, setting)         further stress sets (blind13.cases13); STRESS_SETS = blind13.TESTS
    cases_plant(task, name)             vehicle variants (blind15.cases15); PLANTS = blind15.TESTS,
                                        apply_plant(env, PLANTS[name]) changes the simulated vehicle
    cases_sustained(task, name)         scaled sustained disturbance (blind16.cases16); SUSTAINED_SETS = blind16.TESTS
    run_episode(env, task, case, policy)   fly one episode, returns the result row (bench10.run_episode10)
    worker_init(source, dash_m=.15)     build the FlapSim environment of this process (bench.worker_init)
"""
from ..spike_steer import bench as _B, bench8 as _E, bench9 as _T, bench10 as _T10
from ..spike_steer.blind12 import cases12 as cases_hit, TESTS as HIT_SETS  # noqa: F401
from ..spike_steer.blind13 import cases13 as cases_stress, TESTS as STRESS_SETS  # noqa: F401
from ..spike_steer.blind15 import cases15 as cases_plant, apply_plant, TESTS as PLANTS  # noqa: F401
from ..spike_steer.blind16 import cases16 as cases_sustained, TESTS as SUSTAINED_SETS  # noqa: F401

TASKS = ('evade', 'redirect', 'dash')
cases_nominal = _T.cases9
klass = _T.klass
score = _T10.score10
run_episode = _T10.run_episode10
Episode = _T10.Episode
worker_init = _B.worker_init
load_signs = _E.load_signs
SOURCE = _E.SOURCE                    # relocated robot model, relative to the workspace
prepare_source = _B.prepare_source    # FlapSim's prepare_source


def env():
    """The environment built by worker_init in this process."""
    return _B._ENV
