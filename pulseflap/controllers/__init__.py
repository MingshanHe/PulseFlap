"""The controllers compared in the paper (thin aliases of the experiment modules; behaviour is defined there).

Every controller is a policy object with ``reset(case)`` and ``act(env, now, t, x, cmd, rel) -> (pulse, set_points)``,
flown by ``pulseflap.tasks.run_episode``. All of them share the PID reflex layer, the muscle layer that executes a
pulse (3 ticks at maximum motor target) and the classical behaviour outside manoeuvres.

    PulseLaw              hand-designed pulse law u = -(k_theta*theta + k_q*q_slow), fired inside +-width/2 of 135 or
                          315 deg crank phase; the phase comes from a phase-locked loop on the pitch gyro
                          (pilot.Steer / pilot_v3.SteerV3; the estimator is FlapSim's WingbeatPhase)
    Classical             classical controller: open-loop set-point schedules per direction class, pulse law
                          suppressed while a schedule segment is active (bench9.Classical9). Classical + ES is the
                          same class with the ES-tuned schedules (weights/plan_v11/.../opp_g*.json)
    ClassicalPulseLaw     classical + pulse law: the pulse law keeps firing during manoeuvres
                          (ablate14.ClassicalUnsuppressed)
    SNN                   the spiking command layer (descend8.Learned8, network descend8.Net8)
    SNNNoPulse, SNNNoRef  channel ablations: pulses suppressed / set-points zeroed while in charge
    PPOPolicy             MLP + PPO policy wrapper (v7ppo.PpoPolicy9; flown through v7ppo.PpoLayer9Env, see
                          ppo_blind14.job)
    SNNPlusPulseLaw       SNN + pulse law (see its docstring: reconstructed, not the cluster code)

``load_arms()`` returns the frozen arms of the paper: classical, cls_es_s1-2, snn11_s1-5, snn9_s1-5, snn_bc_s1.
"""
import math
import numpy as np
from ..spike_steer import pilot_v3 as _V, bench as _B, bench9 as _T, descend8 as _D8, ablate14 as _A14, v7ppo as _P
from ..spike_steer.blind12 import load_arms  # noqa: F401  (reads artifacts/spike_steer/ in the workspace)

PulseLaw = _V.SteerV3
Classical = _T.Classical9
ClassicalPulseLaw = _A14.ClassicalUnsuppressed
SNN = _D8.Learned8
SNNNetwork = _D8.Net8
SNNNoPulse = _A14.NoPulse
SNNNoRef = _A14.NoRef
PPOPolicy = _P.PpoPolicy9
PPOEnv = _P.PpoLayer9Env


class SNNPlusPulseLaw(_D8.Learned8):
    """SNN + pulse law: the SNN is in charge during the manoeuvre as usual, and a pulse also starts whenever the
    hand-designed pulse law fires (fire = SNN spike OR pulse law).

    Reconstructed from the description of the cluster experiment (P6). The cluster code was not available when this
    repository was assembled, so this class has NOT been checked against the cluster rows; treat its results as a
    re-implementation, not as a reproduction of Table I's "SNN + pulse law" row.
    """

    def act(self, env, now, t, x, cmd, rel):
        fire, ref = self.net_step(t, x)
        law_pulse, _ = self.steer.step(env, now, t)
        up_pulse, _ = self.steer_up.step(env, now, t)
        on = bool(np.any(cmd))
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        in_charge = on or (self.off_t is not None and t-self.off_t < self.tail)
        if not self.upset and not in_charge and math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) > self.tilt_thr:
            self.upset = True
        if self.upset:
            law_pulse = up_pulse
        if in_charge:
            if fire and self.twitch == _B.TWITCH-1:
                if self.last_onset is not None and t-self.last_onset < self.refr:
                    fire, self.twitch = False, 0
                    self.cancelled += 1
                else:
                    self.last_onset = t
                    self.onsets += 1
            return bool(fire or law_pulse), ref
        return bool(law_pulse), np.zeros(_D8.N_G)
