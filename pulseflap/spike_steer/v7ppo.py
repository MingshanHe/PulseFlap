"""Plan v7 quick check (evade / redirect / dash), PPO reference arm: the "ES search of the SNN" of descend9 is replaced by
"PPO training of an MLP with the same interface"; every other step (cases, gating, objective, centre selection, blind test,
report format) is taken unchanged from bench9 / descend9.

Correspondence (descend9 -> here):
  per-tick logic  bench9.Episode (command channel, redirect mid-course turn, evade random t0, score9 scoring) is reused
            as is: this module only provides a policy object PpoPolicy9 with the same interface as descend8.Learned8
            (reset / act); gating (in charge during the command + 1 s tail), 25 deg righting reflex and 37 ms refractory
            period match Learned8; the only difference is that "decisions come from an MLP, once every 10 ms".
  observation  57-dim = the 16-dim rectified IMU features (first 16 of bench8.features8) of three frames (now, 10 ms ago,
            20 ms ago) + bias + 8-dim rectified command channel (slot 0-1 = evade, slot 2-3 = dash / redirect, as in
            features8). Identical to v6ppo.
  action    3-dim in [-1, 1]: pulse (>0 starts a 3-tick pulse, subject to the refractory period), roll set-point, pitch
            set-point (x 0.7 rad).
  start     bc: descend9 collect classical trajectories aggregated in 10 ms windows, MSE regression onto the MLP
            (= centre number -1, flown in the blind test as its own arm 'mlp_bc').
  search    ES 20 generations x 17 candidates x (4 evade + 4 redirect + 4 dash) = 4,080 episodes -> PPO budget 1/4 =
            1,020 episodes; terminal reward = -3 x weight(1/3) x (score / direction-class normaliser + tilt penalty) =
            -(descend9 per-case objective), no other shaping; cases drawn from the ES pool
            (cases9(train, offset 100_000 + gen*100), gen < 20); a checkpoint every 1/4 of the budget -> 4 + bc = 5 centres
            (matching the SNN's g-1, 4, 9, 14, 19).
  selection 16 cases / task (offset 60_000), no crash and true peak tilt < 70 deg, lowest objective (same rule as
            descend9.stage_reselect).
  blind test bench9 BASES['test'], 16 cases / task; opponent rows from runs/spike_steer/v7/test_opponent.json if present
            (produced on another workstation), otherwise generated locally; the report uses descend9.report and writes
            evaluate_<arm>_s<seed>.json (arm = mlp_ppo / mlp_bc) with the same fields as descend9.
  One implementation difference from v6ppo: every episode end is treated as terminated (time_limit is no longer treated
  as truncated), so PPO does not add a value estimate to the terminal reward.

    python -m pulseflap.spike_steer.v7ppo selftest
    python -m pulseflap.spike_steer.v7ppo bc       --seed 1
    python -m pulseflap.spike_steer.v7ppo finetune --seed 1 --budget-episodes 1020 --n-envs 18
    python -m pulseflap.spike_steer.v7ppo reselect --seed 1 --workers 18
    python -m pulseflap.spike_steer.v7ppo evaluate --seed 1 --workers 18
Input: runs/spike_steer/v7/{baseline.json, collect_train.npz[, test_opponent.json]} (descend9 output directory; override
with the PLAN_V7 environment variable).
Output: runs/ppo_direct/v7ppo/{s<seed>/ckpt_g*.zip, reselect_s<seed>.json, evaluate_<arm>_s<seed>.json, test_opponent.json}.
Full round: V7PPO_PROFILE=full (output runs/ppo_direct/v7full/, 9 centres, selection 32 cases / task, blind test base
2_098_300_000 with 32 cases / task, opponent rows test_opponent_v7full.json).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from experiments.algorithms.neuro_gates import gates
from experiments.algorithms.snn_pid_compare.common import DT
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import descend8 as D8
from . import descend9 as D9
from . import pilot_v3 as V

# Profiles: quick = v7 quick check; full = v7 full round (preregistration_v7full.json: 9 centres, selection 32 cases/task, fresh blind set 32 cases/task).
PROFILES = dict(
    quick=dict(out="runs/ppo_direct/v7ppo", n_centres=4, n_select=D9.N_SELECT, test_base=T.BASES["test"], n_test=D9.N_TEST, opp_name="test_opponent.json"),
    full=dict(out="runs/ppo_direct/v7full", n_centres=8, n_select=32, test_base=2_098_300_000, n_test=32, opp_name="test_opponent_v7full.json"),
    # v8 (preregistration_v8.json): only re-tests the frozen checkpoints of the v7 full round (evaluate --from-dir runs/ppo_direct/v7full), fresh blind set 64 cases/task
    v8=dict(out="runs/ppo_direct/v8", n_centres=8, n_select=32, test_base=2_098_500_000, n_test=64, opp_name="test_opponent_v8.json"),
)
PROFILE = os.environ.get("V7PPO_PROFILE", "quick")
P = PROFILES[PROFILE]
OUT = Path(P["out"])
PLAN = Path(os.environ.get("PLAN_V7", "runs/spike_steer/v7"))
TICKS = 5                                      # one decision every 10 ms
FRAMES = (0, 5, 10)                            # lags of the three frames (ticks)
N_IMU, N_CMD = 16, 8
N_OBS = N_IMU * len(FRAMES) + 1 + N_CMD        # 57
N_GENS = 20                                    # descend9 default generations (size of the case pool)
ES_EPISODES = N_GENS * (D9.ES["pop"] + 1) * D9.ES["cases"] * len(T.TASKS)    # 20 × 17 × 12 = 4 080
MEAN_DECISIONS = sum(T.SECONDS[t] for t in T.TASKS) / len(T.TASKS) / DT / TICKS   # 600
N_CENTRES = P["n_centres"]                     # number of PPO checkpoints (+ bc: quick 5 centres, full 9)
MACHINE = os.environ.get("MACHINE", socket.gethostname())


def load_plan():
    E.load_signs()
    B.DASH_M["value"] = .15
    return json.loads((PLAN / "baseline.json").read_text())


# ----------------------------------------------------------------------------------------- learned layer (the policy of bench9.Episode)
class PpoPolicy9:
    """Learned layer with the same interface and gating as descend8.Learned8; the decision (3-dim action) is set from outside once at the start of every 10 ms window."""

    def __init__(self, base: dict, variant: str = "intact") -> None:
        self.law, self.upset_law, self.variant = base["law"], base["upset_law"], variant

    def reset(self, c):
        self.steer = V.SteerV3(None, self.law, False)
        self.steer.reset()
        self.steer_up = V.SteerV3(None, self.upset_law, False)
        self.steer_up.reset()
        self.tilt_thr, self.tail, self.refr = math.radians(D8.UPSET_TILT_DEG), D8.TAIL_S, D8.REFRACTORY_S
        self.upset, self.off_t, self.was_on, self.last_onset, self.in_charge = False, None, False, None, False
        self.onsets = self.cancelled = 0
        self.twitch = 0
        self.ref = np.zeros(2)
        self.hist = collections.deque(maxlen=max(FRAMES))      # features8 of earlier ticks (for the lagged frames)
        self.action, self.first = np.array([-1., 0., 0.]), False

    def act(self, env, now, t, x, cmd, rel):
        self.hist.append(np.asarray(x, np.float32))
        law_pulse, _ = self.steer.step(env, now, t)
        up_pulse, _ = self.steer_up.step(env, now, t)
        on = bool(np.any(cmd))
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        self.in_charge = on or (self.off_t is not None and t - self.off_t < self.tail)
        if not self.upset and not self.in_charge and math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) > self.tilt_thr:
            self.upset = True
        if self.upset:
            law_pulse = up_pulse
        if self.in_charge and t >= gates.HANDOVER_T:
            if self.first:                                   # first tick of the window: read the action
                a = self.action
                self.ref = B.REF_MAX * np.clip(np.asarray(a[1:3], float), -1., 1.)
                if self.variant == "noref":
                    self.ref = np.zeros(2)
                if float(a[0]) > 0. and self.variant != "nopulse":
                    if self.last_onset is not None and t - self.last_onset < self.refr:
                        self.cancelled += 1
                    else:
                        self.last_onset, self.onsets, self.twitch = t, self.onsets + 1, B.TWITCH
            fire = False
            if self.twitch > 0:
                self.twitch -= 1
                fire = True
            return fire, self.ref
        self.twitch, self.ref = 0, np.zeros(2)
        return bool(law_pulse), np.zeros(2)


# ----------------------------------------------------------------------------------------- gym environment
class PpoLayer9Env(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, *, variant: str = "intact", task_weights=(1., 1., 1.)) -> None:
        super().__init__()
        self.base = load_plan()
        if not (E.SOURCE / "robot_upper7_lower8.xml").exists():
            B.prepare_source(E.SOURCE)
        B.worker_init(str(E.SOURCE), .15)
        self.env = B._ENV
        self.norm = self.base["normalisers"]
        self.pol = PpoPolicy9(self.base, variant)
        w = np.asarray(task_weights, float)
        self.task_p = w / w.sum()
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float32)
        self.ep = None

    # ---- cases: the same pool as ES ----------------------------------------------
    def _draw_case(self, task: str) -> dict:
        gen, i = int(self.np_random.integers(0, N_GENS)), int(self.np_random.integers(0, D9.ES["cases"]))
        return T.cases9(task, T.BASES["train"], D9.ES["cases"], offset=D9.OFF["es"] + gen * 100)[i]

    # ---- observation ---------------------------------------------------------------
    def _cmd_next(self) -> np.ndarray:
        """The command channel that the next bench9.Episode.tick will produce (same rules)."""
        s, c = self.ep.s, self.ep.s["c"]
        rel = self.env.air.steps * DT - c["t0"]
        cmd = np.zeros(4)
        if rel >= 0 and not s["goal_done"]:
            phi = c["dir2_deg"] if (s["task"] == "redirect" and s["phase"] == 2) else c["dir_deg"]
            k = 0 if s["task"] == "evade" else 2
            cmd[k], cmd[k + 1] = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        return cmd

    def _observation(self) -> np.ndarray:
        x = np.asarray(E.features8(self.env.last_reading, self._cmd_next()), np.float32)
        h = self.pol.hist

        def lag(k):                                          # IMU features k ticks ago (the earliest frame if there are fewer)
            return (h[-k] if len(h) >= k else (h[0] if h else x))[:N_IMU]
        return np.concatenate([x[:N_IMU]] + [lag(k) for k in FRAMES[1:]] + [x[N_IMU:D8.N_IN]]).astype(np.float32)

    # ---- episode ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        task = options.get("task") or str(self.np_random.choice(list(T.TASKS), p=self.task_p))
        case = options.get("case") or self._draw_case(task)
        self.ep = T.Episode(self.env, task, case, self.pol)   # reset the simulation, the low-level PID and the learned layer
        return self._observation(), dict(task=task, seed=case["seed"], dir_deg=case["dir_deg"], dir2_deg=case.get("dir2_deg"))

    def step(self, action):
        self.pol.action = np.clip(np.asarray(action, np.float32), -1., 1.)
        for k in range(TICKS):
            self.pol.first = (k == 0)
            self.ep.tick()
            if self.ep.s["done"]:
                break
        s = self.ep.s
        reward, done = 0., bool(s["done"])
        info = dict(task=s["task"], in_charge=self.pol.in_charge, goal_done=s["goal_done"])
        if done:
            res = self.ep.finish()
            res.update(onsets=self.pol.onsets, cancelled=self.pol.cancelled)
            pen = max(0., (res["tilt_peak_deg"] or 0.) - D9.SOFT_TILT_DEG) * D9.PENALTY_PER_DEG
            res["objective"] = res["score"] / self.norm[s["task"]][res["cls"]] + pen
            reward = -D9.WEIGHTS[s["task"]] * 3. * res["objective"]       # weight 1/3 x 3 = 1: reward = -(per-case objective)
            info["metrics"] = res
            info["is_success"] = bool(res["flown"] and res["goal_t"] is not None)
        return self._observation(), float(reward), done, False, info


# ----------------------------------------------------------------------------------------- tools / bc
def seed_dir(seed: int) -> Path:
    d = OUT / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_policy_model(env, seed: int):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
               ent_coef=0.0, policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.5}, verbose=1, seed=seed, device="cpu", tensorboard_log=str(OUT / "tb"))


def bc_dataset(path: Path):
    """descend9 collect per-tick logs -> (observation, target) per 10 ms window: target = [pulse onset in window ? 1 : -1, mean window set-point / 0.7]."""
    d = np.load(path)
    X, L = d["X"], d["lengths"]
    obs, tgt = [], []
    for e in range(X.shape[0]):
        x = X[e, :L[e]]
        imu, bias, cmd8 = x[:, :N_IMU], x[:, N_IMU:N_IMU + 1], x[:, N_IMU + 1:D8.N_IN]
        pulse, ref = x[:, D8.N_IN], x[:, D8.N_IN + 2:D8.N_IN + 4]
        for s in range(max(FRAMES), len(x) - TICKS, TICKS):
            frames = [imu[s - k] for k in FRAMES]
            win = slice(s, s + TICKS)
            onset = bool(pulse[win].max() > 0 and pulse[s - 1] == 0)
            obs.append(np.concatenate(frames + [bias[s], cmd8[s]]))
            tgt.append(np.r_[1. if onset else -1., ref[win].mean(0) / B.REF_MAX])
    return np.asarray(obs, np.float32), np.asarray(tgt, np.float32), d["tasks"]


def bc(a):
    import torch
    torch.set_num_threads(2)
    obs, tgt, tasks = bc_dataset(PLAN / "collect_train.npz")
    print(f"bc data: {len(obs)} windows from {len(tasks)} episodes {dict(zip(*np.unique(tasks, return_counts=True)))}, "
          f"pulse-window rate {np.mean(tgt[:, 0] > 0):.3f}, |ref| mean {np.abs(tgt[:, 1:]).mean(0).round(3).tolist()}", flush=True)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(obs))
    va, tr = idx[:len(obs) // 10], idx[len(obs) // 10:]
    model = make_policy_model(PpoLayer9Env(), a.seed)
    pol = model.policy
    net = torch.nn.Sequential(pol.mlp_extractor.policy_net, pol.action_net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Yt = torch.tensor(obs), torch.tensor(tgt)
    w = torch.tensor([a.pulse_weight, 5., 5.])          # MSE version of descend8/9: BCE(pos weight 3) + 5 x ref MSE
    best, state = None, None
    for epoch in range(a.epochs):
        rng.shuffle(tr)
        losses = []
        for b in range(0, len(tr), 256):
            i = torch.tensor(tr[b:b + 256])
            loss = (w * (net(Xt[i]) - Yt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            iv = torch.tensor(va)
            pv = net(Xt[iv])
            vloss = float((w * (pv - Yt[iv]) ** 2).mean())
            acc = float(((pv[:, 0] > 0) == (Yt[iv, 0] > 0)).float().mean())
            rmse = torch.sqrt(((pv[:, 1:] - Yt[iv, 1:]) ** 2).mean(0)).tolist()
        print(json.dumps(dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=vloss, pulse_acc=acc, ref_rmse=[round(r, 4) for r in rmse])), flush=True)
        if best is None or vloss < best:
            best, state = vloss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    d = seed_dir(a.seed)
    model.save(d / "ckpt_g-1")
    print(f"wrote {d / 'ckpt_g-1.zip'} (best val loss {best:.4f})", flush=True)


# ----------------------------------------------------------------------------------------- finetune
def finetune(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    d = seed_dir(a.seed)
    steps = int(a.budget_episodes * MEAN_DECISIONS)
    every = steps // N_CENTRES

    def make(rank):
        def _m():
            e = PpoLayer9Env()
            e.reset(seed=a.seed * 1000 + rank)
            return Monitor(e, info_keywords=("is_success", "task"))
        return _m
    env = SubprocVecEnv([make(i) for i in range(a.n_envs)], start_method="fork")
    init = a.init_model or (d / "ckpt_g-1.zip")
    model = PPO.load(init, env=env, device="cpu", tensorboard_log=str(OUT / "tb"), seed=a.seed)
    t0 = time.monotonic()

    class Archive(BaseCallback):
        def __init__(self):
            super().__init__()
            self.k = 0

        def _on_step(self):
            if self.k < N_CENTRES and self.num_timesteps >= (self.k + 1) * every:
                self.model.save(d / f"ckpt_g{self.k}")
                row = dict(centre=self.k, timesteps=int(self.num_timesteps), episodes_approx=round(self.num_timesteps / MEAN_DECISIONS), wall_s=round(time.monotonic() - t0))
                (d / "history.jsonl").open("a").write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                self.k += 1
            return True
    print(json.dumps(dict(seed=a.seed, budget_episodes=a.budget_episodes, timesteps=steps, checkpoint_every=every, n_envs=a.n_envs, init=str(init))), flush=True)
    model.learn(total_timesteps=steps, callback=Archive(), tb_log_name=f"v7ppo_s{a.seed}", reset_num_timesteps=True)
    model.save(d / f"ckpt_g{N_CENTRES - 1}")
    env.close()
    print(f"finetune done: {steps} steps, wall {round(time.monotonic() - t0)} s", flush=True)


# ----------------------------------------------------------------------------------------- parallel replay
_ENV = None
_POL = {}


def _init(variant):
    global _ENV
    _ENV = PpoLayer9Env(variant=variant)


def _run(args):
    """("ppo", checkpoint path, task, case, seed) -> deterministic replay of one case; ("classical", {law, schedule}, task, case, _) -> the bench9 opponent."""
    kind, arg, task, case, seed = args
    if kind == "classical":
        return T.job(("opp", task, case, "classical", arg))
    from stable_baselines3 import PPO
    if arg not in _POL:
        _POL[arg] = PPO.load(arg, device="cpu")
    pol = _POL[arg]
    obs, _ = _ENV.reset(seed=seed, options=dict(task=task, case=case))
    done, info = False, {}
    while not done:
        act, _ = pol.predict(obs, deterministic=True)
        obs, _, te, tr, info = _ENV.step(act)
        done = te or tr
    return dict(info["metrics"], controller=Path(arg).stem)


def centre_paths(d: Path):
    return [d / f"ckpt_g{g}.zip" for g in [-1] + list(range(N_CENTRES)) if (d / f"ckpt_g{g}.zip").exists()]


def reselect(a):
    d = seed_dir(a.seed)
    norm = load_plan()["normalisers"]
    cs = {t: T.cases9(t, T.BASES["train"], P["n_select"], offset=D9.OFF["select"]) for t in T.TASKS}
    paths = centre_paths(d)
    jobs = [("ppo", str(p), t, c, int(c["seed"]) % (2 ** 31)) for p in paths for t in T.TASKS for c in cs[t]]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=("intact",)) as ex:
        rows = list(ex.map(_run, jobs, chunksize=1))
    res = []
    for p in paths:
        rs = [r for r in rows if r["controller"] == p.stem]
        obj, rt = D9.objective(rs, norm)
        g = int(p.stem.split("g")[-1])
        e = dict(generation=g, budget_fraction=0. if g < 0 else (g + 1) / N_CENTRES, objective=obj, ratios=rt, crashes=sum(not r["flown"] for r in rs),
                 over_hard=sum((r["tilt_peak_deg"] or 0) >= D9.HARD_TILT_DEG for r in rs), tilt_peak_max=max(r["tilt_peak_deg"] or 0 for r in rs), spec=str(p))
        res.append(e)
        print(json.dumps(e), flush=True)
    ok = [e for e in res if e["crashes"] == 0 and e["over_hard"] == 0]
    sel = min(ok, key=lambda e: e["objective"]) if ok else None
    fallback = None
    if sel is None:                              # descend9 has no fallback rule (selected = null); record a fallback here and mark it
        nocrash = [e for e in res if e["crashes"] == 0]
        sel = min(nocrash or res, key=lambda e: e["objective"])
        fallback = "no eligible centre (descend9 rule would select none); fallback = " + ("zero-crash centres" if nocrash else "all centres")
    (OUT / f"reselect_s{a.seed}.json").write_text(json.dumps(dict(seed=a.seed, machine=MACHINE, profile=PROFILE, n_select=P["n_select"], centres=res, selected=sel, fallback=fallback), indent=1))
    print(json.dumps(dict(selected=sel["generation"], objective=sel["objective"], fallback=fallback)), flush=True)


def evaluate(a):
    """Blind test: the selected PPO centre (mlp_ppo) and the bc start (mlp_bc), each paired with the classical opponent; report = descend9.report."""
    import hashlib
    base = load_plan()
    src = Path(a.from_dir) if a.from_dir else OUT          # --from-dir: frozen re-test, the selected centre and bc both come from another round's output directory (v8)
    d = src / f"s{a.seed}"
    sel = json.loads((src / f"reselect_s{a.seed}.json").read_text())["selected"]
    cs = {t: T.cases9(t, P["test_base"], P["n_test"]) for t in T.TASKS}
    frozen = f"; frozen controller from {src}" if a.from_dir else ""
    arms = [("mlp_ppo", sel["spec"], f"selected PPO centre g{sel['generation']} (budget fraction {sel.get('budget_fraction')}){frozen}"),
            ("mlp_bc", str(d / "ckpt_g-1.zip"), f"imitation start (g-1){frozen}")]
    sha = {arm: hashlib.sha256(Path(spec).read_bytes()).hexdigest()[:16] for arm, spec, _ in arms}
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=("intact",)) as ex:
        opp_file, source = PLAN / P["opp_name"], "descend9 (CloudPC-2)"
        if a.opp_file:
            opp_file, source = Path(a.opp_file), f"given: {a.opp_file}"
        if not opp_file.exists():
            opp_file, source = OUT / P["opp_name"], f"bench9 classical run on {MACHINE}"
            if not opp_file.exists():
                opp_rows = list(ex.map(_run, [("classical", dict(law=base["law"], schedule=base["opponent"][t][T.klass(c)]), t, c, 0)
                                              for t in T.TASKS for c in cs[t]], chunksize=1))
                opp_file.write_text(json.dumps(opp_rows))
        opp_rows = json.loads(opp_file.read_text())
        specs = list(dict.fromkeys(spec for _, spec, _ in arms))          # replay only once when the selected centre is bc
        rows = list(ex.map(_run, [("ppo", spec, t, c, int(c["seed"]) % (2 ** 31)) for spec in specs for t in T.TASKS for c in cs[t]], chunksize=1))
    n_cases = sum(len(cs[t]) for t in T.TASKS)
    by_spec = {spec: rows[i * n_cases:(i + 1) * n_cases] for i, spec in enumerate(specs)}
    for arm, spec, note in arms:
        rr = [dict(r, controller=arm) for r in by_spec[spec]]
        rep = D9.report(rr, opp_rows)
        doc = dict(arm=arm, seed=a.seed, machine=MACHINE, profile=PROFILE, blind_base=P["test_base"], n_per_task=P["n_test"], note=note, opponent_rows_source=source,
                   controller_file=spec, controller_sha256=sha[arm], frozen_from=str(src) if a.from_dir else None,
                   selected=sel if arm == "mlp_ppo" else None, report=rep, rows=rr)
        (OUT / f"evaluate_{arm}_s{a.seed}.json").write_text(json.dumps(doc, indent=1, default=float))
        print(f"v7ppo_s{a.seed} {arm}:", json.dumps({t: round(rep[t]["ratio"], 3) for t in T.TASKS}), "mean", round(rep["manoeuvre_mean"], 3),
              "crashes", sum(rep[t]["crashes"] for t in T.TASKS), "wins", "/".join(str(rep[t]["wins"]) for t in T.TASKS), flush=True)


def rereport(a):
    """Recompute the report of existing evaluate_*_s<seed>.json files with another set of classical opponent rows (--opp-file, e.g. from another machine), without re-running the simulation."""
    opp_rows = json.loads(Path(a.opp_file).read_text())
    for arm in ("mlp_ppo", "mlp_bc"):
        f = OUT / f"evaluate_{arm}_s{a.seed}.json"
        doc = json.loads(f.read_text())
        doc["report"], doc["opponent_rows_source"] = D9.report(doc["rows"], opp_rows), f"given: {a.opp_file}"
        f.write_text(json.dumps(doc, indent=1, default=float))
        rep = doc["report"]
        print(f"v7ppo_s{a.seed} {arm} (rereport):", json.dumps({t: round(rep[t]["ratio"], 3) for t in T.TASKS}), "mean", round(rep["manoeuvre_mean"], 3), flush=True)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    from stable_baselines3.common.env_checker import check_env
    env = PpoLayer9Env()
    check_env(env, warn=True)
    out = {}
    for task in T.TASKS:
        c = T.cases9(task, T.BASES["tune"], 2)[1]
        for name, mode in (("zero", "zero"), ("full", "full")):
            obs, info = env.reset(seed=1, options=dict(task=task, case=c))
            assert obs.shape == (N_OBS,) and abs(obs[N_IMU * 3] - 1.) < 1e-6
            done, n = False, 0
            while not done:
                if mode == "zero":
                    a = np.array([-1., 0., 0.], np.float32)
                else:                                    # full set-point along the current command direction (follows the redirect turn automatically)
                    cmd = env._cmd_next()
                    k = 0 if task == "evade" else 2
                    phi = math.degrees(math.atan2(cmd[k + 1], cmd[k])) if np.any(cmd) else c["dir_deg"]
                    a = np.r_[-1., E.att_dir(phi)].astype(np.float32)
                obs, r, te, tr, info = env.step(a)
                done, n = te or tr, n + 1
            m = info["metrics"]
            assert abs(r + m["objective"]) < 1e-9 and m["task"] == task and m["cls"] == T.klass(c), (r, m["objective"])
            out[(task, name)] = (m["score"], m["flown"], m.get("goal_t"), n)
        print(task, f"dir {c['dir_deg']:.0f}" + (f"->{c['dir2_deg']:.0f}" if c.get("dir2_deg") is not None else ""),
              "zero action: score %.3f flown %s goal_t %s (%d decisions)" % out[(task, "zero")],
              "| full set-point: score %.3f flown %s goal_t %s" % out[(task, "full")][:3])
    assert all(out[(t, "zero")][1] for t in T.TASKS), "zero-action episodes must fly"
    assert out[("evade", "full")][0] < out[("evade", "zero")][0], "set-point along the command must help evade"
    print("v7ppo selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["selftest", "bc", "finetune", "reselect", "evaluate", "rereport"])
    p.add_argument("--from-dir", default=None, help="evaluate: frozen re-test, output directory of another round holding the selected centre / bc (e.g. runs/ppo_direct/v7full)")
    p.add_argument("--opp-file", default=None, help="evaluate / rereport: classical opponent rows file")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pulse-weight", type=float, default=3.)
    p.add_argument("--budget-episodes", type=int, default=ES_EPISODES // 4)
    p.add_argument("--n-envs", type=int, default=18)
    p.add_argument("--init-model", type=Path, default=None)
    p.add_argument("--workers", type=int, default=18)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        return selftest()
    dict(bc=bc, finetune=finetune, reselect=reselect, evaluate=evaluate, rereport=rereport)[a.stage](a)


if __name__ == "__main__":
    main()
