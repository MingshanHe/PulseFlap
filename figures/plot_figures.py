"""Draw the five data figures of the paper from the files in data/ and calibration/ (numpy, scipy, matplotlib only).

    python figures/plot_figures.py                          # all five -> figures/out/*.pdf and *.png
    python figures/plot_figures.py robustness authority     # some of them

  calibration     Fig. 2  calibration/fixed_base_*.csv
  robustness      Fig. 4  data/figures/robustness_crashes.json, data/figures/pulse_law_crashes.json
  authority       Fig. 5  data/cluster/q2_pulse_authority.json
  generalisation  Fig. 6  data/cluster/p1_vehicle_variants.json
  phase           Fig. 7  data/figures/cost_energy_phase.json

Figures are drawn at their printed size for the IEEE two-column template, so font sizes are the sizes on paper.
The overview (Fig. 1) and controller (Fig. 3) diagrams are TikZ sources in figures/tikz/.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[1]
DATA = REPO/'data'
CALIB = REPO/'calibration'
OUT = Path(__file__).resolve().parent/'out'

# ----------------------------------------------------------------------------------------------- style
COL1, COL2 = 3.5, 7.16                     # IEEE single-column / double-column width (inch)
FONT = ['Nimbus Roman', 'TeX Gyre Termes', 'Times New Roman', 'STIXGeneral', 'DejaVu Serif']   # Times-like
FS, FS_SMALL, FS_TINY = 8, 7.5, 7.0        # axis labels / ticks & legends / annotations (pt, as printed)
LW = .8
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
C = dict(classical='#4a4844', cls_es='#c8c5bd', snn='#1764d6', snn_edge='#0b3d8c', ppo='#e98f52',
         snn_light='#cde2fb', window='#ecebe6', grey=MUTED)
DPI_PNG = 300
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': FONT, 'mathtext.fontset': 'stix', 'axes.unicode_minus': True,
    'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'lines.linewidth': LW, 'axes.linewidth': .6,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'grid.color': GRID, 'grid.linewidth': .5, 'grid.alpha': 1.,
    'axes.axisbelow': True, 'legend.frameon': False, 'legend.handlelength': 1.4, 'legend.handletextpad': .5,
    'legend.columnspacing': 1.2, 'legend.borderaxespad': .2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlelocation': 'left', 'axes.titleweight': 'bold',
    'axes.titlepad': 3, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.pad_inches': .015,
    'figure.constrained_layout.use': True, 'figure.constrained_layout.h_pad': .02, 'figure.constrained_layout.w_pad': .02,
    'hatch.linewidth': .5})
BAR_SNN = dict(edgecolor=C['snn_edge'], linewidth=.6)
N = 192

# one style per controller, used in every figure
NAME = dict(classical='Classical', probe='Classical + pulse law', snn='SNN', ppo='MLP + PPO')
STY = dict(
    classical=dict(marker='o', ms=3.6, mfc=C['classical'], mec=C['classical'], mew=.5, color=C['classical'], lw=.7),
    probe=dict(marker='^', ms=4.2, mfc='white', mec=INK, mew=.8, color=INK, lw=.7),
    snn=dict(marker='o', ms=4.4, mfc=C['snn'], mec=C['snn_edge'], mew=.6, color=C['snn_edge'], lw=1.2),
    ppo=dict(marker='s', ms=3.4, mfc=C['ppo'], mec=C['ppo'], mew=.5, color=C['ppo'], lw=1.0))
ORDER = ('classical', 'probe', 'snn', 'ppo')

# rows of Fig. 4 (labels as in the data files -> labels as printed)
ROBUST_ROWS = {'2 mN·m hit': '2 mN·m', '3 mN·m hit': '3 mN·m (confirmatory)', '4.5 mN·m hit': '4.5 mN·m',
               'hit +0–0.15 s': '0–0.15 s after command', 'hit +0.5–1.0 s': '0.5–1.0 s after command',
               'hit before cmd': 'before command', 'double hit': 'double hit', 'roll-axis hit': 'roll axis',
               'pitch-axis hit': 'pitch axis', 'turbulence ×1.25': 'scaled ×1.25', 'turbulence ×1.5': 'scaled ×1.5',
               'IMU noise ×5': 'IMU noise ×5', 'sensor delay 3': 'sensor delay 6 ms', 'mass +12.5 %': 'mass +12.5 %'}
ROBUST_GROUPS = [('Hit magnitude', 0, 3), ('Hit timing', 3, 6), ('Hit type', 6, 9), ('Sustained disturbance', 9, 11),
                 ('Sensors', 11, 13), ('Plant', 13, 14)]
CONFIRMATORY = '3 mN·m hit'

# vehicles of Fig. 6: (key, label, group)
PLANTS = [
    ('lag150', 'actuator lag ×1.5', 'actuator'), ('volt90', 'voltage −10 %', 'actuator'),
    ('kt90', '$k_t$, $k_e$ −10 %', 'actuator'),
    ('wing_mass130', 'wing mass +30 %', 'wings and air'), ('span95', 'span −5 %', 'wings and air'),
    ('chord90', 'chord −10 %', 'wings and air'), ('asym97', 'left span −3 %', 'wings and air'),
    ('density90', 'air density −10 %', 'wings and air'),
    ('lower110', 'mass +10 %', 'lower body'), ('lower125', 'mass +25 %', 'lower body'),
    ('lower_com', '+12.5 %, COM −3 mm', 'lower body')]

# aerodynamic models of Fig. 2: (key, label, colour, line style, marker)
MODELS = [('be', 'Blade element (ours)', '#087F8C', '-', 'o'), ('flappy', 'Flappy', '#4267A5', '--', 's'),
          ('sd', 'Sane–Dickinson', '#C67A23', '-.', '^'), ('ellipsoid', 'MuJoCo ellipsoid', '#8E669A', ':', 'D')]


def load(rel):
    return json.loads((DATA/rel).read_text(encoding='utf-8'))


def read_csv(name):
    with (CALIB/name).open(encoding='utf-8-sig', newline='') as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def pct(x, n=N):
    return np.asarray(x, float)*100./n


def clopper_pearson(k, n, a=.05):
    """Exact two-sided binomial interval (same as pulseflap.eval.clopper_pearson up to bisection precision)."""
    lo = 0. if k == 0 else beta.ppf(a/2, k, n-k+1)
    hi = 1. if k == n else beta.ppf(1-a/2, k+1, n-k)
    return lo, hi


assert [round(100*v, 1) for v in clopper_pearson(5, 192)] == [0.9, 6.0]      # values quoted in the text
assert [round(100*v, 1) for v in clopper_pearson(14, 768)] == [1.0, 3.0]
CI_KW = dict(lw=.7, solid_capstyle='butt')


def ci_bar(ax, y, k, n, color, scale, zorder=2, cap=.13):
    """Thin whisker with end caps: 95 % Clopper-Pearson interval of a single run (k crashes of n)."""
    lo, hi = (scale*v for v in clopper_pearson(int(k), n))
    ax.plot([lo, hi], [y, y], color=color, zorder=zorder, **CI_KW)
    for x in (lo, hi):
        ax.plot([x, x], [y-cap, y+cap], color=color, zorder=zorder, **CI_KW)


def handle(fam, line=True):
    return Line2D([], [], ls='-' if line else 'none', **dict(STY[fam]))


def marker(ax, x, y, fam, zorder=4):
    s = {k: v for k, v in STY[fam].items() if k not in ('color', 'lw')}
    ax.plot([x], [y], ls='none', zorder=zorder, **s)


def hrange(ax, lo, hi, y, fam, zorder=2):
    ax.plot([lo, hi], [y, y], color=STY[fam]['color'], lw=STY[fam]['lw'], solid_capstyle='butt', zorder=zorder)


def vrange(ax, x, lo, hi, fam, zorder=2):
    ax.plot([x, x], [lo, hi], color=STY[fam]['color'], lw=STY[fam]['lw'], solid_capstyle='butt', zorder=zorder)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/f'{name}.pdf', metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    fig.savefig(OUT/f'{name}.png', dpi=DPI_PNG, metadata={'Software': None})
    plt.close(fig)
    print('saved', OUT/name)


# ------------------------------------------------------------------------------------- Fig. 2 calibration
def calibration():
    lift, ph = read_csv('fixed_base_mean_lift.csv'), read_csv('fixed_base_phase_example.csv')
    mape = {m: np.mean([abs(r[m+'_error_percent']) for r in lift]) for m, *_ in MODELS}
    assert [round(mape[m], 2) for m in ('be', 'flappy', 'sd', 'ellipsoid')] == [7.86, 6.87, 3.93, 46.96]
    fig = plt.figure(figsize=(COL2, 2.5))
    sa, sb = fig.subfigures(1, 2, width_ratios=[1.3, 3], wspace=.02)
    a = sa.add_subplot(1, 1, 1)
    gs = sb.add_gridspec(2, 3)
    f = np.array([r['frequency_hz'] for r in lift])
    a.axvspan(12.5, 15.3, color=C['window'], lw=0, zorder=0)
    a.text(15.1, 34, 'held out', ha='right', va='top', fontsize=FS_TINY, color=INK2)
    a.text(12.35, 34, 'fitted', ha='right', va='top', fontsize=FS_TINY, color=INK2)
    for m, lab, col, ls, mk in MODELS:
        a.plot(f, [r[m+'_mN'] for r in lift], color=col, ls=ls, lw=1.1, marker=mk, ms=3.2, mfc='white', mec=col,
               mew=.8, zorder=3)
    a.plot(f, [r['be_calibrated_mN'] for r in lift], color='#087F8C', lw=2.2, alpha=.35, zorder=2,
           solid_capstyle='round')
    a.errorbar(f, [r['experiment_mN'] for r in lift], yerr=[r['experimental_SD_mN'] for r in lift], ls='none',
               marker='o', ms=3.4, color=INK, mfc=INK, capsize=1.8, elinewidth=.8, capthick=.8, zorder=5)
    a.set_xlim(9.85, 15.25); a.set_ylim(5, 35); a.set_yticks([5, 15, 25, 35])
    a.set_xlabel('flapping frequency (Hz)'); a.set_ylabel('mean lift (mN)'); a.grid(axis='y')
    a.set_title('(a)', fontsize=FS_SMALL)
    x = np.r_[[r['aligned_cycle'] for r in ph], 1.]
    lims = dict(Fx=(-300, 300), Fy=(-300, 300), Fz=(-100, 150), Mx=(-16, 16), My=(-16, 16), Mz=(-8, 8))
    ticks = dict(Fx=[-300, 0, 300], Fy=[-300, 0, 300], Fz=[-100, 0, 100], Mx=[-15, 0, 15], My=[-15, 0, 15],
                 Mz=[-8, 0, 8])
    for i, ax_ in enumerate(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']):
        b = sb.add_subplot(gs[i//3, i % 3])
        for m, lab, col, ls, mk in MODELS:
            y = [r[f'{m}_{ax_}'] for r in ph]
            b.plot(x, y+y[:1], color=col, ls=ls, lw=.9)
        y = [r[f'actual_{ax_}'] for r in ph]
        b.plot(x, y+y[:1], color=INK, lw=1.3, zorder=4)
        b.set_xlim(0, 1); b.set_ylim(*lims[ax_]); b.set_yticks(ticks[ax_]); b.set_xticks([0, .5, 1])
        b.axhline(0, color=GRID, lw=.5, zorder=0)
        unit = 'mN' if ax_[0] == 'F' else 'mN·m'
        b.set_title(f'${ax_[0]}_{ax_[1]}$ ({unit})', fontsize=FS_SMALL, fontweight='normal')
        if i < 3:
            b.set_xticklabels([])
        else:
            b.set_xticklabels(['0', '0.5', '1'])
            if i == 4:
                b.set_xlabel('wingbeat cycle (aligned)')
    sb.suptitle('(b) periodic six-axis loads, 15 Hz recording, harmonics 1–4', x=.02, ha='left',
                fontsize=FS_SMALL, fontweight='bold')
    hs = [Line2D([], [], color=INK, marker='o', ms=3.4, lw=1.3)]
    hs += [Line2D([], [], color=col, ls=ls, lw=1.1, marker=mk, ms=3.2, mfc='white', mec=col, mew=.8)
           for m, lab, col, ls, mk in MODELS]
    hs += [Line2D([], [], color='#087F8C', lw=2.2, alpha=.35)]
    labs = ['Measured, mean ± SD'] + [lab for _, lab, *_ in MODELS] + ['Blade element, one gain fitted']
    fig.legend(hs, labs, loc='outside upper center', ncol=6, fontsize=FS_TINY, handlelength=2.2,
               columnspacing=1.0, handletextpad=.4)
    save(fig, 'fig_calibration')


# ------------------------------------------------------------------------------------- Fig. 4 robustness
def robustness():
    S = load('figures/robustness_crashes.json')['sets']
    PR = load('figures/pulse_law_crashes.json')['per_set']
    assert [s['label'] for s in S] == list(ROBUST_ROWS)
    DY = dict(ppo=-.33, classical=-.11, probe=.11, snn=.33)
    ypos, labels, heads, y = [], [], [], 0.
    for name, lo, hi in ROBUST_GROUPS:
        heads.append((y, name)); y += .85
        for i in range(lo, hi):
            ypos.append(y); labels.append(ROBUST_ROWS[S[i]['label']]); y += 1.
        y += .15
    fig, ax = plt.subplots(figsize=(COL1, 4.25))
    ic = [s['label'] for s in S].index(CONFIRMATORY)
    ax.axhspan(ypos[ic]-.5, ypos[ic]+.5, color=C['window'], lw=0, zorder=0)
    for s, yy in zip(S, ypos):
        kc, kp = s['classical'], PR[s['label']]['probe']
        assert PR[s['label']]['classical'] == kc
        ci_bar(ax, yy+DY['classical'], kc, N, C['classical'], 100.)
        ci_bar(ax, yy+DY['probe'], kp, N, INK, 100.)
        for fam, vals in (('ppo', pct(s['ppo'])), ('snn', pct(s['snn']))):
            hrange(ax, vals.min(), vals.max(), yy+DY[fam], fam, zorder=3)
            marker(ax, vals.mean(), yy+DY[fam], fam, zorder=5)
        marker(ax, pct(kc), yy+DY['classical'], 'classical', zorder=5)
        marker(ax, pct(kp), yy+DY['probe'], 'probe', zorder=6)
    for yy, name in heads:
        ax.text(-.02, yy+.1, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=FS_SMALL,
                fontweight='bold', color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=FS_SMALL)
    ax.tick_params(axis='y', length=0); ax.set_ylim(ypos[-1]+.7, -.4); ax.set_xlim(-1, 55)
    ax.set_xlabel('crashed episodes (% of 192)'); ax.grid(axis='x'); ax.spines['left'].set_visible(False)
    fig.legend([handle(f) for f in ORDER], [NAME[f] for f in ORDER], loc='outside upper center', ncol=4,
               fontsize=FS_TINY, columnspacing=1.0, handlelength=1.6, handletextpad=.4)
    save(fig, 'fig_robustness_crashes')


# ------------------------------------------------------------------------------------- Fig. 5 authority
def authority():
    D = load('cluster/q2_pulse_authority.json')
    pl = [p for p in D['plants'] if not p.get('marginal')]
    xs = np.arange(len(pl))

    def plant(lab):
        if lab == 'nominal':
            return 'nominal'
        q, k = lab.split(' x')
        return {'density': 'density', 'voltage': 'voltage'}[q]+f' ×{k}'
    ticks = [f"{p['authority']:.2f}\n{plant(p['label'])}" for p in pl]
    assert [t.split('\n')[0] for t in ticks] == ['0.98', '1.68', '1.77', '2.61', '3.03']
    inom = [p['key'] for p in pl].index('dt5')
    fig, axs = plt.subplots(2, 1, figsize=(COL1, 2.95), sharex=True)
    for ax, key, title, ymax, yt in ((axs[0], 'kick', '(a) 3 mN·m torque hit', 80, [0, 20, 40, 60, 80]),
                                     (axs[1], 'x1.5', '(b) sustained disturbance ×1.5', 40, [0, 10, 20, 30, 40])):
        ax.axvspan(inom-.4, inom+.4, color=C['window'], lw=0, zorder=0)
        for x, p in zip(xs, pl):
            r = p[key]
            for fam, dx in (('snn', -.17), ('ppo', .17)):
                v = np.array(r[fam], float)
                vrange(ax, x+dx, v.min(), v.max(), fam)
                marker(ax, x+dx, v.mean(), fam)
            marker(ax, x, r['probe'], 'probe', zorder=5)
        ax.set_title(title, fontsize=FS_SMALL)
        ax.set_ylim(-2, ymax); ax.set_yticks(yt); ax.set_xlim(-.5, len(pl)-.5)
        ax.grid(axis='y'); ax.tick_params(axis='y', labelsize=FS_SMALL); ax.tick_params(axis='x', length=0)
        ax.set_ylabel('crashes (of 192)', fontsize=FS_SMALL)
    axs[1].set_xticks(xs); axs[1].set_xticklabels(ticks, fontsize=FS_TINY, linespacing=1.1)
    axs[1].set_xlabel('pulse authority (rad/s per burst) and plant', fontsize=FS_SMALL)
    fams = ('probe', 'snn', 'ppo')
    fig.legend([handle(f, line=f != 'probe') for f in fams], [NAME[f] for f in fams], loc='outside upper center',
               ncol=3, fontsize=FS_TINY, handletextpad=.4, columnspacing=1.2, handlelength=1.5)
    save(fig, 'fig_authority')


# ------------------------------------------------------------------------------------- Fig. 6 generalisation
def generalisation():
    D = load('cluster/p1_vehicle_variants.json')
    pl = D['plants']
    assert set(pl) == {k for k, _, _ in PLANTS}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL1, 3.2), sharey=True, gridspec_kw=dict(width_ratios=[1.2, 1]))
    y, ypos, labs, heads, grp = 0., [], [], [], None
    for key, lab, g in PLANTS:
        if g != grp:
            heads.append((y, g)); y += .75; grp = g
        p = pl[key]
        ypos.append(y); labs.append(lab)
        s = np.array(p['snn'], float)
        ci_bar(a1, y-.25, p['classical'], N, C['classical'], N, cap=.11)
        ci_bar(a1, y+.25, p['probe'], N, INK, N, cap=.11)
        hrange(a1, s.min(), s.max(), y, 'snn')
        marker(a1, s.mean(), y, 'snn', zorder=5)
        marker(a1, p['classical'], y-.25, 'classical')
        marker(a1, p['probe'], y+.25, 'probe')
        lo, hi = p['snn_d_probe_range']
        a2.plot([lo, hi], [y, y], color=C['snn_edge'], lw=2.4, solid_capstyle='butt', zorder=2)
        a2.text(.86, y, f"{p['seeds_adv']}/5", va='center', ha='right', fontsize=FS_TINY,
                color=C['snn_edge'] if p['seeds_adv'] else MUTED)
        y += 1.
    for ax in (a1, a2):
        ax.set_ylim(y-.4, -.5); ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    a1.grid(axis='x')
    for yy, name in heads:
        a1.text(-.02, yy+.05, name, transform=a1.get_yaxis_transform(), ha='right', va='center',
                fontsize=FS_SMALL, fontweight='bold', color=INK)
    a1.set_yticks(ypos); a1.set_yticklabels(labs, fontsize=FS_SMALL)
    a1.set_xlim(-.8, 30); a1.set_xticks([0, 10, 20, 30])
    a1.set_xlabel('crashes (of 192)', fontsize=FS_SMALL); a1.set_title('(a)', fontsize=FS_SMALL)
    a2.axvline(0, color=INK2, lw=.7, zorder=1)
    a2.set_xlim(-.55, .88); a2.set_xticks([-.5, 0, .5])
    a2.set_xlabel('SNN paired $\\Delta$ to\nclassical + pulse law', fontsize=FS_SMALL)
    a2.set_title('(b)', fontsize=FS_SMALL)
    a2.text(.86, heads[0][0]+.05, 'seeds', ha='right', va='center', fontsize=FS_TINY, color=INK2)
    a2.text(-.52, heads[0][0]+.05, '← SNN better', ha='left', va='center', fontsize=FS_TINY, color=INK2)
    fams = ('classical', 'probe', 'snn')
    fig.legend([handle(f) for f in fams], [NAME[f] for f in fams], loc='outside lower center', ncol=3,
               fontsize=FS_TINY, handletextpad=.4, columnspacing=1.2, handlelength=1.6)
    save(fig, 'fig_generalisation')


# ------------------------------------------------------------------------------------- Fig. 7 phase
def phase():
    Ph = load('figures/cost_energy_phase.json')['phase']
    fig, ax = plt.subplots(figsize=(COL1, 1.7))
    bins = np.arange(0, 361, 15)
    for c0, name in ((135, 'pitch up'), (315, 'pitch down')):
        ax.axvspan(c0-45, c0+45, color=C['window'], lw=0, zorder=0)
        ax.text(c0, .0232, name, ha='center', va='top', fontsize=FS_TINY, color=INK2)
    ax.hist(Ph['snn_own_pulses_deg'], bins, density=True, color=C['snn'],
            label=f"SNN's own pulses ({len(Ph['snn_own_pulses_deg']):,})", **BAR_SNN)
    ax.hist(Ph['classical_law_deg'], bins, density=True, histtype='step', color=INK, lw=1.,
            label=f"Pulse law ({len(Ph['classical_law_deg']):,})")
    ax.axhline(1/360, color=MUTED, lw=.7, ls=':', label='Uniform')
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_xlim(0, 360); ax.set_ylim(0, .024); ax.set_yticks([0, .01, .02])
    ax.set_xlabel('true crank phase at pulse onset (°)')
    ax.set_ylabel('density (1/°)')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncol=3, fontsize=FS_TINY,
               columnspacing=1.2)
    ax.grid(axis='y')
    save(fig, 'fig_phase')


FIGURES = dict(calibration=calibration, robustness=robustness, authority=authority, generalisation=generalisation,
               phase=phase)

if __name__ == '__main__':
    for name in ([a for a in sys.argv[1:] if not a.startswith('--')] or FIGURES):
        FIGURES[name]()
