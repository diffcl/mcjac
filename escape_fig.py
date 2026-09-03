"""How a rollout leaves the attractor: one outlier of `mc` at seed 36, m = 1.

seed_table.py counts the rollouts that left; this is what leaving looks like.
`mc` seed 36 idx 672 stays on the attractor until t = 139.5, drops down the z
axis in about one time unit, and is held there by a set of states 4.9 from the
true origin, the nearest true fixed point.

    python3 escape_fig.py        # figure/demo_escape4.png

Reads data/escape/escape_mc_s36_idx672.npz and draws that outlier: four
consecutive windows across the transition, in one row, each in phase space over
a grey copy of the whole rollout.
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

GREY_BG = 0.85          # grey copy of the whole rollout behind each snapshot
ELEV, AZIM = 14, -62
FS = 1.45
W_PANEL = 4.3
H_PANEL = 3.5

p = argparse.ArgumentParser()
p.add_argument('--dir', default='data/escape')
p.add_argument('--out', default='figure')
p.add_argument('--fs', type=float, default=FS, help='font / linewidth scale')
args = p.parse_args()
fs = args.fs


def snapshot_panel(fig, pos, x_all, sl, fps, x_inf, lim, title):
    """One phase-space panel (9d-PlotEscapeSnapshots.py)."""
    ax = fig.add_subplot(*pos, projection='3d')
    ax.plot(*x_all.T, lw=0.3 * fs, color=str(GREY_BG), zorder=1)

    seg_x = x_all[sl]
    pts = seg_x.reshape(-1, 1, 3)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=plt.get_cmap('plasma'),
                          norm=plt.Normalize(0.0, 1.0), lw=1.1 * fs, zorder=3)
    lc.set_array(np.linspace(0, 1, len(seg_x) - 1))
    ax.add_collection(lc)
    ax.scatter(*seg_x[0], s=30 * fs ** 2, marker='o', color='k', zorder=4)
    ax.scatter(*fps.T, s=40 * fs ** 2, marker='o', facecolors='none',
               edgecolors='0.35', linewidths=1.0 * fs, zorder=2)
    ax.scatter(*x_inf, s=55 * fs ** 2, marker='X', color='crimson', zorder=5)

    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel(r'$x$', fontsize=12 * fs, labelpad=-3 * fs)
    ax.set_ylabel(r'$y$', fontsize=12 * fs, labelpad=-3 * fs)
    ax.set_zlabel(r'$z$', fontsize=12 * fs, labelpad=-4 * fs)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.zaxis.set_major_locator(plt.MultipleLocator(20))
    ax.tick_params(labelsize=7 * fs, pad=-1.0 * fs)
    ax.set_box_aspect((1, 1, 0.95), zoom=1.22)
    ax.set_title(title, fontsize=9.5 * fs, y=0.99)


d = np.load(os.path.join(args.dir, 'escape_mc_s36_idx672.npz'))
x, le_cum, dt = d['x'], d['le_cum'].astype(np.float64), float(d['dt'])
t_esc, x_inf, fps = float(d['t_esc']), d['x_inf'], d['fps']
lim, label, idx = d['lim'], str(d['label']), int(d['idx'])
edges = [t_esc + o for o in d['offsets']]

T0 = int(d['n_transient']) * dt

os.makedirs(args.out, exist_ok=True)
fig = plt.figure(figsize=(W_PANEL * 4, H_PANEL))
print(f'{label} m=1 idx {idx}:  escape at t = {t_esc + T0:.2f}')
print(f"\n{'window':>18s} {'z min':>8s} {'z max':>8s} {'max|x|':>8s} "
      f"{'lam1 over window':>18s}")
print('-' * 66)
for k in range(4):
    lo, hi = edges[k], edges[k + 1]
    sl = slice(int(lo / dt), int(hi / dt))
    seg = x[sl]
    j0, j1 = sl.start, min(sl.stop, len(le_cum)) - 1
    lam_w = (le_cum[j1] - le_cum[j0]) / ((j1 - j0) * dt)
    snapshot_panel(fig, (1, 4, k + 1), x, sl, fps, x_inf, lim,
                   rf'({chr(97 + k)})  $t \in [{lo + T0:.1f},\,{hi + T0:.1f}]$   '
                   rf'$\lambda_1^{{\rm loc}} = {lam_w:.2f}$')
    print(f'  [{lo + T0:7.2f},{hi + T0:7.2f}] {seg[:, 2].min():8.2f} '
          f'{seg[:, 2].max():8.2f} '
          f'{np.linalg.norm(seg, axis=1).max():8.2f} {lam_w:18.3f}')


fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.0, wspace=0.04)
out = os.path.join(args.out, 'demo_escape4.png')
fig.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print('\nwrote', out)
