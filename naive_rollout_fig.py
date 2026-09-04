"""Lorenz 63 rollout example from the released naive model.

data/naive_m1.npz is the released Lorenz 63 naive model at m=1. 
It stores the MLP weights as a plain npz, 
so l63.py supplies the Tsit5 step and the network and nothing else is needed.

Rolled out for 32,000 steps from the first of the paper's 1,000 evaluation
initial conditions, it gives panels (a), (b), (c) of Figure 1:

    python3 naive_rollout_fig.py          # figure/naive_rollout_fig.png
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from l63 import call_mlp, lorenz63, rollout, tsit5_step

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--weights', default='data/naive_m1.npz')
p.add_argument('--dt', type=float, default=0.01)
p.add_argument('--n_transient', type=int, default=2000, help='panel (b) drops these')
p.add_argument('--n_step', type=int, default=32000)
p.add_argument('--vmap', action='store_true',
               help='roll all 1,000 evaluation ICs out at once '
                    'a different XLA kernel and so a different rounding.')
p.add_argument('--out', default='figure/naive_rollout_fig.png')
args = p.parse_args()


def load_params(path):
    """The npz holds arr_0, arr_1, ... = W, b, W, b, ... of MLP."""
    d = np.load(path)
    arrs = [d[f'arr_{i}'] for i in range(len(d.files) - ('tree_def' in d.files))]
    params = [{'W': jnp.asarray(arrs[i]), 'b': jnp.asarray(arrs[i + 1])}
              for i in range(0, len(arrs), 2)]
    for a, b in zip(params[:-1], params[1:]):        # layer shapes must chain
        assert a['W'].shape[1] == a['b'].shape[0] == b['W'].shape[0]
    return params


def make_batch_x0(seed=23, n_traj=1000):
    """The evaluation ensemble: 1,000 initial conditions in the
    unit cube around (-15, -15, 5), off the attractor. """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_traj)
    return jax.vmap(lambda k: jax.random.uniform(k, (3,), minval=-1.0, maxval=1.0)
                    + jnp.array([-15.0, -15.0, 5.0]))(keys)


params = load_params(args.weights)
print(f'{args.weights}: MLP ' + ' - '.join(
    [str(params[0]['W'].shape[0])] + [str(q['W'].shape[1]) for q in params]))

batch_x0 = make_batch_x0()
x0 = batch_x0[0]
step_net = tsit5_step(lambda z: call_mlp(params, z), args.dt)
if args.vmap:
    x_pred = jax.jit(jax.vmap(rollout(step_net, args.n_step)))(batch_x0)[0]
else:
    x_pred = jax.jit(rollout(step_net, args.n_step))(x0)
x_true = jax.jit(rollout(tsit5_step(lorenz63, args.dt), args.n_step))(x0)

for name, tr in [('true', x_true), ('naive', x_pred)]:
    tr = np.asarray(tr)
    print(f'{name:>6}  max|x| {np.abs(tr[:, 0]).max():9.4g}   '
          f'max|y| {np.abs(tr[:, 1]).max():9.4g}   '
          f'max|z| {np.abs(tr[:, 2]).max():9.4g}')

# - - - the three panels, in the style of hmd/out/Lorenz63.py - - -
t_end = args.n_step * args.dt
t_cut = args.n_transient * args.dt
panels = [(f'(a) full trajectory (true), $t \\in [0, {t_end:g}]$', x_true),
          (f'(b) truncated (true), $t \\in [{t_cut:g}, {t_end:g}]$',
           x_true[args.n_transient:]),
          (f'(c) naive, $t \\in [0, {t_end:g}]$', x_pred)]

fig = plt.figure(figsize=(16, 5))
for i, (title, tr) in enumerate(panels):
    tr = np.asarray(tr)
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    pts = tr.reshape(-1, 1, 3)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(seg, cmap='viridis', lw=0.8,
                          norm=plt.Normalize(0.0, 1.0))
    lc.set_array(np.linspace(0.0, 1.0, len(tr) - 1))
    ax.add_collection(lc)
    ax.set_xlim(tr[:, 0].min(), tr[:, 0].max())
    ax.set_ylim(tr[:, 1].min(), tr[:, 1].max())
    ax.set_zlim(tr[:, 2].min(), tr[:, 2].max())
    ax.set_xlabel(r'$x$', fontsize=14, labelpad=10)
    ax.set_ylabel(r'$y$', fontsize=14, labelpad=10)
    ax.set_zlabel(r'$z$', fontsize=14, labelpad=6)
    ax.set_title(title, fontsize=13, pad=4)
    ax.tick_params(labelsize=9)
    ax.view_init(elev=20, azim=-60)
fig.colorbar(lc, ax=fig.axes, shrink=0.7, pad=0.06).set_label(
    r'Time Evolution (Start $\longrightarrow$ End)', fontsize=11)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=300, bbox_inches='tight', pad_inches=0.1)
print('wrote', args.out)
