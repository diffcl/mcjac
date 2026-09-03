"""Train the five strategies on Lorenz 63, then roll each model out.

All five methods train in a few minutes.

    python3 train_l63.py                 # all five methods
    python3 train_l63.py --method mcjac
    python3 train_l63.py --paper         # Figure 1

"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

from l63 import (call_mlp, init_mlp, lorenz63, make_data, rollout, tsit5_step)
from losses import METHODS, make_loss

matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--method', default='all', choices=['all'] + list(METHODS))
p.add_argument('--epochs', type=int, default=20000)
p.add_argument('--width', type=int, default=256)
p.add_argument('--depth', type=int, default=3)
p.add_argument('--batch', type=int, default=512)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--m', type=int, default=1, help='rollout length in the loss')
p.add_argument('--dt', type=float, default=0.01)
p.add_argument('--seed', type=int, default=32)
p.add_argument('--t_roll', type=float, default=100.0, help='rollout horizon')
p.add_argument('--n_traj', type=int, default=8, help='trajectories, half held out')
p.add_argument('--n_step', type=int, default=8000, help='steps per trajectory')
p.add_argument('--x0', default='heldout', choices=['heldout', 'paper'],
               help="rollout state: a held-out on-attractor state, or the first "
                    "of the paper's 1,000 evaluation initial conditions")
p.add_argument('--out', default='figure/demo_l63.png')
p.add_argument('--paper', action='store_true')
args = p.parse_args()

PAPER = dict(width=512, epochs=10000, n_traj=32, n_step=50000, t_roll=320.0,
             x0='paper', out='figure/paper_l63.png')
if args.paper:
    named = {a.split('=')[0] for a in sys.argv[1:]}
    for k, v in PAPER.items():
        if f'--{k}' not in named:
            setattr(args, k, v)

key = jax.random.PRNGKey(args.seed)
k_data, k_init, k_train = jax.random.split(key, 3)

data, val = make_data(k_data, n_traj=args.n_traj, dt=args.dt,
                      n_step=args.n_step, m=args.m)
step_true = tsit5_step(lorenz63, args.dt)
solve_true = rollout(step_true, args.m)
solve_hat = lambda q, x0: rollout(tsit5_step(lambda z: call_mlp(q, z), args.dt),
                                  args.m)(x0)
print(f'data {data.shape[0]} train / {val.shape[0]} held-out segments'
      f' of {args.m + 1} states')

methods = list(METHODS) if args.method == 'all' else [args.method]
trained = {}
for method in methods:
    params = init_mlp(k_init, [3] + [args.width] * args.depth + [3])
    loss_fn = make_loss(method, lorenz63, call_mlp, solve_true, solve_hat)
    opt = optax.adam(args.lr)
    state = opt.init(params)

    @jax.jit
    def update(params, state, batch, key):
        (lval, terms), grad = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, key)
        upd, state = opt.update(grad, state)
        return optax.apply_updates(params, upd), state, lval, terms

    t0 = time.time()
    for epoch in range(args.epochs):
        k_train, k_b, k_l = jax.random.split(k_train, 3)
        idx = jax.random.randint(k_b, (args.batch,), 0, data.shape[0])
        params, state, lval, terms = update(params, state, data[idx], k_l)
        if epoch % (args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f'{method:>6} {epoch:6d}  L {lval:.4e}  ' + '  '.join(
                f'{k} {float(v):.2e}' for k, v in terms.items() if float(v)))
    trained[method] = params
    print(f'{method:>6} done in {time.time() - t0:.1f} s')

# - - - one long rollout from a single state - - -
def paper_x0(seed=23, n_traj=1000):
    """ 1,000 initial conditions in the unit cube around (-15, -15, 5), 
    off the attractor."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_traj)
    return jax.vmap(lambda k: jax.random.uniform(k, (3,), minval=-1.0, maxval=1.0)
                    + jnp.array([-15.0, -15.0, 5.0]))(keys)


n_roll = int(args.t_roll / args.dt)
x0 = paper_x0()[0] if args.x0 == 'paper' else val[0, 0]
ref = jax.jit(rollout(step_true, n_roll))(x0)
rolls = {m: jax.jit(rollout(tsit5_step(lambda z: call_mlp(q, z), args.dt),
                            n_roll))(x0) for m, q in trained.items()}

# The pre-registered failure test of the paper:
# a rollout fails if it diverges, collapses, or visits one lobe only.
def verdict(tr):
    tr = np.asarray(tr)
    p_lobe = float(np.mean(tr[:, 0] > 0))
    bad = ('diverged' if np.max(np.abs(tr)) > 200.0 else
           'collapsed' if np.std(tr[:, 0]) < 1.0 else
           'one lobe' if not 0.05 < p_lobe < 0.95 else 'ok')
    return (f'max|.| {np.max(np.abs(tr)):7.1f}   std(x) {np.std(tr[:, 0]):6.2f}'
            f'   P(x>0) {p_lobe:5.3f}   {bad}')


print(f'\n rollout over t in [0, {args.t_roll:g}] from one '
      + ('evaluation IC of ' if args.x0 == 'paper' else 'held-out state'))
print(f"{'true':>6}  {verdict(ref)}")
for name, tr in rolls.items():
    print(f'{name:>6}  {verdict(tr)}')

n = 1 + len(rolls)
fig = plt.figure(figsize=(2.6 * n, 2.9))
for i, (name, tr) in enumerate([('true', ref)] + list(rolls.items())):
    tr = np.asarray(tr)
    ax = fig.add_subplot(1, n, i + 1, projection='3d')
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=0.5,
            color='k' if name == 'true' else 'C0')
    ax.set_title(name, fontsize=13, y=0.94)
    ax.set_xlabel(r'$x$', fontsize=12, labelpad=-9)
    ax.set_ylabel(r'$y$', fontsize=12, labelpad=-9)
    ax.set_zlabel(r'$z$', fontsize=12, labelpad=-9)
    if np.max(np.abs(tr)) <= 60.0:
        ax.set_xlim(-25, 25)
        ax.set_ylim(-30, 30)
        ax.set_zlim(0, 55)
    ax.tick_params(labelsize=7, pad=-3)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect((1, 1, 0.95), zoom=1.18)
fig.suptitle(rf'Lorenz 63, rollout over $t \in [0, {args.t_roll:g}]$ '
             f'({args.epochs} epochs, {args.depth}x{args.width} MLP)',
             fontsize=11, y=0.99)
fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.0, wspace=0.12)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=300, bbox_inches='tight', pad_inches=0.05)
print('wrote', args.out)
