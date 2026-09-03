"""Lorenz 63, the Tsit5 step and the MLP
"""

import jax
import jax.numpy as jnp


# - - - true vector field - - -
def lorenz63(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    return jnp.stack([sigma * (x2 - x1),
                      x1 * (rho - x3) - x2,
                      x1 * x2 - beta * x3], axis=-1)


# - - - Tsit5 - - -
# Tsitouras, Ch. (2011), Comput. Math. Appl. 62(2), 770-775.
_A = jnp.array([
    [0, 0, 0, 0, 0, 0],
    [0.161, 0, 0, 0, 0, 0],
    [-0.008480655492356989, 0.335480655492357, 0, 0, 0, 0],
    [2.8971530571054935, -6.359448489975075, 4.362295432869581, 0, 0, 0],
    [5.325864828439257, -11.748883564062828, 7.4955393428898365,
     -0.09249506636175525, 0, 0],
    [5.86145544294642, -12.92096931784711, 8.159367898576159,
     -0.071584973281401, -0.028269050394068383, 0]])
_B = jnp.array([0.09646076681806523, 0.01, 0.4798896504144996,
                1.379008574103742, -3.290069515436081, 2.324710524099774])


def tsit5_step(f, dt):
    """One Tsit5 step of dx/dt = f(x).  f may close over parameters."""
    def step(x0):
        xs, ks = [x0], [f(x0)]
        for i in range(1, 6):
            xs.append(x0 + dt * sum(_A[i][j] * ks[j] for j in range(i)))
            ks.append(f(xs[i]))
        return x0 + dt * sum(_B[i] * ks[i] for i in range(6))
    return step


def rollout(step, n):
    """n steps of `step`, returning [n+1, d] including the initial state."""
    def solve(x0):
        xn, prev = jax.lax.scan(lambda x, _: (step(x), x), x0, None, length=n)
        return jnp.concatenate([prev, xn[None, ...]], axis=0)
    return solve


# - - - MLP - - -
def init_mlp(key, layer_sizes):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [{'W': jax.random.normal(jax.random.split(k)[0], (m, n))
                  * jnp.sqrt(2.0 / m),                      # Kaiming He
             'b': jnp.zeros((n,))}
            for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:]))]


def call_mlp(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(jnp.dot(h, layer['W']) + layer['b'])
    return jnp.dot(h, params[-1]['W']) + params[-1]['b']


# - - - training data: trajectory segments on the attractor - - -
def make_data(key, n_traj=8, dt=0.01, n_step=8000, warmup=1000, m=1, split=0.5):
    """(train, val) arrays of [N, m+1, 3] segments, transient discarded.

    Initial conditions in the unit cube around (-15, -15, 5) and the first
    `warmup` steps dropped, as in the paper, so the segments lie on the
    attractor. 
    """
    x0 = jnp.array([-15.0, -15.0, 5.0]) + jax.random.uniform(
        key, (n_traj, 3), minval=-1.0, maxval=1.0)
    traj = jax.vmap(rollout(tsit5_step(lorenz63, dt), n_step))(x0)  # [T, n+1, 3]
    traj = traj[:, warmup:]
    seg = jnp.stack([traj[:, i:traj.shape[1] - m + i] for i in range(m + 1)], 2)
    n_tr = int(n_traj * split)
    return (seg[:n_tr].reshape(-1, m + 1, 3), seg[n_tr:].reshape(-1, m + 1, 3))
