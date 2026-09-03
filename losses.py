"""The five supervision strategies of the paper, side by side.

Reference implementation for

    S. Kang, H. V. Nguyen, T. Bui-Thanh,
    "Second-order consistency for learning chaotic dynamics 
    via randomized Jacobian matching".

The proposed term is the Jacobian mismatch evaluated at randomly perturbed
states (mcjac). 

The module takes the true vector field, the network and the two one-step maps as arguments; train_l63.py is a runnable example.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


# - - - Mismatch terms - - -
def jacobian_mismatch(f_true, f_hat, params, x):
    """(1/d^2) ||J_hat(x) - J(x)||_F^2 at a single state x.

    Both Jacobians are built explicitly with jax.jacrev, as in hmd/train.py.
    """
    d = x.shape[-1]
    dJ = jax.jacrev(lambda z: f_hat(params, z))(x) - jax.jacrev(f_true)(x)
    return jnp.sum(dJ ** 2) / d ** 2


def hessian_mismatch(f_true, f_hat, params, x):
    """(1/d^3) ||H_hat(x) - H(x)||_F^2, explicit second-order supervision."""
    d = x.shape[-1]
    dH = jax.hessian(lambda z: f_hat(params, z))(x) - jax.hessian(f_true)(x)
    return jnp.sum(dH ** 2) / d ** 3


# - - - The five strategies  - - -
#   L = L_data + a_mc L_mc + a_mcjac L_mcjac + a_jac L_jac + a_hes L_hes
#
# L_data    trajectory matching on the data
# L_jac     ||dJ||^2 on the data states
# L_hes     ||dH||^2 on the data states
# L_mc      trajectory matching from a perturbed initial state
# L_mcjac   ||dJ||^2 on the true trajectory from that state
#
# The weights below are the Lorenz 63 (m = 1) settings.
# lam_mc * (L_mc + lam_jac * L_mcjac), i.e. a_mcjac = lam_mc * lam_jac = 100,
METHODS = {
    'naive':  dict(a_mc=0.0,   a_mcjac=0.0,  a_jac=0.0, a_hes=0.0, sigma=0.0),
    'mc':     dict(a_mc=100.0, a_mcjac=0.0,  a_jac=0.0, a_hes=0.0, sigma=1.0),
    'jac':    dict(a_mc=0.0,   a_mcjac=0.0,  a_jac=1.0, a_hes=0.0, sigma=0.0),
    'hes':    dict(a_mc=0.0,   a_mcjac=0.0,  a_jac=1.0, a_hes=0.5, sigma=0.0),
    'mcjac':  dict(a_mc=100.0, a_mcjac=50.0, a_jac=0.0, a_hes=0.0, sigma=0.5),
}


def make_loss(method, f_true, f_hat, solve_true, solve_hat,
              lam_l2=1e-5, **override):
    """Return loss(params, batch, key) -> (scalar, dict of terms).

    f_true(x)            true vector field,           x: [d]
    f_hat(params, x)     network vector field,        x: [d]
    solve_true(x0)       true m-step rollout,         -> [m+1, d] (includes x0)
    solve_hat(params,x0) network m-step rollout,      -> [m+1, d]
    batch                true trajectory segments,    [B, m+1, d]
    """
    cfg = dict(METHODS[method], **override)
    a_mc, a_mcjac = cfg['a_mc'], cfg['a_mcjac']
    a_jac, a_hes, sigma = cfg['a_jac'], cfg['a_hes'], cfg['sigma']

    def mean_jac(params, states):
        """Mean of (8)/(14) over a [N, d] set of evaluation states."""
        return jnp.mean(jax.vmap(lambda x: jacobian_mismatch(
            f_true, f_hat, params, x))(states))

    def loss(params, batch, key):
        u = batch                                        # [B, m+1, d]
        u_hat = jax.vmap(solve_hat, in_axes=(None, 0))(params, u[:, 0])
        l_data = jnp.mean((u_hat - u) ** 2)
        l2 = sum(jnp.sum(w ** 2) for w in jtu.tree_leaves(params))
        total = l_data + lam_l2 * l2

        terms = dict(data=l_data, mc=0.0, mcjac=0.0, jac=0.0, hes=0.0)
        # Only the mc perturbation draws random numbers.
        k_eps = jax.random.split(key, 3)[0]
        flat = u.reshape(-1, u.shape[-1])                # [B(m+1), d]

        if a_jac > 0.0:                                  # jac, hes
            terms['jac'] = mean_jac(params, flat)
            total += a_jac * terms['jac']

        if a_hes > 0.0:                                  # hes only
            terms['hes'] = jnp.mean(jax.vmap(
                lambda x: hessian_mismatch(f_true, f_hat, params, x))(flat))
            total += a_hes * terms['hes']

        if a_mc > 0.0:                                   # mc, mcjac
            # The perturbation is applied to the initial state only and the
            # true model integrated from there, so the supervision lives on the
            # true trajectory started at u0 + eps.
            v0 = u[:, 0] + sigma * jax.random.normal(k_eps, u[:, 0].shape)
            v = jax.vmap(solve_true)(v0)                 # [B, m+1, d]
            v_hat = jax.vmap(solve_hat, in_axes=(None, 0))(params, v0)
            terms['mc'] = jnp.mean((v_hat - v) ** 2)
            total += a_mc * terms['mc']

            if a_mcjac > 0.0:                            # mcjac
                terms['mcjac'] = mean_jac(
                    params, v.reshape(-1, v.shape[-1]))
                total += a_mcjac * terms['mcjac']

        return total, terms

    return loss
