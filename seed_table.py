"""Training-seed variability of the five objectives at m = 1, from stored evaluations.

Each objective was trained with five seeds (32-36). 
Each resulting model was evaluated using the same 1,000 on-attractor initial conditions.

    n_outlier  robustness  rollouts per 1,000 whose leading exponent misses
                           the reference by more than 0.1 -- every rollout that
                           diverged, collapsed or kept to one lobe is one of
                           them, so this one criterion counts them all
    W1, W1_lj  accuracy    over the rollouts that stayed, except for `naive`,
                           whose rollouts all leave: its row is taken over all
                           1,000 and reports the size of the failure instead

    python3 seed_table.py            # the table

data/onattr_eval_on_<method>_s<seed>_m1.npz
    Per trajectory: 
    `wd` (W^1 between the empirical measure of the rollout and of the truth), 
    `lyap` (1,000 x 3 finite-time exponents),
    the failure flags, the initial conditions, the training config.
data/onattr_lyaptrue_on.npz
    the same spectrum for the true system on the same initial conditions.

"""

import argparse
import os

import numpy as np

SEEDS = [32, 33, 34, 35, 36]
METHODS = ['mc', 'mcjac', 'naive', 'jac', 'hes']

# One criterion for the robustness column: a rollout whose leading exponent
# misses the reference by more than this, counted over all 1,000.  It is a
# superset of the failure flags -- every diverged, collapsed or one-lobe rollout
# also misses the reference -- so the accuracy columns, taken over the
# complement, need no second mask.
OUTLIER_DLAM1 = 0.1

p = argparse.ArgumentParser()
p.add_argument('--dir', default='data')
p.add_argument('--tex', action='store_true', help="the paper's table body")
args = p.parse_args()


def mean_se(x):
    """(mean, standard error) accumulated in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return x.mean(), x.std(ddof=1) / np.sqrt(x.size)


def w1_lam(u, v):
    """W^1 between two equal-size empirical measures."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    v = np.sort(np.asarray(v, dtype=np.float64))
    return float(np.mean(np.abs(u - v)))


def stats(d, lyap_true):
    """One model: the robustness count, and the accuracy over what stayed."""
    fail = np.asarray(d['failed'])
    wd = np.asarray(d['wd'], dtype=np.float64)
    lam = np.asarray(d['lyap'], dtype=np.float64)
    ref = np.asarray(lyap_true, dtype=np.float64)

    ok = ~fail & np.isfinite(wd)
    out = {'fail': 100 * fail.mean()}
    if not ok.any():
        # `naive` leaves on every rollout of every seed, so the conditional
        # mask is over an empty set.  The row is filled unconditionally, over
        # all 1,000 rollouts, and daggered: it reports the size of the failure,
        # not accuracy on the attractor, and is not comparable with the rows
        # that stayed.
        fin = np.isfinite(wd)
        out['n_outlier'] = int(len(fail))
        out['cwd'] = mean_se(wd[fin])[0]
        for j in range(3):
            good = fin & np.isfinite(lam[:, j])
            out[f'cw1{j + 1}'] = (w1_lam(lam[good, j], ref[good, j])
                                  if good.any() else np.nan)
        return out

    ref1 = ref[:, 0].mean()
    outlier = np.isfinite(lam[:, 0]) & (np.abs(lam[:, 0] - ref1) > OUTLIER_DLAM1)
    cond = ok & ~outlier

    out['n_outlier'] = int(outlier.sum())
    out['cwd'] = mean_se(wd[cond])[0]
    for j in range(3):
        good = cond & np.isfinite(lam[:, j])
        out[f'cw1{j + 1}'] = w1_lam(lam[good, j], ref[good, j]) if good.any() else np.nan
    return out


def across_seeds(per_seed, key):
    """(mean, std) over the five trainings, in float64.
    """
    v = np.array([per_seed[s].get(key, np.nan) for s in SEEDS], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0)


lyap_true = np.load(os.path.join(args.dir, 'onattr_lyaptrue_on.npz'))['lyap_true']
data = {}
for meth in METHODS:
    data[meth] = {}
    for s in SEEDS:
        f = os.path.join(args.dir, f'onattr_eval_on_{meth}_s{s}_m1.npz')
        with np.load(f) as d:
            data[meth][s] = stats(d, lyap_true)

KEYS = ['cwd', 'cw11', 'cw12', 'cw13']
NAMES = ['W1', 'W1_lam1', 'W1_lam2', 'W1_lam3']

if args.tex:
    for meth in METHODS:
        per = data[meth]
        noutl = '/'.join(str(per[s]['n_outlier']) for s in SEEDS)
        cells = []
        for k in KEYS:
            m, sd = across_seeds(per, k)
            if not np.isfinite(m):
                cells += ['N/A', 'N/A']
            elif k == 'cwd':
                cells += [f'${m:.4f}$', f'${sd:.4f}$']
            else:
                cells += [f'${m:.2E}$', f'${sd:.2E}$']
        dag = '$^{\\dagger}$' if per[SEEDS[0]]['fail'] >= 100.0 else ''
        print(f'{meth}{dag} & {noutl} & ' + ' & '.join(cells) + r' \\')
else:
    print('\nTraining-seed variability of the Lorenz 63 methods (m = 1), '
          'five seeds, 1,000 on-attractor ICs\n')
    head = (f"{'':7s} {'n_outlier per seed 32/33/34/35/36':>33s} | "
            + ' '.join(f'{n:>21s}' for n in NAMES))
    print(head)
    print('-' * len(head))
    for meth in METHODS:
        per = data[meth]
        noutl = '/'.join(str(per[s]['n_outlier']) for s in SEEDS)
        cells = []
        for k in KEYS:
            m, sd = across_seeds(per, k)
            cells.append('N/A' if not np.isfinite(m)
                         else (f'{m:8.4f} +/- {sd:6.4f}' if k == 'cwd'
                               else f'{m:9.3E} +/- {sd:7.1E}'))
        print(f'{meth:7s} {noutl:>33s} | ' + ' '.join(f'{c:>21s}' for c in cells))
