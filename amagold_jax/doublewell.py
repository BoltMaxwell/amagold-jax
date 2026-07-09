"""Double-well simulation, port of simulation/doublewell_{amagold,sghmc}.m.

U(x) = (x+4)(x+1)(x-1)(x-3)/14 + 0.5, stochastic gradient = exact + N(0,1).
100,000 samples after 1,000 burn-in steps; AMAGOLD uses the amortized M-H
test, SGHMC none. dt = 0.25, C = 0.5, nstep = 10.

Usage: python -m amagold_jax.doublewell [--sampler both] [--nsample 100000] [--plot]
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from .samplers import amagold_kernel, sghmc_kernel

XSTEP = 0.1
GRID = np.arange(-6.0, 6.0 + 1e-9, XSTEP)
DT, C, NSTEP = 0.25, 0.5, 10
BURNIN = 1000


def u_fn(x):
    return (x + 4.0) * (x + 1.0) * (x - 1.0) * (x - 3.0) / 14.0 + 0.5


def grad_u_exact(x):
    return (4.0 * x**3 + 3.0 * x**2 - 26.0 * x - 1.0) / 14.0


def grad_u(key, x):
    return grad_u_exact(x) + jax.random.normal(key)


def run(sampler, nsample, seed=0):
    if sampler == "amagold":
        step = amagold_kernel(u_fn, grad_u, dt=DT, nstep=NSTEP, C=C, mh=True)

        def body(x, key):
            x, acc = step(key, x)
            return x, (x, acc)

        keys = jax.random.split(jax.random.key(seed), nsample + BURNIN)
        _, (xs, accs) = jax.lax.scan(body, jnp.zeros(()), keys)
        return np.asarray(xs[BURNIN:]), float(np.asarray(accs[BURNIN:]).mean())
    step = sghmc_kernel(grad_u, dt=DT, nstep=NSTEP, C=C)

    def body(x, key):
        x = step(key, x)
        return x, x

    keys = jax.random.split(jax.random.key(seed), nsample + BURNIN)
    _, xs = jax.lax.scan(body, jnp.zeros(()), keys)
    return np.asarray(xs[BURNIN:]), None


def hist_density(samples):
    edges = np.concatenate([[-np.inf], (GRID[:-1] + GRID[1:]) / 2, [np.inf]])
    y, _ = np.histogram(samples, bins=edges)
    return y / y.sum() / XSTEP


def true_density():
    y = np.exp(-u_fn(GRID))
    return y / y.sum() / XSTEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=("amagold", "sghmc", "both"), default="both")
    ap.add_argument("--nsample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    samplers = ("amagold", "sghmc") if args.sampler == "both" else (args.sampler,)
    ytrue = true_density()
    for name in samplers:
        xs, acc_rate = run(name, args.nsample, args.seed)
        dens = hist_density(xs)
        l1 = np.abs(dens - ytrue).sum() * XSTEP
        acc = f"  acceptance {acc_rate:.4f}" if acc_rate is not None else ""
        print(f"{name}: L1 distance to true density {l1:.4f}{acc}")
        np.savez(os.path.join(args.out, f"doublewell_{name}.npz"),
                 samples=xs.astype(np.float32), grid=GRID, density=dens, true=ytrue)
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(5, 4))
            plt.plot(GRID, ytrue, "b-", label="True")
            plt.plot(GRID, dens, "r--", label=name.upper())
            plt.xlabel("x"); plt.ylabel("density"); plt.legend()
            plt.tight_layout()
            path = os.path.join(args.out, f"doublewell_{name}.png")
            plt.savefig(path, dpi=150); plt.close()
            print(f"saved {path}")


if __name__ == "__main__":
    main()
