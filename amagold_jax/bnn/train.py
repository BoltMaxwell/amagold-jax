"""MNIST BNN with AMAGOLD / SGHMC / SGD, port of bnn/{train_amagold,train_sghmc,train_sgd}.py.

Model: 784-500-256-10 ReLU MLP with log-softmax NLL. Defaults reproduce the
original scripts: batch 2000, datasize 60000, lr = 5e-4 / 60000 (per-step),
weight decay 5e-4 (added to the sum-scale gradient), T = 10 leapfrog steps,
AMAGOLD beta = 5e-6, SGHMC alpha = 1e-5. AMAGOLD keeps a persistent momentum
buffer across outer iterations (negated on rejection), performs T-1 gradient
updates per outer loop with half position steps at the ends, accumulates
rho = 0.5 sum d_p (buf_old + buf_new), and M-H-tests against the full-data
mean NLL difference times datasize (the prior term is not part of the M-H
energy, exactly as in the original). Evaluation reports the current sample's
test accuracy/NLL (the original does no posterior averaging).

Usage: python -m amagold_jax.bnn.train --sampler amagold --init checkpoints/sgd_init_epoch3.npz
"""

import argparse
import functools
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.sgmcmc import diffusions
from jax.flatten_util import ravel_pytree

from . import data as data_mod

LAYERS = ((28 * 28, 500), (500, 256), (256, 10))


def gaussian_like(key, tree):
    """Flat standard-normal draw + unravel (same construction as blackjax)."""
    flat, unravel = ravel_pytree(tree)
    return unravel(jax.random.normal(key, flat.shape, flat.dtype))


def init_params(key):
    """torch.nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b."""
    params = {}
    for i, (fan_in, fan_out) in enumerate(LAYERS, start=1):
        key, kw, kb = jax.random.split(key, 3)
        bound = 1.0 / math.sqrt(fan_in)
        params[f"w{i}"] = jax.random.uniform(kw, (fan_in, fan_out), jnp.float32,
                                             -bound, bound)
        params[f"b{i}"] = jax.random.uniform(kb, (fan_out,), jnp.float32, -bound, bound)
    return params


def load_params(path):
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in d.files}


def forward(params, x):
    h = jax.nn.relu(x @ params["w1"] + params["b1"])
    h = jax.nn.relu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def nll_mean(params, x, y):
    logp = jax.nn.log_softmax(forward(params, x))
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])


def sum_scale_grad(params, x, y, datasize, weight_decay):
    """Gradient of datasize * mean NLL, plus weight_decay * p (the original's d_p)."""
    g = jax.grad(lambda p: nll_mean(p, x, y) * datasize)(params)
    return jax.tree_util.tree_map(lambda g_, p_: g_ + weight_decay * p_, g, params)


# ---------------------------------------------------------------------------
# AMAGOLD (bnn/amagold.py)
# ---------------------------------------------------------------------------

def make_amagold_outer(datasize, T, weight_decay, beta, lr):
    @jax.jit
    def outer(key, params, buf, xb, yb, x_full, y_full):
        """One outer loop over stacked batches xb (T, B, 784). Returns
        (params, buf, accepted, rho)."""
        old_params = params
        buf_init = buf
        # t = 0: position half step only
        params = jax.tree_util.tree_map(lambda p, b: p + 0.5 * b, params, buf)

        def leapfrog(carry, inp):
            params, buf, rho = carry
            t, x, y, subkey = inp
            d_p = sum_scale_grad(params, x, y, datasize, weight_decay)
            noise = gaussian_like(subkey, buf)
            buf_new = jax.tree_util.tree_map(
                lambda b, g, n: ((1.0 - beta) * b - lr * g
                                 + (lr * beta) ** 0.5 * 2.0 * n) / (1.0 + beta),
                buf, d_p, noise)
            rho = rho + 0.5 * sum(
                jnp.sum(d_p[k] * (buf[k] + buf_new[k])) for k in sorted(buf))
            scale = jnp.where(t == T - 1, 0.5, 1.0)
            params = jax.tree_util.tree_map(lambda p, b: p + scale * b, params, buf_new)
            return (params, buf_new, rho), None

        ts = jnp.arange(1, T)
        keys = jax.random.split(key, T)
        (params, buf, rho), _ = jax.lax.scan(
            leapfrog, (params, buf, jnp.zeros(())), (ts, xb[1:], yb[1:], keys[: T - 1]))

        u_new = nll_mean(params, x_full, y_full)
        u_old = nll_mean(old_params, x_full, y_full)
        a = jnp.exp((u_old - u_new) * datasize + rho)
        accept = jax.random.uniform(keys[T - 1]) <= a
        params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(accept, new, old), params, old_params)
        buf = jax.tree_util.tree_map(
            lambda new, init: jnp.where(accept, new, -init), buf, buf_init)
        return params, buf, accept, rho

    return outer


# ---------------------------------------------------------------------------
# SGHMC (bnn/train_sghmc.py), jax + blackjax backends
# ---------------------------------------------------------------------------

def make_sghmc_epoch(datasize, weight_decay, alpha, lr, backend="jax"):
    if backend == "blackjax":
        h = math.sqrt(lr)
        one_step = diffusions.sghmc(alpha=alpha / h, beta=0.0)

    @jax.jit
    def epoch(key, params, buf, xb, yb):
        """One epoch over stacked shuffled batches xb (nb, B, 784)."""

        def body(carry, inp):
            params, buf = carry
            x, y, subkey = inp
            d_p = sum_scale_grad(params, x, y, datasize, weight_decay)
            if backend == "jax":
                noise = gaussian_like(subkey, buf)
                buf = jax.tree_util.tree_map(
                    lambda b, g, n: (1.0 - alpha) * b - lr * g
                    + (2.0 * lr * alpha) ** 0.5 * n,
                    buf, d_p, noise)
                params = jax.tree_util.tree_map(lambda p, b: p + b, params, buf)
            else:
                g_asc = jax.tree_util.tree_map(jnp.negative, d_p)
                params, buf = one_step(subkey, params, buf, g_asc, h)
            return (params, buf), None

        keys = jax.random.split(key, xb.shape[0])
        (params, buf), _ = jax.lax.scan(body, (params, buf), (xb, yb, keys))
        return params, buf

    return epoch


# ---------------------------------------------------------------------------
# SGD (bnn/train_sgd.py) for generating the initialization checkpoint
# ---------------------------------------------------------------------------

def make_sgd_epoch(lr, momentum, weight_decay):
    @jax.jit
    def epoch(params, buf, xb, yb):
        def body(carry, inp):
            params, buf = carry
            x, y = inp
            g = jax.grad(nll_mean)(params, x, y)
            g = jax.tree_util.tree_map(lambda g_, p_: g_ + weight_decay * p_, g, params)
            buf = jax.tree_util.tree_map(lambda b, g_: momentum * b + g_, buf, g)
            params = jax.tree_util.tree_map(lambda p, b: p - lr * b, params, buf)
            return (params, buf), None

        (params, buf), _ = jax.lax.scan(body, (params, buf), (xb, yb))
        return params, buf

    return epoch


def init_momentum(key, params, scale):
    return jax.tree_util.tree_map(lambda n: scale * n, gaussian_like(key, params))


@jax.jit
def evaluate(params, x, y):
    logp = jax.nn.log_softmax(forward(params, x))
    nll = -jnp.mean(logp[jnp.arange(y.shape[0]), y])
    acc = jnp.mean((jnp.argmax(logp, axis=1) == y).astype(jnp.float32))
    return nll, acc


def shuffled_batches(rng, x, y, batch_size, num_batches):
    perm = rng.permutation(x.shape[0])[: num_batches * batch_size]
    idx = perm.reshape(num_batches, batch_size)
    return jnp.asarray(x[idx]), jnp.asarray(y[idx])


def train(args):
    x_train, y_train = data_mod.load(args.data_path, "train")
    x_test, y_test = data_mod.load(args.data_path, "test")
    datasize = x_train.shape[0]
    lr = args.lr / datasize
    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)

    if args.init:
        params = load_params(args.init)
    else:
        key, sub = jax.random.split(key)
        params = init_params(sub)

    x_test_d, y_test_d = jnp.asarray(x_test), jnp.asarray(y_test)
    x_full, y_full = jnp.asarray(x_train), jnp.asarray(y_train)
    history = []
    t_start = time.perf_counter()

    if args.sampler == "amagold":
        outer = make_amagold_outer(datasize, args.T, args.weight_decay, args.beta, lr)
        key, sub = jax.random.split(key)
        buf = init_momentum(sub, params, math.sqrt(lr))
        succ = 0
        for it in range(1, args.iters + 1):
            xb, yb = shuffled_batches(rng, x_train, y_train, args.batch_size, args.T)
            key, sub = jax.random.split(key)
            params, buf, accept, _ = outer(sub, params, buf, xb, yb, x_full, y_full)
            succ += int(accept)
            if it % args.eval_interval == 0:
                nll, acc = evaluate(params, x_test_d, y_test_d)
                row = {"iter": it, "test-nll": float(nll), "test-acc": float(acc),
                       "accept-rate": succ / it,
                       "wall": time.perf_counter() - t_start}
                history.append(row)
                print(f"[{it}] test-nll:{row['test-nll']:.4f} "
                      f"test-acc:{row['test-acc']:.4f} accept:{row['accept-rate']:.4f}",
                      file=sys.stderr)
    elif args.sampler == "sghmc":
        epoch_fn = make_sghmc_epoch(datasize, args.weight_decay, args.alpha, lr,
                                    args.backend)
        key, sub = jax.random.split(key)
        scale = math.sqrt(lr) if args.backend == "jax" else 1.0  # blackjax buf is v = p/sqrt(lr)
        buf = init_momentum(sub, params, scale)
        nb = datasize // args.batch_size
        for ep in range(1, args.epochs + 1):
            xb, yb = shuffled_batches(rng, x_train, y_train, args.batch_size, nb)
            key, sub = jax.random.split(key)
            params, buf = epoch_fn(sub, params, buf, xb, yb)
            if ep % args.eval_interval == 0:
                nll, acc = evaluate(params, x_test_d, y_test_d)
                row = {"epoch": ep, "test-nll": float(nll), "test-acc": float(acc),
                       "wall": time.perf_counter() - t_start}
                history.append(row)
                print(f"[{ep}] test-nll:{row['test-nll']:.4f} "
                      f"test-acc:{row['test-acc']:.4f}", file=sys.stderr)
                if not np.isfinite(row["test-nll"]):
                    print(f"diverged at epoch {ep}", file=sys.stderr)
                    break
    else:  # sgd
        epoch_fn = make_sgd_epoch(args.lr, 1.0 - args.momentum, args.weight_decay)
        buf = jax.tree_util.tree_map(jnp.zeros_like, params)
        nb = datasize // args.batch_size
        for ep in range(1, args.epochs + 1):
            xb, yb = shuffled_batches(rng, x_train, y_train, args.batch_size, nb)
            params, buf = epoch_fn(params, buf, xb, yb)
            nll, acc = evaluate(params, x_test_d, y_test_d)
            history.append({"epoch": ep, "test-nll": float(nll), "test-acc": float(acc)})
            print(f"[{ep}] test-nll:{float(nll):.4f} test-acc:{float(acc):.4f}",
                  file=sys.stderr)
        if args.save_init:
            os.makedirs(os.path.dirname(args.save_init) or ".", exist_ok=True)
            np.savez(args.save_init, **{k: np.asarray(v) for k, v in params.items()})
            print(f"saved {args.save_init}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = f"{args.sampler}_{args.backend}_seed{args.seed}"
        with open(os.path.join(args.out, f"bnn_{tag}.json"), "w") as f:
            json.dump({"args": vars(args), "history": history}, f)
        print(f"saved {args.out}/bnn_{tag}.json")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/mnist")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--sampler", choices=("amagold", "sghmc", "sgd"), default="amagold")
    ap.add_argument("--backend", choices=("jax", "blackjax"), default="jax",
                    help="sghmc only; amagold has no blackjax equivalent")
    ap.add_argument("--init", default=None, help="npz checkpoint (converted .pt)")
    ap.add_argument("--iters", type=int, default=2500, help="amagold outer iterations")
    ap.add_argument("--epochs", type=int, default=1000, help="sghmc/sgd epochs")
    ap.add_argument("--batch_size", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--beta", type=float, default=5e-6)
    ap.add_argument("--alpha", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.5, help="sgd only (as 1-m)")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--eval_interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--save_init", default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    if args.sampler == "sgd":
        ap.set_defaults()
        if args.lr == 0.0005:
            args.lr = 0.1
        if args.batch_size == 2000:
            args.batch_size = 500
        if args.epochs == 1000:
            args.epochs = 3
    if args.download:
        data_mod.download(args.data_path)
    train(args)


if __name__ == "__main__":
    main()
