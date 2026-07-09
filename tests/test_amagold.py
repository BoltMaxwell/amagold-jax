"""Tests for the AMAGOLD port (pytest or direct run).

Run:  JAX_PLATFORMS=cpu python tests/test_amagold.py
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from amagold_jax import doublewell, samplers
from amagold_jax.bnn import train as bnn_train

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_amagold_exact_gradient_unbiased_at_large_dt():
    """With the M-H test and exact gradients, AMAGOLD must sample N(0,1)
    accurately even at dt = 0.5 (where plain SGHMC is visibly biased)."""
    u = lambda x: 0.5 * x**2
    grad = lambda key, x: x
    step = samplers.amagold_kernel(u, grad, dt=0.5, nstep=10, C=0.5, mh=True)

    def body(x, key):
        x, acc = step(key, x)
        return x, (x, acc)

    _, (xs, accs) = jax.lax.scan(body, jnp.zeros(()),
                                 jax.random.split(jax.random.key(0), 20_000))
    xs = np.asarray(xs[2000:])
    assert abs(xs.mean()) < 0.05, xs.mean()
    assert abs(xs.var() - 1.0) < 0.05, xs.var()
    assert np.asarray(accs).mean() > 0.5


def test_sghmc_biased_where_amagold_is_not():
    """At the double-well settings SGHMC's density error must exceed AMAGOLD's
    substantially (the paper's core claim, reproduced in miniature)."""
    xs_am, _ = doublewell.run("amagold", 20_000, seed=1)
    xs_sg, _ = doublewell.run("sghmc", 20_000, seed=1)
    ytrue = doublewell.true_density()
    l1_am = np.abs(doublewell.hist_density(xs_am) - ytrue).sum() * doublewell.XSTEP
    l1_sg = np.abs(doublewell.hist_density(xs_sg) - ytrue).sum() * doublewell.XSTEP
    assert l1_sg > 2.0 * l1_am, (l1_am, l1_sg)


def test_bnn_forward_matches_torch_checkpoint():
    """The JAX model on the converted checkpoint must reproduce the original
    torch model's outputs on random inputs (model + conversion equivalence)."""
    ckpt_pt = os.path.join(REPO, "bnn", "checkpoints", "sgd_init_epoch3.pt")
    ckpt_npz = os.path.join(REPO, "checkpoints", "sgd_init_epoch3.npz")
    if not (os.path.exists(ckpt_pt) and os.path.exists(ckpt_npz)):
        print("  (skipped: checkpoints not present)")
        return
    import torch
    import torch.nn.functional as F

    sd = torch.load(ckpt_pt, map_location="cpu", weights_only=True)
    x = np.random.default_rng(0).normal(size=(32, 784)).astype(np.float32)
    xt = torch.from_numpy(x)
    h = F.relu(xt @ sd["fc1.weight"].T + sd["fc1.bias"])
    h = F.relu(h @ sd["fc2.weight"].T + sd["fc2.bias"])
    logits_t = (h @ sd["fc3.weight"].T + sd["fc3.bias"]).numpy()
    params = bnn_train.load_params(ckpt_npz)
    logits_j = np.asarray(bnn_train.forward(params, jnp.asarray(x)))
    np.testing.assert_allclose(logits_j, logits_t, rtol=1e-5, atol=1e-5)


def test_bnn_amagold_accepts_at_tiny_step():
    """As lr -> 0 the leapfrog is exact, so the M-H test must accept."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 784)).astype(np.float32)
    y = rng.integers(0, 10, size=40).astype(np.int32)
    key = jax.random.key(0)
    key, sub = jax.random.split(key)
    params = bnn_train.init_params(sub)
    lr = 1e-12
    outer = bnn_train.make_amagold_outer(datasize=40, T=4, weight_decay=5e-4,
                                         beta=5e-6, lr=lr)
    buf = bnn_train.init_momentum(jax.random.key(1), params, float(np.sqrt(lr)))
    xb = jnp.asarray(np.stack([x[:10], x[10:20], x[20:30], x[30:]]))
    yb = jnp.asarray(np.stack([y[:10], y[10:20], y[20:30], y[30:]]))
    accepts = []
    for i in range(5):
        key, sub = jax.random.split(key)
        params, buf, acc, _ = outer(sub, params, buf, xb, yb,
                                    jnp.asarray(x), jnp.asarray(y))
        accepts.append(bool(acc))
    assert all(accepts), accepts


def test_bnn_sghmc_blackjax_mapping_exact():
    """One blackjax SGHMC step must equal the hand-computed original update in
    p-space given the same noise draw: p' = (1 - alpha) p - lr d_p +
    sqrt(2 lr alpha) n, with the position advanced by the previous momentum
    (blackjax's qp ordering; the jax backend uses the original pq ordering,
    a documented splitting difference, so the sequences are not compared
    stepwise)."""
    import math

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 20, 784)).astype(np.float32)
    y = rng.integers(0, 10, size=(1, 20)).astype(np.int32)
    key = jax.random.key(0)
    params = bnn_train.init_params(jax.random.key(1))
    lr, alpha, wd = 0.0005 / 60000, 1e-5, 5e-4
    h = math.sqrt(lr)
    v = bnn_train.init_momentum(jax.random.key(2), params, 1.0)

    epoch = bnn_train.make_sghmc_epoch(60000, wd, alpha, lr, "blackjax")
    p_out, v_out = epoch(key, params, v, jnp.asarray(x), jnp.asarray(y))

    subkey = jax.random.split(key, 1)[0]  # the key the scan body received
    d_p = bnn_train.sum_scale_grad(params, jnp.asarray(x[0]), jnp.asarray(y[0]),
                                   60000, wd)
    noise = bnn_train.gaussian_like(subkey, params)
    for k in sorted(params):
        exp_x = np.asarray(params[k]) + h * np.asarray(v[k])
        exp_p = ((1.0 - alpha) * h * np.asarray(v[k])
                 - lr * np.asarray(d_p[k])
                 + math.sqrt(2.0 * lr * alpha) * np.asarray(noise[k]))
        np.testing.assert_allclose(np.asarray(p_out[k]), exp_x, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(h * np.asarray(v_out[k]), exp_p, rtol=1e-4, atol=1e-10)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
