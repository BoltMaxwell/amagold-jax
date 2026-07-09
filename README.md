amagold-jax
===========

A JAX reimplementation of the experiments from

> Ruqi Zhang, A. Feder Cooper, Christopher De Sa.
> "AMAGOLD: Amortized Metropolis Adjustment for Efficient Stochastic
> Gradient MCMC". AISTATS 2020. [arXiv:2003.00193](https://arxiv.org/abs/2003.00193)

This repository started as a fork of the original code,
[ruqizhang/amagold](https://github.com/ruqizhang/amagold), and now contains
only the JAX port. Every experiment was verified against the original before
removal — including the authors' own cached simulation samples that shipped
in the repo — see [docs/verification.md](docs/verification.md). The port was
written with the assistance of Claude Code; all credit for the method and the
experimental design belongs to the original authors:

```bibtex
@inproceedings{zhang2020amagold,
  title={{AMAGOLD}: Amortized {M}etropolis adjustment for efficient
         stochastic gradient {MCMC}},
  author={Zhang, Ruqi and Cooper, A. Feder and De Sa, Christopher},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2020}
}
```

Scope of this port
------------------

| original | status |
|---|---|
| `simulation/` — double-well density, AMAGOLD vs SGHMC (matlab) | ported (`amagold_jax/doublewell.py`) |
| `bnn/` — MNIST BNN: AMAGOLD, SGHMC, SGD init (PyTorch) | ported (`amagold_jax/bnn/`) |
| SGD-init checkpoint `sgd_init_epoch3.pt` | converted to `checkpoints/sgd_init_epoch3.npz` (bit-equivalent forward pass, tested) |

AMAGOLD has no blackjax implementation, so the kernel is pure JAX; the SGHMC
baseline additionally has a blackjax-backed variant (same diffusion-mapping
and qp-splitting caveats as documented in the sibling
[SGHMC-jax](https://github.com/BoltMaxwell/SGHMC-jax) port).

Install
-------

```bash
pip install -e '.[plot]'
```

Double-well simulation
----------------------

```bash
python -m amagold_jax.doublewell --plot     # AMAGOLD + SGHMC, 100k samples each
```

Reproduces `doublewell_{amagold,sghmc}.m`: dt = 0.25, C = 0.5, 10 leapfrog
steps, 1000 burn-in. AMAGOLD's amortized M-H test removes the step-size bias
that SGHMC exhibits at this dt (L1 density error 0.026 vs 0.177; the authors'
cached runs give 0.025 vs 0.175).

Bayesian NN on MNIST
--------------------

```bash
python -m amagold_jax.bnn.train --download --sampler amagold \
    --init checkpoints/sgd_init_epoch3.npz          # 2500 outer iterations
python -m amagold_jax.bnn.train --sampler sghmc --backend jax   # diverges ~epoch 500,
                                                    # exactly like the original README says
```

Defaults reproduce the original scripts: 784-500-256-10 ReLU MLP, batch 2000,
lr 5e-4 (scaled by 1/60000), weight decay 5e-4, T = 10 leapfrog steps,
AMAGOLD beta 5e-6 / SGHMC alpha 1e-5, initialization from the original SGD
checkpoint, evaluation of the current sample every 10 iterations.

Results (verification against the original)
-------------------------------------------

Full tables in [docs/verification.md](docs/verification.md). Highlights:

- Double-well: JAX matches both the authors' cached samples and a fresh
  Octave rerun of the original matlab within Monte-Carlo noise (moments agree
  to ~3 decimals).
- The JAX BNN forward pass on the converted checkpoint matches the torch
  model to 1e-5 (tested).
- BNN trajectories track the original at every checkpoint (final test acc
  96.23% torch vs 96.34% JAX over 2500 iterations), and the original README's
  documented SGHMC divergence "after about 500 epochs" occurs at epoch 509
  (torch) vs 510 (JAX).

GPU benchmark: original PyTorch vs JAX
--------------------------------------

H100, steady-state, identical configurations (original run with a
device/timing-only patch; details and caveats in
[docs/verification.md](docs/verification.md)):

| workload | original PyTorch | JAX port | speedup |
|---|---|---|---|
| AMAGOLD outer iteration | 884.5 ms | 25.1 ms | **35x** |
| SGHMC epoch | 476.6 ms | 64.2 ms | **7.4x** |
| AMAGOLD iteration on CPU (M-series) | 6.4 s | 0.24 s | ~27x |

The AMAGOLD gap is larger because the original pays a fresh full-dataset
DataLoader pass and a model deepcopy every outer iteration for the amortized
M-H test, while the JAX port keeps data device-resident inside one jitted
step.

Tests
-----

```bash
JAX_PLATFORMS=cpu python tests/test_amagold.py    # or pytest tests/
```
