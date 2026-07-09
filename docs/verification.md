# Verification of the JAX port against the original implementation

The original repo ships two experiments with reference material: the matlab
double-well simulation **with the authors' cached sample files**
(`simulation/*_doublewell.mat`), and the PyTorch MNIST BNN with a committed
SGD-initialization checkpoint. Both were used, plus fresh reruns of the
originals (matlab under GNU Octave 11.3 with the original kernels untouched;
the PyTorch scripts run unmodified under torch 2.x on CPU, and with a
device/timing-only patch on GPU).

## Double-well simulation (100,000 samples, dt = 0.25, C = 0.5, nstep = 10)

L1 distance between the sampled histogram and the analytic density, plus
sample moments:

| source | AMAGOLD L1 | AMAGOLD mean/var | SGHMC L1 | SGHMC mean/var |
|---|---|---|---|---|
| authors' cached samples (in repo) | 0.0247 | -2.122 / 2.934 | 0.1747 | -1.922 / 3.423 |
| fresh Octave rerun (original .m) | 0.0212 | -2.153 / 2.854 | 0.1784 | -1.898 / 3.507 |
| JAX port | 0.0257 | -2.127 / 2.928 | 0.1773 | -1.921 / 3.427 |

All three agree within Monte-Carlo noise; the paper's core claim (AMAGOLD's
M-H correction removes the large-step-size bias that SGHMC suffers at
dt = 0.25) is reproduced quantitatively. The JAX AMAGOLD acceptance rate is
0.724 over the run.

## BNN model equivalence

The JAX model evaluated on the converted original checkpoint
(`sgd_init_epoch3.pt` -> npz) reproduces the torch model's logits to 1e-5 on
random inputs (`tests/test_amagold.py::test_bnn_forward_matches_torch_checkpoint`),
verifying the architecture, parameter conversion, and normalization.

## BNN MNIST (784-500-256-10, batch 2000, lr 5e-4, T = 10, beta 5e-6)

Both start from the same committed SGD-init checkpoint. Original = the
original PyTorch scripts, unmodified (CPU) / device-patched (GPU).

Short-run comparison (60 outer iterations, CPU, independent RNG):

| | test acc @60 | test NLL @60 | acceptance @60 |
|---|---|---|---|
| original PyTorch | 96.31% | 0.1243 | 0.283 |
| JAX port | 96.17% | 0.1260 | 0.317 |

Full runs (2500 AMAGOLD iterations; SGHMC 1000 epochs at lr 5e-4, alpha 1e-5,
where the original README documents divergence after ~500 epochs), H100,
independent RNG streams:

| | acc @500 | acc @2500 | NLL @2500 | cumulative acceptance @500 / @2500 |
|---|---|---|---|---|
| original PyTorch AMAGOLD | 96.25% | 96.23% | 0.1277 | 0.212 / 0.150 |
| JAX AMAGOLD | 96.31% | 96.34% | 0.1186 | 0.168 / 0.088 |

| | first epoch below 90% acc | final state |
|---|---|---|
| original PyTorch SGHMC | 509 | collapsed (9.6% acc) |
| JAX SGHMC (jax backend) | 510 | collapsed (8.9% acc) |
| JAX SGHMC (blackjax backend) | 700 | collapsed (9.8% acc) |

Accuracy and NLL track the original at every checkpoint, and the documented
SGHMC divergence occurs at epoch 509 vs 510 — essentially identical. The
cumulative M-H acceptance decays in both implementations (0.27 -> 0.15 torch,
0.30 -> 0.09 JAX); the late-run gap should be read with the caveat that the
M-H energy is datasize x a float32 full-data mean-NLL difference, so its
exponent carries O(1) rounding noise in *both* implementations, making the
late-run (near-zero-signal) acceptance statistic implementation-sensitive.
The blackjax SGHMC variant diverges later (epoch 700), consistent with its
qp-splitting differences documented in SGHMC-jax.

## GPU benchmark (H100, steady-state, same configurations)

Timing from synchronized wall-clock stamps every 10 iterations/epochs
(`torch.cuda.synchronize()` for PyTorch; JAX walls recorded after blocking
evaluation). The PyTorch numbers are the original implementation as written
(device moved to CUDA and torchvision-free in-memory loaders substituted —
verified byte-identical trajectories on CPU): its per-iteration cost includes
the original's fresh full-data DataLoader pass and model deepcopy per outer
loop, which JAX's jitted, device-resident implementation does not pay.

| workload | original PyTorch | JAX port | speedup |
|---|---|---|---|
| AMAGOLD outer iteration (T=10 leapfrog + full-data M-H) | 884.5 ms | 25.1 ms | 35x |
| SGHMC epoch (30 minibatches) | 476.6 ms | 64.2 ms | 7.4x |
| SGHMC epoch (blackjax backend) | — | 59.2 ms | 8.0x |
| AMAGOLD iteration, CPU (Apple M-series) | 6.4 s | 0.24 s | ~27x |

## Faithfulness notes

- The AMAGOLD BNN update replicates the original exactly: persistent momentum
  across outer loops (negated on rejection), T-1 gradient steps with half
  position steps at both ends, rho = 0.5 sum d_p (buf_old + buf_new),
  semi-implicit friction division by (1 + beta), noise 2 sqrt(lr beta), and
  an M-H energy of datasize x full-data mean-NLL difference plus rho — the
  weight-decay (prior) term is *not* part of the M-H energy, exactly as in
  `bnn/amagold.py`.
- The original evaluates the current sample only (no posterior averaging);
  the port does the same.
- blackjax has no AMAGOLD kernel, so AMAGOLD is pure JAX; the SGHMC baseline
  has a blackjax-backed variant whose coefficient mapping is verified
  algebraically in the tests (same qp-vs-pq splitting caveat as documented in
  SGHMC-jax).
