"""Convert the original PyTorch SGD-init checkpoint to npz for the JAX port.

torch Linear stores weight as (out, in); the JAX params use (in, out).

Usage: python -m amagold_jax.bnn.convert_checkpoint bnn/checkpoints/sgd_init_epoch3.pt \
       checkpoints/sgd_init_epoch3.npz
"""

import sys

import numpy as np
import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    out = {}
    for i in (1, 2, 3):
        out[f"w{i}"] = sd[f"fc{i}.weight"].numpy().T.copy()
        out[f"b{i}"] = sd[f"fc{i}.bias"].numpy().copy()
    np.savez(dst, **out)
    print(f"saved {dst}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
