"""MNIST loading with torchvision-equivalent normalization ((x/255 - 0.1307)/0.3081)."""

import gzip
import os
import struct
import urllib.request

import numpy as np

MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
FILES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}
MEAN, STD = 0.1307, 0.3081


def download(root):
    os.makedirs(root, exist_ok=True)
    for img, lbl in FILES.values():
        for name in (img, lbl):
            path = os.path.join(root, name)
            if os.path.exists(path):
                continue
            last_err = None
            for mirror in MIRRORS:
                try:
                    print(f"downloading {mirror + name}.gz")
                    with urllib.request.urlopen(mirror + name + ".gz") as r:
                        data = gzip.decompress(r.read())
                    with open(path, "wb") as f:
                        f.write(data)
                    break
                except Exception as e:
                    last_err = e
            else:
                raise RuntimeError(f"could not download {name}") from last_err


def load(root, split):
    fname_img, fname_lbl = FILES[split]
    with open(os.path.join(root, fname_lbl), "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
    with open(os.path.join(root, fname_img), "rb") as f:
        _, size, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows * cols)
    x = (images.astype(np.float32) / 255.0 - MEAN) / STD
    return x, labels
