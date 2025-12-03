import os
from pathlib import Path
import urllib.request
import tarfile

# Paths
cifar_dir = Path("cifar-10-batches-py")
cifar_tar = Path("cifar-10-python.tar.gz")
imagenet_file = Path("imagenet_val_25.npz")

# Download CIFAR-10 if not exists
if not cifar_dir.exists():
    print("Downloading CIFAR-10...")
    urllib.request.urlretrieve(
        "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        cifar_tar
    )
    # Extract
    with tarfile.open(cifar_tar) as tar:
        tar.extractall()
    cifar_tar.unlink()  # remove tar file

# Download ImageNet validation subset if not exists
if not imagenet_file.exists():
    print("Downloading ImageNet validation subset...")
    urllib.request.urlretrieve(
        "http://cs231n.stanford.edu/imagenet_val_25.npz",
        imagenet_file
    )

print("Datasets ready!")
