#!/usr/bin/env python3

## Dana Tran
## CS 5330 Project 5
## 04/03/26


import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets

#get mnist test split
def get_mnist_testset(data_dir: Path):
    return datasets.MNIST(root=str(data_dir), train=False, download=True)


#2x3 grid of first six test digits
def plot_first_six_test_digits(testset, output_path: Path):
    indices = list(range(6))

    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    axes = axes.flatten()

    for ax, i in zip(axes, indices):
        img, label = testset[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


#arg + save plot png
def main(argv):
    parser = argparse.ArgumentParser(description="Plot first 6 MNIST test digits")
    parser.add_argument(
        "--data-dir",
        default="mnist_data",
        help="Where to store/download MNIST data (default: mnist_data)",
    )
    parser.add_argument(
        "--output",
        default="plot-first-six-test-digits.png",
        help="Output PNG filename (default: plot-first-six-test-digits.png)",
    )
    args = parser.parse_args(argv[1:])

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    testset = get_mnist_testset(data_dir)
    plot_first_six_test_digits(testset, output_path)

    print(f"Saved plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main(sys.argv)