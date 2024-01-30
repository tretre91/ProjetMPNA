import numpy as np

import argparse

np.set_printoptions(linewidth=300, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Input file containing the matrix")
parser.add_argument("--original", type=str, help="File containing the original matrix")
args = parser.parse_args()

if args.original is not None:
    original = np.loadtxt(args.original, skiprows=1)
    or_eigvals = np.linalg.eigvals(original)
    print("Original:", sorted(or_eigvals, reverse=True, key=np.abs))
    print("")
    print(np.linalg.eig(original)[1])
    print("")

mat = np.loadtxt(args.file, skiprows=1)
eigvals = np.linalg.eigvals(mat)

eigvals, eigvecs = np.linalg.eig(mat)

indices = np.argsort(np.abs(eigvals))

for i in indices[::-1]:
    print(f"{eigvals[i]}: {eigvecs[:, i]}")

