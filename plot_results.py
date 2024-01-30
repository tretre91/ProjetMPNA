#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file", type=argparse.FileType("r"), help="Input file containing the results to plot")
parser.add_argument("-o", type=str, help="Output file for the plot")
parser.add_argument("-s", "--scale", type=str, help="Scale to use for the plot", default="log")
args = parser.parse_args()

results = np.loadtxt(args.file)

fig, ax = plt.subplots(1, dpi=250)

ax.plot(results[:, 0], results[:, 1], linewidth=0.5)
ax.set_xlabel("Itération")
ax.set_ylabel("Epsilon")
ax.set_yscale(args.scale)

if (args.o is not None):
    fig.savefig(args.o)
else:
    plt.show()
