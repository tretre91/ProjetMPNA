#!/usr/bin/env python3

import argparse
import csv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file", type=argparse.FileType("r"), help="csv result file")
parser.add_argument("-o", type=str, help="Output file")
args = parser.parse_args()

reader = csv.DictReader(args.file, delimiter=";")


nb_threads = []
mean = []
stddev = []
nb_threads_blas = []
mean_blas = []
stddev_blas = []

for row in reader:
    s = row["s"]
    m = row["m"]
    n = row["n"]
    if row["version"] == "blas":
        nb_threads_blas.append(int(row["n"]))
        mean_blas.append(float(row["mean"]))
        stddev_blas.append(float(row["dev"]))
    else:
        nb_threads.append(int(row["n"]))
        mean.append(float(row["mean"]))
        stddev.append(float(row["dev"]))

fig, ax = plt.subplots(1, dpi=250)

ax.plot(nb_threads_blas, mean_blas, label="BLAS")
ax.plot(nb_threads, mean, label="classique")

ax.set_xlabel("taille de matrice")
ax.set_ylabel("Temps d'exécution (s)")
ax.set_title(f"Temps d'exécution séquentiel de l'algorithme PRR, pour le calcul de {s} valeurs propres dans un sous espace de taille {m}", wrap=True)

if args.o is not None:
    fig.savefig(args.o)
else:
    plt.show()
