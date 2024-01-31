import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

df = pd.read_csv('results/time.dat', delimiter=';')

df_blas = df[df["version"] == "blas"] 
df_noblas = df[df["version"] == "no_blas"] 

time_blas = df_blas["time"]
time_noblas = df_noblas["time"]
nthreads = df_blas["nthreads"]

plt.figure(figsize=(8, 5))
plt.plot(nthreads, time_blas, label="blas" )
plt.scatter(nthreads, time_blas )
plt.plot(nthreads, time_noblas, label="noblas" )
plt.scatter(nthreads, time_noblas )


plt.xlabel("n threads")
plt.ylabel("time (s)")
# plt.yscale("log", base=10)
plt.xscale("log", base=2)
plt.title("todo")
plt.savefig('img/time.png')
plt.show()



