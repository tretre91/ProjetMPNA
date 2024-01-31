# generer dossier de build

a la racine executez la commande cmake suivante :
```
#cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Il peut necessaire de preciser le `vendor` de la lib Lapack :

```
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DBLA_VENDOR=OpenBLAS
```

# compiler

```
cd build
make
```

# executer

l'executable se nomme `prr`, et `prr_blas` pour la version faisant des appels a blas.
Un exemple d'execution :
```
OMP_NUM_THREADS=4 ./prr ../data/papier.mat -s 2 -m 4 -v -i 100
OMP_NUM_THREADS=8 ./prr_bench_blas   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100
```




# Comment compiler sur ROMEO

```
git clone git@github.com:tretre91/ProjetMPNA.git prr_project
cd prr_project
git checkout omp

module load llvm/11.0.0
module load cmake/3.27.9_spack2021_gcc-12.2.0-rvyk
module load gcc/10.2.0_spack2021_gcc-10.2.0-gdnn
module load 2018/netlib-lapack/3.8.0-gcc-9.1.0-dpxqu7i
module load openblas/0.3.25_spack2021_gcc-12.2.0-l4ru


cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DBLA_VENDOR=OpenBLAS
cd build
sbatch ../scripts/exec.slurm
```



