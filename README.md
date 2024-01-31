# Générer le dossier de build

À la racine, exécutez la commande cmake suivante :
```
cmake -S . -B build
```

Il peut être necessaire de preciser le fournisseur de la bibliothèque BLAS/LAPACK à utiliser pour que CMake puisse la trouver (voir [la documentation de CMake](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors) pour une liste des valeurs acceptées) :

```
cmake -S . -B build -DBLA_VENDOR=OpenBLAS
```

# Compiler

```
cd build
make      # ou cmake --build .
```

# Exécuter

l'executable se nomme `prr`, et `prr_blas` pour la version faisant des appels a blas.
Un exemple d'exécution :
```
OMP_NUM_THREADS=4 ./prr ../data/papier.mat -s 2 -m 4 -v -i 100
OMP_NUM_THREADS=8 ./prr_blas   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100
```

# Comment compiler sur ROMEO

```
git clone https://github.com/tretre91/ProjetMPNA.git prr_project
cd prr_project
git checkout omp

module load llvm/11.0.0
module load cmake/3.27.9_spack2021_gcc-12.2.0-rvyk
module load gcc/10.2.0_spack2021_gcc-10.2.0-gdnn
module load 2018/netlib-lapack/3.8.0-gcc-9.1.0-dpxqu7i
module load openblas/0.3.25_spack2021_gcc-12.2.0-l4ru

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DBLA_VENDOR=OpenBLAS
cd build
sbatch ../scripts/exec.slurm
```



