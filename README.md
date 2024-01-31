# HOW TO BUILD ON ROMEO

```
git clone git@github.com:tretre91/ProjetMPNA.git prr_project
cd prr_project
git checkout omp
module load llvm/11.0.0
module load cmake/3.27.9_spack2021_gcc-12.2.0-rvyk
module load netlib-lapack/3.9.1/gcc-11.2.0
module load openblas/0.3.25_spack2021_gcc-12.2.0-l4ru
module load gcc/10.2.0_spack2021_gcc-10.2.0-gdnn
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cd build
sbatch ../scripts/exec.slurm
```

