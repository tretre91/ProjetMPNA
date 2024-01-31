# HOW TO BUILD ON ROMEO

```
git clone git@github.com:tretre91/ProjetMPNA.git prr_project
cd prr_project
git checkout omp

module load llvm/11.0.0
module load cmake/3.27.9_spack2021_gcc-12.2.0-rvyk
module load gcc/10.2.0_spack2021_gcc-10.2.0-gdnn
module load 2018/netlib-lapack/3.8.0-gcc-9.1.0-dpxqu7i
module load openblas/0.3.25_spack2021_gcc-12.2.0-l4ru


#cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DBLA_VENDOR=OpenBLAS
cd build
sbatch ../scripts/exec.slurm
```



