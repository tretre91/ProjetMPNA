#!/usr/bin/env bash

#SBATCH --output output.out
#SBATCH --error output.err
#SBATCH --nodes=1
#SBATCH --job-name=prr
#SBATCH --time="00:05:00"
#SBATCH --ntasks=20
#SBATCH --exclusive
#SBATCH --partition=short

# #SBATCH --output=<jobname>-%j.out
# #SBATCH --error=<jobname>-%j.err
# #SBATCH --ntasks-per-node=<nctile>

set -x
module purge

module load llvm/11.0.0
module load cmake/3.27.9_spack2021_gcc-12.2.0-rvyk
module load gcc/10.2.0_spack2021_gcc-10.2.0-gdnn
module load 2018/netlib-lapack/3.8.0-gcc-9.1.0-dpxqu7i
module load openblas/0.3.25_spack2021_gcc-12.2.0-l4ru

cd ${SLURM_SUBMIT_DIR}

make -B

echo "nthreads;n;m;s;time;dev;version" > data.dat
OMP_NUM_THREADS=1 ./prr_bench_blas   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=2 ./prr_bench_blas   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=4 ./prr_bench_blas   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=8 ./prr_bench_blas   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=16 ./prr_bench_blas  ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat


OMP_NUM_THREADS=1 ./prr_bench   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=2 ./prr_bench   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=4 ./prr_bench   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=8 ./prr_bench   ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat
OMP_NUM_THREADS=16 ./prr_bench  ../data/1138_bus.mtx -s 4 -m 16 -v -i 100 >> data.dat


