#!/usr/bin/env bash

#SBATCH --output output.out
#SBATCH --error output.err
#SBATCH --nodes=1
#SBATCH --job-name=<jobname>
#SBATCH --time="00:05:00"
#SBATCH --ntasks=20
#SBATCH --exclusive

# #SBATCH --output=<jobname>-%j.out
# #SBATCH --error=<jobname>-%j.err
# #SBATCH --ntasks-per-node=<nctile>

set -x
module purge

module load cmake/3.21.4/gcc-11.2.0
module load netlib-lapack/3.9.1/gcc-11.2.0
module load openblas/0.3.8/gcc-9.2.0

cd ${SLURM_SUBMIT_DIR}

make -B

# cuda-gdb ./memcheck.exe # lancer l'éxécutable

# time OMP_NUM_THREADS=1 ./prr ../data/test4.mat -s 2 -m 4 -i 100 
# time OMP_NUM_THREADS=2 ./prr ../data/test4.mat -s 2 -m 4 -i 100
# time OMP_NUM_THREADS=4 ./prr ../data/test4.mat -s 2 -m 4 -i 100
# time OMP_NUM_THREADS=8 ./prr ../data/test4.mat -s 2 -m 4 -i 100
#
# time OMP_NUM_THREADS=1 ./prr_blas ../data/test4.mat -s 2 -m 4 -i 100
# time OMP_NUM_THREADS=2 ./prr_blas ../data/test4.mat -s 2 -m 4 -i 100
# time OMP_NUM_THREADS=4 ./prr_blas ../data/test4.mat -s 2 -m 4 -i 100
# time OMP_NUM_THREADS=8 ./prr_blas ../data/test4.mat -s 2 -m 4 -i 100

OMP_NUM_THREADS=8 maqao.intel64 oneview -sampling-rate=high -R1 -xp=maqao_prr_cblas --replace --include-kernel sampling-rate=high -- ./prr_blas ../data/test4.mat -s 2 -m 4 -v -i 100
OMP_NUM_THREADS=8 maqao.intel64 oneview -sampling-rate=high -R1 -xp=maqao_prr       --replace --include-kernel sampling-rate=high -- ./prr      ../data/test4.mat -s 2 -m 4 -v -i 100
