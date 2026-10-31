#!/bin/bash
#SBATCH --job-name=rs-hw2-experiment
#SBATCH --partition=gpu24
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/experiment-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/experiment-%j.err
set -euo pipefail
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2020
RS_REPO=/home/minseok/data/semester/1/RS/RS_HW2
: "${RS_SOURCE_COMMIT:?Submit with a specific source commit}"
RS_SNAPSHOT="$RS_REPO/snapshots/${RS_SOURCE_COMMIT}_${SLURM_JOB_ID}"
mkdir -p "$RS_SNAPSHOT"
git -C "$RS_REPO" archive "$RS_SOURCE_COMMIT" | tar -x -C "$RS_SNAPSHOT"
cd "$RS_SNAPSHOT"
/home/minseok/.venvs/rs-hw2/bin/python -m unittest discover -s tests -v
/home/minseok/.venvs/rs-hw2/bin/python -m rs_hw2.run --root "$RS_REPO" --config configs/experiment.json --device cuda "$@"
