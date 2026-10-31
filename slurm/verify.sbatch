#!/bin/bash
#SBATCH --job-name=rs-hw2-verify
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/verify-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/verify-%j.err
set -euo pipefail
export PYTHONPATH=/home/minseok/data/semester/1/RS/RS_HW2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/minseok/.venvs/rs-hw2/bin/python scripts/verify_artifacts.py
