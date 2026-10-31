#!/bin/bash
#SBATCH --job-name=rs-hw2-kaggle
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/kaggle-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/kaggle-%j.err
set -euo pipefail
/home/minseok/.venvs/rs-kaggle/bin/python scripts/kaggle_workflow.py "$@"
