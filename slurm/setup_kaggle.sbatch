#!/bin/bash
#SBATCH --job-name=rs-kaggle-setup
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/kaggle-setup-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/kaggle-setup-%j.err
set -euo pipefail
if [ ! -e /home/minseok/.venvs/rs-kaggle ]; then
  /home/minseok/.conda/envs/aimers/bin/python -m venv /home/minseok/.venvs/rs-kaggle
fi
/home/minseok/.venvs/rs-kaggle/bin/python -m pip install kaggle
/home/minseok/.venvs/rs-kaggle/bin/python -m pip freeze > logs/kaggle-environment-freeze.txt
