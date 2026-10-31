#!/bin/bash
#SBATCH --job-name=rs-hw2-report
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/report-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/report-%j.err
set -euo pipefail
/home/minseok/.venvs/rs-report/bin/python scripts/build_report.py
/home/minseok/.venvs/rs-report/bin/python scripts/inspect_report.py
