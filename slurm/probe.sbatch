#!/bin/bash
#SBATCH --job-name=rs-hw2-env-probe
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/probe-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/probe-%j.err
set -euo pipefail
for prefix in /home/minseok/.conda/envs/minseok /home/minseok/.conda/envs/aimers /home/minseok/.conda/envs/bot; do
  "$prefix/bin/python" - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
print('PREFIX', sys.prefix, 'PYTHON', sys.version)
for package in ['torch', 'numpy', 'scipy', 'matplotlib', 'pandas', 'pytest', 'reportlab', 'kaggle']:
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, 'NOT_INSTALLED')
PY
done
