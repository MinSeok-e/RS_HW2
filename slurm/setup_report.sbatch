#!/bin/bash
#SBATCH --job-name=rs-hw2-report-setup
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/report-setup-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/report-setup-%j.err
set -euo pipefail
if [ ! -e /home/minseok/.venvs/rs-report ]; then
  /home/minseok/.conda/envs/aimers/bin/python -m venv /home/minseok/.venvs/rs-report
fi
/home/minseok/.venvs/rs-report/bin/python -m pip install reportlab pypdf pypdfium2
/home/minseok/.venvs/rs-report/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
import hashlib
base = Path('/home/minseok/.cache/rs-hw2-fonts')
base.mkdir(parents=True, exist_ok=True)
for name in ['NanumGothic-Regular.ttf', 'NanumGothic-Bold.ttf']:
    target = base / name
    if not target.exists():
        with urlopen('https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/' + name, timeout=30) as response:
            target.write_bytes(response.read())
    print(name, hashlib.sha256(target.read_bytes()).hexdigest())
PY
/home/minseok/.venvs/rs-report/bin/python -m pip freeze > logs/report-environment-freeze.txt
