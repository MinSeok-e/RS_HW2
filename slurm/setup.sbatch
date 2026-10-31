#!/bin/bash
#SBATCH --job-name=rs-hw2-setup
#SBATCH --partition=gpu24
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/minseok/data/semester/1/RS/RS_HW2
#SBATCH --output=/home/minseok/data/semester/1/RS/RS_HW2/logs/setup-%j.out
#SBATCH --error=/home/minseok/data/semester/1/RS/RS_HW2/logs/setup-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
if [ ! -e /home/minseok/.venvs/rs-hw2 ]; then
  /home/minseok/.conda/envs/bot/bin/python -m venv --system-site-packages /home/minseok/.venvs/rs-hw2
fi
/home/minseok/.venvs/rs-hw2/bin/python - <<'PY'
import sys, torch, numpy, scipy, matplotlib
print('Python:', sys.version)
print('Venv:', sys.prefix, 'Base:', sys.base_prefix)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('numpy:', numpy.__version__, 'scipy:', scipy.__version__, 'matplotlib:', matplotlib.__version__)
assert torch.cuda.is_available(), 'Requested GPU is not visible to PyTorch'
print('GPU:', torch.cuda.get_device_name())
x = torch.tensor([1., 2., 3.], device='cuda')
assert x.sum().item() == 6
print('GPU operation: OK')
PY
/home/minseok/.venvs/rs-hw2/bin/python -m pip freeze > logs/environment-freeze.txt
