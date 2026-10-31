"""Verify saved checkpoints recreate selected validation metrics and final CSVs."""
import csv
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from rs_hw2.data import load_data, sha256
from rs_hw2.evaluation import evaluate, topk, write_submission
from rs_hw2.models import NeuMF, neumf_scores
from rs_hw2.run import set_seed
from rs_hw2.provenance import write_json

if not os.getenv('SLURM_JOB_ID'):
    raise RuntimeError('Use a compute node')
root = Path('/home/minseok/data/semester/1/RS/RS_HW2').resolve()
run = Path(json.loads((root / 'results/latest_experiment.json').read_text())['run_dir'])
summary = json.loads((run / 'summary.json').read_text())
data = load_data(root / 'data')
assert data.hashes == summary['data']['file_sha256']
assert data.summary()['split_sha256'] == summary['data']['split_sha256']
full = data.matrix(np.arange(len(data.users)))
checkpoints = root / 'checkpoints' / run.name
set_seed(2020)
torch.set_num_threads(2)
state = np.load(checkpoints / 'ease_full.npz')
np.testing.assert_array_equal(state['user_ids'], data.user_ids)
np.testing.assert_array_equal(state['item_ids'], data.item_ids)
np.testing.assert_array_equal(state['x'], full)
ease = state['x'] - state['ax'] / state['inverse_diagonal'][None, :]
checks = {}
for model_name in ('ease', 'neumf'):
    if model_name == 'ease':
        scores = ease
    else:
        checkpoint = torch.load(checkpoints / 'neumf_full.pt', map_location='cuda', weights_only=True)
        config = checkpoint['config']
        model = NeuMF(*data.shape, config['mf_dim'], config['mlp_dim'], config['hidden']).cuda()
        model.load_state_dict(checkpoint['state_dict'])
        np.testing.assert_array_equal(checkpoint['user_ids'], data.user_ids)
        np.testing.assert_array_equal(checkpoint['item_ids'], data.item_ids)
        scores = neumf_scores(model, *data.shape, device='cuda')
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'submission.csv'
        write_submission(path, data, topk(scores, full, full.any(axis=0)), full)
        assert sha256(path) == summary['submissions'][model_name]['sha256']
        assert path.read_bytes() == Path(summary['submissions'][model_name]['path']).read_bytes()
        checks[model_name] = {'checkpoint_reproduces_csv_exactly': True, 'sha256': sha256(path)}
checkpoint = torch.load(checkpoints / (summary['neumf']['trial'] + '.pt'), map_location='cuda', weights_only=True)
config = checkpoint['config']
model = NeuMF(*data.shape, config['mf_dim'], config['mlp_dim'], config['hidden']).cuda()
model.load_state_dict(checkpoint['state_dict'])
scores = neumf_scores(model, *data.shape, device='cuda')
metrics, _ = evaluate(scores, data.matrix(data.train_rows), data.targets())
for key in ('ndcg@20', 'recall@20', 'hit@20'):
    assert abs(metrics[key] - summary['neumf'][key]) < 1e-12
checks['validation_checkpoint'] = metrics
checks['slurm_job_id'] = os.environ['SLURM_JOB_ID']
write_json(run / 'artifact_verification.json', checks)
print(json.dumps(checks, indent=2))
