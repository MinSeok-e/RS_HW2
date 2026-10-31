"""Validation sweeps, selection, full-data refits, and checked submissions."""
import argparse
import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .data import load_data, sha256
from .evaluation import evaluate, topk, write_submission
from .models import NeuMF, ease_scores, neumf_scores, sample_epoch
from .provenance import start_run, write_csv, write_json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train_neumf(data, rows, config, seed, device, directory, checkpoint_path, targets=None, epochs=None):
    directory.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    seen = data.matrix(rows, dtype=bool)
    model = NeuMF(*data.shape, mf_dim=config['mf_dim'], mlp_dim=config['mlp_dim'], hidden=config['hidden']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    best_value, best_epoch, stale = -1.0, 0, 0
    history, best_per_user, best_metrics = [], None, None
    maximum = epochs if epochs is not None else config['max_epochs']
    start = time.monotonic()
    for epoch in range(1, maximum + 1):
        epoch_start = time.monotonic()
        model.train()
        u, i, y = sample_epoch(data.users[rows], data.items[rows], seen, config['negatives'], rng)
        u = torch.from_numpy(u).long().to(device)
        i = torch.from_numpy(i).long().to(device)
        y = torch.from_numpy(y).to(device)
        total_loss = 0.0
        for offset in range(0, len(u), config['batch_size']):
            sl = slice(offset, offset + config['batch_size'])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(u[sl], i[sl]), y[sl])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite NeuMF training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(y[sl])
        record = {'epoch': epoch, 'bce_loss': total_loss / len(u)}
        if targets is not None:
            scores = neumf_scores(model, *data.shape, device=device)
            metrics, per_user = evaluate(scores, seen, targets)
            record.update(metrics)
            if metrics['ndcg@20'] > best_value:
                best_value, best_epoch, stale = metrics['ndcg@20'], epoch, 0
                best_metrics, best_per_user = metrics, per_user
                torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
                            'seed': seed, 'user_ids': data.user_ids.tolist(), 'item_ids': data.item_ids.tolist()}, checkpoint_path)
            else:
                stale += 1
        record['seconds'] = time.monotonic() - epoch_start
        history.append(record)
        write_csv(directory / 'history.csv', history)
        logging.info('NeuMF %s epoch %d/%d loss %.6f%s time %.2fs', directory.name, epoch, maximum,
                     record['bce_loss'], f' valid_ndcg@20={metrics["ndcg@20"]:.6f}' if targets is not None else '', record['seconds'])
        if targets is not None and stale >= config['patience']:
            logging.info('Early stop: %d epochs without improvement', config['patience'])
            break
    if targets is None:
        best_epoch = maximum
        torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': maximum,
                    'seed': seed, 'user_ids': data.user_ids.tolist(), 'item_ids': data.item_ids.tolist()}, checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        write_csv(directory / 'best_per_user.csv', [{'userId': int(data.user_ids[r['user_index']]), **r} for r in best_per_user])
    result = {'best_epoch': best_epoch, 'epochs_run': len(history), 'seconds': time.monotonic() - start,
              'metrics': best_metrics, 'config': config, 'checkpoint': str(checkpoint_path)}
    write_json(directory / 'result.json', result)
    return model, result


def make_plots(run_dir, ease_results, neumf_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogx([r['regularization'] for r in ease_results], [r['ndcg@20'] for r in ease_results], 'o-')
    ax.set(xlabel='EASE regularization (lambda)', ylabel='Validation NDCG@20', title='Chronological holdout: EASE')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / 'ease_validation.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    import csv
    for result in neumf_results:
        path = run_dir / result['trial'] / 'history.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        ax.plot([int(r['epoch']) for r in rows], [float(r['ndcg@20']) for r in rows], label=result['trial'])
    ax.set(xlabel='Epoch', ylabel='Validation NDCG@20', title='Chronological holdout: NeuMF')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / 'neumf_validation.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Working repository, data and output destination')
    parser.add_argument('--config', type=Path, default=Path('configs/experiment.json'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run this experiment on a Slurm compute node')
    root = args.root.resolve()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = copy.deepcopy(config)
        config['ease_lambdas'] = [100.0]
        config['neumf_trials'] = config['neumf_trials'][:1]
        config['neumf_defaults']['max_epochs'] = 2
    config['smoke'] = args.smoke
    torch.set_num_threads(min(4, int(os.getenv('SLURM_CPUS_PER_TASK', '1'))))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A GPU was requested but torch.cuda.is_available() is false')
    device = torch.device(args.device)
    run_dir, checkpoint_dir, manifest = start_run(root, 'smoke' if args.smoke else 'experiment', config)
    if args.device == 'cuda':
        manifest['gpu'] = {'name': torch.cuda.get_device_name(), 'cuda': torch.version.cuda}
        write_json(run_dir / 'manifest.json', manifest)
    start = time.monotonic()
    data = load_data(root / 'data', config['valid_fraction'])
    summary = data.summary()
    write_json(run_dir / 'data_summary.json', summary)
    logging.info('Data summary: %s', json.dumps(summary))
    # Split membership is raw interaction data and stays in ignored data/.
    split_dir = root / 'data' / 'splits'
    split_dir.mkdir(exist_ok=True)
    np.savez_compressed(split_dir / f'{summary["split_sha256"]}.npz', train_rows=data.train_rows, valid_rows=data.valid_rows)
    targets = data.targets()
    fit = data.matrix(data.train_rows)
    popularity_scores = np.broadcast_to(fit.sum(axis=0), fit.shape).copy()
    popularity, _ = evaluate(popularity_scores, fit, targets)
    write_json(run_dir / 'popularity_reference.json', popularity)
    logging.info('Popularity sanity reference: %s', popularity)
    ease_results = []
    best_ease = None
    for regularization in config['ease_lambdas']:
        t0 = time.monotonic()
        scores, _ = ease_scores(fit, regularization)
        metrics, per_user = evaluate(scores, fit, targets)
        row = {'regularization': regularization, **metrics, 'seconds': time.monotonic() - t0}
        ease_results.append(row)
        logging.info('EASE: %s', row)
        if best_ease is None or row['ndcg@20'] > best_ease['ndcg@20']:
            best_ease = row.copy()
            write_csv(run_dir / 'ease_best_per_user.csv', [{'userId': int(data.user_ids[r['user_index']]), **r} for r in per_user])
    write_csv(run_dir / 'ease_sweep.csv', ease_results)
    write_json(run_dir / 'ease_selected.json', best_ease)
    neumf_results = []
    for n, override in enumerate(config['neumf_trials']):
        trial_config = {**config['neumf_defaults'], **override}
        trial_name = f'neumf_{n:02d}'
        model, result = train_neumf(data, data.train_rows, trial_config, config['seed'], device,
                                   run_dir / trial_name, checkpoint_dir / f'{trial_name}.pt', targets=targets)
        row = {'trial': trial_name, 'best_epoch': result['best_epoch'], 'epochs_run': result['epochs_run'],
               **result['metrics'], 'seconds': result['seconds'], 'config': trial_config}
        neumf_results.append(row)
        write_json(run_dir / 'neumf_sweep.json', neumf_results)
        del model
    best_neumf = max(neumf_results, key=lambda row: row['ndcg@20'])
    write_json(run_dir / 'neumf_selected.json', best_neumf)
    write_csv(run_dir / 'neumf_sweep.csv', [{**row, 'config': json.dumps(row['config'], sort_keys=True)} for row in neumf_results])
    make_plots(run_dir, ease_results, neumf_results)
    all_rows = np.arange(len(data.users))
    full = data.matrix(all_rows)
    logging.info('Refit EASE on all %d interactions, lambda=%s', len(all_rows), best_ease['regularization'])
    full_scores, state = ease_scores(full, best_ease['regularization'])
    np.savez_compressed(checkpoint_dir / 'ease_full.npz', **state, user_ids=data.user_ids, item_ids=data.item_ids)
    submissions = {}
    for name, scores in [('ease', full_scores)]:
        target = root / 'submissions' / f'{run_dir.name}_{name}.csv'
        check = write_submission(target, data, topk(scores, full, full.any(axis=0)), full)
        submissions[name] = {'path': str(target), 'sha256': sha256(target), 'checks': check}
    logging.info('Refit NeuMF on all interactions, %d epochs', best_neumf['best_epoch'])
    model, final_result = train_neumf(data, all_rows, best_neumf['config'], config['seed'], device,
                                    run_dir / 'neumf_full', checkpoint_dir / 'neumf_full.pt', epochs=best_neumf['best_epoch'])
    full_scores = neumf_scores(model, *data.shape, device=device)
    target = root / 'submissions' / f'{run_dir.name}_neumf.csv'
    check = write_submission(target, data, topk(full_scores, full, full.any(axis=0)), full)
    submissions['neumf'] = {'path': str(target), 'sha256': sha256(target), 'checks': check}
    result = {'status': 'completed', 'run_id': run_dir.name, 'smoke': args.smoke,
              'source_commit': manifest['source_commit'], 'data': summary,
              'popularity_reference': popularity, 'ease': best_ease, 'neumf': best_neumf,
              'submissions': submissions, 'seconds': time.monotonic() - start,
              'completed_utc': datetime.now(timezone.utc).isoformat()}
    write_json(run_dir / 'summary.json', result)
    write_json(root / 'results' / ('latest_smoke.json' if args.smoke else 'latest_experiment.json'),
               {'run_dir': str(run_dir), 'summary': str(run_dir / 'summary.json')})
    logging.info('COMPLETED: %s', json.dumps(result))


if __name__ == '__main__':
    main()
