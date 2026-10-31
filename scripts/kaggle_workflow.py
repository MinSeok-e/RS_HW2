"""Official API access with explicit submission action and duplicate protection."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

COMPETITION = 'skku-esw-4024-recommender-system'


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, root, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError('Submission file differs from completed experiment manifest')
    with (root / 'data/train.csv').open() as f:
        train = list(csv.DictReader(f))
    known, seen = set(), {}
    for row in train:
        known.add(row['movieId'])
        seen.setdefault(row['userId'], set()).add(row['movieId'])
    with (root / 'data/test.csv').open() as f:
        test = list(csv.DictReader(f))
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'recommendations']:
            raise ValueError('Incorrect header')
        rows = list(reader)
    if len(rows) != len(test):
        raise ValueError('Incorrect row count')
    for row, target in zip(rows, test):
        movies = row['recommendations'].split()
        if row['id'] != target['id'] or len(movies) != 20 or len(set(movies)) != 20:
            raise ValueError('Incorrect IDs or recommendation count')
        if not set(movies).issubset(known) or set(movies) & seen[target['userId']]:
            raise ValueError('Unknown or already observed movie')


def describe(submission):
    return {'submission_id': submission.ref, 'file_name': submission.file_name,
            'description': submission.description, 'date': str(submission.date),
            'public_score': submission.public_score,
            'status': getattr(submission.status, 'name', str(submission.status)),
            'error_description': submission.error_description,
            'team_name': submission.team_name, 'url': submission.url}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'submit', 'status'])
    parser.add_argument('--root', type=Path, default=Path('/home/minseok/data/semester/1/RS/RS_HW2'))
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Use a Slurm compute node for Kaggle CLI/API execution')
    root = args.root.resolve()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    existing = api.competition_submissions(COMPETITION, page_size=100) or []
    if args.action == 'check':
        files = api.competition_list_files(COMPETITION)
        print(json.dumps({'authentication': 'ok', 'competition': COMPETITION,
                          'data_files': [f.name for f in files.files],
                          'existing_submissions': len(existing)}, indent=2))
        return
    latest = json.loads((root / 'results/latest_experiment.json').read_text())
    summary = json.loads(Path(latest['summary']).read_text())
    if summary['smoke'] or summary['status'] != 'completed':
        raise ValueError('Only a completed full experiment may be submitted')
    ledger_path = root / 'submissions/kaggle_records.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    by_name = {s.file_name: s for s in existing}
    # Prepare and verify both files before any external write.
    for model in ('ease', 'neumf'):
        details = summary['submissions'][model]
        verify(Path(details['path']), root, details['sha256'])
    for model in ('ease', 'neumf'):
        details = summary['submissions'][model]
        path = Path(details['path'])
        key = summary['run_id'] + ':' + model
        record = ledger.get(key, {'model': model, 'run_id': summary['run_id'], 'file_name': path.name,
                                  'sha256': details['sha256'], 'source_commit': summary['source_commit']})
        if path.name in by_name:
            record.update(describe(by_name[path.name]))
            ledger[key] = record
            write(ledger_path, ledger)
            print(json.dumps(record))
            continue
        if args.action == 'status':
            print(json.dumps({'model': model, 'status': record.get('status', 'NOT_SUBMITTED')}))
            continue
        if record.get('submission_attempted_utc'):
            raise RuntimeError('A prior attempt is not visible in API results. Inspect before retrying; no automatic duplicate submission.')
        record['submission_attempted_utc'] = datetime.now(timezone.utc).isoformat()
        record['status'] = 'SUBMITTING'
        ledger[key] = record
        write(ledger_path, ledger)
        response = api.competition_submit(str(path),
                f'HW2 {model.upper()} | source {summary["source_commit"][:12]} | sha256 {details["sha256"][:12]}',
                COMPETITION, quiet=True)
        record['submission_id'] = response.ref
        record['message'] = response.message
        record['status'] = 'SUBMITTED'
        ledger[key] = record
        write(ledger_path, ledger)
        print(json.dumps(record))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # HTTP exception representations can contain request details; do not print credentials/headers.
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        print(f'Kaggle operation failed: {type(error).__name__}; HTTP status={status}', file=sys.stderr)
        if isinstance(error, (ValueError, RuntimeError)):
            print(str(error), file=sys.stderr)
        raise SystemExit(1)
