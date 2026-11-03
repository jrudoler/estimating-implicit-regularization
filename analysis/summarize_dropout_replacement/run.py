"""Summarize the historical ridge-replacement conditions reported in Appendix H."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
from statistics import median

CONDITIONS = {'target': (0.3, 'none', 0.), 'ablated': (0., 'none', 0.),
              'ridge_T': (0., 'ridge', 3.556925e-5),
              'ridge_T10x': (0., 'ridge', 3.556925e-4)}
SEEDS = {123, 42, 666, 314, 17, 111, 325, 643, 432, 51}


def summarize(inputs: list[Path]) -> dict:
    grouped = {name: {} for name in CONDITIONS}
    sources = []
    for path in inputs:
        row = json.loads(path.read_text())
        cfg = row['config']
        name, seed = cfg['condition'], cfg['seed']
        if name not in CONDITIONS or seed not in SEEDS or seed in grouped[name]:
            raise ValueError(f'Unexpected or duplicate configuration: {path}')
        if (cfg['dropout'], cfg['penalty_family'], cfg['penalty_coef']) != CONDITIONS[name]:
            raise ValueError(f'Wrong replacement condition: {path}')
        for key, expected in {'depth': 3, 'width': 256, 'max_epochs': 500,
                              'patience': 50, 'lr': .05, 'momentum': .9,
                              'batch_size': 2048}.items():
            if cfg[key] != expected:
                raise ValueError(f'Wrong {key}: {path}')
        grouped[name][seed] = row
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {}
    for name, rows in grouped.items():
        if set(rows) != SEEDS:
            raise ValueError(f'Incomplete seed coverage for {name}')
        summary[name] = {'n': len(rows),
                         'median_test_accuracy': median(r['test']['test/acc'] for r in rows.values()),
                         'median_epochs': median(r['stationarity']['epochs_run'] for r in rows.values())}
    return {'conditions': summary, 'sources': sources,
            'scope': 'Historical validation-stopped experiment; not the Figure 5 loss-plateau grid.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    summary = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    logging.info('Verified and summarized %d historical runs', len(args.inputs))


if __name__ == '__main__':
    main()
