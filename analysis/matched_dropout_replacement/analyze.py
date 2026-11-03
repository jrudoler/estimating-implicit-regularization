"""Verify and summarize all paired replacements; test metrics never select runs."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from core.loss_plateau_verification import verify_history
from core.scalar_estimation import file_sha256

SEEDS = [123, 42, 666, 314, 17, 111, 325, 643, 432, 51]


def describe(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    return {'values': values, 'mean': float(x.mean()), 'median': float(np.median(x)),
            'min': float(x.min()), 'max': float(x.max())}


def paired_difference(values: list[float]) -> dict[str, Any]:
    result = describe(values)
    rng = np.random.default_rng(20261103)
    x = np.asarray(values)
    estimates = x[rng.integers(0,len(x),size=(20000,len(x)))].mean(axis=1)
    result['mean_bootstrap_ci95'] = np.quantile(estimates,[.025,.975]).tolist()
    return result


def summarize(paths: list[Path], references: Path) -> dict[str, Any]:
    rows: dict[tuple[str,int], dict[str, Any]] = {}
    sources = []
    blocks = 0
    runners = set()
    for path in paths:
        r = json.loads(path.read_text())
        signature = r['signature']
        seed, condition = r['config']['seed'], signature['condition']
        key = (condition,seed)
        if condition not in ('ablated','ridge') or seed not in SEEDS or key in rows:
            raise ValueError('Unexpected/duplicate run')
        target_path = references / f'd3_w256_do0.3_s{seed}' / 'result.json'
        baseline_path = references / f'd3_w256_do0.0_s{seed}' / 'result.json'
        t = json.loads(target_path.read_text())
        if signature['target_sha256'] != file_sha256(target_path) or signature['baseline_sha256'] != file_sha256(baseline_path):
            raise ValueError('Changed reference result')
        if signature['protocol'] != t['protocol'] or signature['frozen_source_sha256'] != t['signature']['code_sha256']:
            raise ValueError('Protocol/source mismatch')
        expected_coef = t['endpoint']['ridge']['scale_star_clamped'] if condition == 'ridge' else 0.
        if signature['coefficient'] != expected_coef or r['config']['coefficient'] != expected_coef:
            raise ValueError('Wrong replacement coefficient')
        if r['initial_checkpoint_sha256'] != t['trajectory'][0]['provenance']['checkpoint_sha256']:
            raise ValueError('Wrong starting weights')
        if r['split_indices_sha256'] != t['signature']['split_indices_sha256']:
            raise ValueError('Wrong training split')
        if r['status'] != 'loss_plateau':
            raise ValueError('Budget exhaustion requires separate interpretation')
        checkpoint = path.parent / Path(r['endpoint']['provenance']['checkpoint']).name
        if file_sha256(checkpoint) != r['endpoint']['provenance']['checkpoint_sha256']:
            raise ValueError('Changed published endpoint')
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['signature'] != signature or payload['completed_epochs'] != r['endpoint']['epoch']:
            raise ValueError('Checkpoint/result mismatch')
        replay = {**r, 'protocol': signature['protocol'],
                  'endpoint': {**r['endpoint'], 'lr': payload['optimizer_states'][0]['param_groups'][0]['lr']}}
        blocks += verify_history(replay)
        if condition == 'ablated' and r['metrics']['relative_parameter_distance_to_saved_baseline'] > 1e-6:
            raise ValueError('No-dropout replay differs materially from the saved final-grid baseline')
        runners.add(signature['runner_sha256'])
        rows[key] = r
        sources.append({'path': str(path),'sha256': file_sha256(path), 'checkpoint_sha256': file_sha256(checkpoint)})
    if set(rows) != {(c,s) for c in ['ablated','ridge'] for s in SEEDS} or len(runners) != 1:
        raise ValueError('Incomplete experiment or mixed runner versions')
    out: dict[str, Any] = {'seeds': SEEDS,'n':len(rows),'sources':sources,'verified_loss_blocks':blocks,
                           'runner_sha256':next(iter(runners)), 'conditions':{}, 'paired_comparisons':{}}
    for name in ['ablated','ridge']:
        rs = [rows[name,s] for s in SEEDS]
        summary: dict[str, Any] = {'epochs': describe([r['endpoint']['epoch'] for r in rs]),
                                  'coefficient':describe([r['config']['coefficient'] for r in rs]),
                                  'relative_parameter_distance_to_target':describe([r['metrics']['relative_parameter_distance_to_target'] for r in rs]),
                                  'baseline_replay_distance':describe([r['metrics']['relative_parameter_distance_to_saved_baseline'] for r in rs])}
        for split in ['train','test']:
            summary[split] = {key:describe([r['metrics'][split][key] for r in rs])
                              for key in ['accuracy','cross_entropy','target_accuracy','prediction_disagreement','mean_probability_l1']}
        out['conditions'][name] = summary
        out['paired_comparisons'][name + '_minus_target_test_accuracy'] = paired_difference([
            r['metrics']['test']['accuracy']-r['metrics']['test']['target_accuracy'] for r in rs])
    for seed in SEEDS:
        for split in ['train','test']:
            if rows['ablated',seed]['metrics'][split]['target_accuracy'] != rows['ridge',seed]['metrics'][split]['target_accuracy']:
                raise ValueError('Inconsistent evaluation of the common target')
    out['paired_comparisons']['ridge_minus_ablated_test_accuracy'] = paired_difference([
        rows['ridge',s]['metrics']['test']['accuracy'] - rows['ablated',s]['metrics']['test']['accuracy'] for s in SEEDS])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--references',type=Path,default=Path('data/generated/dropout_loss_plateau_absolute'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    result=summarize(args.inputs,args.references)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    logging.info('Verified %d matched results and %d loss blocks',result['n'],result['verified_loss_blocks'])


if __name__ == '__main__':
    main()
