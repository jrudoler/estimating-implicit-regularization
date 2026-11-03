"""Matched dropout replacement using the final Figure 5 protocol and endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

from core.data import InMemoryBatches, MNISTLightningDataModule
from core.dropout_continuation import atomic_json, restore_rng
from core.loss_plateau import LossPlateau
from core.models import DeepReLUClassifier
from core.scalar_estimation import file_sha256, model_sha256, parameters64, ridge_estimate

LOG = logging.getLogger(__name__)


class TensorBatches:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> None:
        self.x, self.y, self.batch_size = x, y, batch_size

    def __iter__(self):
        return iter(zip(self.x.split(self.batch_size), self.y.split(self.batch_size)))


def train_epoch(model: nn.Module, batches: Any, optimizer: torch.optim.Optimizer,
                coefficient: float) -> float:
    """Identical shuffled CE loop, plus the exact gradient of lambda ||theta||^2.

    CE alone feeds the unchanged stopping controller. Both weights and biases
    receive the penalty; the coefficient is fixed before replacement training.
    """
    model.train()
    order = torch.randperm(len(batches.y), device=batches.y.device)
    loss_sum = torch.zeros((), device=batches.y.device)
    for index in order.split(batches.batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batches.x[index]), batches.y[index])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite cross-entropy')
        loss.backward()
        if coefficient:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.grad.add_(parameter, alpha=2 * coefficient)
        optimizer.step()
        loss_sum += loss.detach() * len(index)
    return float(loss_sum / len(batches.y))


def vector_from_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().cpu().double().flatten() for value in state.values()])


def load_verified(path: Path, digest: str, device: str) -> dict[str, Any]:
    if file_sha256(path) != digest:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    return torch.load(path, map_location=device, weights_only=False)


def predictions(model: nn.Module, batches: Any) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x).detach().cpu() for x, _ in batches])


def compare_logits(logits: torch.Tensor, target: torch.Tensor,
                   labels: torch.Tensor) -> dict[str, float]:
    p = logits.double().softmax(1)
    q = target.double().softmax(1)
    return {
        'accuracy': float((logits.argmax(1) == labels.cpu()).double().mean()),
        'cross_entropy': float(F.cross_entropy(logits.double(), labels.cpu())),
        'target_accuracy': float((target.argmax(1) == labels.cpu()).double().mean()),
        'prediction_disagreement': float((logits.argmax(1) != target.argmax(1)).double().mean()),
        'mean_probability_l1': float((p - q).abs().sum(1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--condition', choices=['ablated', 'ridge'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/raw'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = json.loads(args.target.read_text())
    baseline = json.loads(args.baseline.read_text())
    cfg, protocol = target['config'], target['protocol']
    if target['status'] != 'loss_plateau' or baseline['status'] != 'loss_plateau':
        raise ValueError('Require completed final-grid reference endpoints')
    if (cfg['depth'], cfg['width'], cfg['dropout']) != (3, 256, .3):
        raise ValueError('Unexpected target configuration')
    if baseline['config'] != {**cfg, 'dropout': 0.} or baseline['protocol'] != protocol:
        raise ValueError('Control is not matched to the target protocol/seed')
    # Imports are deliberately taken from the immutable Figure 5 source bundle.
    import core.models
    source_root = Path(core.models.__file__).resolve().parents[2]
    for name, digest in target['signature']['code_sha256'].items():
        if file_sha256(source_root / name) != digest:
            raise ValueError(f'Changed Figure 5 source: {name}')
    coefficient = target['endpoint']['ridge']['scale_star_clamped'] if args.condition == 'ridge' else 0.
    if coefficient is None or coefficient < 0:
        raise ValueError('Invalid nonnegative ridge fit')
    signature = {
        'study': 'matched_dropout_replacement', 'condition': args.condition,
        'target_sha256': file_sha256(args.target), 'baseline_sha256': file_sha256(args.baseline),
        'coefficient': coefficient, 'protocol': protocol,
        'runner_sha256': file_sha256(Path(__file__)),
        'frozen_source_sha256': target['signature']['code_sha256'],
        'stopping_metric': 'sample-weighted training cross-entropy, excluding explicit penalty',
        'torch_version': torch.__version__,
    }
    if torch.__version__ != target['endpoint']['provenance']['versions']['torch']:
        raise ValueError('PyTorch version differs from Figure 5')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path, latest = args.output_dir / 'result.json', args.output_dir / 'latest.ckpt'
    if result_path.exists():
        if json.loads(result_path.read_text())['signature'] != signature:
            raise ValueError('Refusing to overwrite a different replacement experiment')
        LOG.info('Completed replacement already exists')
        return
    torch.manual_seed(cfg['seed'])
    dm = MNISTLightningDataModule(root=args.data_root, batch_size=2048, num_workers=0)
    dm.setup('fit')
    batches = InMemoryBatches(dm._train_dataset, args.device)
    split_hash = hashlib.sha256(torch.tensor(dm._train_dataset.indices, dtype=torch.int64).numpy().tobytes()).hexdigest()
    if split_hash != target['signature']['split_indices_sha256']:
        raise ValueError('Training split differs from target')
    initial_record = target['trajectory'][0]['provenance']
    initial = load_verified(Path(initial_record['checkpoint']), initial_record['checkpoint_sha256'], args.device)
    baseline_initial_record = baseline['trajectory'][0]['provenance']
    baseline_initial = load_verified(Path(baseline_initial_record['checkpoint']), baseline_initial_record['checkpoint_sha256'], 'cpu')
    if not torch.equal(vector_from_state(initial['state_dict']), vector_from_state(baseline_initial['state_dict'])):
        raise ValueError('Target and baseline initial parameter vectors differ')
    if not torch.equal(initial['cpu_rng'].cpu(), baseline_initial['cpu_rng'].cpu()):
        raise ValueError('Target and baseline initial CPU streams differ')
    if any(not torch.equal(a.cpu(), b.cpu()) for a,b in zip(initial['cuda_rng'], baseline_initial['cuda_rng'])):
        raise ValueError('Target and baseline initial GPU streams differ')
    model = DeepReLUClassifier(**initial_record['model_hparams']).to(args.device)
    model.load_state_dict(initial['state_dict'])
    if model_sha256(model) != initial_record['model_sha256']:
        raise ValueError('Wrong initial model')
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    optimizer = torch.optim.SGD(model.parameters(), lr=protocol['initial_lr'], momentum=protocol['momentum'])
    optimizer.load_state_dict(initial['optimizer_states'][0])
    stopper = LossPlateau(**protocol['stopping'])
    restore_rng(initial)
    epoch, history, trajectory, previous_seconds = 0, [], [], 0.
    if latest.exists():
        saved = torch.load(latest, map_location=args.device, weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('Recovery signature mismatch')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer_states'][0])
        stopper = LossPlateau(**saved['stopper'])
        epoch, history, trajectory = saved['completed_epochs'], saved['history'], saved['trajectory']
        previous_seconds = saved['elapsed_seconds']
        restore_rng(saved)
    begin = time.monotonic()

    def save(path: Path) -> None:
        payload = {'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                   'optimizer_states': [optimizer.state_dict()], 'cpu_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all(), 'stopper': stopper.state_dict(),
                   'signature': signature, 'completed_epochs': epoch, 'history': history,
                   'trajectory': trajectory, 'elapsed_seconds': previous_seconds + time.monotonic() - begin}
        pending = path.with_suffix('.tmp')
        torch.save(payload, pending)
        pending.replace(path)

    while epoch < protocol['max_epochs'] and (not history or history[-1]['action'] != 'stop'):
        loss = train_epoch(model, batches, optimizer, coefficient)
        epoch += 1
        block = stopper.update(loss, optimizer.param_groups[0]['lr'])
        if block is not None:
            history.append(block)
            for group in optimizer.param_groups:
                group['lr'] = block['next_lr']
            atomic_json(args.output_dir / 'progress.json', {'epoch': epoch, 'block': block, 'condition': args.condition})
            LOG.info('seed=%s condition=%s epoch=%d CE=%.9g action=%s', cfg['seed'], args.condition,
                     epoch, block['mean_train_loss'], block['action'])
        if epoch % protocol['checkpoint_every'] == 0:
            path = args.output_dir / f'trajectory_epoch_{epoch:06d}.ckpt'
            save(path)
            trajectory.append({'epoch': epoch, 'checkpoint': str(path), 'sha256': file_sha256(path)})
            save(latest)
    status = 'loss_plateau' if history[-1]['action'] == 'stop' else 'budget_exhausted'
    endpoint_path = args.output_dir / f'endpoint_epoch_{epoch:06d}.ckpt'
    save(endpoint_path)
    save(latest)
    endpoint = {'epoch': epoch, 'status': status, 'ridge': ridge_estimate(model, batches),
                'provenance': {'checkpoint': str(endpoint_path), 'checkpoint_sha256': file_sha256(endpoint_path),
                               'model_sha256': model_sha256(model)}}
    # Evaluate only after stopping. Test data never enter optimization or stopping.
    target_record = target['endpoint']['provenance']
    target_path = args.target.parent / Path(target_record['checkpoint']).name
    target_payload = load_verified(target_path, target_record['checkpoint_sha256'], args.device)
    target_model = DeepReLUClassifier(**target_record['model_hparams']).to(args.device)
    target_model.load_state_dict(target_payload['state_dict'])
    if model_sha256(target_model) != target_record['model_sha256']:
        raise ValueError('Wrong evaluation target')
    test_dataset = MNIST(root=args.data_root, train=False, download=False)
    test_batches = TensorBatches(test_dataset.data.to(args.device).float().div_(255).unsqueeze(1),
                                 test_dataset.targets.to(args.device))
    metrics = {}
    for name, data in [('train', batches), ('test', test_batches)]:
        metrics[name] = compare_logits(predictions(model, data), predictions(target_model, data), data.y)
    reference_path = args.baseline.parent / Path(baseline['endpoint']['provenance']['checkpoint']).name
    reference = load_verified(reference_path, baseline['endpoint']['provenance']['checkpoint_sha256'], 'cpu')
    theta, target_theta = parameters64(model), parameters64(target_model)
    metrics['relative_parameter_distance_to_target'] = float((theta-target_theta).norm()/target_theta.norm())
    metrics['relative_parameter_distance_to_saved_baseline'] = float((theta-vector_from_state(reference['state_dict'])).norm()/vector_from_state(reference['state_dict']).norm())
    result = {'signature': signature, 'config': {**cfg, 'dropout': 0., 'coefficient': coefficient},
              'status': status, 'endpoint': endpoint, 'history': history, 'trajectory': trajectory,
              'initial_checkpoint_sha256': initial_record['checkpoint_sha256'],
              'initialization_matches_saved_baseline': True, 'split_indices_sha256': split_hash,
              'metrics': metrics, 'elapsed_seconds': previous_seconds + time.monotonic()-begin}
    atomic_json(result_path, result)
    LOG.info('Completed seed=%s condition=%s epoch=%d metrics=%s', cfg['seed'], args.condition, epoch, metrics)


if __name__ == '__main__':
    main()
