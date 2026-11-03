"""Verify the complete absolute-loss grid and node-local trajectory retention."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.dropout_continuation import atomic_json
from core.loss_plateau_verification import verify_grid
from core.scalar_estimation import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--pilot-review", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    receipt = verify_grid(args.results_dir, scratch_root=args.scratch_root, expected_node=args.expected_node,
                          pilot_review=args.pilot_review, task_file=args.task_file)
    receipt["entrypoint_sha256"] = file_sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, receipt)
    logging.info("Full-grid audit passed for %d models; saved %s", receipt["n"], args.output)


if __name__ == "__main__":
    main()
