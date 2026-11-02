"""Build the bounded 30-run control design, reusing the saved 0.3 trajectories."""
import argparse
import json
from pathlib import Path
from core.dropout_grid import SEEDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    spec_dir = args.output_root / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for rate in (0., .3, .5):
        for seed in SEEDS:
            tag = f"d3_w256_do{rate}_s{seed}"
            cfg = {"depth": 3, "width": 256, "dropout": rate, "seed": seed}
            prior = args.source_root / "data/generated/dropout_trajectory_lightning/results" / f"{tag}.json"
            spec = {"config": cfg}
            if prior.exists():
                data = json.loads(prior.read_text())
                if any(data["config"].get(k) != v for k, v in cfg.items()):
                    raise ValueError(f"Trajectory/config mismatch: {prior}")
                if sorted(r["fraction"] for r in data["trajectory"]) != [.5, .8, 1.]:
                    raise ValueError(f"Missing/duplicate trajectory fractions: {prior}")
                spec["checkpoints"] = [{"fraction": r["fraction"], "checkpoint": str((args.source_root / r["checkpoint"]).resolve())} for r in data["trajectory"]]
                if not all(Path(r["checkpoint"]).is_file() for r in spec["checkpoints"]):
                    raise FileNotFoundError(f"Saved trajectory has missing checkpoint: {prior}")
                spec["endpoint_kind"] = "saved_corrected_grid_replay"
            else:
                grid = args.source_root / "data/generated/closed_form_dropout_grid/results" / f"{tag}.json"
                grid_data = json.loads(grid.read_text())
                if any(grid_data["config"].get(k) != v for k, v in cfg.items()):
                    raise ValueError(f"Grid/config mismatch: {grid}")
                spec.update(endpoint_kind="explicit_replacement_replay", checkpoints=[],
                            endpoint_epochs=grid_data["stationarity"]["epochs_run"],
                            checkpoint_dir=str((args.output_root / "checkpoints" / tag).resolve()),
                            replay_record=str((args.output_root / "replays" / f"{tag}.json").resolve()))
            (spec_dir / f"{tag}.json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
