#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Sweep YAML must define a mapping at the top level.")
    return data


def parameter_cardinality(spec: Any) -> int:
    if not isinstance(spec, dict):
        raise ValueError("parameter definition must be a mapping.")

    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list):
            raise ValueError("'values' must be a list.")
        if not values:
            raise ValueError("'values' list cannot be empty.")
        return len(values)

    if "value" in spec:
        return 1

    if "distribution" in spec:
        raise ValueError(
            "sweep uses a sampled distribution; grid-size counting is unsupported."
        )

    raise ValueError("parameter must define either 'values' or 'value'.")


def count_wandb_grid_combinations(config: dict[str, Any]) -> int:
    parameters = config.get("parameters")
    if parameters is None:
        raise ValueError("Sweep YAML must include a 'parameters' section.")
    if not isinstance(parameters, dict):
        raise ValueError("'parameters' section must be a mapping.")

    counts = []
    for name, spec in parameters.items():
        try:
            counts.append(parameter_cardinality(spec))
        except ValueError as error:
            raise ValueError(f"parameter '{name}': {error}") from error

    return math.prod(counts) if counts else 1


def main(yaml_path: Path | str) -> None:
    path = Path(yaml_path)
    if not path.exists():
        sys.exit(f"Error: {path} not found.")

    config = load_yaml(path)
    total = count_wandb_grid_combinations(config)
    return total


if __name__ == "__main__":
    total = main(sys.argv[1] if len(sys.argv) > 1 else "parameters.yaml")
    print(total)
