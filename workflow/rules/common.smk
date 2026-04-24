# Shared paths and config loading.

configfile: "config/sweeps.yaml"


# Device override for training rules. Defaults to autodetect: the training
# scripts pick CUDA > MPS > CPU on their own. Override at the CLI:
#   uv run snakemake --config device=cpu figures
#   uv run snakemake --config device=cuda --profile workflow/profiles/slurm paper
def device_arg() -> str:
    value = config.get("device")
    return f"--device {value}" if value else ""
