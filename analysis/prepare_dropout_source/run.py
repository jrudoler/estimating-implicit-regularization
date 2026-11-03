"""Restore the immutable training source used by the manuscript dropout study."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import tarfile

ARCHIVE_SHA256 = "15fb6023d4d207bf5eafb51e68998958528767a4c74790a57ae7c107ac18e278"


def restore(archive: Path, manifest: Path, output_dir: Path) -> None:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("The manuscript requires the original source-v1 archive")
    expected = set(json.loads(manifest.read_text()))
    with tarfile.open(archive) as handle:
        members = [m for m in handle.getmembers() if m.isfile()]
        if {m.name for m in members} != expected:
            raise ValueError("Source manifest does not match archive")
        for member in members:
            target = output_dir / member.name
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError("Archive path escapes output directory")
            data = handle.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Refusing to replace modified source: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    logging.info("Verified/restored %d frozen source files", len(expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    restore(args.archive, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
