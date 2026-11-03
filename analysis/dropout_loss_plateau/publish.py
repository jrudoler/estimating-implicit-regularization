"""Publish final endpoints from scratch without filling the home quota."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket

from core.artifact_storage import publish_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    publish_endpoint(args.source, args.destination, node=socket.gethostname(), lock_path=args.lock)


if __name__ == "__main__":
    main()
