#!/usr/bin/env python

"""Bootstrap optional third_party repositories for ODISE."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict


THIRD_PARTY_ROOT_REPOS: Dict[str, str] = {
    "latent-diffusion": "https://github.com/CompVis/latent-diffusion.git",
    "taming-transformers": "https://github.com/CompVis/taming-transformers.git",
}


def _run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_repo(name: str, destination: Path) -> None:
    url = THIRD_PARTY_ROOT_REPOS[name]
    marker = destination / ".git"

    if destination.exists():
        if not marker.exists():
            raise RuntimeError(
                f"{destination} already exists but is not a git repository. "
                "Please move/rename it before retrying."
            )
        _run(["git", "-C", str(destination), "fetch", "--all"])
        return

    _run(["git", "clone", "--depth", "1", url, str(destination)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=".", help="Repository root where third_party/ lives (default: '.')"
    )
    parser.add_argument("--all", action="store_true", help="Clone all optional repos.")
    parser.add_argument(
        "--latent-diffusion",
        action="store_true",
        help="Clone optional latent-diffusion integration.",
    )
    parser.add_argument(
        "--taming-transformers",
        action="store_true",
        help="Clone optional taming-transformers integration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh existing checkouts by fetching remotes.",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()
    third_party_root = root / "third_party"
    third_party_root.mkdir(parents=True, exist_ok=True)
    os.chdir(third_party_root)

    selected = []
    if args.all:
        selected = sorted(THIRD_PARTY_ROOT_REPOS.keys())
    else:
        if args.latent_diffusion:
            selected.append("latent-diffusion")
        if args.taming_transformers:
            selected.append("taming-transformers")

    if not selected:
        raise SystemExit(
            "No repository selected. Use --all, --latent-diffusion, or --taming-transformers."
        )

    for repo in selected:
        destination = third_party_root / repo
        _ensure_repo(repo, destination)

    if args.force:
        for repo in selected:
            destination = third_party_root / repo
            if (destination / ".git").exists():
                _run(["git", "-C", str(destination), "pull", "--ff-only"])

    print("Bootstrap completed:", ", ".join(selected))


if __name__ == "__main__":
    main()
