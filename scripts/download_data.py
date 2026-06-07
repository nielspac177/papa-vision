#!/usr/bin/env python
"""Download the PlantVillage *potato* subset (3 classes) into ``data/potato/``.

Primary source: the canonical PlantVillage GitHub mirror
(``spMohanty/PlantVillage-Dataset``), fetched via the GitHub contents API +
raw.githubusercontent.com. This needs no API token and no heavy ``datasets``
dependency — only the Python standard library.

Fallbacks:
    --synthetic   generate a small synthetic stand-in (offline / CI).

The script is idempotent: classes already populated are skipped.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Make the package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from papavision import CLASSES  # noqa: E402
from papavision.data import make_synthetic_dataset  # noqa: E402
from papavision.utils import DATA_DIR, get_logger  # noqa: E402

log = get_logger()

REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"
SUBDIR = "raw/color"  # color images live here
# The GitHub *contents* API returns up to 1000 entries for a directory in a single
# response (it does not paginate reliably beyond that). Each potato class has
# <= 1000 images, so one request per class suffices.
API = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"


def _auth_token() -> str | None:
    """Find a GitHub token: env var first, then the `gh` CLI. Optional but lifts
    the API rate limit from 60/hour (anonymous) to 5000/hour."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _headers() -> dict:
    h = {"User-Agent": "papa-vision-downloader", "Accept": "application/vnd.github+json"}
    token = _auth_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


HEADERS = _headers()


def _get(url: str, headers: dict | None = None, retries: int = 3, timeout: int = 30) -> bytes:
    """HTTP GET with simple exponential-backoff retries."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last_err})")


def list_class_files(cls: str) -> list[dict]:
    """List image entries for one class directory via the GitHub contents API."""
    url = API.format(repo=REPO, path=f"{SUBDIR}/{cls}", branch=BRANCH)
    data = json.loads(_get(url, HEADERS).decode())
    if isinstance(data, dict) and data.get("message"):
        raise RuntimeError(f"GitHub API error for {cls}: {data['message']}")
    return [e for e in data if e.get("type") == "file"]


def download_class(cls: str, out_dir: Path) -> int:
    """Download every image for one class; returns the number written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*"))
    if len(existing) >= 50:  # already populated
        log.info("  %s already has %d images — skipping.", cls, len(existing))
        return len(existing)

    entries = list_class_files(cls)
    log.info("  %s: %d images to fetch", cls, len(entries))
    n = 0
    for i, e in enumerate(entries):
        dest = out_dir / e["name"]
        if dest.exists():
            n += 1
            continue
        try:
            dest.write_bytes(_get(e["download_url"], HEADERS))
            n += 1
        except RuntimeError as err:
            log.warning("  failed %s (%s)", e["name"], err)
        if (i + 1) % 200 == 0:
            log.info("    ... %d/%d", i + 1, len(entries))
    return n


def download_from_github(root: Path) -> dict[str, int]:
    """Download all three potato classes from the GitHub mirror."""
    counts = {}
    for cls in CLASSES:
        counts[cls] = download_class(cls, root / cls)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the PlantVillage potato subset.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate a synthetic stand-in instead of downloading.")
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "potato"),
                        help="Output directory (default: data/potato).")
    args = parser.parse_args()
    root = Path(args.out)

    if args.synthetic:
        make_synthetic_dataset(root, n_per_class=120, size=128, seed=0)
        log.info("Synthetic potato dataset ready at %s", root)
        return

    log.info("Downloading PlantVillage potato subset from %s ...", REPO)
    try:
        counts = download_from_github(root)
    except Exception as e:  # noqa: BLE001 — give an actionable message, then fall back
        log.error("GitHub download failed: %s", e)
        log.error("Falling back to a synthetic dataset so the pipeline still runs.")
        log.error("Re-run with network access for real images, or use --synthetic.")
        make_synthetic_dataset(root, n_per_class=120, size=128, seed=0)
        return

    total = sum(counts.values())
    log.info("Done. %d images: %s", total, counts)
    if total < 100:
        log.warning("Very few images downloaded — check your network connection.")


if __name__ == "__main__":
    main()
