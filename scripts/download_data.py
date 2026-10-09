"""Download the public SkillCorner open data (A-League 2024/25) at a pinned commit.

Standard library only (no git, git-lfs or curl needed). Small files come from
raw.githubusercontent.com. Tracking files are Git LFS objects fetched from
media.githubusercontent.com, and each is verified against the SHA-256 in its LFS pointer.
Files already present with the right hash are skipped.

Usage: python scripts/download_data.py [--rev COMMIT] [--dest data/skillcorner]
The data is NOT redistributed by this repository (see DECISIONS.md D-002).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "SkillCorner/opendata"
REV = "4340d274572876239c154c90bc507a9b3250a656"  # pinned upstream commit (14 Sep 2026)
RAW = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"
MEDIA = "https://media.githubusercontent.com/media/{repo}/{rev}/{path}"
RETRIES = 4  # network retries per file, exponential backoff from 2 s
CHUNK = 1 << 20


def _get(url: str, dest: Path) -> None:
    for attempt in range(RETRIES + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                while chunk := r.read(CHUNK):
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == RETRIES:
                raise
            time.sleep(2 ** (attempt + 1))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _fetch(rel: str, root: Path, rev: str) -> str:
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith("_tracking_extrapolated.jsonl"):
        ptr = urllib.request.urlopen(RAW.format(repo=REPO, rev=rev, path=rel), timeout=60).read()
        m = re.search(rb"oid sha256:([0-9a-f]{64})", ptr)
        if m is None:
            raise RuntimeError(f"{rel}: expected a Git LFS pointer")
        want = m.group(1).decode()
        if dest.exists() and _sha256(dest) == want:
            return f"ok (cached) {rel}"
        _get(MEDIA.format(repo=REPO, rev=rev, path=rel), dest)
        if _sha256(dest) != want:
            dest.unlink()
            raise RuntimeError(f"{rel}: checksum mismatch")
        return f"ok {rel}"
    if not dest.exists():
        _get(RAW.format(repo=REPO, rev=rev, path=rel), dest)
    return f"ok {rel}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", default=REV)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "data/skillcorner"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    root = Path(a.dest)
    for rel in ("data/matches.json", "data/bodypose/MANIFEST.json", "LICENSE", "README.md"):
        print(_fetch(rel, root, a.rev))
    ids = [m["id"] for m in json.loads((root / "data/matches.json").read_text())]
    rels = [
        f"data/matches/{i}/{i}_{kind}"
        for i in ids
        for kind in (
            "match.json",
            "dynamic_events.csv",
            "phases_of_play.csv",
            "tracking_extrapolated.jsonl",
        )
    ]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda r: _fetch(r, root, a.rev), rels):
            print(msg)
    print(f"SkillCorner open data ready in {root} ({len(ids)} matches, rev {a.rev[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
