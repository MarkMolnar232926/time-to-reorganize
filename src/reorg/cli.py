"""Command-line interface: ``reorg <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from reorg import __version__
from reorg.io import list_match_ids, load_dynamic_events, load_phases
from reorg.turnovers import LossParams, find_losses


def _losses(cfg: dict, out: Path | None) -> None:
    root = cfg["data"]["root"]
    params = LossParams.from_config(cfg)
    frames = []
    for mid in list_match_ids(root):
        res = find_losses(load_dynamic_events(root, mid), load_phases(root, mid), params)
        res.insert(0, "match_id", mid)
        frames.append(res)
    df = pd.concat(frames, ignore_index=True)
    print(df.groupby("match_id").size().to_string())
    print(f"total losses: {len(df)}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="reorg", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_l = sub.add_parser("losses", help="detect possession losses in every match")
    p_l.add_argument("--out", type=Path, default=None, help="optional CSV output path")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cmd == "losses":
        _losses(cfg, args.out)


if __name__ == "__main__":
    main()
