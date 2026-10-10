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


def _episodes(cfg: dict, out: Path, teams_out: Path) -> None:
    from reorg.api import run_league
    from reorg.reorganisation import team_summary

    res = run_league(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.episodes.to_csv(out, index=False)
    teams = team_summary(res.episodes)
    teams.insert(
        1,
        "team",
        teams["losing_team_id"].map(
            res.episodes.drop_duplicates("losing_team_id").set_index("losing_team_id")["team"]
        ),
    )
    teams.to_csv(teams_out, index=False)
    print(teams.round(3).to_string(index=False))


def _team_cards(cfg: dict, out: Path, team: str | None) -> None:
    import numpy as np

    from reorg.api import run_league
    from reorg.profiles import plot_team_card, team_profile

    res = run_league(cfg)
    out.mkdir(parents=True, exist_ok=True)
    grid = np.round(np.arange(0, res.params.horizon_s + 1e-9, 0.1), 1)
    seed = int(cfg["prototype"]["seed"])
    n_boot = int(cfg["prototype"]["n_boot"])
    teams = [team] if team else sorted(res.episodes["team"].dropna().unique())
    for t in teams:
        p = team_profile(res.episodes, t, 6.0, grid, n_boot, seed)
        path = out / f"{t.replace(' ', '_')}.png"
        plot_team_card(p, path)
        print(f"wrote {path}")


def _episode(cfg: dict, match_id: int, frame: int, out: Path, video: bool) -> None:
    from reorg.api import run_league
    from reorg.reorganisation import episodes_for_match
    from reorg.viz import EpisodeView, animate_episode, plot_snapshots

    res = run_league(cfg, keep_traces=False)
    md = res.match(match_id)
    e, tr = episodes_for_match(md, res.reference, res.tau, res.params, keep_traces=True)
    if frame not in set(e["frame_loss"]):
        near = e.iloc[(e["frame_loss"] - frame).abs().argsort()[:5]]["frame_loss"].tolist()
        raise SystemExit(f"no loss at frame {frame} in match {match_id}; nearest: {near}")
    row = e[e["frame_loss"] == frame].iloc[0]
    t = tr[f"{match_id}_{frame}"]
    v = EpisodeView(md, res.reference, int(row["losing_team_id"]), frame)
    team = res.episodes.loc[res.episodes["losing_team_id"] == row["losing_team_id"], "team"].iloc[0]
    status = (
        f"back in shape after {row['T_r']:.1f} s"
        if row["reorganised"]
        else f"not back in shape ({row['end_reason']} after {row['window_s']:.1f} s)"
    )
    title = f"{team} lose the ball (match {match_id}, frame {frame}): {status}"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"episode_{match_id}_{frame}"
    plot_snapshots(v, D=t.D, tau=t.tau, path=f"{stem}.png", suptitle=title, fps=res.params.fps)
    print(f"wrote {stem}.png")
    if video:
        animate_episode(
            v,
            t.D,
            t.tau,
            t.end,
            f"{stem}.mp4",
            fps=res.params.fps,
            title=title,
            t_r=row["T_r"] if row["reorganised"] else None,
        )
        print(f"wrote {stem}.mp4")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="reorg", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_l = sub.add_parser("losses", help="detect possession losses in every match")
    p_l.add_argument("--out", type=Path, default=None, help="optional CSV output path")
    p_e = sub.add_parser("episodes", help="time-to-reorganise summary for every loss")
    p_e.add_argument("--out", type=Path, default=Path("outputs/episodes.csv"))
    p_e.add_argument("--teams-out", type=Path, default=Path("outputs/teams.csv"))
    p_t = sub.add_parser("team-cards", help="one-page 'what goes wrong' card per team")
    p_t.add_argument("--out", type=Path, default=Path("outputs/teams"))
    p_t.add_argument("--team", default=None, help="only this team (name as in the data)")
    p_v = sub.add_parser("episode", help="snapshots (and optional video) of one loss")
    p_v.add_argument("--match", type=int, required=True)
    p_v.add_argument("--frame", type=int, required=True, help="frame of the loss")
    p_v.add_argument("--out", type=Path, default=Path("outputs/episodes"))
    p_v.add_argument("--video", action="store_true", help="also render an MP4 (needs ffmpeg)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cmd == "losses":
        _losses(cfg, args.out)
    elif args.cmd == "episodes":
        _episodes(cfg, args.out, args.teams_out)
    elif args.cmd == "team-cards":
        _team_cards(cfg, args.out, args.team)
    elif args.cmd == "episode":
        _episode(cfg, args.match, args.frame, args.out, args.video)


if __name__ == "__main__":
    main()
