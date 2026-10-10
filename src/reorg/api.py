"""High-level API.

Example::

    import reorg
    cfg = reorg.load_config("config.yaml")
    res = reorg.run_league(cfg)                       # all matches under cfg["data"]["root"]
    res.episodes.head()                               # one row per possession loss
    reorg.team_summary(res.episodes)                  # per-team reorganisation speed
    reorg.plot_reorganisation(res, 2007721, 18757, "episode.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from reorg.io import list_match_ids
from reorg.pipeline import MatchData, prepare_match
from reorg.reference import Reference, build_reference, organised_frames
from reorg.reorganisation import ReorgParams, episodes_for_match, team_taus


def load_config(path: str | Path = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass
class LeagueResult:
    matches: list[MatchData]
    reference: Reference
    tau: dict[int, float]
    episodes: pd.DataFrame
    params: ReorgParams
    traces: dict = field(default_factory=dict)

    def match(self, match_id: int) -> MatchData:
        return next(m for m in self.matches if m.meta.match_id == match_id)


def run_league(
    cfg: dict,
    root: str | Path | None = None,
    match_ids: list[int] | None = None,
    min_frame_det: float | None = None,
    team_specific: bool = True,
    continuous: bool = True,
    keep_traces: bool = False,
) -> LeagueResult:
    """Detect losses, build references and summarise every episode for a set of matches.

    Works unchanged for any number of matches in the SkillCorner open-data layout.
    """
    root = str(root or cfg["data"]["root"])
    cache = cfg["data"].get("cache")
    ids = match_ids or list_match_ids(root)
    mds = [prepare_match(root, m, cfg, cache) for m in ids]
    ref = build_reference(organised_frames(mds), cfg, team_specific=team_specific)
    params = ReorgParams.from_config(cfg)
    taus = team_taus(mds, ref, params, continuous=continuous)
    eps, traces = [], {}
    names = {}
    for md in mds:
        names[md.meta.home_team_id] = md.meta.home_team_name
        names[md.meta.away_team_id] = md.meta.away_team_name
        e, tr = episodes_for_match(
            md,
            ref,
            taus,
            params,
            min_frame_det=min_frame_det,
            keep_traces=keep_traces,
            continuous=continuous,
        )
        eps.append(e)
        traces.update(tr)
    episodes = pd.concat(eps, ignore_index=True)
    episodes.insert(3, "team", episodes["losing_team_id"].map(names))
    return LeagueResult(mds, ref, taus, episodes, params, traces)


def plot_reorganisation(res: LeagueResult, match_id: int, frame_loss: int, path=None):
    """Pitch snapshots (loss, +3 s, +6 s) and the D(t) trace for one episode."""
    from reorg.reorganisation import episodes_for_match
    from reorg.viz import EpisodeView, plot_snapshots

    md = res.match(match_id)
    e, tr = episodes_for_match(md, res.reference, res.tau, res.params, keep_traces=True)
    row = e[e["frame_loss"] == frame_loss].iloc[0]
    t = tr[f"{match_id}_{frame_loss}"]
    v = EpisodeView(md, res.reference, int(row["losing_team_id"]), frame_loss)
    status = (
        f"T_r = {row['T_r']:.1f} s"
        if row["reorganised"]
        else f"not reorganised ({row['end_reason']})"
    )
    return plot_snapshots(v, D=t.D, tau=t.tau, path=path, suptitle=status, fps=res.params.fps)
