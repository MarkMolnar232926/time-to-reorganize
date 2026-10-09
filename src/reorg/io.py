"""Thin parsers for SkillCorner open-data match files.

Schemas were verified against the actual files (see ``docs/DATA_AUDIT.md``):

* ``{id}_match.json``: lineups (``players[].id`` is the id used in tracking), pitch size,
  ``home_team_side`` per period, ``match_periods`` with start/end frames.
* ``{id}_tracking_extrapolated.jsonl``: one JSON object per frame at 10 fps with keys
  ``frame, timestamp, period, ball_data, possession, image_corners_projection, player_data``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GK_ROLE_ACRONYM = "GK"


@dataclass(frozen=True)
class MatchMeta:
    """Match-level metadata needed by the pipeline."""

    match_id: int
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    pitch_length: float
    pitch_width: float
    home_team_side: tuple[str, ...]
    periods: pd.DataFrame
    players: pd.DataFrame

    def team_id_for_group(self, group: str) -> int:
        """Map a tracking ``possession.group`` value ('home team'/'away team') to a team id."""
        return {"home team": self.home_team_id, "away team": self.away_team_id}[group]


def match_dir(root: str | Path, match_id: int) -> Path:
    return Path(root) / "matches" / str(match_id)


def list_match_ids(root: str | Path) -> list[int]:
    """Match ids that have a folder under ``{root}/matches``."""
    return sorted(int(p.name) for p in (Path(root) / "matches").iterdir() if p.is_dir())


def load_match_meta(root: str | Path, match_id: int) -> MatchMeta:
    path = match_dir(root, match_id) / f"{match_id}_match.json"
    m = json.loads(path.read_text())
    rows = []
    for p in m["players"]:
        pt = p.get("playing_time") or {}
        tot = pt.get("total") or {}
        role = p.get("player_role") or {}
        rows.append(
            {
                "player_id": p["id"],
                "team_id": p["team_id"],
                "number": p.get("number"),
                "short_name": p.get("short_name"),
                "role": role.get("acronym"),
                "position_group": role.get("position_group"),
                "is_gk": role.get("acronym") == GK_ROLE_ACRONYM,
                "start_frame": tot.get("start_frame"),
                "end_frame": tot.get("end_frame"),
                "minutes_played": tot.get("minutes_played"),
            }
        )
    return MatchMeta(
        match_id=int(m["id"]),
        home_team_id=int(m["home_team"]["id"]),
        away_team_id=int(m["away_team"]["id"]),
        home_team_name=m["home_team"]["short_name"],
        away_team_name=m["away_team"]["short_name"],
        pitch_length=float(m["pitch_length"]),
        pitch_width=float(m["pitch_width"]),
        home_team_side=tuple(m["home_team_side"]),
        periods=pd.DataFrame(m["match_periods"]),
        players=pd.DataFrame(rows),
    )


def _parse_timestamp(ts: str | None) -> float:
    if ts is None:
        return np.nan
    h, mi, s = ts.split(":")
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_tracking(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a tracking JSONL file into a frame table and a long player-frame table.

    Returns
    -------
    frames : one row per frame: ``frame, period, t (s within period clock), ball_x, ball_y,
        ball_z, ball_detected, poss_group, poss_player_id, n_players, n_detected``.
    players : one row per player per frame: ``frame, player_id, x, y, is_detected``.
    """
    f_cols: dict[str, list] = {
        k: []
        for k in (
            "frame",
            "period",
            "t",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_detected",
            "poss_group",
            "poss_player_id",
            "n_players",
            "n_detected",
        )
    }
    p_frame: list[int] = []
    p_id: list[int] = []
    p_x: list[float] = []
    p_y: list[float] = []
    p_det: list[bool] = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            fr = d["frame"]
            ball = d.get("ball_data") or {}
            poss = d.get("possession") or {}
            pdata = d.get("player_data") or []
            f_cols["frame"].append(fr)
            f_cols["period"].append(d.get("period"))
            f_cols["t"].append(_parse_timestamp(d.get("timestamp")))
            f_cols["ball_x"].append(ball.get("x"))
            f_cols["ball_y"].append(ball.get("y"))
            f_cols["ball_z"].append(ball.get("z"))
            f_cols["ball_detected"].append(ball.get("is_detected"))
            f_cols["poss_group"].append(poss.get("group"))
            f_cols["poss_player_id"].append(poss.get("player_id"))
            f_cols["n_players"].append(len(pdata))
            nd = 0
            for p in pdata:
                p_frame.append(fr)
                p_id.append(p["player_id"])
                p_x.append(p["x"])
                p_y.append(p["y"])
                det = bool(p.get("is_detected"))
                p_det.append(det)
                nd += det
            f_cols["n_detected"].append(nd)
    frames = pd.DataFrame(f_cols)
    for c in ("ball_x", "ball_y", "ball_z"):
        frames[c] = pd.to_numeric(frames[c], errors="coerce").astype("float64")
    frames["period"] = frames["period"].astype("Int8")
    frames["poss_player_id"] = frames["poss_player_id"].astype("Int64")
    frames["ball_detected"] = frames["ball_detected"].astype("boolean")
    players = pd.DataFrame(
        {
            "frame": np.asarray(p_frame, dtype=np.int32),
            "player_id": np.asarray(p_id, dtype=np.int64),
            "x": np.asarray(p_x, dtype=np.float32),
            "y": np.asarray(p_y, dtype=np.float32),
            "is_detected": np.asarray(p_det, dtype=bool),
        }
    )
    return frames, players


def load_tracking(
    root: str | Path, match_id: int, cache: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tracking for a match, using a parquet cache if ``cache`` is given."""
    if cache is not None:
        cdir = Path(cache)
        fp, pp = cdir / f"{match_id}_frames.parquet", cdir / f"{match_id}_players.parquet"
        if fp.exists() and pp.exists():
            return pd.read_parquet(fp), pd.read_parquet(pp)
    path = match_dir(root, match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    frames, players = parse_tracking(path)
    if cache is not None:
        cdir.mkdir(parents=True, exist_ok=True)
        frames.to_parquet(fp, index=False)
        players.to_parquet(pp, index=False)
    return frames, players


def load_dynamic_events(root: str | Path, match_id: int) -> pd.DataFrame:
    """Dynamic events CSV. NB: x/y are NOT scaled to pitch size (SkillCorner README)."""
    path = match_dir(root, match_id) / f"{match_id}_dynamic_events.csv"
    return pd.read_csv(path, low_memory=False)


def load_phases(root: str | Path, match_id: int) -> pd.DataFrame:
    """Phases-of-play CSV (one row per phase, defined only while the ball is in play)."""
    return pd.read_csv(match_dir(root, match_id) / f"{match_id}_phases_of_play.csv")
