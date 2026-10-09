"""Coordinate normalisation and per-team frame arrays.

Every team is expressed in its *own* frame: it attacks towards +x, so its own goal line is at
``x = -pitch_length / 2``. Within a period this is a 180° rotation (x, y) -> (-x, -y) for the
team attacking right-to-left, which keeps "left"/"right" relative to the attacking direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reorg.io import MatchMeta

N_OUTFIELD = 10
BALL_FFILL_FRAMES = (
    10  # bridge ball-position gaps up to 1 s (ball x/y null in ~2% of in-play frames)
)


@dataclass
class TeamFrames:
    """Outfield positions of one team for every frame from kick-off to full time, in its own frame.

    Arrays are aligned with ``frame``. Missing tracking (broadcast replays) gives NaN rows.
    ``X``, ``Y``, ``det`` have shape (n_frames, 10). ``player_ids`` gives the slot occupants.
    """

    match_id: int
    team_id: int
    frame: np.ndarray
    period: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    det: np.ndarray
    player_ids: np.ndarray
    ball_x: np.ndarray
    ball_y: np.ndarray
    pitch_length: float
    pitch_width: float

    @property
    def tracked(self) -> np.ndarray:
        return np.isfinite(self.X).all(axis=1)

    def index_of(self, frames: np.ndarray | int) -> np.ndarray:
        """Positional index of frame number(s); frames are contiguous so this is an offset."""
        return np.asarray(frames) - self.frame[0]


def attack_sign(meta: MatchMeta, team_id: int, period: int) -> int:
    """+1 if ``team_id`` attacks left-to-right (+x) in ``period``, else -1."""
    home_ltr = meta.home_team_side[period - 1] == "left_to_right"
    is_home = team_id == meta.home_team_id
    return 1 if home_ltr == is_home else -1


def team_frames(
    meta: MatchMeta, frames: pd.DataFrame, players: pd.DataFrame, team_id: int
) -> TeamFrames:
    """Build :class:`TeamFrames` for ``team_id`` from parsed tracking.

    Goalkeepers (lineup role ``GK``) are excluded. Slots are filled in player-id order per frame;
    shape components are permutation-invariant so slot identity does not matter for them.
    """
    fr = frames.sort_values("frame")
    inp = fr["period"].notna().to_numpy()
    first, last = np.flatnonzero(inp)[[0, -1]]
    fr = fr.iloc[first : last + 1].reset_index(drop=True)
    if not (np.diff(fr["frame"].to_numpy()) == 1).all():
        raise ValueError("frames are expected to be contiguous")
    f0 = int(fr["frame"].iloc[0])
    n = len(fr)
    # period 0 = between periods (half-time); those frames are masked out downstream.
    period = fr["period"].fillna(0).to_numpy(dtype=np.int64)
    sign = np.array([1, *(attack_sign(meta, team_id, p) for p in (1, 2))])[period]

    roster = meta.players
    outfield = set(roster.loc[(roster["team_id"] == team_id) & ~roster["is_gk"], "player_id"])
    p = players[players["player_id"].isin(outfield)]
    p = p[(p["frame"] >= f0) & (p["frame"] < f0 + n)].sort_values(["frame", "player_id"])
    slot = p.groupby("frame").cumcount().to_numpy()
    keep = slot < N_OUTFIELD
    p, slot = p[keep], slot[keep]
    idx = p["frame"].to_numpy() - f0

    X = np.full((n, N_OUTFIELD), np.nan)
    Y = np.full((n, N_OUTFIELD), np.nan)
    det = np.zeros((n, N_OUTFIELD), dtype=bool)
    pids = np.zeros((n, N_OUTFIELD), dtype=np.int64)
    X[idx, slot] = p["x"].to_numpy()
    Y[idx, slot] = p["y"].to_numpy()
    det[idx, slot] = p["is_detected"].to_numpy()
    pids[idx, slot] = p["player_id"].to_numpy()
    # A frame with fewer than 10 outfield players (e.g. red card) is kept but flagged as
    # untracked for the 10-player shape; this does not occur in the 2024/25 sample.
    incomplete = np.isnan(X).any(axis=1)
    X[incomplete] = np.nan
    Y[incomplete] = np.nan

    s = sign[:, None]
    bx = fr["ball_x"].ffill(limit=BALL_FFILL_FRAMES).to_numpy(dtype=float)
    by = fr["ball_y"].ffill(limit=BALL_FFILL_FRAMES).to_numpy(dtype=float)
    return TeamFrames(
        match_id=meta.match_id,
        team_id=team_id,
        frame=fr["frame"].to_numpy(),
        period=period,
        X=X * s,
        Y=Y * s,
        det=det,
        player_ids=pids,
        ball_x=bx * sign,
        ball_y=by * sign,
        pitch_length=meta.pitch_length,
        pitch_width=meta.pitch_width,
    )
