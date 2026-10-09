"""Outcome ("dangerous attack conceded") definition and landmark datasets for H2/H4.

Written and unit-tested on synthetic data *before* ANALYSIS_PLAN.md is frozen. It must not be
run on the real matches until the freeze (GATE 2), see ANALYSIS_PLAN.md.

Dangerous attack (ANALYSIS_PLAN.md §O): within ``danger_window_s`` after the loss, while the
gaining team still has the ball in open play (before the losing team's next control or a dead
ball), either
* a shot: a gaining-team ``player_possession`` ending in ``end_type == "shot"``, or
* a box entry: the ball moves from outside to inside the losing team's penalty area and stays
  there at least ``min_box_frames`` frames.
If the ball is already inside the box at the loss, only a shot counts.

Why landmarks: a shot ends the T_r observation window, so using T_r itself as the exposure
would bias the association (episodes with early danger cannot have long observed T_r). A
landmark analysis at time l uses only episodes still in open play and danger-free at l, measures
the exposure at l, and counts danger after l.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PENALTY_DEPTH_M = 16.5
PENALTY_HALF_WIDTH_M = 20.16


@dataclass(frozen=True)
class OutcomeParams:
    fps: float
    danger_window_s: float
    min_box_frames: int

    @classmethod
    def from_config(cls, cfg: dict) -> OutcomeParams:
        oc = cfg["outcomes"]
        return cls(
            fps=float(cfg["data"]["fps"]),
            danger_window_s=float(oc["danger_window_s"]),
            min_box_frames=int(oc["min_box_frames"]),
        )


def in_own_box(ball_x: np.ndarray, ball_y: np.ndarray, L: float) -> np.ndarray:
    """Ball inside the defending team's own penalty area (defending frame: own goal at -L/2)."""
    return (ball_x <= -L / 2 + PENALTY_DEPTH_M) & (np.abs(ball_y) <= PENALTY_HALF_WIDTH_M)


def first_danger(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    shot_frames: np.ndarray,
    stop: int,
    L: float,
    params: OutcomeParams,
) -> tuple[float, str]:
    """Time (s) and type of the first dangerous event in one episode; (nan, "") if none.

    ``ball_x``/``ball_y`` start at the loss frame (index 0) in the defending team's frame.
    ``shot_frames`` are gaining-team shot frames relative to the loss. ``stop`` is the relative
    frame where open play under the gaining team's possession ends (regain or dead ball).
    """
    horizon = min(int(round(params.danger_window_s * params.fps)), stop)
    t_shot = np.inf
    s = np.asarray(shot_frames)
    s = s[(s >= 0) & (s <= horizon)]
    if s.size:
        t_shot = float(s.min())
    t_box = np.inf
    inside = in_own_box(ball_x[: horizon + 1], ball_y[: horizon + 1], L)
    inside = np.where(np.isfinite(ball_x[: horizon + 1]), inside, False)
    if inside.size and not inside[0]:
        run = 0
        for j, v in enumerate(inside):
            run = run + 1 if v else 0
            if run >= params.min_box_frames:
                t_box = float(j - run + 1)
                break
    t = min(t_shot, t_box)
    if not np.isfinite(t):
        return np.nan, ""
    kind = "shot" if t_shot <= t_box else "box_entry"
    return t / params.fps, kind


def landmark_sample(episodes: pd.DataFrame, landmark_s: float) -> pd.DataFrame:
    """Episodes eligible for a landmark analysis at ``landmark_s``.

    Eligible = disorganised at loss (D-015), open-play window still running at the landmark
    (no regain, dead ball, shot or cap before it), exposure defined at the landmark, and no
    dangerous event at or before the landmark. Outcome = danger after the landmark (within the
    danger window). Requires columns ``window_s``, ``D_{l}s``, ``t_danger``.
    """
    col = f"D_{landmark_s:g}s"
    e = episodes[episodes["disorganised_at_loss"]].copy()
    e = e[(e["window_s"] > landmark_s) & np.isfinite(e[col])]
    e = e[~(e["t_danger"] <= landmark_s)]
    e["y"] = np.isfinite(e["t_danger"]).astype(int)
    e["still_disorganised"] = (e[col] > e["tau"]).astype(int)
    e["log_D"] = np.log(e[col])
    return e


def add_danger(md, episodes: pd.DataFrame, params: OutcomeParams) -> pd.DataFrame:
    """Attach ``t_danger`` (s, NaN if none) and ``danger_type`` to one match's episodes.

    Open play under the gaining team ends at the episode window end (regain, dead ball, period
    end, or the opponent's shot, which itself counts as danger). DO NOT RUN on real data before
    ANALYSIS_PLAN.md is frozen.
    """
    ev = md.events
    shots = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "shot")]
    out = episodes.copy()
    t_d, kinds = [], []
    for r in out.itertuples(index=False):
        tf = md.teams[r.losing_team_id].tf
        i0 = int(r.frame_loss - md.frame[0])
        stop = int(round(r.window_s * params.fps))
        n = int(round(params.danger_window_s * params.fps)) + 1
        sh = shots[(shots["team_id"] == r.gaining_team_id) & (shots["period"] == r.period)]
        rel = sh["frame_end"].to_numpy() - r.frame_loss
        t, k = first_danger(
            tf.ball_x[i0 : i0 + n], tf.ball_y[i0 : i0 + n], rel, stop, tf.pitch_length, params
        )
        t_d.append(t)
        kinds.append(k)
    out["t_danger"] = t_d
    out["danger_type"] = kinds
    return out
