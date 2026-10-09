"""Per-match preparation shared by reference building and episode extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from reorg.io import (
    MatchMeta,
    load_dynamic_events,
    load_match_meta,
    load_phases,
    load_tracking,
)
from reorg.normalise import TeamFrames, team_frames
from reorg.shape import shape_components
from reorg.turnovers import LossParams, find_losses, in_play_intervals, team_possession_runs


@dataclass
class TeamData:
    tf: TeamFrames
    S: pd.DataFrame  # shape components, aligned with tf.frame
    cell: np.ndarray  # ball context cell (-1 = unknown)
    organised: np.ndarray  # bool: frame usable for the team's reference block
    out_of_possession: np.ndarray  # bool: phase says the opponent is in possession


@dataclass
class MatchData:
    meta: MatchMeta
    frame: np.ndarray
    in_play: np.ndarray
    phase_team: np.ndarray  # team in possession per frame from phases (-1 = no phase)
    phase_oop_type: np.ndarray  # out-of-possession phase label per frame ("" = none)
    events: pd.DataFrame
    runs: pd.DataFrame
    losses: pd.DataFrame
    frames_table: pd.DataFrame
    teams: dict[int, TeamData] = field(default_factory=dict)


def phase_per_frame(frame: np.ndarray, phases: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Team in possession and out-of-possession phase label for every frame.

    Boundary frames shared by two phases are given to the later phase.
    """
    ph = phases.sort_values("frame_start")
    starts = ph["frame_start"].to_numpy()
    ends = ph["frame_end"].to_numpy()
    k = np.searchsorted(starts, frame, side="right") - 1
    ok = (k >= 0) & (frame <= ends[np.clip(k, 0, None)])
    team = np.full(len(frame), -1, dtype=np.int64)
    oop = np.full(len(frame), "", dtype=object)
    team[ok] = ph["team_in_possession_id"].to_numpy()[k[ok]]
    oop[ok] = ph["team_out_of_possession_phase_type"].to_numpy()[k[ok]]
    return team, oop


def ball_context(
    ball_x: np.ndarray, ball_y: np.ndarray, L: float, W: float, n_ch: int, n_dp: int
) -> np.ndarray:
    """Context cell = depth_index * n_ch + channel_index in the defending team's frame.

    Depth 0 = the team's own third; channel 0 = its right-hand side when attacking +x (y < 0).
    Unknown ball position gives -1.
    """
    ok = np.isfinite(ball_x) & np.isfinite(ball_y)
    dp = np.clip(np.floor((ball_x + L / 2) / (L / n_dp)), 0, n_dp - 1)
    ch = np.clip(np.floor((ball_y + W / 2) / (W / n_ch)), 0, n_ch - 1)
    cell = np.where(ok, dp * n_ch + ch, -1)
    return np.nan_to_num(cell, nan=-1).astype(np.int64)


def prepare_match(
    root: str | Path, match_id: int, cfg: dict, cache: str | Path | None
) -> MatchData:
    """Load one match and compute everything per team that the metric needs."""
    meta = load_match_meta(root, match_id)
    fr, pl = load_tracking(root, match_id, cache=cache)
    ph = load_phases(root, match_id)
    ev = load_dynamic_events(root, match_id)
    params = LossParams.from_config(cfg)
    losses = find_losses(ev, ph, params)
    runs = team_possession_runs(ev)
    rc = cfg["reference"]
    fps = float(cfg["data"]["fps"])
    settle = int(rc["settled_after_loss_s"] * fps)
    organised_labels = set(rc["organised_phases"])

    teams: dict[int, TeamData] = {}
    md: MatchData | None = None
    for team_id in (meta.home_team_id, meta.away_team_id):
        tf = team_frames(meta, fr, pl, team_id)
        if md is None:
            s0, e0 = in_play_intervals(ph)
            k = np.searchsorted(s0, tf.frame, side="right") - 1
            in_play = (k >= 0) & (tf.frame <= e0[np.clip(k, 0, None)])
            p_team, p_oop = phase_per_frame(tf.frame, ph)
            md = MatchData(meta, tf.frame, in_play, p_team, p_oop, ev, runs, losses, fr)
        S = shape_components(tf.X, tf.Y, tf.ball_x, tf.ball_y, tf.pitch_length)
        cell = ball_context(
            tf.ball_x, tf.ball_y, tf.pitch_length, tf.pitch_width, rc["n_channels"], rc["n_depths"]
        )
        oop = (md.phase_team != team_id) & (md.phase_team != -1)
        recent = np.zeros(len(tf.frame), dtype=bool)
        for fl in losses.loc[losses["losing_team_id"] == team_id, "frame_loss"]:
            i = int(fl - tf.frame[0])
            recent[i : i + settle] = True
        organised = (
            oop
            & np.isin(md.phase_oop_type, list(organised_labels))
            & md.in_play
            & tf.tracked
            & (cell >= 0)
            & (tf.period > 0)
            & ~recent
        )
        teams[team_id] = TeamData(tf, S, cell, organised, oop)
    assert md is not None
    md.teams = teams
    return md
