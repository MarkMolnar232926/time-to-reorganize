"""Shape distance D(t), time to reorganise T_r and per-episode summaries.

D(t) = sqrt( sum_k w_k z_k(t)^2 / sum_k w_k ),  z_k = (S_k(t) - R_k(c(t))) / sigma_k(c(t)),

i.e. the weighted root-mean-square number of typical deviations from the team's own organised
block in the *current* ball context c(t) (the target moves with the ball).

T_r = time from loss to the first moment D <= tau that then holds for ``hold_s`` seconds, with
the whole hold inside the open-play window. The window ends at the first of: regain by the
losing team (competing event), a shot by the opponent, the ball going dead, period end, or the
cap ``horizon_s``. Frames with missing tracking (or, under a reliability regime, too few
detected defenders) have undefined D and break a hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from reorg.pipeline import MatchData, TeamData
from reorg.reference import Reference

# Episode end codes for survival analysis.
REORGANISED, REGAIN, CENSORED = 1, 2, 0


@dataclass(frozen=True)
class ReorgParams:
    fps: float
    horizon_s: float
    hold_s: float
    tau_quantile: float
    debt_window_s: float
    horizons_s: tuple[float, ...]
    ball_speed_window_s: float
    savgol_window: int
    savgol_order: int
    weights: dict[str, float]

    @classmethod
    def from_config(cls, cfg: dict) -> ReorgParams:
        rc = cfg["reorganisation"]
        return cls(
            fps=float(cfg["data"]["fps"]),
            horizon_s=float(rc["horizon_s"]),
            hold_s=float(rc["hold_s"]),
            tau_quantile=float(rc["tau_quantile"]),
            debt_window_s=float(rc["debt_window_s"]),
            horizons_s=tuple(float(h) for h in rc["decomposition_horizons_s"]),
            ball_speed_window_s=float(rc["ball_speed_window_s"]),
            savgol_window=int(rc["savgol_window_frames"]),
            savgol_order=int(rc["savgol_polyorder"]),
            weights={k: float(v) for k, v in cfg["shape"]["weights"].items()},
        )


def z_scores(td: TeamData, ref: Reference, continuous: bool = True) -> np.ndarray:
    """Per-frame standardised deviations (n_frames, K) from the team's reference.

    ``continuous`` interpolates the reference in ball position (D-011); otherwise the
    reference of the discrete context cell is used (sensitivity analysis).
    """
    tf = td.tf
    if continuous:
        C, Sg = ref.lookup_continuous(
            tf.team_id, tf.ball_x, tf.ball_y, tf.pitch_length, tf.pitch_width
        )
    else:
        C, Sg = ref.lookup(tf.team_id, td.cell)
    S = td.S[list(ref.components)].to_numpy(float)
    return (S - C) / Sg


def shape_distance(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted RMS of z across components; NaN if any component is NaN."""
    w = weights / weights.sum()
    return np.sqrt((w * z**2).sum(axis=1))


def team_tau(D: np.ndarray, organised: np.ndarray, q: float) -> float:
    """tau = q-quantile of D over the team's own organised frames."""
    vals = D[organised & np.isfinite(D)]
    return float(np.quantile(vals, q)) if vals.size else np.nan


def first_hold(below: np.ndarray, hold: int, end: int) -> int:
    """First index j with below[j : j + hold] all True and j + hold <= end; -1 if none.

    ``below`` must be False where D is undefined.
    """
    run = 0
    for j in range(min(end, len(below))):
        run = run + 1 if below[j] else 0
        if run >= hold:
            return j - hold + 1
    return -1


def time_to_reorganise(D: np.ndarray, tau: float, end: int, hold: int) -> tuple[int, int]:
    """T_r in frames for one episode trace ``D`` (index 0 = loss) and end code.

    Returns ``(t_frames, code)``: if reorganised, ``t`` is the start of the confirmed hold and
    code is REORGANISED. Otherwise ``t = max(end - hold, 0)`` (the last moment a hold could
    have started and still been confirmed) and code is left for the caller (regain/censored).
    """
    below = np.isfinite(D) & (D <= tau)
    j = first_hold(below, hold, end)
    if j >= 0:
        return j, REORGANISED
    return max(end - hold, 0), -1


@dataclass
class EpisodeTrace:
    episode_id: str
    D: np.ndarray
    z: np.ndarray
    det_frac: np.ndarray
    end: int
    tau: float


def episodes_for_match(
    md: MatchData,
    ref: Reference,
    tau: dict[int, float],
    params: ReorgParams,
    min_frame_det: float | None = None,
    keep_traces: bool = False,
) -> tuple[pd.DataFrame, dict[str, EpisodeTrace]]:
    """Summarise every primary loss in a match.

    ``min_frame_det``: reliability regime. Frames whose fraction of detected defending outfield
    players is below this value get undefined D (None = use all frames).
    """
    fps = params.fps
    H = int(round(params.horizon_s * fps))
    hold = int(round(params.hold_s * fps))
    w = np.array([params.weights[k] for k in ref.components])
    s0 = md.frame[0]
    # Dead-ball boundaries: end of the in-play stretch containing each frame.
    ip = md.in_play.astype(int)
    next_dead = np.full(len(ip), len(ip))
    nd = len(ip)
    for i in range(len(ip) - 1, -1, -1):
        if not ip[i]:
            nd = i
        next_dead[i] = nd
    ev = md.events
    shots = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "shot")]
    runs = md.runs
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows, traces = [], {}
    for r in md.losses.itertuples(index=False):
        A = int(r.losing_team_id)
        td = md.teams[A]
        if A not in cache:
            z = z_scores(td, ref)
            D = shape_distance(z, w)
            det = td.tf.det.mean(axis=1)
            cache[A] = (z, D, det)
        z, D, det = cache[A]
        i0 = int(r.frame_loss - s0)
        # Window end candidates (relative frames).
        reg = runs[
            (runs["period"] == r.period)
            & (runs["team_id"] == A)
            & (runs["frame_start"] > r.frame_loss)
        ]["frame_start"]
        t_regain = int(reg.iloc[0] - r.frame_loss) if len(reg) else np.inf
        sh = shots[
            (shots["team_id"] == r.gaining_team_id)
            & (shots["period"] == r.period)
            & (shots["frame_end"] >= r.frame_loss)
        ]["frame_end"]
        t_shot = int(sh.iloc[0] - r.frame_loss) if len(sh) else np.inf
        t_dead = int(next_dead[i0] - i0)
        per_end = np.flatnonzero(td.tf.period[i0:] != r.period)
        t_period = int(per_end[0]) if per_end.size else len(td.tf.period) - i0
        cands = {
            "regain": t_regain,
            "shot": t_shot,
            "dead_ball": t_dead,
            "period_end": t_period,
            "cap": H,
        }
        reason = min(cands, key=lambda k: cands[k])
        end = int(min(cands.values()))

        Dw = D[i0 : i0 + H + 1].copy()
        detw = det[i0 : i0 + H + 1]
        if min_frame_det is not None:
            Dw[detw < min_frame_det] = np.nan
        Dw[end + 1 :] = np.nan
        t_fr, code = time_to_reorganise(Dw, tau[A], end, hold)
        if code != REORGANISED:
            code = REGAIN if reason == "regain" else CENSORED
        row = {
            "match_id": md.meta.match_id,
            "period": r.period,
            "losing_team_id": A,
            "gaining_team_id": int(r.gaining_team_id),
            "frame_loss": int(r.frame_loss),
            "a_possession_s": r.a_possession_s,
            "a_end_type": r.a_end_type,
            "end_reason": reason,
            "window_s": end / fps,
            "code": code,
            "reorganised": code == REORGANISED,
            "time_s": t_fr / fps,
            "T_r": t_fr / fps if code == REORGANISED else np.nan,
            "tau": tau[A],
            "D0": Dw[0],
            "frac_D_missing": float(np.isnan(Dw[: end + 1]).mean()),
        }
        nwin = min(end, int(params.debt_window_s * fps))
        seg = Dw[: nwin + 1]
        row["debt"] = float(np.nansum(seg)) / fps
        row["debt_above_tau"] = float(np.nansum(np.clip(seg - tau[A], 0, None))) / fps
        row["debt_window_s"] = nwin / fps
        for h in params.horizons_s:
            k = int(round(h * fps))
            row[f"D_{h:g}s"] = Dw[k] if k <= end else np.nan
            zk = z[i0 + k] if k <= end else np.full(len(ref.components), np.nan)
            share = w * zk**2 / np.sum(w * zk**2) if np.all(np.isfinite(zk)) else zk * np.nan
            for c, zz, sh_ in zip(ref.components, zk, share, strict=True):
                row[f"z_{c}_{h:g}s"] = zz
                row[f"share_{c}_{h:g}s"] = sh_
        n10 = min(end, int(10 * fps))
        row["reliability_10s"] = float(np.mean(detw[: n10 + 1]))
        row["reliability_window"] = float(np.mean(detw[: end + 1]))
        # Covariates at loss (defending team's frame).
        row["ball_x"] = td.tf.ball_x[i0]
        row["ball_y"] = td.tf.ball_y[i0]
        row["cell"] = int(td.cell[i0])
        row["goal_side_at_loss"] = td.S["goal_side"].iloc[i0]
        row["ball_speed_3s"] = _ball_speed(td, i0, params)
        rows.append(row)
        if keep_traces:
            eid = f"{md.meta.match_id}_{int(r.frame_loss)}"
            traces[eid] = EpisodeTrace(eid, Dw, z[i0 : i0 + H + 1], detw, end, tau[A])
    return pd.DataFrame(rows), traces


def _ball_speed(td: TeamData, i0: int, p: ReorgParams) -> float:
    """Mean ball speed (m/s) over the first ``ball_speed_window_s`` after the loss."""
    n = int(round(p.ball_speed_window_s * p.fps))
    bx = pd.Series(td.tf.ball_x[i0 : i0 + n + 1]).interpolate(limit_direction="both").to_numpy()
    by = pd.Series(td.tf.ball_y[i0 : i0 + n + 1]).interpolate(limit_direction="both").to_numpy()
    if len(bx) < p.savgol_window or not np.isfinite(bx).all():
        return np.nan
    vx = savgol_filter(bx, p.savgol_window, p.savgol_order, deriv=1) * p.fps
    vy = savgol_filter(by, p.savgol_window, p.savgol_order, deriv=1) * p.fps
    return float(np.mean(np.hypot(vx, vy)))


def team_taus(match_data: list, ref: Reference, params: ReorgParams) -> dict[int, float]:
    """tau per team: ``tau_quantile`` of D over all of the team's organised frames."""
    w = np.array([params.weights[k] for k in ref.components])
    vals: dict[int, list] = {}
    for md in match_data:
        for tid, td in md.teams.items():
            D = shape_distance(z_scores(td, ref), w)
            vals.setdefault(tid, []).append(D[td.organised & np.isfinite(D)])
    return {t: float(np.quantile(np.concatenate(v), params.tau_quantile)) for t, v in vals.items()}
