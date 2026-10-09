"""Possession-loss detection.

Two detectors. ``find_losses`` (proposed primary, pending GATE 0) uses dynamic events; see
its docstring. ``find_losses_tracking`` uses the tracking ``possession.group`` field and is
kept as a cross-check only: the audit found that field lags and misses short spells.

Tracking-field rule: a loss for team A happens when the tracking
``possession.group`` passes from A to B, B keeps it for at least ``min_sustain_s``, the
null-possession gap between A's last frame and B's first frame is at most ``max_null_gap_s``,
and the ball stays in play (inside a phase of play) throughout the transfer.

Everything here works on plain arrays/DataFrames so it can be tested on synthetic sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LossParams:
    fps: float = 10.0
    min_sustain_s: float = 1.0
    max_null_gap_s: float = 1.0
    require_ball_in_play: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> LossParams:
        lc = cfg["losses"]
        return cls(
            fps=float(cfg["data"]["fps"]),
            min_sustain_s=float(lc["min_sustain_s"]),
            max_null_gap_s=float(lc["max_null_gap_s"]),
            require_ball_in_play=bool(lc["require_ball_in_play"]),
        )


def in_play_mask(frames: pd.Series, phases: pd.DataFrame) -> np.ndarray:
    """Boolean mask: frame lies inside some phase of play (phases exist only with ball in play)."""
    f = frames.to_numpy()
    mask = np.zeros(len(f), dtype=bool)
    starts = phases["frame_start"].to_numpy()
    ends = phases["frame_end"].to_numpy()
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    idx = np.searchsorted(starts, f, side="right") - 1
    ok = idx >= 0
    mask[ok] = f[ok] <= ends[idx[ok]]
    return mask


def in_play_intervals(phases: pd.DataFrame, max_gap_frames: int = 1) -> tuple[np.ndarray, ...]:
    """Merge phases of play into contiguous ball-in-play intervals (inclusive frame bounds).

    Consecutive phases share a boundary frame at a change of possession, so phases whose gap is
    at most ``max_gap_frames`` belong to the same in-play stretch.
    """
    ph = phases.sort_values("frame_start")
    s0, e0 = ph["frame_start"].to_numpy(), ph["frame_end"].to_numpy()
    starts: list[int] = []
    ends: list[int] = []
    for a, b in zip(s0, e0, strict=True):
        if starts and a - ends[-1] <= max_gap_frames:
            ends[-1] = max(ends[-1], b)
        else:
            starts.append(a)
            ends.append(b)
    return np.asarray(starts), np.asarray(ends)


def possession_spells(labels: np.ndarray) -> pd.DataFrame:
    """Run-length encode a per-frame label array (``None``/NaN = no possession).

    Returns a DataFrame with ``label, i0, i1`` (inclusive positional indices) for every run,
    including null runs (label ``None``).
    """
    lab = pd.Series(labels, dtype="object").where(pd.notna(labels), None).to_numpy()
    n = len(lab)
    if n == 0:
        return pd.DataFrame(columns=["label", "i0", "i1"])
    change = np.ones(n, dtype=bool)
    change[1:] = lab[1:] != lab[:-1]
    i0 = np.flatnonzero(change)
    i1 = np.append(i0[1:] - 1, n - 1)
    return pd.DataFrame({"label": lab[i0], "i0": i0, "i1": i1})


def _merge_same_team(spells: list[list], gap_max_f: int) -> list[list]:
    out: list[list] = []
    for lab, i0, i1 in spells:
        if out and out[-1][0] == lab and (i0 - out[-1][2] - 1) <= gap_max_f:
            out[-1][2] = i1
        else:
            out.append([lab, i0, i1])
    return out


def detect_losses_from_labels(
    labels: np.ndarray,
    in_play: np.ndarray,
    params: LossParams,
    max_null_gap_s: float | None = None,
) -> pd.DataFrame:
    """Detect possession losses in one continuous sequence (a single period).

    Parameters
    ----------
    labels : per-frame team label in possession, or None.
    in_play : per-frame boolean ball-in-play mask.
    max_null_gap_s : override for ``params.max_null_gap_s`` (for sensitivity counts).

    Returns
    -------
    DataFrame with one row per loss: ``losing, gaining, i_last_a`` (last frame A held it),
    ``i_loss`` (first frame B holds it), ``gap_frames``, ``b_sustain_frames``.
    """
    gap_max = params.max_null_gap_s if max_null_gap_s is None else max_null_gap_s
    gap_max_f = int(round(gap_max * params.fps))
    sustain_f = int(round(params.min_sustain_s * params.fps))
    spells = possession_spells(labels)
    team = spells[spells["label"].notna()].reset_index(drop=True)

    # 1) Drop opponent "blips" shorter than the sustain threshold that sit between two spells
    #    of the same team (deflections, flicks): A -> B(blip) -> A is not a loss.
    # 2) Merge consecutive same-team spells separated by at most ``gap_max_f`` null frames.
    spl = [list(r) for r in team[["label", "i0", "i1"]].itertuples(index=False)]
    spl = _merge_same_team(spl, gap_max_f)
    changed = True
    while changed:
        changed = False
        for k in range(1, len(spl) - 1):
            short = (spl[k][2] - spl[k][1] + 1) < sustain_f
            if short and spl[k - 1][0] == spl[k + 1][0] != spl[k][0]:
                spl[k - 1][2] = spl[k + 1][2]
                del spl[k : k + 2]
                changed = True
                break
    cleaned = spl

    rows = []
    for a, b in zip(cleaned[:-1], cleaned[1:], strict=False):
        if a[0] == b[0]:
            continue
        gap = b[1] - a[2] - 1
        sustain = b[2] - b[1] + 1
        if gap > gap_max_f or sustain < sustain_f:
            continue
        if params.require_ball_in_play and not in_play[a[2] : b[1] + sustain_f].all():
            continue
        rows.append(
            {
                "losing": a[0],
                "gaining": b[0],
                "i_last_a": a[2],
                "i_loss": b[1],
                "gap_frames": gap,
                "b_sustain_frames": sustain,
            }
        )
    return pd.DataFrame(
        rows, columns=["losing", "gaining", "i_last_a", "i_loss", "gap_frames", "b_sustain_frames"]
    )


def find_losses_tracking(
    frames: pd.DataFrame,
    phases: pd.DataFrame,
    params: LossParams | None = None,
    max_null_gap_s: float | None = None,
) -> pd.DataFrame:
    """Detect possession losses for a whole match from parsed tracking frames.

    ``frames`` is the frame table from :func:`reorg.io.parse_tracking`. Returns one row per loss
    with ``period, losing, gaining, frame_last_a, frame_loss, gap_frames, b_sustain_frames``.
    """
    params = params or LossParams()
    fr = frames[frames["period"].notna()].sort_values("frame").reset_index(drop=True)
    play = in_play_mask(fr["frame"], phases)
    out = []
    for period, idx in fr.groupby("period").indices.items():
        sub = fr.iloc[idx]
        res = detect_losses_from_labels(
            sub["poss_group"].to_numpy(dtype=object), play[idx], params, max_null_gap_s
        )
        if res.empty:
            continue
        fnums = sub["frame"].to_numpy()
        res["frame_last_a"] = fnums[res["i_last_a"].to_numpy()]
        res["frame_loss"] = fnums[res["i_loss"].to_numpy()]
        res["period"] = int(period)
        out.append(res.drop(columns=["i_last_a", "i_loss"]))
    cols = [
        "period",
        "losing",
        "gaining",
        "frame_last_a",
        "frame_loss",
        "gap_frames",
        "b_sustain_frames",
    ]
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)[cols]


def losses_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """Cross-check losses from dynamic events: consecutive ``player_possession`` events by
    different teams with no game interruption between them (open-play team change).

    Returns ``period, losing_team_id, gaining_team_id, frame_loss`` (start of the gaining
    team's first possession) and ``end_type`` of the losing possession.
    """
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    rows = []
    for _, grp in pp.groupby("period"):
        a, b = grp.iloc[:-1], grp.iloc[1:]
        change = a["team_id"].to_numpy() != b["team_id"].to_numpy()
        no_stop = a["game_interruption_after"].isna().to_numpy() & (
            b["game_interruption_before"].isna().to_numpy()
        )
        keep = change & no_stop
        rows.append(
            pd.DataFrame(
                {
                    "period": b["period"].to_numpy()[keep],
                    "losing_team_id": a["team_id"].to_numpy()[keep],
                    "gaining_team_id": b["team_id"].to_numpy()[keep],
                    "frame_loss": b["frame_start"].to_numpy()[keep],
                    "end_type": a["end_type"].to_numpy()[keep],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def losses_from_phases(phases: pd.DataFrame, max_gap_frames: int = 1) -> pd.DataFrame:
    """Cross-check losses from phases of play: consecutive phases (contiguous in play) whose
    team in possession changes."""
    ph = phases.sort_values("frame_start")
    a, b = ph.iloc[:-1], ph.iloc[1:]
    contiguous = (b["frame_start"].to_numpy() - a["frame_end"].to_numpy()) <= max_gap_frames
    same_period = a["period"].to_numpy() == b["period"].to_numpy()
    change = a["team_in_possession_id"].to_numpy() != b["team_in_possession_id"].to_numpy()
    keep = contiguous & same_period & change
    return pd.DataFrame(
        {
            "period": b["period"].to_numpy()[keep],
            "losing_team_id": a["team_in_possession_id"].to_numpy()[keep],
            "gaining_team_id": b["team_in_possession_id"].to_numpy()[keep],
            "frame_loss": b["frame_start"].to_numpy()[keep],
            "losing_phase": a["team_in_possession_phase_type"].to_numpy()[keep],
        }
    )


def match_losses(
    a: pd.DataFrame, b: pd.DataFrame, tol_frames: int, team_col_a: str, team_col_b: str
) -> np.ndarray:
    """For each loss in ``a``, whether ``b`` has a loss by the same losing team within
    ``tol_frames`` frames."""
    out = np.zeros(len(a), dtype=bool)
    for i, (team, fr) in enumerate(zip(a[team_col_a], a["frame_loss"], strict=True)):
        cand = b[b[team_col_b] == team]["frame_loss"].to_numpy()
        out[i] = cand.size > 0 and np.abs(cand - fr).min() <= tol_frames
    return out


def team_possession_runs(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``player_possession`` events into team-possession runs.

    A run is a maximal sequence of consecutive player possessions by one team, within a period,
    with no game interruption between them. Returns one row per run with ``period, team_id,
    frame_start`` (first control), ``frame_end`` (last control), ``n_possessions``,
    ``interrupted_after`` (run ended with a stoppage) and ``end_type`` of its last possession.
    """
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    runs: list[dict] = []
    for period, grp in pp.groupby("period"):
        cur: dict | None = None
        for r in grp.itertuples(index=False):
            new_run = (
                cur is None
                or r.team_id != cur["team_id"]
                or cur["interrupted_after"]
                or pd.notna(r.game_interruption_before)
            )
            if new_run:
                if cur is not None:
                    runs.append(cur)
                cur = {
                    "period": int(period),
                    "team_id": int(r.team_id),
                    "frame_start": int(r.frame_start),
                    "frame_end": int(r.frame_end),
                    "n_possessions": 0,
                    "interrupted_before": pd.notna(r.game_interruption_before),
                }
            cur["frame_end"] = int(r.frame_end)
            cur["n_possessions"] += 1
            cur["interrupted_after"] = pd.notna(r.game_interruption_after)
            cur["end_type"] = r.end_type
        if cur is not None:
            runs.append(cur)
    return pd.DataFrame(runs)


def find_losses(
    events: pd.DataFrame, phases: pd.DataFrame, params: LossParams | None = None
) -> pd.DataFrame:
    """Open-play possession losses from dynamic events (primary candidate after the audit).

    A loss for team A at frame ``frame_loss`` = first control by team B, where B's run directly
    follows A's run with no stoppage in between, B keeps the ball for at least
    ``min_sustain_s`` (until B's next change of team, or the end of B's run if it ends in a
    stoppage), and the ball is in play (inside a phase of play) from A's last control through
    the sustain window.

    Returns ``period, losing_team_id, gaining_team_id, frame_last_a, frame_loss,
    a_possession_s, b_hold_s, a_end_type``.
    """
    params = params or LossParams()
    runs = team_possession_runs(events)
    sustain_f = int(round(params.min_sustain_s * params.fps))
    cols = [
        "period",
        "losing_team_id",
        "gaining_team_id",
        "frame_last_a",
        "frame_loss",
        "a_possession_s",
        "b_hold_s",
        "a_end_type",
    ]
    if runs.empty:
        return pd.DataFrame(columns=cols)
    starts, ends = in_play_intervals(phases)

    def in_play(f0: int, f1: int) -> bool:
        k = np.searchsorted(starts, f0, side="right") - 1
        return k >= 0 and ends[k] >= f1

    rows = []
    for period, grp in runs.groupby("period"):
        g = grp.reset_index(drop=True)
        for k in range(len(g) - 1):
            a, b = g.iloc[k], g.iloc[k + 1]
            if a["team_id"] == b["team_id"] or a["interrupted_after"] or b["interrupted_before"]:
                continue
            nxt = g.iloc[k + 2] if k + 2 < len(g) else None
            if nxt is not None and not b["interrupted_after"]:
                b_hold_end = int(nxt["frame_start"])
            else:
                b_hold_end = int(b["frame_end"])
            hold = b_hold_end - int(b["frame_start"])
            if hold < sustain_f:
                continue
            if params.require_ball_in_play and not in_play(
                int(a["frame_end"]), int(b["frame_start"]) + sustain_f
            ):
                continue
            rows.append(
                {
                    "period": int(period),
                    "losing_team_id": int(a["team_id"]),
                    "gaining_team_id": int(b["team_id"]),
                    "frame_last_a": int(a["frame_end"]),
                    "frame_loss": int(b["frame_start"]),
                    "a_possession_s": (int(a["frame_end"]) - int(a["frame_start"])) / params.fps,
                    "b_hold_s": hold / params.fps,
                    "a_end_type": a["end_type"],
                }
            )
    return pd.DataFrame(rows, columns=cols)
