"""Non-parametric survival estimators for reorganisation episodes.

Codes follow :mod:`reorg.reorganisation`: 1 = reorganised, 2 = regain (competing event),
0 = censored (dead ball, shot, period end, cap). At tied times, events precede censoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _step_eval(times: np.ndarray, values: np.ndarray, grid: np.ndarray, start: float) -> np.ndarray:
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], start)


def cumulative_incidence(
    time: np.ndarray, code: np.ndarray, grid: np.ndarray, causes: tuple[int, ...] = (1, 2)
) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence per cause, evaluated on ``grid``.

    Also returns ``surv``: overall event-free probability (no cause has occurred).
    """
    time = np.asarray(time, float)
    code = np.asarray(code, int)
    ut = np.unique(time[code > 0])
    S = 1.0
    cif = {c: 0.0 for c in causes}
    vals = {c: [] for c in causes}
    svals = []
    for t in ut:
        n = np.sum(time >= t)
        at = time == t
        d_all = np.sum(at & (code > 0))
        for c in causes:
            cif[c] += S * np.sum(at & (code == c)) / n
            vals[c].append(cif[c])
        S *= 1 - d_all / n
        svals.append(S)
    out = {"t": grid}
    for c in causes:
        out[f"cif_{c}"] = _step_eval(ut, np.asarray(vals[c]), grid, 0.0)
    out["surv"] = _step_eval(ut, np.asarray(svals), grid, 1.0)
    return pd.DataFrame(out)


def kaplan_meier(time: np.ndarray, event: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival S(t) on ``grid`` for a binary event (others censored)."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    ut = np.unique(time[event])
    S, vals = 1.0, []
    for t in ut:
        n = np.sum(time >= t)
        S *= 1 - np.sum((time == t) & event) / n
        vals.append(S)
    return _step_eval(ut, np.asarray(vals), grid, 1.0)


def bootstrap_cif(
    df: pd.DataFrame,
    grid: np.ndarray,
    n_boot: int,
    seed: int,
    cluster: str = "match_id",
    causes: tuple[int, ...] = (1, 2),
    ci: float = 0.95,
) -> pd.DataFrame:
    """Cluster (match-level) bootstrap percentile intervals for the cumulative incidences."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    est = cumulative_incidence(df["time_s"], df["code"], grid, causes)
    boots = {c: [] for c in causes}
    for _ in range(n_boot):
        samp = pd.concat([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        b = cumulative_incidence(samp["time_s"], samp["code"], grid, causes)
        for c in causes:
            boots[c].append(b[f"cif_{c}"].to_numpy())
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    for c in causes:
        arr = np.vstack(boots[c])
        est[f"cif_{c}_lo"] = np.percentile(arr, lo, axis=0)
        est[f"cif_{c}_hi"] = np.percentile(arr, hi, axis=0)
    return est
