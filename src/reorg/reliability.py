"""Tracking-reliability layer.

SkillCorner broadcast tracking extrapolates off-camera players (``is_detected = False``).
Central defenders are the least-detected outfield role (see docs/DATA_AUDIT.md), so shape
features of the back line are often extrapolated. Every result is therefore recomputed under
three regimes:

* ``all``:       every tracked frame counts;
* ``detected``:  D(t) is undefined in frames where fewer than ``frame_min_detected`` of the 10
                 defenders are detected (such frames break a reorganisation hold);
* ``reliable``:  only episodes whose mean detected fraction over the first 10 s is at least
                 ``episode_min_reliability``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reorg.survival import cumulative_incidence

REGIMES = ("all", "detected", "reliable")


def regime_episodes(eps_all: pd.DataFrame, eps_detected: pd.DataFrame, cfg: dict) -> dict:
    """Map regime name -> episode table (``reliable`` filters the ``all`` table)."""
    thr = float(cfg["reliability"]["episode_min_reliability"])
    return {
        "all": eps_all,
        "detected": eps_detected,
        "reliable": eps_all[eps_all["reliability_10s"] >= thr],
    }


def reliability_report(regimes: dict, horizons_s=(3.0, 6.0, 10.0)) -> pd.DataFrame:
    """Per regime: episode count, share reorganised/regained, CIF of reorganisation at horizons."""
    rows = []
    grid = np.asarray(horizons_s, float)
    for name, df in regimes.items():
        cif = cumulative_incidence(df["time_s"], df["code"], grid)
        row = {
            "regime": name,
            "episodes": len(df),
            "reorganised": float((df["code"] == 1).mean()),
            "regain_first": float((df["code"] == 2).mean()),
            "median_T_r_observed": float(df["T_r"].median()),
        }
        for h, v in zip(horizons_s, cif["cif_1"], strict=True):
            row[f"cif_reorg_{h:g}s"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)
