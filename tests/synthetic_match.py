"""Write a small synthetic match in the SkillCorner open-data layout (no real data).

Possession alternates every ``spell_s`` seconds. After each loss, the losing team starts
pushed high and stretched and moves linearly into its block over ``converge_s`` seconds, so the
true time to reorganise is known. The ball stays near the centre spot, so the only thing that
changes is the defending shape. Used by the end-to-end test and the CI smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 10
L, W = 105.0, 68.0
HOME, AWAY = 100, 200
BLOCK_X = np.array([-35, -35, -35, -35, -22, -22, -22, -22, -10, -10], float)
DRIFT_M = 0.6
DRIFT_PERIOD_S = 15.0
BLOCK_Y = np.array([-15, -5, 5, 15, -15, -5, 5, 15, -5, 5], float)


def _players(team_id: int, base: int) -> list[dict]:
    out = []
    for k in range(11):
        acr = "GK" if k == 0 else "CM"
        out.append(
            {
                "id": base + k,
                "team_id": team_id,
                "number": k + 1,
                "short_name": f"P{base + k}",
                "player_role": {
                    "acronym": acr,
                    "position_group": "Other" if k == 0 else "Midfield",
                },
                "playing_time": {
                    "total": {"start_frame": 0, "end_frame": 10**6, "minutes_played": 90}
                },
            }
        )
    return out


def write_match(
    root: Path,
    match_id: int = 1,
    minutes: float = 8.0,
    spell_s: float = 30.0,
    converge_s: float = 5.0,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, 10)
    d = root / "matches" / str(match_id)
    d.mkdir(parents=True, exist_ok=True)
    n = int(minutes * 60 * FPS)
    half = n // 2
    periods = [
        {"period": 1, "name": "period_1", "start_frame": 0, "end_frame": half - 1},
        {"period": 2, "name": "period_2", "start_frame": half, "end_frame": n - 1},
    ]
    meta = {
        "id": match_id,
        "home_team": {"id": HOME, "short_name": "Home FC"},
        "away_team": {"id": AWAY, "short_name": "Away FC"},
        "pitch_length": L,
        "pitch_width": W,
        "home_team_side": ["left_to_right", "right_to_left"],
        "match_periods": periods,
        "date_time": "2026-01-01T00:00:00Z",
        "players": _players(HOME, 1000) + _players(AWAY, 2000),
    }
    (d / f"{match_id}_match.json").write_text(json.dumps(meta))

    spell = int(spell_s * FPS)
    conv = int(converge_s * FPS)
    poss, phases, events = [], [], []
    for f0 in range(0, n, spell):
        team = HOME if (f0 // spell) % 2 == 0 else AWAY
        f1 = min(f0 + spell - 1, n - 1)
        per = 1 if f0 < half else 2
        if per == 1 and f1 >= half:
            f1 = half - 1
        phases.append(
            {
                "frame_start": f0,
                "frame_end": f1 + 1 if f1 + 1 < n else f1,
                "period": per,
                "team_in_possession_id": team,
                "team_in_possession_phase_type": "create",
                "team_out_of_possession_phase_type": "medium_block",
            }
        )
        for a in range(f0, f1, 20):
            events.append(
                {
                    "event_type": "player_possession",
                    "period": per,
                    "frame_start": a,
                    "frame_end": min(a + 15, f1),
                    "team_id": team,
                    "end_type": "pass",
                    "game_interruption_before": None,
                    "game_interruption_after": None,
                }
            )
        poss.append((f0, f1, team))

    with open(d / f"{match_id}_tracking_extrapolated.jsonl", "w") as fh:
        for fr in range(n):
            per = 1 if fr < half else 2
            spell_i = fr // spell
            in_poss = HOME if spell_i % 2 == 0 else AWAY
            k = fr - spell_i * spell  # frames since the start of this spell (= since the loss)
            a = min(k / conv, 1.0)
            # Ball nearly still near the centre spot, so only the defending shape changes and the
            # true T_r is the convergence time (the reference itself is tested elsewhere).
            bx = 0.3 * np.sin(fr / 50)
            by = 0.3 * np.cos(fr / 70)
            pdata = []
            for team, base in ((HOME, 1000), (AWAY, 2000)):
                sign = 1 if (team == HOME) == (per == 1) else -1  # own frame -> pitch frame
                if team == in_poss:
                    x, y = BLOCK_X + 30, BLOCK_Y * 1.5
                else:
                    x = (1 - a) * (BLOCK_X * 0.4 + 25) + a * BLOCK_X
                    y = (1 - a) * BLOCK_Y * 1.8 + a * BLOCK_Y
                    # Smooth, slow per-player drift (real positions are autocorrelated; iid
                    # per-frame jitter would make any 2 s hold below a quantile threshold rare).
                    x = x + DRIFT_M * np.sin(2 * np.pi * fr / (DRIFT_PERIOD_S * FPS) + phase)
                    y = y + DRIFT_M * np.cos(2 * np.pi * fr / (DRIFT_PERIOD_S * FPS) + phase)
                gk = {"x": -50.0 * sign, "y": 0.0, "player_id": base, "is_detected": False}
                pdata.append(gk)
                for j in range(10):
                    pdata.append(
                        {
                            "x": float(x[j] * sign),
                            "y": float(y[j] * sign),
                            "player_id": base + 1 + j,
                            "is_detected": bool(j % 3),
                        }
                    )
            s = fr / FPS + (0 if per == 1 else 45 * 60 - half / FPS)
            rec = {
                "frame": fr,
                "timestamp": f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}",
                "period": per,
                "ball_data": {"x": float(bx), "y": float(by), "z": 0.0, "is_detected": True},
                "possession": {
                    "player_id": None,
                    "group": "home team" if in_poss == HOME else "away team",
                },
                "image_corners_projection": {},
                "player_data": pdata,
            }
            fh.write(json.dumps(rec) + "\n")
    pd.DataFrame(events).to_csv(d / f"{match_id}_dynamic_events.csv", index=False)
    pd.DataFrame(phases).to_csv(d / f"{match_id}_phases_of_play.csv", index=False)
