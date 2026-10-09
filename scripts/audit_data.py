"""Data audit for the SkillCorner open data (Phase 0).

Writes machine-readable tables to ``outputs/audit/`` and a generated Markdown section to
``docs/audit_generated.md`` which ``docs/DATA_AUDIT.md`` summarises.

Usage: python scripts/audit_data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reorg.io import (
    list_match_ids,
    load_dynamic_events,
    load_match_meta,
    load_phases,
    load_tracking,
    match_dir,
)
from reorg.turnovers import (
    LossParams,
    find_losses,
    find_losses_tracking,
    in_play_mask,
    losses_from_events,
    losses_from_phases,
    match_losses,
    team_possession_runs,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "audit"
# Audit-only descriptive constants (not used by the metric pipeline).
MATCH_TOL_S = 2.0  # tolerance when matching losses across sources
POST_LOSS_S = 10.0  # window used to describe post-loss tracking coverage/detection
WINDOW_HORIZONS_S = (5.0, 10.0, 15.0, 20.0)  # uncensored-window horizons to report
OFF_PITCH_MARGIN_M = 5.0  # positions further than this outside the pitch are flagged
TRACK_LAG_MAX_S = 30.0  # search window for the tracking possession field to switch team


def audit_match(args: tuple[str, str, int, dict]) -> dict:
    root, cache, mid, cfg = args
    params = LossParams.from_config(cfg)
    fps = params.fps
    meta = load_match_meta(root, mid)
    fr, pl = load_tracking(root, mid, cache=cache)
    ph = load_phases(root, mid)
    ev = load_dynamic_events(root, mid)
    gid = {meta.home_team_id: "home team", meta.away_team_id: "away team"}
    tid = {v: k for k, v in gid.items()}

    f = fr[fr["period"].notna()].sort_values("frame").reset_index(drop=True)
    f["play"] = in_play_mask(f["frame"], ph)
    f["tracked"] = f["n_players"] > 0
    fp = f[f["play"]]

    # Possession vs phase team agreement.
    phs = ph.sort_values("frame_start")
    idx = np.searchsorted(phs["frame_start"].to_numpy(), fp["frame"].to_numpy(), "right") - 1
    ph_team = phs["team_in_possession_id"].map(gid).to_numpy()[idx]
    has_poss = fp["poss_group"].notna().to_numpy()
    agree = float(np.mean(fp["poss_group"].to_numpy()[has_poss] == ph_team[has_poss]))

    # Player-level joins.
    players = meta.players
    pl = pl.merge(
        players[["player_id", "team_id", "is_gk", "position_group"]], on="player_id", how="left"
    )
    missing_lineup = int(pl["team_id"].isna().sum())
    pl = pl.merge(f[["frame", "period", "play"]], on="frame", how="inner")
    plp = pl[pl["play"]]
    det_team = plp.groupby("team_id")["is_detected"].mean()
    det_period = plp.groupby("period")["is_detected"].mean()
    det_pos = plp.groupby("position_group")["is_detected"].mean()
    det_gk = plp.groupby("is_gk")["is_detected"].mean()

    # Anomalies.
    dup = int(pl.duplicated(["frame", "player_id"]).sum())
    per_team = pl.groupby(["frame", "team_id"]).size()
    team_frames_not_11 = int((per_team != 11).sum())
    gk_per_team = pl[pl["is_gk"]].groupby(["frame", "team_id"]).size()
    team_frames = pl.groupby(["frame", "team_id"]).ngroups
    team_frames_no_gk = int(team_frames - len(gk_per_team))
    off = (pl["x"].abs() > meta.pitch_length / 2 + OFF_PITCH_MARGIN_M) | (
        pl["y"].abs() > meta.pitch_width / 2 + OFF_PITCH_MARGIN_M
    )

    # Attacking direction: home GK mean x per period should be negative when home attacks L->R.
    gkx = pl[pl["is_gk"]].groupby(["team_id", "period"])["x"].mean()
    dir_ok = True
    for i, side in enumerate(meta.home_team_side, start=1):
        if (meta.home_team_id, i) in gkx.index:
            sign = -1 if side == "left_to_right" else 1
            dir_ok &= bool(np.sign(gkx[(meta.home_team_id, i)]) == sign)

    # Event coordinate frame check: events mirrored by attacking_side.
    e = ev[ev["event_type"] == "player_possession"][
        ["frame_start", "player_id", "x_start", "y_start", "attacking_side"]
    ].dropna(subset=["x_start"])
    j = e.merge(
        pl[["frame", "player_id", "x", "y"]],
        left_on=["frame_start", "player_id"],
        right_on=["frame", "player_id"],
    )
    s = np.where(j["attacking_side"] == "right_to_left", -1.0, 1.0)
    ev_dx_med = float(np.median(np.abs(j["x_start"] * s - j["x"]))) if len(j) else np.nan
    ev_dx_raw = float(np.median(np.abs(j["x_start"] - j["x"]))) if len(j) else np.nan

    # Losses: primary (dynamic events) + sensitivity + cross-checks (phases, tracking field).
    lp = find_losses(ev, ph, params)
    sens = {}
    for su in (0.5, 2.0):
        p2 = LossParams(
            fps=fps,
            min_sustain_s=su,
            max_null_gap_s=params.max_null_gap_s,
            require_ball_in_play=params.require_ball_in_play,
        )
        sens[f"sustain{su}"] = len(find_losses(ev, ph, p2))
    le_raw = losses_from_events(ev)
    lph = losses_from_phases(ph)
    lt = find_losses_tracking(fr, ph, params)
    lt["losing_team_id"] = lt["losing"].map(tid)
    tol = int(MATCH_TOL_S * fps)
    p_in_ph = match_losses(lp, lph, tol, "losing_team_id", "losing_team_id")
    ph_in_p = match_losses(lph, lp, tol, "losing_team_id", "losing_team_id")
    p_in_t = match_losses(lp, lt, tol, "losing_team_id", "losing_team_id")
    t_in_p = match_losses(lt, lp, tol, "losing_team_id", "losing_team_id")
    # Lag of the tracking possession field behind the event-defined loss.
    track_team = fr.set_index("frame")["poss_group"].map(tid)
    lags = []
    for r in lp.itertuples(index=False):
        seg = track_team.loc[r.frame_last_a : r.frame_last_a + int(TRACK_LAG_MAX_S * fps)]
        hit = np.flatnonzero(seg.to_numpy() == r.gaining_team_id)
        lags.append((seg.index[hit[0]] - r.frame_loss) / fps if len(hit) else np.nan)
    lags = np.asarray(lags, dtype=float)

    # Post-loss coverage / censoring for primary losses.
    runs = team_possession_runs(ev)
    frame_to_pos = pd.Series(np.arange(len(f)), index=f["frame"])
    play = f["play"].to_numpy()
    tracked = f["tracked"].to_numpy()
    per = f["period"].to_numpy()
    horizon_f = int(max(WINDOW_HORIZONS_S) * fps)
    post_f = int(POST_LOSS_S * fps)
    ep = []
    pl_idx = pl.set_index("frame")
    for r in lp.itertuples(index=False):
        i0 = int(frame_to_pos[r.frame_loss])
        # Regain = next control by the losing team (any length).
        nxt = runs[
            (runs["period"] == r.period)
            & (runs["frame_start"] > r.frame_loss)
            & (runs["team_id"] == r.losing_team_id)
        ]["frame_start"]
        regain_i = int(frame_to_pos.get(nxt.iloc[0], len(f))) if len(nxt) else len(f)
        stop = i0
        while (
            stop < len(f)
            and stop - i0 < horizon_f
            and play[stop]
            and per[stop] == r.period
            and stop < regain_i
        ):
            stop += 1
        window_s = (stop - i0) / fps
        reason = (
            "cap"
            if stop - i0 >= horizon_f
            else "regain"
            if stop >= regain_i
            else "dead_ball_or_period_end"
        )
        cov = float(tracked[i0 : i0 + post_f].mean())
        frames_win = f["frame"].to_numpy()[i0 : min(i0 + post_f, len(f))]
        sub = pl_idx.loc[pl_idx.index.intersection(frames_win)]
        sub = sub[(sub["team_id"] == r.losing_team_id) & (~sub["is_gk"].astype(bool))]
        det = float(sub["is_detected"].mean()) if len(sub) else np.nan
        ep.append(
            {
                "match_id": mid,
                "period": r.period,
                "losing_team_id": r.losing_team_id,
                "frame_loss": r.frame_loss,
                "a_possession_s": r.a_possession_s,
                "a_end_type": r.a_end_type,
                "window_s": window_s,
                "censor": reason,
                "tracked_frac_10s": cov,
                "def_detected_frac_10s": det,
            }
        )
    eps = pd.DataFrame(ep)

    # Organised-defence frames per team (phase label only; the >=15 s rule is applied below).
    org = set(cfg["reference"]["organised_phases"])
    settle_f = int(cfg["reference"]["settled_after_loss_s"] * fps)
    org_rows = {}
    for team in (meta.home_team_id, meta.away_team_id):
        oop = phs[
            (phs["team_in_possession_id"] != team)
            & phs["team_out_of_possession_phase_type"].isin(org)
        ]
        mask = np.zeros(len(f), dtype=bool)
        fnum = f["frame"].to_numpy()
        for a0, a1 in zip(oop["frame_start"], oop["frame_end"], strict=True):
            mask |= (fnum >= a0) & (fnum <= a1)
        mask &= tracked
        n_org = int(mask.sum())
        recent = np.zeros(len(f), dtype=bool)
        for fl in lp[lp["losing_team_id"] == team]["frame_loss"]:
            i = int(frame_to_pos[fl])
            recent[i : i + settle_f] = True
        org_rows[team] = (n_org, int((mask & ~recent).sum()))

    out = {
        "inventory": {
            "match_id": mid,
            "date": json.loads((match_dir(root, mid) / f"{mid}_match.json").read_text())[
                "date_time"
            ][:10],
            "home": meta.home_team_name,
            "away": meta.away_team_name,
            "pitch": f"{meta.pitch_length:g}x{meta.pitch_width:g}",
            "home_side_p1": meta.home_team_side[0],
            "events": len(ev),
            "phases": len(ph),
            "frames": len(fr),
            "frames_in_period": len(f),
            "in_play_frac": round(float(f["play"].mean()), 3),
            "tracked_frac_all": round(float(f["tracked"].mean()), 3),
            "tracked_frac_in_play": round(float(fp["tracked"].mean()), 3),
            "poss_group_set_in_play": round(float(has_poss.mean()), 3),
            "poss_player_set_in_play": round(float(fp["poss_player_id"].notna().mean()), 3),
            "poss_agrees_phase": round(agree, 3),
            "ball_xy_null_in_play": round(float(fp["ball_x"].isna().mean()), 3),
            "ball_detected_in_play": round(float(fp["ball_detected"].astype(float).mean()), 3),
        },
        "detection": {
            "match_id": mid,
            "all": round(float(plp["is_detected"].mean()), 3),
            "home": round(float(det_team.get(meta.home_team_id, np.nan)), 3),
            "away": round(float(det_team.get(meta.away_team_id, np.nan)), 3),
            "p1": round(float(det_period.get(1, np.nan)), 3),
            "p2": round(float(det_period.get(2, np.nan)), 3),
            "gk": round(float(det_gk.get(True, np.nan)), 3),
            **{f"pos:{k}": round(float(v), 3) for k, v in det_pos.items()},
        },
        "anomalies": {
            "match_id": mid,
            "players_not_in_lineup": missing_lineup,
            "dup_player_frame": dup,
            "team_frames_not_11": team_frames_not_11,
            "team_frames_no_gk": team_frames_no_gk,
            "off_pitch_player_frames": int(off.sum()),
            "direction_matches_home_team_side": dir_ok,
            "event_xy_median_err_mirrored_m": round(ev_dx_med, 2),
            "event_xy_median_err_raw_m": round(ev_dx_raw, 2),
            "frame_step_not_1": int((fr["frame"].diff().dropna() != 1).sum()),
        },
        "losses": {
            "match_id": mid,
            "events_raw_team_changes": len(le_raw),
            "primary_events": len(lp),
            **sens,
            "phases_based": len(lph),
            "tracking_field": len(lt),
            "primary_in_phases": round(float(p_in_ph.mean()), 3),
            "phases_in_primary": round(float(ph_in_p.mean()), 3),
            "primary_in_tracking": round(float(p_in_t.mean()), 3),
            "tracking_in_primary": round(float(t_in_p.mean()), 3),
            "track_lag_median_s": round(float(np.nanmedian(lags)), 2),
            "track_lag_p90_s": round(float(np.nanpercentile(lags, 90)), 2),
            "track_never_switch": round(float(np.isnan(lags).mean()), 3),
        },
        "episodes": eps,
        "organised": [
            {"match_id": mid, "team_id": t, "org_frames": v[0], "org_frames_settled": v[1]}
            for t, v in org_rows.items()
        ],
        "teams": [
            (meta.home_team_id, meta.home_team_name),
            (meta.away_team_id, meta.away_team_name),
        ],
        "vocab": {
            "event_type": ev["event_type"].value_counts().to_dict(),
            "event_subtype": ev["event_subtype"].value_counts().to_dict(),
            "end_type": ev["end_type"].value_counts().to_dict(),
            "start_type": ev["start_type"].value_counts().to_dict(),
            "game_interruption_before": ev["game_interruption_before"].value_counts().to_dict(),
            "game_interruption_after": ev["game_interruption_after"].value_counts().to_dict(),
            "in_possession_phase": ph["team_in_possession_phase_type"].value_counts().to_dict(),
            "out_of_possession_phase": (
                ph["team_out_of_possession_phase_type"].value_counts().to_dict()
            ),
            "poss_group": fr["poss_group"].value_counts(dropna=False).to_dict(),
            "primary_losing_end_type": lp["a_end_type"].value_counts(dropna=False).to_dict(),
        },
        "n_players_dist": fr["n_players"].value_counts().to_dict(),
    }
    return out


def _sum_dicts(ds: list[dict]) -> dict:
    tot: dict = {}
    for d in ds:
        for k, v in d.items():
            key = "null" if k is None or (isinstance(k, float) and np.isnan(k)) else k
            tot[key] = tot.get(key, 0) + v
    return dict(sorted(tot.items(), key=lambda kv: -kv[1]))


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    root = str(ROOT / cfg["data"]["root"])
    cache = str(ROOT / cfg["data"]["cache"])
    mids = list_match_ids(root)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(audit_match, [(root, cache, m, cfg) for m in mids]))

    OUT.mkdir(parents=True, exist_ok=True)
    inv = pd.DataFrame([r["inventory"] for r in res])
    det = pd.DataFrame([r["detection"] for r in res])
    ano = pd.DataFrame([r["anomalies"] for r in res])
    los = pd.DataFrame([r["losses"] for r in res])
    eps = pd.concat([r["episodes"] for r in res], ignore_index=True)
    org = pd.DataFrame([o for r in res for o in r["organised"]])
    teams = dict(t for r in res for t in r["teams"])
    org["team"] = org["team_id"].map(teams)
    eps["team"] = eps["losing_team_id"].map(teams)
    for name, df in (
        ("inventory", inv),
        ("detection", det),
        ("anomalies", ano),
        ("losses", los),
        ("episodes", eps),
        ("organised", org),
    ):
        df.to_csv(OUT / f"{name}.csv", index=False)

    pose_ids = set(json.loads((Path(root) / "bodypose" / "MANIFEST.json").read_text())["matches"])
    inv.insert(1, "pose", inv["match_id"].astype(str).isin(pose_ids))

    vocab = {k: _sum_dicts([r["vocab"][k] for r in res]) for k in res[0]["vocab"]}
    nplay = _sum_dicts([r["n_players_dist"] for r in res])
    (OUT / "vocab.json").write_text(json.dumps(vocab, indent=1, default=str))

    # Team-level summaries.
    team_matches = org.groupby("team")["match_id"].nunique()
    team_org = org.groupby("team")[["org_frames", "org_frames_settled"]].sum()
    team_org["org_minutes_settled"] = (
        team_org["org_frames_settled"] / cfg["data"]["fps"] / 60
    ).round(1)
    team_eps = eps.groupby("team").size().rename("primary_losses")
    team_tab = pd.concat(
        [team_matches.rename("matches"), team_eps, team_org["org_minutes_settled"]], axis=1
    ).reset_index()

    usable = eps[eps["tracked_frac_10s"] >= 0.9]
    hz = {f">={h:g}s": round(float((eps["window_s"] >= h).mean()), 3) for h in WINDOW_HORIZONS_S}
    det_q = eps["def_detected_frac_10s"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3)

    lines = ["<!-- generated by scripts/audit_data.py; do not edit by hand -->", ""]
    lines += ["### A. Inventory", "", md_table(inv), ""]
    lines += [
        "### B. Detection rate (`is_detected`) of player-frames, ball in play",
        "",
        md_table(det),
        "",
    ]
    lines += ["### C. Anomaly checks", "", md_table(ano), ""]
    lines += [f"`n_players` per frame across all files: {nplay}", ""]
    cnt = [
        c
        for c in los.columns
        if not c.startswith(("primary_in", "phases_in", "tracking_in", "track_"))
    ][1:]
    lines += [
        "### D. Possession-loss counts",
        "",
        md_table(los),
        "",
        f"Totals: {los[cnt].sum().to_dict()}",
        f"Means of agreement/lag columns: "
        f"{los.drop(columns=['match_id'] + cnt).mean().round(3).to_dict()}",
        "",
    ]
    lines += [
        "### E. Primary-loss episodes: censoring and coverage",
        "",
        f"- Episodes: {len(eps)}; with >=90% of first 10 s tracked: {len(usable)}",
        f"- Uncensored window length >= horizon (fraction of episodes): {hz}",
        f"- Censoring reason within 20 s: {eps['censor'].value_counts().to_dict()}",
        f"- Losing team's possession before the loss < 2 s: "
        f"{round(float((eps['a_possession_s'] < 2).mean()), 3)}",
        f"- Defending outfield detection fraction over first 10 s (quantiles 10/25/50/75/90):"
        f" {det_q.to_dict()}",
        "",
    ]
    lines += ["### F. Teams", "", md_table(team_tab), ""]
    lines += ["### G. Vocabularies (counts across all 20 games)", ""]
    for k, v in vocab.items():
        lines += [f"- **{k}**: {v}"]
    (ROOT / "docs" / "audit_generated.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
